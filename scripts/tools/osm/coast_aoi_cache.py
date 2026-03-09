#!/usr/bin/env python3
"""OSM/Coast AOI cache builder from GNSS track.

Goal
----
Given a slamboat-style YAML config (like configs/config_MOANA.yaml), read the GNSS track
(lat/lon/time) using the column definitions in the YAML, then:
  1) Build an AOI polygon (buffer around the track).
  2) Download OSM features (water polygons, coastline, buildings) for that AOI.
  3) Save to a GeoPackage in a cache directory.

Notes
-----
- Requires: geopandas, shapely, pyproj
- For OSM download: osmnx (recommended). If missing, the script will error with
  installation instructions.

Outputs
-------
- <cache_aoi_dir>/aoi_<YYYYmmdd_HHMMSS>_<epsg>.gpkg with layers:
    aoi, track, water, coastline, buildings
- <cache_aoi_dir>/aoi_<...>_meta.json

"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import yaml
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Polygon


# -----------------------------
# GNSS reading from YAML config
# -----------------------------

def _resolve_path(cfg_path: Path, p: str, data_root: Optional[str]) -> Path:
    cand1 = Path(p).expanduser()
    if cand1.exists():
        return cand1

    if data_root:
        cand2 = Path(data_root).expanduser() / p
        if cand2.exists():
            return cand2

    # common: config in configs/, data relative to repo root
    cand3 = cfg_path.parent.parent / p
    if cand3.exists():
        return cand3

    # fallback: return as-is (will error later)
    return cand1


def _infer_utm_crs(lat: float, lon: float) -> CRS:
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    if lat >= 0:
        return CRS.from_epsg(32600 + zone)
    return CRS.from_epsg(32700 + zone)


def _hemi_sign(h: str) -> float:
    hh = (h or "").strip().upper()
    if hh in ("S", "W"):
        return -1.0
    return 1.0


def read_gnss_track_from_config(cfg_path: Path, sensor_key: str = "gnss0") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t[s], lat[deg], lon[deg]) from YAML config file."""

    cfg = yaml.safe_load(cfg_path.read_text())
    data_root = str(cfg.get("paths", {}).get("data_root", "")) or None

    s = cfg.get("sensors", {}).get(sensor_key)
    if not s or s.get("kind") != "gnss":
        raise ValueError(f"Config has no sensors.{sensor_key} of kind 'gnss'.")

    fcfg = s.get("file", {})
    path = _resolve_path(cfg_path, str(fcfg.get("path")), data_root=data_root)
    if not path.exists():
        raise FileNotFoundError(f"GNSS file not found: {path}")

    delim = str(fcfg.get("delimiter", ","))
    if delim == "\\t":
        delim = "\t"

    has_header = bool(fcfg.get("has_header", True))
    cols = fcfg.get("columns", {})
    tcol = fcfg.get("time", {}).get("col")
    if tcol is None:
        raise ValueError("Config sensors.gnss0.file.time.col is required.")

    # required
    lat_col = cols.get("lat_deg", {}).get("col")
    lon_col = cols.get("lon_deg", {}).get("col")
    if lat_col is None or lon_col is None:
        raise ValueError("Config must define columns.lat_deg.col and columns.lon_deg.col")

    lat_hemi_col = cols.get("lat_hemi", {}).get("col")
    lon_hemi_col = cols.get("lon_hemi", {}).get("col")

    # fast-ish reader
    if has_header:
        import csv

        with path.open("r", newline="") as fp:
            rdr = csv.DictReader(fp, delimiter=delim)
            if rdr.fieldnames is None:
                raise ValueError(f"GNSS file has no header but config says has_header=true: {path}")
            fn = {k.strip(): k for k in rdr.fieldnames}

            def _col(k):
                if isinstance(k, int):
                    raise ValueError("When has_header=true, columns must be names, not indices.")
                return fn[str(k)]

            # time can be name
            tname = _col(tcol)
            latname = _col(lat_col)
            lonname = _col(lon_col)
            latH = _col(lat_hemi_col) if lat_hemi_col is not None else None
            lonH = _col(lon_hemi_col) if lon_hemi_col is not None else None

            tt, lat, lon = [], [], []
            for r in rdr:
                tt.append(float(r[tname]))
                la = float(r[latname])
                lo = float(r[lonname])
                if latH is not None:
                    la *= _hemi_sign(r.get(latH, "N"))
                if lonH is not None:
                    lo *= _hemi_sign(r.get(lonH, "E"))
                lat.append(la)
                lon.append(lo)

    else:
        import csv

        tt, lat, lon = [], [], []
        with path.open("r", newline="") as fp:
            rdr = csv.reader(fp, delimiter=delim)
            for row in rdr:
                if not row:
                    continue
                try:
                    t = float(row[int(tcol)])
                    la = float(row[int(lat_col)])
                    lo = float(row[int(lon_col)])
                except Exception:
                    continue

                if lat_hemi_col is not None:
                    la *= _hemi_sign(row[int(lat_hemi_col)])
                if lon_hemi_col is not None:
                    lo *= _hemi_sign(row[int(lon_hemi_col)])

                tt.append(t)
                lat.append(la)
                lon.append(lo)

    tt = np.asarray(tt, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    if tt.size < 5:
        raise ValueError(f"Too few GNSS samples read from {path} (n={tt.size}).")

    # sort by time
    idx = np.argsort(tt)
    return tt[idx], lat[idx], lon[idx]


# -----------------------------
# OSM download and cache
# -----------------------------

@dataclass
class AOIResult:
    aoi_gpkg: Path
    meta_json: Path
    epsg: int


def _make_aoi_polygon_utm(xy: np.ndarray, buffer_m: float) -> Polygon:
    track = LineString([(float(x), float(y)) for x, y in xy])
    # buffer around track then take envelope (keeps downloads bounded)
    return track.buffer(float(buffer_m)).envelope


def _to_gdf(geom, crs: CRS) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": [geom]}, crs=crs)


def build_aoi_cache(
    cfg_path: Path,
    cache_aoi_dir: Path,
    buffer_m: float = 1500.0,
    sensor_key: str = "gnss0",
    tags_water: Optional[List[Dict[str, object]]] = None,
    tags_coast: Optional[Dict[str, object]] = None,
    tags_buildings: Optional[Dict[str, object]] = None,
) -> AOIResult:
    """Create AOI + OSM layers cache from config-defined GNSS track."""

    t, lat, lon = read_gnss_track_from_config(cfg_path, sensor_key=sensor_key)
    utm = _infer_utm_crs(float(np.median(lat)), float(np.median(lon)))
    epsg = int(utm.to_epsg() or 0)
    if epsg == 0:
        raise ValueError("Failed to infer EPSG for UTM CRS.")

    tf = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
    xy = np.array([tf.transform(lo, la) for la, lo in zip(lat, lon)], dtype=np.float64)

    aoi_utm = _make_aoi_polygon_utm(xy, buffer_m=float(buffer_m))

    # Convert AOI polygon to lat/lon for osmnx query
    tf_inv = Transformer.from_crs(utm, "EPSG:4326", always_xy=True)
    aoi_latlon = Polygon([tf_inv.transform(x, y) for x, y in np.asarray(aoi_utm.exterior.coords)])

    # Tags
    if tags_water is None:
        tags_water = [
            {"natural": "water"},
            {"waterway": "riverbank"},
            {"landuse": "reservoir"},
            {"landuse": "basin"},
            {"natural": "bay"},
        ]
    if tags_coast is None:
        tags_coast = {"natural": "coastline"}
    if tags_buildings is None:
        tags_buildings = {"building": True}

    # osmnx is optional (but recommended)
    try:
        import osmnx as ox
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: osmnx. Install it with:\n"
            "  pip install osmnx\n"
            "or (conda-forge):\n"
            "  conda install -c conda-forge osmnx\n"
        ) from e

    # Download features
    water_frames = []
    for tag in tags_water:
        g = ox.features_from_polygon(aoi_latlon, tags=tag)
        if len(g) > 0:
            water_frames.append(g)
    g_water = gpd.GeoDataFrame(pd.concat(water_frames, axis=0), crs="EPSG:4326") if water_frames else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    g_coast = ox.features_from_polygon(aoi_latlon, tags=tags_coast)
    g_build = ox.features_from_polygon(aoi_latlon, tags=tags_buildings)

    # Reproject to UTM
    g_water = g_water.to_crs(utm)
    g_coast = g_coast.to_crs(utm)
    g_build = g_build.to_crs(utm)

    # Keep only geometry column to avoid huge attribute payload
    g_water = gpd.GeoDataFrame({"geometry": g_water.geometry}, crs=utm)
    g_coast = gpd.GeoDataFrame({"geometry": g_coast.geometry}, crs=utm)
    g_build = gpd.GeoDataFrame({"geometry": g_build.geometry}, crs=utm)

    # Track/AOI
    g_aoi = _to_gdf(aoi_utm, utm)
    g_track = gpd.GeoDataFrame({"t": t, "geometry": [LineString([(float(x), float(y)) for x, y in xy])]}, crs=utm)

    cache_aoi_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_gpkg = cache_aoi_dir / f"aoi_{stamp}_epsg{epsg}.gpkg"
    out_meta = cache_aoi_dir / f"aoi_{stamp}_epsg{epsg}_meta.json"

    # Write layers
    g_aoi.to_file(out_gpkg, layer="aoi", driver="GPKG")
    g_track.to_file(out_gpkg, layer="track", driver="GPKG")
    if len(g_water) > 0:
        g_water.to_file(out_gpkg, layer="water", driver="GPKG")
    if len(g_coast) > 0:
        g_coast.to_file(out_gpkg, layer="coastline", driver="GPKG")
    if len(g_build) > 0:
        g_build.to_file(out_gpkg, layer="buildings", driver="GPKG")

    meta = {
        "cfg": str(cfg_path),
        "sensor_key": sensor_key,
        "gnss_samples": int(t.size),
        "t_min": float(t.min()),
        "t_max": float(t.max()),
        "buffer_m": float(buffer_m),
        "epsg": epsg,
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    return AOIResult(aoi_gpkg=out_gpkg, meta_json=out_meta, epsg=epsg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--cache_aoi_dir", type=Path, default=Path(".OSM/cache_aoi"))
    ap.add_argument("--buffer_m", type=float, default=1500.0)
    ap.add_argument("--sensor_key", type=str, default="gnss0")

    args = ap.parse_args()

    res = build_aoi_cache(
        cfg_path=args.config,
        cache_aoi_dir=args.cache_aoi_dir,
        buffer_m=float(args.buffer_m),
        sensor_key=args.sensor_key,
    )

    print(f"[OK] AOI cache created:\n  gpkg: {res.aoi_gpkg}\n  meta: {res.meta_json}\n  epsg: {res.epsg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
