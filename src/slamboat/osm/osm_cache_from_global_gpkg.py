#!/usr/bin/env python3
"""
Optimized AOI cache builder from an *offline* global GPKG (water polygons + coastline)
and a GNSS track defined in a slamboat-style YAML config.

Fixes the common WSL "Killed" (OOM) symptom by:
  - computing AOI in UTM, then transforming AOI envelope back to EPSG:4326
  - using GeoPandas read_file(..., bbox=...) to load only features intersecting that bbox
  - only then reprojecting to UTM and performing exact clip

Expected global GPKG layers:
  - water_polygons (polygons)
  - coastline (lines)

Outputs:
  <cache_aoi_dir>/aoi_<stamp>_epsgXXXX.gpkg with layers: aoi, track, water, coastline
  <cache_aoi_dir>/aoi_<stamp>_epsgXXXX_meta.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import yaml
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Polygon, box


def _resolve_path(cfg_path: Path, p: str, data_root: Optional[str]) -> Path:
    cand1 = Path(p).expanduser()
    if cand1.exists():
        return cand1
    if data_root:
        cand2 = Path(data_root).expanduser() / p
        if cand2.exists():
            return cand2
    cand3 = cfg_path.parent.parent / p
    if cand3.exists():
        return cand3
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
        raise ValueError("Config sensors.<gnss>.file.time.col is required.")

    lat_col = cols.get("lat_deg", {}).get("col")
    lon_col = cols.get("lon_deg", {}).get("col")
    if lat_col is None or lon_col is None:
        raise ValueError("Config must define columns.lat_deg.col and columns.lon_deg.col")

    lat_hemi_col = cols.get("lat_hemi", {}).get("col")
    lon_hemi_col = cols.get("lon_hemi", {}).get("col")

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
                kk = str(k).strip()
                if kk not in fn:
                    raise KeyError(f"Column '{kk}' not found in GNSS header. Available: {list(fn.keys())}")
                return fn[kk]

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

    idx = np.argsort(tt)
    return tt[idx], lat[idx], lon[idx]


def _make_aoi_polygon_utm(xy: np.ndarray, buffer_m: float) -> Polygon:
    track = LineString([(float(x), float(y)) for x, y in xy])
    return track.buffer(float(buffer_m)).envelope


def _ensure_epsg4326(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    elif g.crs.to_string().upper() != "EPSG:4326":
        g = g.to_crs("EPSG:4326")
    return g


def _read_layer_bbox(path: Path, layer: str, bbox_ll: Tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    # bbox in EPSG:4326: (minx, miny, maxx, maxy)
    try:
        return gpd.read_file(path, layer=layer, bbox=bbox_ll, engine="pyogrio")
    except Exception:
        return gpd.read_file(path, layer=layer, bbox=bbox_ll)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--global_gpkg", type=Path, required=True)
    ap.add_argument("--cache_aoi_dir", type=Path, default=Path(".OSM/cache_aoi"))
    ap.add_argument("--buffer_m", type=float, default=1500.0)
    ap.add_argument("--sensor_key", type=str, default="gnss0")
    ap.add_argument("--water_layer", type=str, default="water_polygons")
    ap.add_argument("--coast_layer", type=str, default="coastline")
    args = ap.parse_args()

    t, lat, lon = read_gnss_track_from_config(args.config, sensor_key=args.sensor_key)
    utm = _infer_utm_crs(float(np.median(lat)), float(np.median(lon)))
    epsg = int(utm.to_epsg() or 0)
    if epsg == 0:
        raise ValueError("Failed to infer EPSG for UTM CRS.")

    tf_fwd = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
    tf_inv = Transformer.from_crs(utm, "EPSG:4326", always_xy=True)

    xy = np.array([tf_fwd.transform(lo, la) for la, lo in zip(lat, lon)], dtype=np.float64)
    aoi_utm = _make_aoi_polygon_utm(xy, buffer_m=float(args.buffer_m))

    # AOI bounds -> lat/lon bounds for cheap prefiltering at read-time
    minx, miny, maxx, maxy = aoi_utm.bounds
    lon_min, lat_min = tf_inv.transform(minx, miny)
    lon_max, lat_max = tf_inv.transform(maxx, maxy)
    bbox_ll = (min(lon_min, lon_max), min(lat_min, lat_max), max(lon_min, lon_max), max(lat_min, lat_max))

    # Read only intersecting features from global GPKG
    water_ll = _read_layer_bbox(args.global_gpkg, args.water_layer, bbox_ll)
    coast_ll = _read_layer_bbox(args.global_gpkg, args.coast_layer, bbox_ll)
    water_ll = _ensure_epsg4326(water_ll)
    coast_ll = _ensure_epsg4326(coast_ll)

    # Reproject to UTM
    water = water_ll.to_crs(utm) if len(water_ll) else gpd.GeoDataFrame({"geometry": []}, crs=utm)
    coast = coast_ll.to_crs(utm) if len(coast_ll) else gpd.GeoDataFrame({"geometry": []}, crs=utm)

    # Fast bbox filter in UTM, then exact clip
    aoi_box = box(*aoi_utm.bounds)
    if len(water):
        water = water[water.intersects(aoi_box)]
        if len(water):
            water = gpd.clip(water, aoi_utm)
    if len(coast):
        coast = coast[coast.intersects(aoi_box)]
        if len(coast):
            coast = gpd.clip(coast, aoi_utm)

    water = gpd.GeoDataFrame({"geometry": water.geometry}, crs=utm) if len(water) else gpd.GeoDataFrame({"geometry": []}, crs=utm)
    coast = gpd.GeoDataFrame({"geometry": coast.geometry}, crs=utm) if len(coast) else gpd.GeoDataFrame({"geometry": []}, crs=utm)

    g_aoi = gpd.GeoDataFrame({"geometry": [aoi_utm]}, crs=utm)
    g_track = gpd.GeoDataFrame(
        {"t_min": [float(t.min())], "t_max": [float(t.max())],
         "geometry": [LineString([(float(x), float(y)) for x, y in xy])]},
        crs=utm
    )

    args.cache_aoi_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_gpkg = args.cache_aoi_dir / f"aoi_{stamp}_epsg{epsg}.gpkg"
    out_meta = args.cache_aoi_dir / f"aoi_{stamp}_epsg{epsg}_meta.json"

    g_aoi.to_file(out_gpkg, layer="aoi", driver="GPKG")
    g_track.to_file(out_gpkg, layer="track", driver="GPKG")
    if len(water):
        water.to_file(out_gpkg, layer="water", driver="GPKG")
    if len(coast):
        coast.to_file(out_gpkg, layer="coastline", driver="GPKG")

    meta = {
        "cfg": str(args.config),
        "global_gpkg": str(args.global_gpkg),
        "sensor_key": args.sensor_key,
        "gnss_samples": int(t.size),
        "buffer_m": float(args.buffer_m),
        "epsg": epsg,
        "bbox_ll": [float(v) for v in bbox_ll],
        "layers": {"water": int(len(water)), "coastline": int(len(coast))},
        "layer_names": {"water_layer": args.water_layer, "coast_layer": args.coast_layer},
    }
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"[OK] AOI cache created:\n  gpkg: {out_gpkg}\n  meta: {out_meta}\n  epsg: {epsg}\n  bbox_ll: {bbox_ll}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
