#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Tuple, Optional, Iterable

import cv2
import numpy as np
import open3d as o3d

import json

def dump_cfar_npz(out_dir: Path, stamp: float, band: str, *,
                 I_norm: np.ndarray,
                 mask: np.ndarray,
                 weight: np.ndarray,
                 mask_persist: Optional[np.ndarray] = None,
                 water_mask: Optional[np.ndarray] = None,
                 baseline_per_range: Optional[np.ndarray] = None,
                 meta: Optional[Dict] = None) -> None:
    """Write CFAR outputs to a compressed NPZ + sidecar JSON.

    Arrays are saved as:
      - I_norm: float32 (H,W)
      - mask: uint8 (H,W)  (0/1)
      - weight: float32 (H,W)
      - mask_persist: uint8 (H,W) optional
      - water_mask: uint8 (H,W) optional
      - baseline_per_range: float32 (W,) optional
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{band}_{stamp:.6f}"
    npz_path = out_dir / f"cfar_{tag}.npz"
    js_path  = out_dir / f"cfar_{tag}.json"

    arrays: Dict[str, np.ndarray] = {
        "I_norm": I_norm.astype(np.float32, copy=False),
        "mask": mask.astype(np.uint8, copy=False),
        "weight": weight.astype(np.float32, copy=False),
    }
    if mask_persist is not None:
        arrays["mask_persist"] = mask_persist.astype(np.uint8, copy=False)
    if water_mask is not None:
        arrays["water_mask"] = water_mask.astype(np.uint8, copy=False)
    if baseline_per_range is not None:
        arrays["baseline_per_range"] = baseline_per_range.astype(np.float32, copy=False)

    np.savez_compressed(npz_path, **arrays)

    meta = dict(meta or {})
    meta.update({
        "stamp": float(stamp),
        "band": str(band),
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
        "dtypes": {k: str(v.dtype) for k, v in arrays.items()},
    })
    js_path.write_text(json.dumps(meta, indent=2))

import yaml
from PIL import Image
from scipy.signal import wiener

# OSM / geo stack (available in the project environment)
import geopandas as gpd
from pyproj import CRS, Transformer
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.strtree import STRtree
from shapely import affinity


class PolarToCartesianCV2:
    """
    Remap polar image I(H,W) [rows=azimuth, cols=range] to cartesian grid (G,G).
    Convention:
      x = r*sin(theta), y = r*cos(theta), theta=0 at +Y
    """

    def __init__(
        self,
        H: int,
        W: int,
        R_max_m: float,
        grid_size: int,
        azimuth_ccw: bool = False,
        az_offset_deg: float = 0.0,
        mirror_x: bool = False,
    ):
        self.H = int(H)
        self.W = int(W)
        self.R = float(R_max_m)
        self.G = int(grid_size)
        self.azimuth_ccw = bool(azimuth_ccw)
        self.az_offset = float(az_offset_deg)
        self.mirror_x = bool(mirror_x)

        x_lin = np.linspace(-self.R, self.R, self.G, dtype=np.float32)
        y_lin = np.linspace(-self.R, self.R, self.G, dtype=np.float32)
        X, Y = np.meshgrid(x_lin, y_lin)

        r = np.sqrt(X**2 + Y**2, dtype=np.float32)
        theta = np.arctan2(X, Y).astype(np.float32)  # 0 at +Y

        if not self.azimuth_ccw:
            theta = (2.0 * np.pi) - theta

        theta = theta + np.deg2rad(self.az_offset).astype(np.float32)
        theta = np.mod(theta, 2.0 * np.pi)

        # Guard origin (avoid seam/singularity at r≈0 where atan2 is ill-defined)
        theta[r < 1e-6] = 0.0

        az_idx = (theta / (2.0 * np.pi)) * self.H - 0.5
        dr = self.R / self.W
        rad_idx = (r / dr) - 0.5

        self.map_x = np.clip(rad_idx, 0, self.W - 1).astype(np.float32)  # src col
        self.map_y = np.mod(az_idx, self.H).astype(np.float32)  # src row
        self.outside = r > self.R

    def remap(self, I_polar: np.ndarray, interp: int) -> np.ndarray:
        src = I_polar.astype(np.float32, copy=False)
        cart = cv2.remap(
            src,
            self.map_x,
            self.map_y,
            interpolation=interp,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        cart[self.outside] = 0.0
        if self.mirror_x:
            cart = np.ascontiguousarray(np.fliplr(cart))
        return cart


def _to_seconds_from_name(p: Path) -> float:
    m = re.search(r"(\d+)", p.stem)
    if not m:
        raise ValueError(f"Cannot parse timestamp from filename: {p.name}")
    t = float(m.group(1))
    # heuristic (ns/us/ms/s)
    if t > 1e17:
        return t / 1e9
    if t > 1e14:
        return t / 1e6
    if t > 1e11:
        return t / 1e3
    return t


def _deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _unwrap_angles_rad(a: np.ndarray) -> np.ndarray:
    return np.unwrap(a.astype(np.float64)).astype(np.float64)


def load_polar_png(path: Path, H: int, W: int) -> np.ndarray:
    """Returns polar image as float32 with shape (H, W): rows=azimuth, cols=range."""
    I = np.array(Image.open(path).convert("L"), dtype=np.float32)
    h, w = I.shape
    if (h, w) == (W, H):
        I = I.T
        h, w = I.shape
    if (h, w) != (H, W):
        I = np.array(Image.fromarray(I.astype(np.uint8)).resize((W, H), Image.BILINEAR), dtype=np.float32)
    return I


def _robust_norm01(I: np.ndarray, p_lo: float = 5.0, p_hi: float = 95.0) -> np.ndarray:
    lo = float(np.percentile(I, p_lo))
    hi = float(np.percentile(I, p_hi))
    J = (I - lo) / max(hi - lo, 1e-6)
    return np.clip(J, 0.0, 1.0).astype(np.float32)


def range_comp(I: np.ndarray, R_max_m: float, gamma: float) -> np.ndarray:
    _, W = I.shape
    r = (np.arange(W, dtype=np.float32) + 0.5) / float(W)
    gain = np.power(np.clip(r, 1e-6, 1.0), gamma)
    return I * gain[None, :]


def log_compress(I: np.ndarray, alpha: float) -> np.ndarray:
    return np.log1p(alpha * np.clip(I, 0.0, None))


def ca_cfar_1d(x: np.ndarray, training: int, guard: int, k: float) -> np.ndarray:
    """CA-CFAR (additive) along range: thr[r] = mu + k*sigma."""
    W = x.shape[0]
    T = int(training)
    G = int(guard)

    c1 = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    c2 = np.concatenate([[0.0], np.cumsum((x * x), dtype=np.float64)])
    thr = np.zeros(W, dtype=np.float32)

    for r in range(W):
        l0 = r - (G + T)
        l1 = r - G
        r0 = r + G + 1
        r1 = r + G + 1 + T

        l0c = max(0, l0)
        l1c = max(0, l1)
        r0c = min(W, r0)
        r1c = min(W, r1)

        n = 0
        s1 = 0.0
        s2 = 0.0
        if l1c > l0c:
            s1 += c1[l1c] - c1[l0c]
            s2 += c2[l1c] - c2[l0c]
            n += (l1c - l0c)
        if r1c > r0c:
            s1 += c1[r1c] - c1[r0c]
            s2 += c2[r1c] - c2[r0c]
            n += (r1c - r0c)

        if n > 1:
            mu = s1 / n
            var = max(s2 / n - mu * mu, 0.0)
            sigma = math.sqrt(var)
            thr[r] = float(mu + k * sigma)
        else:
            thr[r] = 0.0

    return thr


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def build_cfar_mask_and_weight(
    I: np.ndarray, training: int, guard: int, k: float, soft_tau: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-azimuth CA-CFAR along range; returns (mask, weight in [0,1])."""
    H, W = I.shape
    mask = np.zeros((H, W), dtype=bool)
    w = np.zeros((H, W), dtype=np.float32)
    tau = max(float(soft_tau), 1e-6)

    for a in range(H):
        x = I[a, :]
        thr = ca_cfar_1d(x, training=training, guard=guard, k=k)
        d = x - thr
        mask[a, :] = d > 0.0
        w[a, :] = sigmoid(d / tau).astype(np.float32)

    return mask, w


def polar_indices_to_points(
    mask: np.ndarray,
    weight: np.ndarray,
    *,
    R_max_m: float,
    azimuth_ccw: bool,
    az_offset_deg: float,
    min_range_m: float,
    max_range_m: float,
    max_points: int,
    mirror_x: bool,
) -> np.ndarray:
    """Convert selected polar bins -> (x,y,0) points. Picks up to max_points by weight."""
    H, W = mask.shape
    dr = float(R_max_m) / float(W)

    idx = np.argwhere(mask)
    if idx.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    scores = weight[idx[:, 0], idx[:, 1]]
    order = np.argsort(scores)[::-1]

    pts: List[Tuple[float, float, float]] = []
    for k in order:
        a = int(idx[k, 0])
        r = int(idx[k, 1])

        rng = (r + 0.5) * dr
        if rng < min_range_m or rng > max_range_m:
            continue

        theta = (a + 0.5) * (2.0 * math.pi / float(H))
        if not azimuth_ccw:
            theta = (2.0 * math.pi) - theta
        theta = (theta + _deg2rad(az_offset_deg)) % (2.0 * math.pi)

        x = rng * math.sin(theta)
        y = rng * math.cos(theta)
        if mirror_x:
            x = -x

        pts.append((x, y, 0.0))
        if len(pts) >= int(max_points):
            break

    if not pts:
        return np.zeros((0, 3), dtype=np.float64)

    return np.array(pts, dtype=np.float64)


def icp_2d(src_pts: np.ndarray, tgt_pts: np.ndarray, max_corr: float, max_iters: int) -> Tuple[np.ndarray, float, float]:
    """Returns (T_4x4, fitness, rmse). Points are (N,3) with z=0."""
    src = o3d.geometry.PointCloud()
    tgt = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(src_pts)
    tgt.points = o3d.utility.Vector3dVector(tgt_pts)

    result = o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        max_correspondence_distance=float(max_corr),
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iters)),
    )
    return result.transformation, float(result.fitness), float(result.inlier_rmse)


def T_to_dxdy_dyaw(T: np.ndarray) -> Tuple[float, float, float]:
    tx = float(T[0, 3])
    ty = float(T[1, 3])
    yaw = math.atan2(float(T[1, 0]), float(T[0, 0]))
    return tx, ty, _wrap_pi(yaw)


@dataclass(frozen=True)
class Pose2D:
    t: float
    x: float
    y: float
    yaw: float  # rad


def _infer_utm_crs(lat: float, lon: float) -> CRS:
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    if lat >= 0:
        return CRS.from_epsg(32600 + zone)
    return CRS.from_epsg(32700 + zone)


def read_gnss_csv(path: Path) -> List[Tuple[float, float, float, Optional[float]]]:
    """Read GNSS CSV with at least (t, lat, lon). Optional yaw/course degrees."""
    rows: List[Tuple[float, float, float, Optional[float]]] = []
    with path.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        if rdr.fieldnames is None:
            raise ValueError(f"GNSS CSV has no header: {path}")
        cols = {c.strip().lower(): c for c in rdr.fieldnames}

        def pick(*names: str) -> Optional[str]:
            for n in names:
                if n in cols:
                    return cols[n]
            return None

        c_t = pick("t", "time", "timestamp", "stamp", "sec", "secs")
        c_lat = pick("lat", "latitude")
        c_lon = pick("lon", "longitude", "lng")
        c_yaw = pick("yaw", "course", "cog", "heading")
        if c_t is None or c_lat is None or c_lon is None:
            raise ValueError(
                f"GNSS CSV must include time+lat+lon. Found: {rdr.fieldnames}. "
                "Expected columns like t/time/timestamp, lat, lon."
            )

        for r in rdr:
            t = float(r[c_t])
            lat = float(r[c_lat])
            lon = float(r[c_lon])
            yaw_deg = float(r[c_yaw]) if (c_yaw is not None and r.get(c_yaw, "") != "") else None
            rows.append((t, lat, lon, yaw_deg))

    rows.sort(key=lambda z: z[0])
    return rows


def poses_from_gnss(rows: List[Tuple[float, float, float, Optional[float]]]) -> List[Pose2D]:
    if len(rows) < 2:
        raise ValueError("Need >=2 GNSS samples to derive yaw (or provide yaw column).")

    t0, lat0, lon0, _ = rows[0]
    utm = _infer_utm_crs(lat0, lon0)
    tf = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
    xy = np.array([tf.transform(lon, lat) for (_, lat, lon, _) in rows], dtype=np.float64)
    tt = np.array([t for (t, _, _, _) in rows], dtype=np.float64)
    # localize
    xy = xy - xy[0:1, :]

    yaw_deg_arr = np.array([y if y is not None else np.nan for (_, _, _, y) in rows], dtype=np.float64)
    if np.isfinite(yaw_deg_arr).any():
        yaw_rad = np.deg2rad(np.where(np.isfinite(yaw_deg_arr), yaw_deg_arr, np.nan))
        # fill missing by course from finite difference
        dx = np.gradient(xy[:, 0])
        dy = np.gradient(xy[:, 1])
        course = np.arctan2(dy, dx)
        yaw_rad = np.where(np.isfinite(yaw_rad), yaw_rad, course)
    else:
        dx = np.gradient(xy[:, 0])
        dy = np.gradient(xy[:, 1])
        yaw_rad = np.arctan2(dy, dx)

    yaw_rad = _unwrap_angles_rad(yaw_rad)
    poses = [Pose2D(t=float(tt[i]), x=float(xy[i, 0]), y=float(xy[i, 1]), yaw=float(yaw_rad[i])) for i in range(len(rows))]
    return poses


def interp_pose(poses: List[Pose2D], t: float) -> Pose2D:
    if t <= poses[0].t:
        return poses[0]
    if t >= poses[-1].t:
        return poses[-1]

    # binary search
    lo, hi = 0, len(poses) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if poses[mid].t <= t:
            lo = mid
        else:
            hi = mid

    p0, p1 = poses[lo], poses[hi]
    a = (t - p0.t) / max(p1.t - p0.t, 1e-9)
    x = (1 - a) * p0.x + a * p1.x
    y = (1 - a) * p0.y + a * p1.y
    yaw = (1 - a) * p0.yaw + a * p1.yaw
    return Pose2D(t=float(t), x=float(x), y=float(y), yaw=float(yaw))


@dataclass
class Sim2:
    s: float = 1.0
    yaw: float = 0.0  # rad
    tx: float = 0.0
    ty: float = 0.0

    @staticmethod
    def from_json(path: Path) -> "Sim2":
        d = json.loads(path.read_text())
        # accept a few common key spellings
        s = float(d.get("s", d.get("scale", 1.0)))
        yaw_deg = d.get("yaw_deg", d.get("yaw", 0.0))
        yaw = float(_deg2rad(float(yaw_deg))) if abs(float(yaw_deg)) > 2 * math.pi else float(yaw_deg)
        tx = float(d.get("tx_m", d.get("tx", 0.0)))
        ty = float(d.get("ty_m", d.get("ty", 0.0)))
        return Sim2(s=s, yaw=yaw, tx=tx, ty=ty)

    def apply_geom(self, g):
        # scale+rotate about origin, then translate
        out = affinity.scale(g, xfact=self.s, yfact=self.s, origin=(0.0, 0.0))
        out = affinity.rotate(out, angle=math.degrees(self.yaw), origin=(0.0, 0.0))
        out = affinity.translate(out, xoff=self.tx, yoff=self.ty)
        return out


class OSMRadarMasker:
    """Use OSM water polygons to build a water mask in POLAR (cheap) and estimate sea-clutter noise."""

    def __init__(
        self,
        water_union_geom,
        buildings_geoms: Optional[List] = None,
        *,
        H: int,
        W: int,
        R_max_m: float,
        azimuth_ccw: bool,
        az_offset_deg: float,
        mirror_x: bool,
        water_margin_m: float,
        water_min_range_m: float,
    ):
        self.H = int(H)
        self.W = int(W)
        self.R_max = float(R_max_m)
        self.dr = float(R_max_m) / float(W)
        self.azimuth_ccw = bool(azimuth_ccw)
        self.az_offset = float(az_offset_deg)
        self.mirror_x = bool(mirror_x)
        self.water_margin = float(water_margin_m)
        self.water_min_range = float(water_min_range_m)

        self._water_geom = water_union_geom
        self._water_is_polygon = self._water_geom.geom_type in {"Polygon", "MultiPolygon"}
        self._water_prepared = prep(self._water_geom) if self._water_is_polygon else None

        self._buildings: List = buildings_geoms or []
        self._b_tree: Optional[STRtree] = STRtree(self._buildings) if self._buildings else None

        # Precompute unit rays in RADAR frame for each azimuth row (consistent with polar_indices_to_points)
        theta = (np.arange(self.H, dtype=np.float64) + 0.5) * (2.0 * np.pi / float(self.H))
        if not self.azimuth_ccw:
            theta = (2.0 * np.pi) - theta
        theta = (theta + _deg2rad(self.az_offset)) % (2.0 * np.pi)
        ux = np.sin(theta)
        uy = np.cos(theta)
        if self.mirror_x:
            ux = -ux
        self._u_radar = np.stack([ux, uy], axis=1)  # (H,2)

        self._ranges = (np.arange(self.W, dtype=np.float64) + 0.5) * self.dr

    def _inside_water(self, x: float, y: float) -> bool:
        if not self._water_is_polygon:
            raise RuntimeError("_inside_water called but water geometry is not a polygon")
        # covers() is more stable near boundaries than contains()
        return bool(self._water_prepared.covers(Point(float(x), float(y))))

    def water_front_ranges(self, pose_w: Pose2D, n_iter: int = 12) -> np.ndarray:
        """For each azimuth, return first range (meters) where we leave water (approx).

        Two modes:
          - water polygon available: binary search on point-in-polygon (fast)
          - coastline lines only: direct ray/lines intersection (slower but works)
        """
        c = math.cos(pose_w.yaw)
        s = math.sin(pose_w.yaw)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        u_w = (R @ self._u_radar.T).T  # (H,2)

        x0, y0 = pose_w.x, pose_w.y
        inside0 = self._inside_water(x0, y0) if self._water_is_polygon else True  # assume vessel starts on water

        front = np.empty(self.H, dtype=np.float64)
        for i in range(self.H):
            dx, dy = float(u_w[i, 0]), float(u_w[i, 1])
            xF = x0 + self.R_max * dx
            yF = y0 + self.R_max * dy
            if self._water_is_polygon:
                insideF = self._inside_water(xF, yF)

                if inside0 == insideF:
                    front[i] = self.R_max if inside0 else 0.0
                    continue

                lo, hi = 0.0, self.R_max
                for _ in range(int(n_iter)):
                    mid = 0.5 * (lo + hi)
                    xm = x0 + mid * dx
                    ym = y0 + mid * dy
                    if self._inside_water(xm, ym) == inside0:
                        lo = mid
                    else:
                        hi = mid
                front[i] = hi
            else:
                # Coastline-only: ray intersection with (Multi)LineString.
                ray = LineString([(x0, y0), (xF, yF)])
                inter = ray.intersection(self._water_geom)

                # Extract candidate points from any intersection geometry
                pts: List[Tuple[float, float]] = []
                gt = inter.geom_type
                if gt == "Point":
                    pts = [(inter.x, inter.y)]
                elif gt == "MultiPoint":
                    pts = [(g.x, g.y) for g in inter.geoms]
                elif gt == "GeometryCollection":
                    for g in inter.geoms:
                        if g.geom_type == "Point":
                            pts.append((g.x, g.y))
                        elif g.geom_type in {"LineString", "MultiLineString"}:
                            # overlap: take the nearest coordinate on the overlap
                            if g.geom_type == "LineString":
                                xs, ys = g.coords[0]
                                pts.append((xs, ys))
                            else:
                                for gg in g.geoms:
                                    xs, ys = gg.coords[0]
                                    pts.append((xs, ys))
                elif gt in {"LineString", "MultiLineString"}:
                    # overlap: take nearest start
                    if gt == "LineString":
                        xs, ys = inter.coords[0]
                        pts = [(xs, ys)]
                    else:
                        pts = [(g.coords[0][0], g.coords[0][1]) for g in inter.geoms]

                if not pts:
                    front[i] = self.R_max
                else:
                    # pick nearest intersection along the ray
                    dmin = self.R_max
                    for (xp, yp) in pts:
                        d = math.hypot(float(xp) - x0, float(yp) - y0)
                        if 1e-3 < d < dmin:
                            dmin = d
                    front[i] = dmin

        return front

    def water_mask_polar(self, pose_w: Pose2D) -> np.ndarray:
        front = self.water_front_ranges(pose_w)
        # conservative margin: keep a buffer around shoreline out of the water ROI
        front_eff = np.maximum(front - self.water_margin, 0.0)
        wmask = self._ranges[None, :] <= front_eff[:, None]
        # also exclude very near range from water ROI (ring artifacts / near-field)
        wmask &= (self._ranges[None, :] >= self.water_min_range)
        return wmask

    def estimate_water_baseline_and_noise(self, I_raw: np.ndarray, wmask: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        """Return (baseline_per_range[W], robust_noise_var or None)."""
        # baseline per range from water-only bins
        J = np.where(wmask, I_raw, np.nan)
        baseline = np.nanmedian(J, axis=0)
        idx = np.where(np.isfinite(baseline))[0]
        if idx.size >= 2:
            baseline = np.interp(np.arange(self.W), idx, baseline[idx]).astype(np.float32)
        else:
            baseline = np.median(I_raw, axis=0).astype(np.float32)

        # residual variance on water after baseline removal
        I_res = np.clip(I_raw - baseline[None, :], 0.0, None)
        vv = I_res[wmask]
        if vv.size < 500:
            return baseline, None
        med = float(np.median(vv))
        mad = float(np.median(np.abs(vv - med)))
        sigma = 1.4826 * mad
        var = float(max(sigma * sigma, 0.0))
        return baseline, var

    def filter_points_by_buildings(self, pts_radar: np.ndarray, pose_w: Pose2D, max_dist_m: float) -> np.ndarray:
        if self._b_tree is None or pts_radar.shape[0] == 0:
            return pts_radar

        c = math.cos(pose_w.yaw)
        s = math.sin(pose_w.yaw)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        xy_w = (R @ pts_radar[:, 0:2].T).T + np.array([pose_w.x, pose_w.y])[None, :]

        keep = np.zeros((xy_w.shape[0],), dtype=bool)
        for i in range(xy_w.shape[0]):
            p = Point(float(xy_w[i, 0]), float(xy_w[i, 1]))
            g_near = self._b_tree.nearest(p)
            d = float(p.distance(g_near))
            keep[i] = d <= float(max_dist_m)

        # avoid collapsing the cloud completely
        if int(keep.sum()) < 50:
            return pts_radar
        return pts_radar[keep, :]


def load_osm_union(
    gpkg: Path,
    *,
    water_layers: List[str],
    buildings_layers: List[str],
    target_crs: CRS,
    sim2: Sim2,
) -> Tuple:
    water_geoms: List = []
    for lyr in water_layers:
        gdf = gpd.read_file(gpkg, layer=lyr)
        if gdf.empty:
            continue
        if gdf.crs is None:
            raise ValueError(f"Layer {lyr} in {gpkg} has no CRS")
        gdf = gdf.to_crs(target_crs)
        gdf["geometry"] = gdf["geometry"].apply(sim2.apply_geom)
        water_geoms.extend([g for g in gdf.geometry.values if g is not None and not g.is_empty])
    if not water_geoms:
        raise ValueError(f"No geometries loaded from water_layers={water_layers} in {gpkg}")

    water_union = unary_union(water_geoms)

    buildings: List = []
    for lyr in buildings_layers:
        gdf = gpd.read_file(gpkg, layer=lyr)
        if gdf.empty:
            continue
        if gdf.crs is None:
            raise ValueError(f"Layer {lyr} in {gpkg} has no CRS")
        gdf = gdf.to_crs(target_crs)
        gdf["geometry"] = gdf["geometry"].apply(sim2.apply_geom)
        buildings.extend([g for g in gdf.geometry.values if g is not None and not g.is_empty])

    return water_union, buildings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/config_MOANA.yaml"))
    ap.add_argument("--input_dir", type=Path, required=True, help="Folder with polar PNGs for ONE band.")
    ap.add_argument("--band", choices=["xband", "wband"], required=True)
    ap.add_argument("--output_csv", type=Path, default=Path("outputs/radar_delta.csv"))
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--debug_dir", type=Path, default=Path(""))
    ap.add_argument("--debug_n", type=int, default=1)
    ap.add_argument("--cfar_out_dir", type=Path, default=Path(""), help="If set (non-empty), dump CFAR arrays (*.npz + *.json) per frame.")
    ap.add_argument("--cfar_only", action="store_true", help="Only compute and dump CFAR results (skip ICP/CSV).")

    # Optional OSM-guided water masking (for sea clutter / noise estimation)
    ap.add_argument("--gnss_csv", type=Path, default=None, help="CSV with t,lat,lon[,yaw/course].")
    ap.add_argument("--osm_gpkg", type=Path, default=None, help="AOI GeoPackage with water/buildings layers.")
    ap.add_argument(
        "--osm_water_layers",
        type=str,
        default="water,coastline",
        help="Comma-separated layer names to treat as water polygons/lines.",
    )
    ap.add_argument(
        "--osm_buildings_layers",
        type=str,
        default="buildings",
        help="Comma-separated layer names to treat as buildings polygons (optional).",
    )
    ap.add_argument("--osm_sim2_json", type=Path, default=None, help="Optional Sim(2) calibration JSON.")
    ap.add_argument("--water_margin_m", type=float, default=8.0)
    ap.add_argument("--water_min_range_m", type=float, default=20.0)
    ap.add_argument("--filter_by_buildings", action="store_true")
    ap.add_argument("--buildings_max_dist_m", type=float, default=20.0)

    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    rp = cfg.get("radar_processing", {})

    H = int(rp.get("H", 400))
    W = int(rp.get("W", 3424))
    R_max_m = float(rp.get("R_max_m", 600.0))
    azimuth_ccw = bool(rp.get("azimuth_ccw", False))
    az_offset_deg = float(rp.get("az_offset_deg", 0.0))
    mirror_x = bool(rp.get("mirror_x", False))

    grid_size = int(rp.get("grid_size", 600))
    remap_cart = PolarToCartesianCV2(
        H=H,
        W=W,
        R_max_m=R_max_m,
        grid_size=grid_size,
        azimuth_ccw=azimuth_ccw,
        az_offset_deg=az_offset_deg,
        mirror_x=mirror_x,
    )

    rc = rp.get("range_comp", {})
    rc_enabled = bool(rc.get("enabled", True))
    rc_gamma = float(rc.get("gamma", 1.5))

    lc = rp.get("log_compress", {})
    lc_enabled = bool(lc.get("enabled", True))
    lc_alpha = float(lc.get("alpha", 1.0))

    # Wiener in POLAR before CFAR
    wien = rp.get("wiener", {})
    wien_enable = bool(wien.get("enabled", True))
    wien_mysize = tuple(wien.get("mysize", (5, 21)))
    wien_noise_default = wien.get("noise", None)  # can be overridden by OSM-estimated noise

    cfar = rp.get("cfar", {})
    training = int(cfar.get("training", 24))
    guard = int(cfar.get("guard", 6))
    k = float(cfar.get("k", 1.6))
    soft_tau = float(cfar.get("soft_tau", 0.15))

    pers = rp.get("persistence", {})
    pers_enabled = bool(pers.get("enabled", True))
    K = int(pers.get("K", 5))
    min_hits = int(pers.get("min_hits", 3))

    peaks = rp.get("peaks", {})
    max_points = int(peaks.get("max_points", 2000))
    min_range_m = float(peaks.get("min_range_m", 10.0))
    max_range_m = float(peaks.get("max_range_m", R_max_m * 0.92))

    icp = rp.get("icp", {})
    max_corr = float(icp.get("max_corr_dist_m", 3.0))
    max_iters = int(icp.get("max_iters", 50))

    pngs = sorted(args.input_dir.glob("*.png"))
    if not pngs:
        raise ValueError(f"No png files found in {args.input_dir}")

    items = [(_to_seconds_from_name(p), p) for p in pngs]
    items.sort(key=lambda x: x[0])
    items = items[:: max(1, int(args.stride))]
    if int(args.max_frames) > 0:
        items = items[: int(args.max_frames)]

    out = args.output_csv
    out.parent.mkdir(parents=True, exist_ok=True)

    # Optional OSM-guided mask engine
    poses_w: Optional[List[Pose2D]] = None
    masker: Optional[OSMRadarMasker] = None
    if args.gnss_csv is not None and args.osm_gpkg is not None:
        gnss_rows = read_gnss_csv(args.gnss_csv)
        poses_w = poses_from_gnss(gnss_rows)

        # CRS used by poses_from_gnss is UTM inferred from first gnss row
        _, lat0, lon0, _ = gnss_rows[0]
        utm = _infer_utm_crs(lat0, lon0)

        sim2 = Sim2()
        if args.osm_sim2_json is not None:
            sim2 = Sim2.from_json(args.osm_sim2_json)

        water_layers = [s.strip() for s in str(args.osm_water_layers).split(",") if s.strip()]
        buildings_layers = [s.strip() for s in str(args.osm_buildings_layers).split(",") if s.strip()]
        water_union, buildings = load_osm_union(
            args.osm_gpkg,
            water_layers=water_layers,
            buildings_layers=buildings_layers,
            target_crs=utm,
            sim2=sim2,
        )

        masker = OSMRadarMasker(
            water_union,
            buildings_geoms=buildings,
            H=H,
            W=W,
            R_max_m=R_max_m,
            azimuth_ccw=azimuth_ccw,
            az_offset_deg=az_offset_deg,
            mirror_x=mirror_x,
            water_margin_m=float(args.water_margin_m),
            water_min_range_m=float(args.water_min_range_m),
        )

        print(
            f"[OSM] enabled: gpkg={args.osm_gpkg} water_layers={water_layers} buildings_layers={buildings_layers} "
            f"sim2=(s={sim2.s:.3f}, yaw_deg={math.degrees(sim2.yaw):.2f}, tx={sim2.tx:.2f}, ty={sim2.ty:.2f})"
        )

    mask_hist: Deque[np.ndarray] = deque(maxlen=max(1, K))

    def preprocess_one(t: float, path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        I_raw = load_polar_png(path, H=H, W=W)
        I = I_raw.copy()

        wmask = None
        baseline = None
        wien_noise = wien_noise_default

        if poses_w is not None and masker is not None:
            pose = interp_pose(poses_w, t)
            wmask = masker.water_mask_polar(pose)
            baseline, var = masker.estimate_water_baseline_and_noise(I_raw, wmask)
            I = np.clip(I - baseline[None, :], 0.0, None)
            if var is not None:
                wien_noise = var

        # Wiener2D (adaptive) on polar image BEFORE CFAR
        if wien_enable:
            I = wiener(I, mysize=wien_mysize, noise=wien_noise)
            I = np.nan_to_num(I, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        if rc_enabled:
            I = range_comp(I, R_max_m=R_max_m, gamma=rc_gamma)
        if lc_enabled:
            I = log_compress(I, alpha=lc_alpha)

        # robust normalize
        I = I - np.percentile(I, 5.0)
        I = np.clip(I, 0.0, None)
        p95 = np.percentile(I, 95.0)
        if p95 > 0:
            I_norm = np.clip(I / p95, 0.0, 1.0).astype(np.float32)
        else:
            I_norm = I.astype(np.float32)

        mask, w = build_cfar_mask_and_weight(I_norm, training=training, guard=guard, k=k, soft_tau=soft_tau)

        # hard remove water candidates (keep only land/structures)
        if wmask is not None:
            mask &= ~wmask
            # also damp weights over water to avoid point picking if mask is later softened
            w = np.where(wmask, 0.0, w)

        return I_raw, I_norm, mask, w, wmask, baseline

    dbg_dir = args.debug_dir if str(args.debug_dir) != "" else None
    cfar_dir = args.cfar_out_dir if str(args.cfar_out_dir) != "" else None
    dumped = 0


    if args.cfar_only:
        dumped = 0
        for (t, p) in items:
            Iraw, In, m, w, wmask, baseline = preprocess_one(t, p)
            mask_hist.append(m)

            mp = None
            if pers_enabled:
                stack = np.stack(list(mask_hist), axis=0)
                mp = (np.sum(stack, axis=0) >= min_hits)

            if dbg_dir is not None and dumped < int(args.debug_n):
                tag = f"{args.band}_{t:.3f}"
                debug_dir = Path(dbg_dir)
                debug_dir.mkdir(parents=True, exist_ok=True)
                Iraw_cart = remap_cart.remap(_robust_norm01(Iraw), interp=cv2.INTER_LINEAR)
                Image.fromarray((Iraw_cart * 255.0).astype(np.uint8)).save(debug_dir / f"cart_raw_{tag}.png")
                I_cart = remap_cart.remap(In, interp=cv2.INTER_LINEAR)
                m_cart = remap_cart.remap(m.astype(np.float32), interp=cv2.INTER_NEAREST) > 0.5
                Icu8 = (np.clip(I_cart, 0.0, 1.0) * 255.0).astype(np.uint8)
                Image.fromarray(Icu8).save(debug_dir / f"cart_pre_{tag}.png")
                Image.fromarray((m_cart.astype(np.uint8) * 255)).save(debug_dir / f"cart_mask_{tag}.png")
                dumped += 1

            if cfar_dir is not None:
                dump_cfar_npz(
                    Path(cfar_dir), t, args.band,
                    I_norm=In, mask=m, weight=w, mask_persist=mp,
                    water_mask=wmask, baseline_per_range=baseline,
                    meta={"source_png": str(p)}
                )

        print(f"[OK] CFAR-only dump complete: dir={cfar_dir} frames={len(items)} band={args.band}")
        return 0

    with out.open("w", newline="") as f:
        wcsv = csv.DictWriter(
            f,
            fieldnames=[
                "t",
                "band",
                "dx",
                "dy",
                "dz",
                "droll",
                "dpitch",
                "dyaw",
                "quality",
                "rmse_m",
                "inlier_ratio",
                "dt_s",
            ],
        )
        wcsv.writeheader()

        t0, p0 = items[0]
        I0raw, I0n, m0, w0, w0mask, b0 = preprocess_one(t0, p0)
        mask_hist.append(m0)

        m0p = None
        if pers_enabled and len(mask_hist) >= 1:
            m0p = np.sum(np.stack(list(mask_hist), axis=0), axis=0) >= min_hits

        if dbg_dir is not None and dumped < int(args.debug_n):
            tag = f"{args.band}_{t0:.3f}"
            # cart-only debug (same as earlier script)
            debug_dir = Path(dbg_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            Iraw_cart = remap_cart.remap(_robust_norm01(I0raw), interp=cv2.INTER_LINEAR)
            Image.fromarray((Iraw_cart * 255.0).astype(np.uint8)).save(debug_dir / f"cart_raw_{tag}.png")
            I_cart = remap_cart.remap(I0n, interp=cv2.INTER_LINEAR)
            m_cart = remap_cart.remap(m0.astype(np.float32), interp=cv2.INTER_NEAREST) > 0.5
            Icu8 = (np.clip(I_cart, 0.0, 1.0) * 255.0).astype(np.uint8)
            Image.fromarray(Icu8).save(debug_dir / f"cart_pre_{tag}.png")
            Image.fromarray((m_cart.astype(np.uint8) * 255)).save(debug_dir / f"cart_mask_{tag}.png")
            dumped += 1

        if cfar_dir is not None:
            dump_cfar_npz(Path(cfar_dir), t0, args.band, I_norm=I0n, mask=m0, weight=w0, mask_persist=m0p, water_mask=w0mask, baseline_per_range=b0, meta={"source_png": str(p0)})

        pts0 = polar_indices_to_points(
            (m0p if m0p is not None else m0),
            w0,
            R_max_m=R_max_m,
            azimuth_ccw=azimuth_ccw,
            az_offset_deg=az_offset_deg,
            min_range_m=min_range_m,
            max_range_m=max_range_m,
            max_points=max_points,
            mirror_x=mirror_x,
        )

        if args.filter_by_buildings and poses_w is not None and masker is not None:
            pts0 = masker.filter_points_by_buildings(pts0, interp_pose(poses_w, t0), max_dist_m=float(args.buildings_max_dist_m))

        for (t1, p1) in items[1:]:
            I1raw, I1n, m1, w1, w1mask, b1 = preprocess_one(t1, p1)
            mask_hist.append(m1)

            m1p = None
            if pers_enabled:
                stack = np.stack(list(mask_hist), axis=0)
                m1p = np.sum(stack, axis=0) >= min_hits

            if dbg_dir is not None and dumped < int(args.debug_n):
                tag = f"{args.band}_{t1:.3f}"
                debug_dir = Path(dbg_dir)
                debug_dir.mkdir(parents=True, exist_ok=True)
                Iraw_cart = remap_cart.remap(_robust_norm01(I1raw), interp=cv2.INTER_LINEAR)
                Image.fromarray((Iraw_cart * 255.0).astype(np.uint8)).save(debug_dir / f"cart_raw_{tag}.png")
                I_cart = remap_cart.remap(I1n, interp=cv2.INTER_LINEAR)
                m_cart = remap_cart.remap(m1.astype(np.float32), interp=cv2.INTER_NEAREST) > 0.5
                Icu8 = (np.clip(I_cart, 0.0, 1.0) * 255.0).astype(np.uint8)
                Image.fromarray(Icu8).save(debug_dir / f"cart_pre_{tag}.png")
                Image.fromarray((m_cart.astype(np.uint8) * 255)).save(debug_dir / f"cart_mask_{tag}.png")
                dumped += 1

            if cfar_dir is not None:
                dump_cfar_npz(Path(cfar_dir), t1, args.band, I_norm=I1n, mask=m1, weight=w1, mask_persist=m1p, water_mask=w1mask, baseline_per_range=b1, meta={"source_png": str(p1)})

            pts1 = polar_indices_to_points(
                (m1p if m1p is not None else m1),
                w1,
                R_max_m=R_max_m,
                azimuth_ccw=azimuth_ccw,
                az_offset_deg=az_offset_deg,
                min_range_m=min_range_m,
                max_range_m=max_range_m,
                max_points=max_points,
                mirror_x=mirror_x,
            )
            if args.filter_by_buildings and poses_w is not None and masker is not None:
                pts1 = masker.filter_points_by_buildings(pts1, interp_pose(poses_w, t1), max_dist_m=float(args.buildings_max_dist_m))

            dt = float(t1 - t0)
            if dt <= 0.0 or pts0.shape[0] < 50 or pts1.shape[0] < 50:
                t0, p0, pts0 = t1, p1, pts1
                continue

            T, fitness, rmse = icp_2d(pts0, pts1, max_corr=max_corr, max_iters=max_iters)
            dx, dy, dyaw = T_to_dxdy_dyaw(T)

            wcsv.writerow(
                {
                    "t": f"{t1:.6f}",
                    "band": args.band,
                    "dx": f"{dx:.6f}",
                    "dy": f"{dy:.6f}",
                    "dz": "0.0",
                    "droll": "0.0",
                    "dpitch": "0.0",
                    "dyaw": f"{dyaw:.9f}",
                    "quality": f"{fitness:.6f}",
                    "rmse_m": f"{rmse:.6f}",
                    "inlier_ratio": f"{fitness:.6f}",
                    "dt_s": f"{dt:.6f}",
                }
            )

            t0, p0, pts0 = t1, p1, pts1

    print(f"[OK] wrote {out} for band={args.band} frames={len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
