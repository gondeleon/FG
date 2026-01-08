#!/usr/bin/env python3
"""
Evaluate slamboat results:
- Loads outputs/state_trace.csv (state in IMU/LiDAR frame)
- Loads GNSS log (lat/lon) and converts to local ENU-like (E,N,Up) using UTM
- Applies lever-arm/extrinsics to compare estimated GPS antenna position vs measured GPS position
- Reports error stats and plots.

Assumptions:
- State pose is in ENU world (x=East, y=North, z=Up)
- Extrinsics JSON contains transforms from AHRS -> sensor (as per your dataset text)
- State frame is IMU/LiDAR (Option A). We therefore convert AHRS->sensor extrinsics into BASE->sensor
  by selecting BASE = "lidar" or "imu" (whichever you use as state) and chaining:
     T_base_sensor = inv(T_ahrs_base) * T_ahrs_sensor
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from pyproj import CRS, Transformer
except ImportError as e:
    raise SystemExit(
        "Missing dependency pyproj. Install with: pip install pyproj"
    ) from e


# ----------------------------
# Quaternion / SE(3) utilities
# ----------------------------

def quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    """Quaternion to rotation matrix. q = [qx,qy,qz,qw]."""
    qx, qy, qz, qw = q
    # Normalize defensively
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n == 0.0:
        return np.eye(3)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n

    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz

    return np.array([
        [1 - 2*(yy+zz),     2*(xy-wz),       2*(xz+wy)],
        [2*(xy+wz),         1 - 2*(xx+zz),   2*(yz-wx)],
        [2*(xz-wy),         2*(yz+wx),       1 - 2*(xx+yy)]
    ], dtype=float)


@dataclass(frozen=True)
class SE3:
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)

    def inv(self) -> "SE3":
        Rt = self.R.T
        return SE3(R=Rt, t=-(Rt @ self.t))

    def compose(self, other: "SE3") -> "SE3":
        return SE3(R=self.R @ other.R, t=self.R @ other.t + self.t)


def load_extrinsics_as_se3(extr_path: Path) -> dict[str, SE3]:
    """
    Load extrinsics.json and return a dict of SE3 transforms.

    Supports rig-aware format:
      {
        "convention": {...},
        "frames": {
          "gps":   {"quaternion": [...], "translation": [...]},
          "lidar": {"quaternion": [...], "translation": [...]},
          "imu":   {"same_as": "lidar"}
        }
      }

    Returns transforms as SE3 with keys equal to frame names.
    """
    raw = json.loads(extr_path.read_text())

    if "frames" not in raw:
        raise SystemExit("extrinsics.json must contain a top-level 'frames' field")

    frames = raw["frames"]
    out: dict[str, SE3] = {}

    # First pass: load explicit transforms
    for name, v in frames.items():
        if "same_as" in v:
            continue
        if "quaternion" not in v or "translation" not in v:
            raise SystemExit(f"Frame '{name}' must define quaternion + translation")
        q = np.array(v["quaternion"], dtype=float)
        t = np.array(v["translation"], dtype=float)
        out[name] = SE3(R=quat_xyzw_to_R(q), t=t)

    # Second pass: resolve same_as
    for name, v in frames.items():
        if "same_as" in v:
            ref = v["same_as"]
            if ref not in out:
                raise SystemExit(f"Frame '{name}' refers to unknown frame '{ref}' via same_as")
            out[name] = out[ref]

    return out


def utm_transformer_for_latlon(lat: float, lon: float) -> Transformer:
    """
    Create transformer WGS84 -> UTM for the zone where (lat,lon) lies.
    """
    zone = int((lon + 180.0) / 6.0) + 1
    is_north = lat >= 0.0
    epsg = 32600 + zone if is_north else 32700 + zone
    crs_utm = CRS.from_epsg(epsg)
    crs_wgs84 = CRS.from_epsg(4326)
    return Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True)


def unwrap_deg(a: np.ndarray) -> np.ndarray:
    """Unwrap degrees for continuous yaw plots."""
    return np.rad2deg(np.unwrap(np.deg2rad(a)))


# ----------------------------
# Data loaders
# ----------------------------

def load_gnss_txt(path: Path) -> pd.DataFrame:
    """
    Expected columns (tab-separated; no header):
    t_unix, gps_time, lat, lat_hemi, lon, lon_hemi, heading_deg, quality, nsat, hdop, geoid_h
    """
    df = pd.read_csv(path, sep=r"\s+|\t|,", header=None, engine="python")
    df.columns = [
        "t", "gps_time", "lat", "lat_hemi", "lon", "lon_hemi",
        "heading_deg", "quality", "nsat", "hdop", "geoid_h"
    ]
    # Apply hemisphere
    df["lat"] = df["lat"] * df["lat_hemi"].map({"N": 1.0, "S": -1.0}).astype(float)
    df["lon"] = df["lon"] * df["lon_hemi"].map({"E": 1.0, "W": -1.0}).astype(float)
    return df


def load_state_trace(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Minimal robustness: accept either (qx,qy,qz,qw) or (qw,qx,qy,qz) but prefer xyzw.
    cols = set(df.columns)
    if {"qx", "qy", "qz", "qw"}.issubset(cols):
        pass
    else:
        raise SystemExit(
            f"Unexpected columns in {path}. Need qx,qy,qz,qw. Found: {sorted(df.columns)}"
        )
    return df


# ----------------------------
# Main evaluation
# ----------------------------

def main(
    trace_csv: Path,
    gnss_txt: Path,
    extrinsics_json: Path,
    base_frame: str = "lidar",
    gps_name: str = "gps",
) -> None:
    # Load
    st = load_state_trace(trace_csv)
    gn = load_gnss_txt(gnss_txt)
    ex = load_extrinsics_as_se3(extrinsics_json)

    if base_frame not in ex:
        raise SystemExit(f"base_frame '{base_frame}' not found in extrinsics. Keys={list(ex.keys())}")
    if gps_name not in ex:
        raise SystemExit(f"gps_name '{gps_name}' not found in extrinsics. Keys={list(ex.keys())}")

    # Convert AHRS->sensor extrinsics into BASE->sensor
    T_ahrs_base = ex[base_frame]
    T_ahrs_gps = ex[gps_name]
    T_base_gps = T_ahrs_base.inv().compose(T_ahrs_gps)  # base -> gps
    t_base_gps = T_base_gps.t  # expressed in base frame

    # GNSS -> UTM -> local EN (origin at first GNSS point)
    tr = utm_transformer_for_latlon(float(gn["lat"].iloc[0]), float(gn["lon"].iloc[0]))
    e, n = tr.transform(gn["lon"].to_numpy(), gn["lat"].to_numpy())
    e0, n0 = float(e[0]), float(n[0])
    gn["x_e"] = e - e0
    gn["y_n"] = n - n0

    # Time align: nearest-neighbor GNSS sample for each state timestamp
    # (For better results: interpolate GNSS; this is enough for sanity checks.)
    gn_t = gn["t"].to_numpy()
    gn_xy = gn[["x_e", "y_n"]].to_numpy()

    st_t = st["t"].to_numpy()
    idx = np.searchsorted(gn_t, st_t, side="left")
    idx = np.clip(idx, 0, len(gn_t) - 1)

    gps_meas_xy = gn_xy[idx, :]
    

    # Estimated GPS antenna position from state:
    # p_gps_est = p_base + R_wb * t_base_gps
    p_base = st[["x", "y", "z"]].to_numpy()
    q = st[["qx", "qy", "qz", "qw"]].to_numpy()

    p_gps_est = np.zeros((len(st), 3), dtype=float)
    for i in range(len(st)):
        Rwb = quat_xyzw_to_R(q[i])
        p_gps_est[i] = p_base[i] + (Rwb @ t_base_gps)
    # ---- Origin alignment (constant offset) ----
    # The estimator's ENU origin (ref_lat/ref_lon) is not necessarily the same as
    # the UTM local origin we defined (first GNSS sample). Align them by removing
    # a constant translation computed at the first matched sample.
    offset = gps_meas_xy[0] - p_gps_est[0, :2]
    gps_meas_xy = gps_meas_xy - offset
    # Error in EN plane
    err = p_gps_est[:, :2] - gps_meas_xy
    e_norm = np.linalg.norm(err, axis=1)

    # Report
    print(f"Loaded states: {len(st)} | GNSS: {len(gn)}")
    print(f"base_frame={base_frame} gps_name={gps_name}")
    print(f"t_base_gps (base coords) = {t_base_gps} | norm={np.linalg.norm(t_base_gps):.3f} m")

    s = pd.Series(e_norm, name="|e| [m]")
    print("\nPosition error stats (after extrinsics compensation):")
    print(s.describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

    # Speed sanity
    if {"vx", "vy", "vz"}.issubset(set(st.columns)):
        v = st[["vx", "vy", "vz"]].to_numpy()
        speed = np.linalg.norm(v, axis=1)
        ss = pd.Series(speed, name="|v| [m/s]")
        print("\nSpeed stats:")
        print(ss.describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

    # Plots
    plt.figure()
    plt.plot(p_base[:, 0], p_base[:, 1], label="Estimated BASE (IMU/LiDAR)")
    plt.plot(gps_meas_xy[:, 0], gps_meas_xy[:, 1], label="GPS measured (local UTM)")
    plt.plot(p_gps_est[:, 0], p_gps_est[:, 1], label="GPS from estimate (BASE+lever arm)")
    plt.axis("equal")
    plt.xlabel("x [m] East")
    plt.ylabel("y [m] North")
    plt.title("Trajectory comparison")
    plt.legend()

    plt.figure()
    plt.plot(st_t, e_norm)
    plt.xlabel("t [s unix]")
    plt.ylabel("|e| [m]")
    plt.title("GPS position error (extrinsics-compensated)")

    plt.show()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, default=Path("outputs/state_trace.csv"))
    ap.add_argument("--gnss", type=Path, required=True)
    ap.add_argument("--extr", type=Path, default=Path("configs/extrinsics.json"))
    ap.add_argument("--base-frame", type=str, default="lidar", help="State frame name in extrinsics.json")
    ap.add_argument("--gps-name", type=str, default="gps", help="GPS sensor key in extrinsics.json")
    args = ap.parse_args()

    main(
        trace_csv=args.trace,
        gnss_txt=args.gnss,
        extrinsics_json=args.extr,
        base_frame=args.base_frame,
        gps_name=args.gps_name,
    )
