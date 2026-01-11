from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d


LIDAR_DTYPE = np.dtype([
    ("x", np.float32),
    ("y", np.float32),
    ("z", np.float32),
    ("intensity", np.float32),
    ("time", np.uint32),
    ("reflectivity", np.uint16),
    ("ambient", np.uint16),
    ("range", np.uint32),
])


def ts_from_name_ns(p: Path) -> float:
    # filename like 1626412888374006294.bin  -> seconds
    return int(p.stem) * 1e-9


def load_bin_points(p: Path, max_range_m: Optional[float] = 80.0) -> np.ndarray:
    scan = np.fromfile(p, dtype=LIDAR_DTYPE)
    pts = np.stack([scan["x"], scan["y"], scan["z"]], axis=-1).astype(np.float64)

    if max_range_m is not None:
        r = scan["range"].astype(np.float64)  # usually mm for many sensors; but we treat as meters if already meters
        # Heuristic: if values look like millimeters, convert
        if np.nanmedian(r) > 500.0:  # likely mm
            r = r * 1e-3
        pts = pts[r <= float(max_range_m)]

    # Drop non-finite
    m = np.isfinite(pts).all(axis=1)
    return pts[m]


def to_pcd(pts: np.ndarray, voxel_size: float) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    return pcd


def icp_i_T_j(pcd_i: o3d.geometry.PointCloud, pcd_j: o3d.geometry.PointCloud,
              max_corr: float, max_iters: int) -> tuple[np.ndarray, float, float, int]:
    """
    Return i_T_j (transform that maps points in frame j into frame i), plus metrics.
    We run ICP with source=pcd_j, target=pcd_i so Open3D returns T_target_source = i_T_j.
    """
    init = np.eye(4)

    # Point-to-point ICP (safe baseline). Point-to-plane can be better but needs normals.
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iters))
    res = o3d.pipelines.registration.registration_icp(
        source=pcd_j,
        target=pcd_i,
        max_correspondence_distance=float(max_corr),
        init=init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=crit,
    )
    T = np.asarray(res.transformation, dtype=float)
    fitness = float(res.fitness)
    rmse = float(res.inlier_rmse)
    corr = int(len(res.correspondence_set))
    return T, fitness, rmse, corr


def rot_to_rpy_zyx(R: np.ndarray) -> tuple[float, float, float]:
    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    pitch = math.asin(max(-1.0, min(1.0, -float(R[2, 0]))))
    roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
    yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    return roll, pitch, yaw


@dataclass
class Params:
    voxel_size_m: float = 0.25
    max_corr_dist_m: float = 1.5
    max_iters: int = 50
    max_range_m: float = 80.0
    step_s: float = 1.0  # keyframe period


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--points_dir", type=str, default="data/points")
    ap.add_argument("--out_csv", type=str, default="outputs/lidar_delta.csv")
    ap.add_argument("--frame_id", type=str, default="lidar")
    ap.add_argument("--step_s", type=float, default=1.0)
    ap.add_argument("--voxel_size_m", type=float, default=0.25)
    ap.add_argument("--max_corr_dist_m", type=float, default=1.5)
    ap.add_argument("--max_iters", type=int, default=50)
    ap.add_argument("--max_range_m", type=float, default=80.0)
    args = ap.parse_args()

    prm = Params(
        voxel_size_m=args.voxel_size_m,
        max_corr_dist_m=args.max_corr_dist_m,
        max_iters=args.max_iters,
        max_range_m=args.max_range_m,
        step_s=args.step_s,
    )

    pts_dir = Path(args.points_dir)
    bins = sorted(pts_dir.glob("*.bin"))
    if not bins:
        raise SystemExit(f"No .bin files in {pts_dir}")

    times = np.array([ts_from_name_ns(p) for p in bins], dtype=float)

    # Choose keyframes at fixed period (nearest bin for each time grid)
    t0 = float(times[0])
    tN = float(times[-1])
    grid = np.arange(t0, tN + 1e-9, prm.step_s)

    idx = np.searchsorted(times, grid, side="left")
    idx = np.clip(idx, 0, len(bins) - 1)

    # Deduplicate (if multiple grid points map to same scan)
    key_idx = [int(idx[0])]
    for k in idx[1:]:
        k = int(k)
        if k != key_idx[-1]:
            key_idx.append(k)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "frame_id", "dx", "dy", "dz", "droll", "dpitch", "dyaw",
                    "rmse_m", "fitness", "correspondences", "dt_s"])

        # Preload first keyframe cloud
        i0 = key_idx[0]
        t_i = ts_from_name_ns(bins[i0])
        pts_i = load_bin_points(bins[i0], max_range_m=prm.max_range_m)
        pcd_i = to_pcd(pts_i, prm.voxel_size_m)

        for j_idx in key_idx[1:]:
            t_j = ts_from_name_ns(bins[j_idx])
            pts_j = load_bin_points(bins[j_idx], max_range_m=prm.max_range_m)
            pcd_j = to_pcd(pts_j, prm.voxel_size_m)

            T_i_j, fitness, rmse, corr = icp_i_T_j(pcd_i, pcd_j, prm.max_corr_dist_m, prm.max_iters)

            R = T_i_j[:3, :3]
            t = T_i_j[:3, 3]
            roll, pitch, yaw = rot_to_rpy_zyx(R)

            dt_s = float(t_j - t_i)
            w.writerow([f"{t_j:.9f}", args.frame_id,
                        f"{t[0]:.6f}", f"{t[1]:.6f}", f"{t[2]:.6f}",
                        f"{roll:.6f}", f"{pitch:.6f}", f"{yaw:.6f}",
                        f"{rmse:.6f}", f"{fitness:.6f}", str(corr), f"{dt_s:.6f}"])

            # Move forward
            pcd_i = pcd_j
            t_i = t_j

    print(f"Wrote: {out} (rows={len(key_idx)-1})")


if __name__ == "__main__":
    main()
