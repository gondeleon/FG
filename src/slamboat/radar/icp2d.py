from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    o3d = None
    _HAS_O3D = False


@dataclass
class ICP2DResult:
    dx: float
    dy: float
    yaw_rad: float
    fitness: float
    rmse: float
    n_src: int
    n_tgt: int


def _se2_to_T(dx: float, dy: float, yaw: float) -> np.ndarray:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    T[0, 3] = dx
    T[1, 3] = dy
    return T


def run_icp_2d(
    src_xy: np.ndarray,
    tgt_xy: np.ndarray,
    init_dx: float = 0.0,
    init_dy: float = 0.0,
    init_yaw: float = 0.0,
    max_corr_m: float = 8.0,
    max_iter: int = 50,
) -> ICP2DResult:
    if not _HAS_O3D:
        raise RuntimeError("Open3D not installed; cannot run ICP refinement.")

    src = src_xy.astype(np.float64)
    tgt = tgt_xy.astype(np.float64)

    pcd_src = o3d.geometry.PointCloud()
    pcd_tgt = o3d.geometry.PointCloud()

    # embed in 3D (z=0)
    pcd_src.points = o3d.utility.Vector3dVector(np.c_[src, np.zeros((src.shape[0], 1))])
    pcd_tgt.points = o3d.utility.Vector3dVector(np.c_[tgt, np.zeros((tgt.shape[0], 1))])

    init = _se2_to_T(init_dx, init_dy, init_yaw)

    reg = o3d.pipelines.registration.registration_icp(
        pcd_src,
        pcd_tgt,
        max_corr_m,
        init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )

    T = reg.transformation
    yaw = float(np.arctan2(T[1, 0], T[0, 0]))
    dx = float(T[0, 3])
    dy = float(T[1, 3])

    return ICP2DResult(
        dx=dx,
        dy=dy,
        yaw_rad=yaw,
        fitness=float(reg.fitness),
        rmse=float(reg.inlier_rmse),
        n_src=int(src.shape[0]),
        n_tgt=int(tgt.shape[0]),
    )
