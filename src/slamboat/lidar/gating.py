from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from slamboat.config import LidarGatingSpec
from slamboat.lidar.icp_types import IcpResult


def _rot_angle_deg(R: np.ndarray) -> float:
    """Return SO(3) rotation angle in degrees from a rotation matrix."""
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"R must be 3x3, got {R.shape}")
    c = (float(np.trace(R)) - 1.0) * 0.5
    c = max(-1.0, min(1.0, c))
    return float(math.degrees(math.acos(c)))


def gate_icp_result(result: IcpResult, cfg: LidarGatingSpec) -> Tuple[bool, List[str], Dict[str, float]]:
    """
    Decide whether an ICP result is good enough to add a BetweenFactorPose3.

    Returns:
      ok: bool
      reasons: list[str] (non-empty if rejected)
      summary: dict[str, float] with key scalars used for logging/CSV
    """
    T = np.asarray(result.delta_T, dtype=float)
    t = T[:3, 3]
    R = T[:3, :3]

    trans_m = float(np.linalg.norm(t))
    rot_deg = float(_rot_angle_deg(R))

    rmse = result.metrics.rmse_m
    inlier = result.metrics.inlier_ratio
    dt_s = result.metrics.dt_s

    reasons: List[str] = []

    if cfg.max_translation_m is not None and trans_m > float(cfg.max_translation_m):
        reasons.append(f"translation {trans_m:.3f} m > {float(cfg.max_translation_m):.3f} m")

    if cfg.max_rotation_deg is not None and rot_deg > float(cfg.max_rotation_deg):
        reasons.append(f"rotation {rot_deg:.3f} deg > {float(cfg.max_rotation_deg):.3f} deg")

    if cfg.max_rmse_m is not None and rmse is not None and float(rmse) > float(cfg.max_rmse_m):
        reasons.append(f"rmse {float(rmse):.3f} m > {float(cfg.max_rmse_m):.3f} m")

    if cfg.min_inlier_ratio is not None and inlier is not None and float(inlier) < float(cfg.min_inlier_ratio):
        reasons.append(f"inlier_ratio {float(inlier):.3f} < {float(cfg.min_inlier_ratio):.3f}")

    if dt_s is not None:
        if cfg.dt_min_s is not None and float(dt_s) < float(cfg.dt_min_s):
            reasons.append(f"dt {float(dt_s):.3f} s < {float(cfg.dt_min_s):.3f} s")
        if cfg.dt_max_s is not None and float(dt_s) > float(cfg.dt_max_s):
            reasons.append(f"dt {float(dt_s):.3f} s > {float(cfg.dt_max_s):.3f} s")

    summary = {
        "trans_m": trans_m,
        "rot_deg": rot_deg,
        "rmse_m": float(rmse) if rmse is not None else float("nan"),
        "inlier_ratio": float(inlier) if inlier is not None else float("nan"),
        "dt_s": float(dt_s) if dt_s is not None else float("nan"),
    }

    ok = len(reasons) == 0
    return ok, reasons, summary
