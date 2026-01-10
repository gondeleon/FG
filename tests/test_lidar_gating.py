import math
import numpy as np

from slamboat.config import LidarGatingSpec
from slamboat.lidar.icp_types import IcpMetrics, IcpResult
from slamboat.lidar.gating import gate_icp_result


def _T(tx=0.0, ty=0.0, tz=0.0, yaw_deg=0.0) -> np.ndarray:
    c = math.cos(math.radians(yaw_deg))
    s = math.sin(math.radians(yaw_deg))
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = np.array([tx, ty, tz], dtype=float)
    return T


def test_gate_accepts_nominal():
    cfg = LidarGatingSpec(max_translation_m=3.0, max_rotation_deg=15.0, max_rmse_m=1.0, min_inlier_ratio=0.2)
    res = IcpResult(_T(tx=1.0, yaw_deg=5.0), IcpMetrics(rmse_m=0.2, inlier_ratio=0.8, dt_s=1.0))
    ok, reasons, _ = gate_icp_result(res, cfg)
    assert ok
    assert reasons == []


def test_gate_rejects_translation():
    cfg = LidarGatingSpec(max_translation_m=1.0, max_rotation_deg=179.0)
    res = IcpResult(_T(tx=2.0), IcpMetrics(dt_s=1.0))
    ok, reasons, _ = gate_icp_result(res, cfg)
    assert not ok
    assert any("translation" in r for r in reasons)


def test_gate_rejects_rotation():
    cfg = LidarGatingSpec(max_translation_m=100.0, max_rotation_deg=10.0)
    res = IcpResult(_T(yaw_deg=45.0), IcpMetrics(dt_s=1.0))
    ok, reasons, _ = gate_icp_result(res, cfg)
    assert not ok
    assert any("rotation" in r for r in reasons)


def test_gate_rejects_rmse():
    cfg = LidarGatingSpec(max_rmse_m=0.3)
    res = IcpResult(_T(tx=0.1), IcpMetrics(rmse_m=0.9, dt_s=1.0))
    ok, reasons, _ = gate_icp_result(res, cfg)
    assert not ok
    assert any("rmse" in r for r in reasons)


def test_gate_rejects_inlier_ratio():
    cfg = LidarGatingSpec(min_inlier_ratio=0.5)
    res = IcpResult(_T(tx=0.1), IcpMetrics(inlier_ratio=0.1, dt_s=1.0))
    ok, reasons, _ = gate_icp_result(res, cfg)
    assert not ok
    assert any("inlier_ratio" in r for r in reasons)


def test_gate_rejects_dt_bounds():
    cfg = LidarGatingSpec(dt_min_s=0.5, dt_max_s=2.0)
    res = IcpResult(_T(tx=0.1), IcpMetrics(dt_s=5.0))
    ok, reasons, _ = gate_icp_result(res, cfg)
    assert not ok
    assert any(r.startswith("dt") for r in reasons)
