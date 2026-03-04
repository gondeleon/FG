from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import json
import numpy as np

from .phasecorr import estimate_yaw_xy_phasecorr, PhaseCorrResult
from .icp2d import run_icp_2d, _HAS_O3D


@dataclass
class RadarOdomResult:
    method: str
    dt_s: float
    dx: float
    dy: float
    yaw_rad: float
    psr: float | None
    icp_fitness: float | None
    icp_rmse: float | None


def load_clean_npz(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    if "meta_json" in out:
        out["meta"] = json.loads(str(out["meta_json"]))
    return out


def _extract_points_from_cart(
    I_cart: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    q: float = 0.995,
    max_points: int = 4000,
) -> np.ndarray:
    """
    Simple “rigid points” extraction: take top intensities.
    Returns Nx2 in meters in the cartesian grid coordinates.
    """
    A = I_cart.astype(np.float32)
    thr = float(np.quantile(A, q))
    mask = A >= thr
    idx = np.argwhere(mask)
    if idx.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # idx is (row=y, col=x)
    yy = ys[idx[:, 0]]
    xx = xs[idx[:, 1]]
    pts = np.c_[xx, yy].astype(np.float32)

    if pts.shape[0] > max_points:
        sel = np.random.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[sel]
    return pts

def wrap_pi(a: float) -> float:
    return (a + np.pi) % (2*np.pi) - np.pi

def consistency_cf(dx: float, dy: float, yaw: float) -> float:
    phi = float(np.arctan2(dy, dx))
    return float(np.exp(-abs(wrap_pi(phi - yaw))))

def estimate_pair(
    npz_ref: Path,
    npz_cur: Path,
    method: str,
    cart_res: int = 512,
    r_max: float = 600.0,
    px_size_m: float | None = None,
    icp_max_corr_m: float = 8.0,
    psr_min: float = 8.0,
    cf_min: float = 0.35,
) -> tuple[RadarOdomResult, dict]:
    """
    method:
      - "icp_raw"
      - "icp_clean"
      - "icp_clean_nowater"
      - "hybrid_clean" (phase corr on clean + ICP refine)
      - "hybrid_clean_nowater" (phase corr on clean with water mask + ICP refine)
      - "hybrid_raw"   (phase corr on raw + ICP refine)  (útil como ablation)
    """
    from .radar_polar_clean import polar_to_cart_bilinear, polar_to_cart_nearest  # reuse your functions

    t0 = time.perf_counter()

    d0 = load_clean_npz(npz_ref)
    d1 = load_clean_npz(npz_cur)

    I0_raw = d0["I_pol_f"]
    I1_raw = d1["I_pol_f"]
    I0_cln = d0["I_clean_f"]
    I1_cln = d1["I_clean_f"]
    W0_pol = d0["mask_water_pol"].astype(np.uint8)
    W1_pol = d1["mask_water_pol"].astype(np.uint8)
    # build cart
    C0_raw, xs, ys = polar_to_cart_bilinear(I0_raw, r_max=r_max, cart_res=cart_res)
    C1_raw, _, _ = polar_to_cart_bilinear(I1_raw, r_max=r_max, cart_res=cart_res)
    C0_cln, _, _ = polar_to_cart_bilinear(I0_cln, r_max=r_max, cart_res=cart_res)
    C1_cln, _, _ = polar_to_cart_bilinear(I1_cln, r_max=r_max, cart_res=cart_res)
    W0_cart, _, _ = polar_to_cart_nearest(W0_pol, r_max=r_max, cart_res=cart_res)
    W1_cart, _, _ = polar_to_cart_nearest(W1_pol, r_max=r_max, cart_res=cart_res)
    W0 = (W0_cart > 0.5).astype(np.float32)
    W1 = (W1_cart > 0.5).astype(np.float32)

    C0_cln_now = C0_cln * (1.0 - W0)
    C1_cln_now = C1_cln * (1.0 - W1)

    if px_size_m is None:
        # grid spans [-r_max, r_max] with cart_res samples
        px_size_m = float((2.0 * r_max) / (cart_res - 1))

    dbg = {
        "ref": str(npz_ref),
        "cur": str(npz_cur),
        "cart_res": int(cart_res),
        "r_max": float(r_max),
        "px_size_m": float(px_size_m),
    }

    psr = None
    icp_fit = None
    icp_rmse = None

    if method == "icp_raw" or method == "icp_clean":
        if not _HAS_O3D:
            raise RuntimeError("Open3D is required for ICP baselines.")
        A = C0_raw if method == "icp_raw" else C0_cln
        B = C1_raw if method == "icp_raw" else C1_cln

        P0 = _extract_points_from_cart(A, xs, ys)
        P1 = _extract_points_from_cart(B, xs, ys)
        icp = run_icp_2d(P1, P0, init_dx=0.0, init_dy=0.0, init_yaw=0.0, max_corr_m=icp_max_corr_m)
        dx, dy, yaw = icp.dx, icp.dy, icp.yaw_rad
        icp_fit, icp_rmse = icp.fitness, icp.rmse

        dbg["icp"] = icp.__dict__

    elif method == "icp_clean_nowater":
        if not _HAS_O3D:
            raise RuntimeError("Open3D is required for ICP baselines.")
        A = C0_cln_now
        B = C1_cln_now
        P0 = _extract_points_from_cart(A, xs, ys)
        P1 = _extract_points_from_cart(B, xs, ys)
        icp = run_icp_2d(P1, P0, init_dx=0.0, init_dy=0.0, init_yaw=0.0, max_corr_m=icp_max_corr_m)
        dx, dy, yaw = icp.dx, icp.dy, icp.yaw_rad
        icp_fit, icp_rmse = icp.fitness, icp.rmse
        dbg["icp"] = icp.__dict__

    elif method == "hybrid_clean" or method == "hybrid_raw":
        A = C0_cln if method == "hybrid_clean" else C0_raw
        B = C1_cln if method == "hybrid_clean" else C1_raw

        pc: PhaseCorrResult = estimate_yaw_xy_phasecorr(A, B, px_size_m=px_size_m, window=True)
        cf = consistency_cf(pc.dx, pc.dy, pc.yaw_rad)

        use_phase = (pc.psr >= psr_min) and (cf >= cf_min)
        if not use_phase:
            dx0, dy0, yaw0 = 0.0, 0.0, 0.0
            rejected = True
        else:
            dx0, dy0, yaw0 = pc.dx, pc.dy, pc.yaw_rad
            rejected = False

        psr = pc.psr
        dbg["phasecorr"] = {
            "dx": float(dx0),
            "dy": float(dy0),
            "yaw_rad": float(yaw0),
            "psr": float(pc.psr),
            "peak": pc.peak,
            "cf": float(cf),
            "rejected": rejected,
        }

        dx, dy, yaw = dx0, dy0, yaw0

        if _HAS_O3D:
            P0 = _extract_points_from_cart(A, xs, ys)
            P1 = _extract_points_from_cart(B, xs, ys)
            icp = run_icp_2d(P1, P0, init_dx=dx0, init_dy=dy0, init_yaw=yaw0, max_corr_m=icp_max_corr_m)
            dx, dy, yaw = icp.dx, icp.dy, icp.yaw_rad
            icp_fit, icp_rmse = icp.fitness, icp.rmse
            dbg["icp"] = icp.__dict__

        dbg["corr_surface"] = pc.corr  # big array; caller may save as image

    elif method == "hybrid_clean_nowater":
        A = C0_cln_now
        B = C1_cln_now

        pc: PhaseCorrResult = estimate_yaw_xy_phasecorr(A, B, px_size_m=px_size_m, window=True)
        cf = consistency_cf(pc.dx, pc.dy, pc.yaw_rad)

        use_phase = (pc.psr >= psr_min) and (cf >= cf_min)
        if not use_phase:
            dx0, dy0, yaw0 = 0.0, 0.0, 0.0
            rejected = True
        else:
            dx0, dy0, yaw0 = pc.dx, pc.dy, pc.yaw_rad
            rejected = False

        psr = pc.psr
        dbg["phasecorr"] = {
            "dx": float(dx0),
            "dy": float(dy0),
            "yaw_rad": float(yaw0),
            "psr": float(pc.psr),
            "peak": pc.peak,
            "cf": float(cf),
            "rejected": rejected,
        }

        dx, dy, yaw = dx0, dy0, yaw0

        if _HAS_O3D:
            P0 = _extract_points_from_cart(A, xs, ys)
            P1 = _extract_points_from_cart(B, xs, ys)
            icp = run_icp_2d(P1, P0, init_dx=dx0, init_dy=dy0, init_yaw=yaw0, max_corr_m=icp_max_corr_m)
            dx, dy, yaw = icp.dx, icp.dy, icp.yaw_rad
            icp_fit, icp_rmse = icp.fitness, icp.rmse
            dbg["icp"] = icp.__dict__

        dbg["corr_surface"] = pc.corr

    else:
        raise ValueError(f"Unknown method: {method}")

    dt = time.perf_counter() - t0

    res = RadarOdomResult(
        method=method,
        dt_s=float(dt),
        dx=float(dx),
        dy=float(dy),
        yaw_rad=float(yaw),
        psr=None if psr is None else float(psr),
        icp_fitness=None if icp_fit is None else float(icp_fit),
        icp_rmse=None if icp_rmse is None else float(icp_rmse),
    )
    return res, dbg
