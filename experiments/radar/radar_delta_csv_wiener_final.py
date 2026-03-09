#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Tuple, Optional
import cv2
from scipy.signal import wiener
from scipy.ndimage import map_coordinates, median_filter
from pathlib import Path
import numpy as np
from PIL import Image
import yaml

import open3d as o3d

import json
import cv2
import numpy as np
from pathlib import Path

def _write_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = img
    if x.dtype != np.uint8:
        x = np.clip(x, 0.0, 1.0)
        x = (255.0 * x).astype(np.uint8)
    cv2.imwrite(str(path), x)

def save_debug_pngs(debug_dir: Path, idx: int, stamp: float, band: str,
                    I_norm: np.ndarray, mask: np.ndarray, weight: np.ndarray,
                    polar_to_cart_fn=None, *, do_cart: bool, do_polar: bool) -> None:
    debug_dir = Path(debug_dir) / band
    tag = f"{idx:04d}_{stamp:.6f}"

    # Polar quicklooks
    if do_polar:
        _write_png(debug_dir / f"polar_I_norm_{tag}.png", I_norm)
        _write_png(debug_dir / f"polar_mask_{tag}.png", mask.astype(np.float32))
        _write_png(debug_dir / f"polar_weight_{tag}.png", weight)

    # Cartesian quicklooks
    if do_cart and (polar_to_cart_fn is not None):
        I_c = polar_to_cart_fn(I_norm)
        m_c = polar_to_cart_fn(mask.astype(np.float32))
        w_c = polar_to_cart_fn(weight)

        _write_png(debug_dir / f"cart_I_norm_{tag}.png", I_c)
        _write_png(debug_dir / f"cart_mask_{tag}.png", m_c)
        _write_png(debug_dir / f"cart_weight_{tag}.png", w_c)


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


class PolarToCartesianCV2:
    """
    Remap polar image I(H,W) [rows=azimuth, cols=range] to cartesian grid (G,G).
    Convention:
      x = r*sin(theta), y = r*cos(theta), theta=0 at +Y
    """
    def __init__(self, H: int, W: int, R_max_m: float, grid_size: int,
                 azimuth_ccw: bool = False, az_offset_deg: float = 0.0,
                 mirror_x: bool = False):
        self.H = int(H); self.W = int(W)
        self.R = float(R_max_m)
        self.G = int(grid_size)
        self.azimuth_ccw = bool(azimuth_ccw)
        self.az_offset = float(az_offset_deg)
        self.mirror_x = bool(mirror_x)

        # cart grid coords in meters
        x_lin = np.linspace(-self.R, self.R, self.G, dtype=np.float32)
        y_lin = np.linspace(-self.R, self.R, self.G, dtype=np.float32)
        X, Y = np.meshgrid(x_lin, y_lin)

        r = np.sqrt(X**2 + Y**2, dtype=np.float32)
        theta = np.arctan2(X, Y).astype(np.float32)  # 0 at +Y

        if not self.azimuth_ccw:
            theta = (2.0*np.pi) - theta

        theta = theta + np.deg2rad(self.az_offset).astype(np.float32)
        theta = np.mod(theta, 2.0*np.pi)

        # Guard origin (avoid seam/singularity at r≈0 where atan2 is ill-defined)
        theta[r < 1e-6] = 0.0

        # map to polar indices
        az_idx = (theta / (2.0*np.pi)) * self.H - 0.5
        dr = self.R / self.W
        rad_idx = (r / dr) - 0.5

        self.map_x = np.clip(rad_idx, 0, self.W - 1).astype(np.float32)  # src col
        self.map_y = np.mod(az_idx, self.H).astype(np.float32)           # src row
        self.outside = (r > self.R)

    def remap(self, I_polar: np.ndarray, interp: int) -> np.ndarray:
        # I_polar is (H,W)
        src = I_polar.astype(np.float32, copy=False)
        cart = cv2.remap(
            src, self.map_x, self.map_y,
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


def load_polar_png(path: Path, H: int, W: int) -> np.ndarray:
    """Returns polar image as float32 with shape (H, W): rows=azimuth, cols=range."""
    I = np.array(Image.open(path).convert("L"), dtype=np.float32)
    h, w = I.shape
    # common case: stored as (W,H) == (range, azimuth)
    if (h, w) == (W, H):
        I = I.T
        h, w = I.shape
    if (h, w) != (H, W):
        I = np.array(Image.fromarray(I.astype(np.uint8)).resize((W, H), Image.BILINEAR), dtype=np.float32)
    return I  # (H,W)

def _robust_norm01(I: np.ndarray, p_lo: float = 5.0, p_hi: float = 95.0) -> np.ndarray:
    lo = float(np.percentile(I, p_lo))
    hi = float(np.percentile(I, p_hi))
    J = (I - lo) / max(hi - lo, 1e-6)
    return np.clip(J, 0.0, 1.0).astype(np.float32)

def _save_u8(path: Path, img_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_u8).save(path)

def save_cfar_debug(debug_dir: Path, tag: str,
                    I_raw: np.ndarray, I_norm: np.ndarray, mask: np.ndarray, weight: np.ndarray,
                    pers_mask: Optional[np.ndarray] = None,
                    remapper_cart: Optional[PolarToCartesianCV2] = None) -> None:

    # """
    # Save:
    #   - pre: normalized intensity
    #   - mask/weight
    #   - side-by-side pre vs overlay(mask)
    #   - (optional) persistence overlay
    # """
    """    
    Save ONLY cartesian debug:
      - cart_pre / cart_mask / cart_weight / cart_prepost
      - (optional) cart_pers_prepost
    """
    debug_dir.mkdir(parents=True, exist_ok=True)

    if remapper_cart is None:
        raise ValueError("save_cfar_debug called without remapper_cart (cart-only mode)")
    
    Iraw_cart = remapper_cart.remap(_robust_norm01(I_raw), interp=cv2.INTER_LINEAR)
    Iru8 = (Iraw_cart * 255.0).astype(np.uint8)
    _save_u8(debug_dir / f"cart_raw_{tag}.png", Iru8)
    I_cart = remapper_cart.remap(I_norm, interp=cv2.INTER_LINEAR)    
    m_cart = remapper_cart.remap(mask.astype(np.float32), interp=cv2.INTER_NEAREST) > 0.5
    w_cart = remapper_cart.remap(weight, interp=cv2.INTER_LINEAR)

    Icu8 = (np.clip(I_cart, 0.0, 1.0) * 255.0).astype(np.uint8)
    _save_u8(debug_dir / f"cart_pre_{tag}.png", Icu8)

    mcu8 = (m_cart.astype(np.uint8) * 255)
    _save_u8(debug_dir / f"cart_mask_{tag}.png", mcu8)

    wcu8 = (np.clip(w_cart, 0.0, 1.0) * 255.0).astype(np.uint8)
    _save_u8(debug_dir / f"cart_weight_{tag}.png", wcu8)

    pre_rgb = np.stack([Icu8, Icu8, Icu8], axis=-1).astype(np.uint8)
    over = pre_rgb.copy()
    over[m_cart, 0] = 255
    over[m_cart, 1] = (over[m_cart, 1] * 0.25).astype(np.uint8)
    over[m_cart, 2] = (over[m_cart, 2] * 0.25).astype(np.uint8)
    side = np.concatenate([pre_rgb, over], axis=1)
    _save_u8(debug_dir / f"cart_prepost_{tag}.png", side)

    if pers_mask is not None:
        pm_cart = remapper_cart.remap(pers_mask.astype(np.float32), interp=cv2.INTER_NEAREST) > 0.5
        pover = pre_rgb.copy()
        pover[pm_cart, 1] = 255
        pover[pm_cart, 0] = (pover[pm_cart, 0] * 0.25).astype(np.uint8)
        pover[pm_cart, 2] = (pover[pm_cart, 2] * 0.25).astype(np.uint8)
        side2 = np.concatenate([pre_rgb, pover], axis=1)
        _save_u8(debug_dir / f"cart_pers_prepost_{tag}.png", side2)
 
def range_comp(I: np.ndarray, R_max_m: float, gamma: float, *, mode: str = "mul", r0: float = 0.02) -> np.ndarray:
    """Range compensation in polar domain.
    - mode='mul': legacy behavior (multiply by r^gamma), attenuates near range.
    - mode='inv': inverse compensation (multiply by (max(r,r0))^-gamma), boosts near range.
    """
    H, W = I.shape
    r = (np.arange(W, dtype=np.float32) + 0.5) / float(W)  # [0..1]
    rr = np.clip(r, float(r0), 1.0)
    mode = (mode or "mul").lower()
    if mode == "inv":
        gain = np.power(rr, -float(gamma))
        gain = gain / np.max(gain)  # keep scale bounded
    else:
        gain = np.power(np.clip(r, 1e-6, 1.0), float(gamma))
    return I * gain[None, :]


def log_compress(I: np.ndarray, alpha: float) -> np.ndarray:
    return np.log1p(alpha * np.clip(I, 0.0, None))


def mad_cfar_1d_fast(x: np.ndarray, training: int, guard: int, k_mad: float) -> np.ndarray:
    """Fast robust CFAR using median_filter approximation.
    Window size = 2*(training+guard)+1.
    Note: includes guard/cell-under-test in stats; for median/MAD this is often acceptable with modest guard.
    """
    T = int(training)
    G = int(guard)
    size = int(2 * (T + G) + 1)
    if size < 3:
        return np.zeros_like(x, dtype=np.float32)
    x32 = x.astype(np.float32, copy=False)
    med = median_filter(x32, size=size, mode='nearest')
    mad = median_filter(np.abs(x32 - med), size=size, mode='nearest')
    thr = med + float(k_mad) * (1.4826 * mad)
    return thr.astype(np.float32, copy=False)


def mad_cfar_1d_exact(x: np.ndarray, training: int, guard: int, k_mad: float) -> np.ndarray:
    """Exact robust CFAR (slow):
      thr[r] = median(training cells) + k_mad * (1.4826 * MAD)
    with training cells excluding guard.
    """
    W = x.shape[0]
    T = int(training)
    G = int(guard)
    thr = np.zeros(W, dtype=np.float32)
    for r in range(W):
        l0 = r - (G + T)
        l1 = r - G
        r0 = r + G + 1
        r1 = r + G + 1 + T
        l0c = max(0, l0); l1c = max(0, l1)
        r0c = min(W, r0); r1c = min(W, r1)
        left = x[l0c:l1c] if l1c > l0c else None
        right = x[r0c:r1c] if r1c > r0c else None
        if left is None and right is None:
            thr[r] = 0.0
            continue
        tr = right if left is None else left if right is None else np.concatenate([left, right], axis=0)
        if tr.size < 8:
            thr[r] = 0.0
            continue
        med = np.median(tr)
        mad = np.median(np.abs(tr - med))
        thr[r] = float(med + float(k_mad) * (1.4826 * mad))
    return thr

def ca_cfar_1d(x: np.ndarray, training: int, guard: int, k: float) -> np.ndarray:
    """
    CA-CFAR (aditivo) sobre un vector 1D:
      thr[r] = mu + k*sigma
    donde mu y sigma se estiman sobre training cells excluyendo guard.
    """
    W = x.shape[0]
    T = int(training)
    G = int(guard)

    # cumsum para mu y sigma
    c1 = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    c2 = np.concatenate([[0.0], np.cumsum((x * x), dtype=np.float64)])
    thr = np.zeros(W, dtype=np.float32)

    for r in range(W):
        l0 = r - (G + T)
        l1 = r - G
        r0 = r + G + 1
        r1 = r + G + 1 + T

        l0c = max(0, l0); l1c = max(0, l1)
        r0c = min(W, r0); r1c = min(W, r1)

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

# def ca_cfar_1d(x: np.ndarray, training: int, guard: int, k: float) -> np.ndarray:
#     """
#     CA-CFAR threshold for 1D signal x over range bins.
#     threshold[r] = mean(training cells excluding guard) * k
#     """
#     W = x.shape[0]
#     T = int(training)
#     G = int(guard)

#     # cumulative sum for fast window means
#     c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
#     thr = np.zeros(W, dtype=np.float32)

#     for r in range(W):
#         l0 = r - (G + T)
#         l1 = r - G
#         r0 = r + G + 1
#         r1 = r + G + 1 + T

#         # clamp
#         l0c = max(0, l0); l1c = max(0, l1)
#         r0c = min(W, r0); r1c = min(W, r1)

#         n = 0
#         s = 0.0
#         if l1c > l0c:
#             s += c[l1c] - c[l0c]
#             n += (l1c - l0c)
#         if r1c > r0c:
#             s += c[r1c] - c[r0c]
#             n += (r1c - r0c)

#         mu = (s / n) if n > 0 else 0.0
#         thr[r] = float(mu * k)

#     return thr


def sigmoid(z: np.ndarray) -> np.ndarray:
    # numerically stable sigmoid (avoid overflow in exp)
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def build_cfar_mask_and_weight(I: np.ndarray, training: int, guard: int,
                               k: float, soft_tau: float,
                               method: str = "gauss", k_mad: float = 6.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute CFAR along range for each azimuth row.
    Outputs:
      - mask: bool (hard detections)
      - weight: float32 in [0,1] (soft detections)
    method:
      - 'gauss'    : CA-CFAR with thr = mu + k*sigma (fast, via cumsum)
      - 'mad'      : MAD-CFAR fast (median_filter approximation)
      - 'mad_exact': MAD-CFAR exact (slow)
    """
    H, W = I.shape
    mask = np.zeros((H, W), dtype=bool)
    w = np.zeros((H, W), dtype=np.float32)
    tau = max(float(soft_tau), 1e-6)

    method = (method or "gauss").lower()

    for a in range(H):
        x = I[a, :]
        if method in ("mad", "mad_fast", "mad_filter"):
            thr = mad_cfar_1d_fast(x, training=training, guard=guard, k_mad=k_mad)
        elif method in ("mad_exact", "mad_loop"):
            thr = mad_cfar_1d_exact(x, training=training, guard=guard, k_mad=k_mad)
        else:
            thr = ca_cfar_1d(x, training=training, guard=guard, k=k)

        d = x - thr
        mask[a, :] = d > 0.0
        w[a, :] = sigmoid(d / tau).astype(np.float32)

    return mask, w


def polar_indices_to_points(mask: np.ndarray, weight: np.ndarray, *,
                            R_max_m: float, azimuth_ccw: bool, az_offset_deg: float,
                            min_range_m: float, max_range_m: float,
                            max_points: int, mirror_x: bool) -> np.ndarray:
    """
    Convert selected polar bins -> (x,y,0) points. Picks up to max_points by weight.
    """
    H, W = mask.shape
    dr = float(R_max_m) / float(W)

    idx = np.argwhere(mask)
    if idx.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    # score by weight
    scores = weight[idx[:, 0], idx[:, 1]]
    order = np.argsort(scores)[::-1]

    pts: List[Tuple[float, float, float]] = []
    for k in order:
        a = int(idx[k, 0])  # azimuth index
        r = int(idx[k, 1])  # range index

        rng = (r + 0.5) * dr
        if rng < min_range_m or rng > max_range_m:
            continue

        theta = (a + 0.5) * (2.0 * math.pi / float(H))
        if not azimuth_ccw:
            theta = (2.0 * math.pi) - theta
        theta = theta + _deg2rad(az_offset_deg)
        theta = theta % (2.0 * math.pi)

        # convention: x=r*sin(theta), y=r*cos(theta)
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
    """
    Returns (T_4x4, fitness, rmse). Points are (N,3) with z=0.
    """
    src = o3d.geometry.PointCloud()
    tgt = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(src_pts)
    tgt.points = o3d.utility.Vector3dVector(tgt_pts)

    # small downsample for stability (optional)
    # src = src.voxel_down_sample(voxel_size=0.5)
    # tgt = tgt.voxel_down_sample(voxel_size=0.5)

    result = o3d.pipelines.registration.registration_icp(
        src, tgt,
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/config_MOANA.yaml"))
    ap.add_argument("--input_dir", type=Path, required=True, help="Folder with polar PNGs for ONE band.")
    ap.add_argument("--band", choices=["xband", "wband"], required=True)
    ap.add_argument("--output_csv", type=Path, default=Path("outputs/radar_delta.csv"))
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=0, help="If >0, process only first N frames.")
    ap.add_argument("--debug_dir", type=Path, default=Path(""), help="If set (non-empty), save CFAR debug images here.")
    ap.add_argument("--debug_n", type=int, default=1, help="How many frames to dump debug images for.")
    ap.add_argument("--cfar_out_dir", type=Path, default=Path(""), help="If set (non-empty), dump CFAR arrays (*.npz + *.json) per frame.")
    ap.add_argument("--cfar_only", action="store_true", help="Only compute and dump CFAR results (skip ICP/CSV).")     
    ap.add_argument("--debug_cart", action="store_true",
                        help="Save cartesian debug PNGs.")
    ap.add_argument("--debug_polar", action="store_true",
                        help="Save polar debug PNGs.")
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
        H=H, W=W, R_max_m=R_max_m, grid_size=grid_size,
        azimuth_ccw=azimuth_ccw, az_offset_deg=az_offset_deg, mirror_x=mirror_x
    )

    rc = rp.get("range_comp", {})
    rc_enabled = bool(rc.get("enabled", True))
    rc_gamma = float(rc.get("gamma", 1.5))
    rc_mode = str(rc.get("mode", "mul"))
    rc_r0 = float(rc.get("r0", 0.02))

    lc = rp.get("log_compress", {})
    lc_enabled = bool(lc.get("enabled", True))
    lc_alpha = float(lc.get("alpha", 1.0))

    # Wiener (2-D adaptive noise removal) in POLAR domain, before CFAR
    wien = rp.get("wiener", {})
    wien_enable = bool(wien.get("enabled", True))
    wien_mysize = tuple(wien.get("mysize", (5, 21)))  # (azimuth, range)
    wien_noise = wien.get("noise", None)  # None => auto-estimate

    cfar = rp.get("cfar", {})
    cfar_method = str(cfar.get("method", "gauss"))
    training = int(cfar.get("training", 24))
    guard = int(cfar.get("guard", 6))
    k = float(cfar.get("k", 1.6))
    k_mad = float(cfar.get("k_mad", 6.0))
    cfar_kz = int(cfar.get("kz", 3))  # for 2D CFAR variants
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
    items = items[::max(1, int(args.stride))]
    if int(args.max_frames) > 0:
        items = items[: int(args.max_frames)]
    out = args.output_csv
    out.parent.mkdir(parents=True, exist_ok=True)

    mask_hist: Deque[np.ndarray] = deque(maxlen=max(1, K))

    # def preprocess_one(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #     I_raw = load_polar_png(path, H=H, W=W)
    #     I = I_raw.astype(np.float32, copy=True)

    #     # 1) Wiener (si aplica)
    #     if wien_enable:
    #         I = wiener(I, mysize=wien_mysize, noise=wien_noise)
    #         I = np.nan_to_num(I, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    #     # 2) Range compensation (ideal: OFF durante tuning)
    #     if rc_enabled:
    #         I = range_comp(I, R_max_m=R_max_m, gamma=rc_gamma, mode=rc_mode, r0=rc_r0)

    #     # 3) Log compress
    #     if lc_enabled:
    #         I = log_compress(I, alpha=lc_alpha)

    #     # ---------- B) 2D BACKGROUND REMOVAL (apply BEFORE normalization) ----------
    #     # background suave (azimuth, range). Aumentá range si persisten anillos.
    #     bg = median_filter(I, size=(21, 151))  # probar (15,101), (21,151), (31,201)
    #     I_pre = I - bg
    #     I_pre[I_pre < 0.0] = 0.0

    #     # 4) Normalización robusta (sobre I_pre)
    #     I_pre = I_pre - np.percentile(I_pre, 5.0)
    #     I_pre = np.clip(I_pre, 0.0, None)
    #     p95 = np.percentile(I_pre, 95.0)
    #     if p95 > 0:
    #         I_norm = np.clip(I_pre / p95, 0.0, 1.0).astype(np.float32)
    #     else:
    #         I_norm = I_pre.astype(np.float32)

    #     # 5) CFAR
    #     mask, w = build_cfar_mask_and_weight(
    #         I_norm, training=training, guard=guard, k=k, soft_tau=soft_tau,
    #         method=cfar_method, k_mad=k_mad
    #     )
    #     return I_raw, I_norm, mask, w
    
    # def preprocess_one(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #     I_raw = load_polar_png(path, H=H, W=W)
    #     I = I_raw.astype(np.float32, copy=True)

    #     # 1) Wiener (si aplica)
    #     if wien_enable:
    #         I = wiener(I, mysize=wien_mysize, noise=wien_noise)
    #         I = np.nan_to_num(I, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    #     # 2) Range compensation (ideal: OFF durante tuning)
    #     if rc_enabled:
    #         I = range_comp(I, R_max_m=R_max_m, gamma=rc_gamma, mode=rc_mode, r0=rc_r0)

    #     # 3) Log compress
    #     if lc_enabled:
    #         I = log_compress(I, alpha=lc_alpha)

    #     # ---------- A) RADIAL BASELINE (apply BEFORE normalization) ----------
    #     # percentil bajo por rango + smoothing en rango
    #     p = 15  # probar 10, 15, 20
    #     baseline = np.percentile(I, p, axis=0).astype(np.float32)
    #     baseline_s = median_filter(baseline, size=51)  # 31/51/71
    #     I_pre = I - baseline_s[None, :]
    #     I_pre[I_pre < 0.0] = 0.0

    #     # 4) Normalización robusta (sobre I_pre)
    #     I_pre = I_pre - np.percentile(I_pre, 5.0)
    #     I_pre = np.clip(I_pre, 0.0, None)
    #     p95 = np.percentile(I_pre, 95.0)
    #     if p95 > 0:
    #         I_norm = np.clip(I_pre / p95, 0.0, 1.0).astype(np.float32)
    #     else:
    #         I_norm = I_pre.astype(np.float32)

    #     # 5) CFAR
    #     mask, w = build_cfar_mask_and_weight(
    #         I_norm, training=training, guard=guard, k=k, soft_tau=soft_tau,
    #         method=cfar_method, k_mad=k_mad
    #     )
    #     return I_raw, I_norm, mask, w

    def preprocess_one(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        I_raw = load_polar_png(path, H=H, W=W)
        I = I_raw.astype(np.float32, copy=True)

        # 1) Wiener
        if wien_enable:
            I = wiener(I, mysize=wien_mysize, noise=wien_noise)
            I = np.nan_to_num(I, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        # 2) Range comp / log
        if rc_enabled:
            I = range_comp(I, R_max_m=R_max_m, gamma=rc_gamma, mode=rc_mode, r0=rc_r0)
        if lc_enabled:
            I = log_compress(I, alpha=lc_alpha)

        # 3) Smooth Radial Baseline Removal on I (not normalized) ----------
        # Esto ayuda a rings, pero NO arregla spokes por sí solo.
        p = 20  # 10..30
        base_r = np.percentile(I, p, axis=0).astype(np.float32)
        base_r = median_filter(base_r, size=51)
        I = I - base_r[None, :]
        I[I < 0.0] = 0.0

        # 4) AZIMUTH ROBUST WHITENING (key for spokes) ----------
        # Robust z-score per azimuth row
        row_med = np.median(I, axis=1, keepdims=True).astype(np.float32)
        row_mad = np.median(np.abs(I - row_med), axis=1, keepdims=True).astype(np.float32)
        row_sigma = 1.4826 * row_mad + 1e-6  # robust std

        Z = (I - row_med) / row_sigma
        Z = np.clip(Z, 0.0, None)  # nos quedamos con “excesos” positivos

        # 5) Normalize for debug/weight maps (0..1) ----------
        # No uses P5/P95 sobre I_raw; sobre Z es estable
        z95 = np.percentile(Z, 95.0)
        if z95 > 0:
            I_norm = np.clip(Z / z95, 0.0, 1.0).astype(np.float32)
        else:
            I_norm = Z.astype(np.float32)

        # 6) Detection: threshold in Z space (MUY rápido) ----------
        # k_z ~ 3..6. Arrancá en 4.0.
        k_z = float(cfar_kz) if "cfar_kz" in globals() else 4.0
        mask = Z > k_z

        # weight suave desde Z (evita overflow)
        tau = float(soft_tau)
        zz = np.clip((Z - k_z) / max(tau, 1e-6), -50.0, 50.0)
        w = 1.0 / (1.0 + np.exp(-zz))
        w = w.astype(np.float32)

        return I_raw, I_norm, mask, w

    with out.open("w", newline="") as f:
        wcsv = csv.DictWriter(
            f,
            fieldnames=["t","band","dx","dy","dz","droll","dpitch","dyaw","quality","rmse_m","inlier_ratio","dt_s"],
        )
        wcsv.writeheader()

        # init
        t0, p0 = items[0]
        I0raw, I0n, m0, w0 = preprocess_one(p0)
        mask_hist.append(m0)

        # persistence on first
        m0p = None
        if pers_enabled and len(mask_hist) >= 1:
            m0p = (np.sum(np.stack(list(mask_hist), axis=0), axis=0) >= min_hits)
 
        dbg_dir = args.debug_dir if str(args.debug_dir) != "" else None
        cfar_dir = args.cfar_out_dir if str(args.cfar_out_dir) != "" else None
        dumped = 0
        if dbg_dir is not None and dumped < int(args.debug_n):
            tag = f"{args.band}_{t0:.3f}"
            save_cfar_debug(Path(dbg_dir), tag, I0raw, I0n, m0, w0, pers_mask=m0p, remapper_cart=remap_cart)
            dumped += 1

        if cfar_dir is not None:
            dump_cfar_npz(
                Path(cfar_dir), t0, args.band,
                I_norm=I0n, mask=m0, weight=w0, mask_persist=m0p,
                meta={"source_png": str(p0)}
            )


        if args.cfar_only:

            # Dump CFAR results for all frames and exit (no ICP / no CSV).

            for (t1, p1) in items[1:]:

                I1raw, I1n, m1, w1 = preprocess_one(p1)

                mask_hist.append(m1)


                m1p = None

                if pers_enabled:

                    stack = np.stack(list(mask_hist), axis=0)

                    m1p = (np.sum(stack, axis=0) >= min_hits)


                if dbg_dir is not None and dumped < int(args.debug_n):

                    tag = f"{args.band}_{t1:.3f}"

                    save_cfar_debug(Path(dbg_dir), tag, I1raw, I1n, m1, w1, pers_mask=m1p, remapper_cart=remap_cart)

                    dumped += 1


                if cfar_dir is not None:

                    dump_cfar_npz(

                        Path(cfar_dir), t1, args.band,

                        I_norm=I1n, mask=m1, weight=w1, mask_persist=m1p,

                        meta={"source_png": str(p1)}

                    )


            print(f"[OK] CFAR-only dump complete: dir={cfar_dir} frames={len(items)} band={args.band}")

            return 0



        pts0 = polar_indices_to_points(
            (m0p if m0p is not None else m0), w0,
            R_max_m=R_max_m, azimuth_ccw=azimuth_ccw, az_offset_deg=az_offset_deg,
            min_range_m=min_range_m, max_range_m=max_range_m,
            max_points=max_points, mirror_x=mirror_x,
        )

        for (t1, p1) in items[1:]:
            I1raw, I1n, m1, w1 = preprocess_one(p1)
            mask_hist.append(m1)

            m1p = None
            if pers_enabled:
                stack = np.stack(list(mask_hist), axis=0)
                m1p = (np.sum(stack, axis=0) >= min_hits)

            if dbg_dir is not None and dumped < int(args.debug_n):
                tag = f"{args.band}_{t1:.3f}"
                save_cfar_debug(Path(dbg_dir), tag, I1raw, I1n, m1, w1, pers_mask=m1p, remapper_cart=remap_cart)
                dumped += 1

            pts1 = polar_indices_to_points(
                (m1p if m1p is not None else m1), w1,
                R_max_m=R_max_m, azimuth_ccw=azimuth_ccw, az_offset_deg=az_offset_deg,
                min_range_m=min_range_m, max_range_m=max_range_m,
                max_points=max_points, mirror_x=mirror_x,
            )

            dt = float(t1 - t0)
            if dt <= 0.0 or pts0.shape[0] < 50 or pts1.shape[0] < 50:
                # not enough points yet
                t0, p0, pts0 = t1, p1, pts1
                continue

            # Estimate transform that maps pts0 -> pts1
            T, fitness, rmse = icp_2d(pts0, pts1, max_corr=max_corr, max_iters=max_iters)
            dx, dy, dyaw = T_to_dxdy_dyaw(T)

            wcsv.writerow({
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
            })

            t0, p0, pts0 = t1, p1, pts1

    print(f"[OK] wrote {out} for band={args.band} frames={len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
