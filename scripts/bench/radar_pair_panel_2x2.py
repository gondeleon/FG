#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from slamboat.radar.odometry import estimate_pair, load_clean_npz

try:
    from scipy.ndimage import rotate as nd_rotate
    from scipy.ndimage import shift as nd_shift
except Exception as e:
    raise RuntimeError("This script requires SciPy: pip install scipy") from e


def _norm01(x: np.ndarray, qlo=0.01, qhi=0.995) -> np.ndarray:
    x = x.astype(np.float32)
    lo = float(np.quantile(x, qlo))
    hi = float(np.quantile(x, qhi))
    y = (x - lo) / (hi - lo + 1e-9)
    return np.clip(y, 0.0, 1.0)


def _warp_se2_image(
    img: np.ndarray,
    dx_m: float,
    dy_m: float,
    yaw_rad: float,
    px_size_m: float,
    yaw_sign: float = 1.0,
) -> np.ndarray:
    """
    Applies SE(2) transform to the *image*:
      1) rotate by yaw (degrees)
      2) shift by (dy, dx) in pixels

    Conventions assumed from your pipeline:
      - x right, y down
      - dx,dy in meters in image axes
    """
    ang_deg = float(np.rad2deg(yaw_sign * yaw_rad))
    out = nd_rotate(img, ang_deg, reshape=False, order=1, mode="nearest")
    sx = float(dx_m / px_size_m)  # cols
    sy = float(dy_m / px_size_m)  # rows
    out = nd_shift(out, shift=(sy, sx), order=1, mode="nearest")
    return out


def _rgb_blend(ref01: np.ndarray, cur01: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """
    Visual:
      - ref in grayscale
      - cur in red channel
    """
    ref_rgb = np.stack([ref01, ref01, ref01], axis=-1)
    cur_rgb = np.zeros_like(ref_rgb)
    cur_rgb[..., 0] = cur01  # red
    out = (1.0 - alpha) * ref_rgb + alpha * cur_rgb
    return np.clip(out, 0.0, 1.0)


def make_panel(
    ref_npz: Path,
    cur_npz: Path,
    out_png: Path,
    cart_res: int = 512,
    r_max: float = 600.0,
    alpha: float = 0.55,
    yaw_sign: float = 1.0,
):
    # Build raw_cart for visualization (same for all methods)
    from slamboat.radar.radar_polar_clean import polar_to_cart_bilinear

    d0 = load_clean_npz(ref_npz)
    d1 = load_clean_npz(cur_npz)

    C0_raw, xs, ys = polar_to_cart_bilinear(d0["I_pol_f"], r_max=r_max, cart_res=cart_res)
    C1_raw, _, _ = polar_to_cart_bilinear(d1["I_pol_f"], r_max=r_max, cart_res=cart_res)

    px_size_m = float((2.0 * r_max) / (cart_res - 1))

    ref01 = _norm01(C0_raw)
    cur01 = _norm01(C1_raw)

    methods = [
        "icp_raw",
        "icp_clean_nowater",
        "hybrid_clean",
        "hybrid_clean_nowater",
    ]

    results = []
    blends = []

    for m in methods:
        res, _ = estimate_pair(ref_npz, cur_npz, method=m, cart_res=cart_res, r_max=r_max)
        results.append(res)

        cur_w = _warp_se2_image(cur01, res.dx, res.dy, res.yaw_rad, px_size_m=px_size_m, yaw_sign=yaw_sign)
        blends.append(_rgb_blend(ref01, cur_w, alpha=alpha))

    # Plot 2x2
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for ax, m, res, img_rgb in zip(axes, methods, results, blends):
        ax.imshow(img_rgb, origin="lower")
        title = (
            f"{m}\n"
            f"dt={res.dt_s*1000:.0f} ms | yaw={np.rad2deg(res.yaw_rad):.2f} deg | "
            f"dx={res.dx:.2f} m dy={res.dy:.2f} m"
        )
        extra = []
        if res.psr is not None:
            extra.append(f"PSR={res.psr:.1f}")
        if res.icp_rmse is not None:
            extra.append(f"rmse={res.icp_rmse:.3f}")
        if res.icp_fitness is not None:
            extra.append(f"fit={res.icp_fitness:.3f}")
        if extra:
            title += "\n" + " | ".join(extra)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(f"Radar pair: {ref_npz.name} -> {cur_npz.name}  (visual base: raw_cart)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_npz", type=Path, required=True)
    ap.add_argument("--cur_npz", type=Path, required=True)
    ap.add_argument("--out_png", type=Path, required=True)
    ap.add_argument("--cart_res", type=int, default=512)
    ap.add_argument("--r_max", type=float, default=600.0)
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--yaw_sign", type=float, default=1.0, help="Flip sign if rotation looks wrong (use -1).")
    args = ap.parse_args()

    make_panel(
        args.ref_npz,
        args.cur_npz,
        args.out_png,
        cart_res=args.cart_res,
        r_max=args.r_max,
        alpha=args.alpha,
        yaw_sign=args.yaw_sign,
    )
    print(f"[OK] wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
