#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import csv
import numpy as np

from slamboat.radar.odometry import load_clean_npz
from slamboat.radar.radar_polar_clean import polar_to_cart_nearest, polar_to_cart_bilinear


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return float(inter / (uni + 1e-9))


def centroid_xy(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    # mask is cart (H,W), xs len W, ys len H
    idx = np.argwhere(mask > 0.5)
    if idx.size == 0:
        return (np.nan, np.nan)
    yy = ys[idx[:, 0]]
    xx = xs[idx[:, 1]]
    return (float(np.mean(xx)), float(np.mean(yy)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, default=Path("outputs/watermask_qc.csv"))
    ap.add_argument("--cart_res", type=int, default=512)
    ap.add_argument("--r_max", type=float, default=600.0)
    ap.add_argument("--max_frames", type=int, default=2000)
    args = ap.parse_args()

    npzs = sorted(args.npz_dir.glob("*.npz"))[: args.max_frames]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    prev_w_pol = None
    prev_w_cart = None

    with args.out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "npz",
            "water_frac_pol", "water_frac_cart",
            "iou_pol_prev", "iou_cart_prev",
            "cx_cart_m", "cy_cart_m",
            "mean_I_raw_in_water", "mean_I_raw_out_water"
        ])

        for p in npzs:
            d = load_clean_npz(p)
            I_pol = d["I_pol_f"].astype(np.float32)
            W_pol = d["mask_water_pol"].astype(np.uint8)

            water_frac_pol = float((W_pol > 0).mean())

            # cart mask
            W_cart, xs, ys = polar_to_cart_nearest(W_pol, r_max=args.r_max, cart_res=args.cart_res)
            W_cart = (W_cart > 0.5).astype(np.uint8)
            water_frac_cart = float(W_cart.mean())

            # cart intensity for stats
            I_cart, _, _ = polar_to_cart_bilinear(I_pol, r_max=args.r_max, cart_res=args.cart_res)

            if W_cart.sum() > 0:
                mean_in = float(I_cart[W_cart > 0].mean())
            else:
                mean_in = np.nan
            if (W_cart == 0).sum() > 0:
                mean_out = float(I_cart[W_cart == 0].mean())
            else:
                mean_out = np.nan

            iou_pol_prev = iou(W_pol > 0, prev_w_pol > 0) if prev_w_pol is not None else np.nan
            iou_cart_prev = iou(W_cart > 0, prev_w_cart > 0) if prev_w_cart is not None else np.nan

            cx, cy = centroid_xy(W_cart, xs, ys)

            w.writerow([
                str(p),
                f"{water_frac_pol:.4f}", f"{water_frac_cart:.4f}",
                "" if np.isnan(iou_pol_prev) else f"{iou_pol_prev:.4f}",
                "" if np.isnan(iou_cart_prev) else f"{iou_cart_prev:.4f}",
                "" if np.isnan(cx) else f"{cx:.2f}",
                "" if np.isnan(cy) else f"{cy:.2f}",
                "" if np.isnan(mean_in) else f"{mean_in:.4f}",
                "" if np.isnan(mean_out) else f"{mean_out:.4f}",
            ])

            prev_w_pol = W_pol
            prev_w_cart = W_cart

    print(f"[OK] wrote {args.out_csv}")
    print("Tip: flag mask drift if IoU drops systematically (<0.7) or centroid jumps a lot frame-to-frame.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
