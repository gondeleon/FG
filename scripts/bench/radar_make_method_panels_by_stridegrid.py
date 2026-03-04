#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from slamboat.radar.odometry import estimate_pair, load_clean_npz
from slamboat.radar.radar_polar_clean import polar_to_cart_bilinear, polar_to_cart_nearest

from scipy.ndimage import rotate as nd_rotate
from scipy.ndimage import shift as nd_shift
from pyproj import Transformer

def wrap_pi(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi

def norm01(x: np.ndarray, qlo=0.01, qhi=0.995) -> np.ndarray:
    x = x.astype(np.float32)
    lo = float(np.quantile(x, qlo))
    hi = float(np.quantile(x, qhi))
    return np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)

def warp(img01: np.ndarray, dx_m: float, dy_m: float, yaw_rad: float, px: float, yaw_sign: float) -> np.ndarray:
    ang_deg = float(np.rad2deg(yaw_sign * yaw_rad))
    out = nd_rotate(img01, ang_deg, reshape=False, order=1, mode="nearest")
    sx = float(dx_m / px)
    sy = float(dy_m / px)
    out = nd_shift(out, shift=(sy, sx), order=1, mode="nearest")
    return out

def blend(ref01: np.ndarray, cur01: np.ndarray, alpha: float) -> np.ndarray:
    ref_rgb = np.stack([ref01, ref01, ref01], axis=-1)
    cur_rgb = np.zeros_like(ref_rgb)
    cur_rgb[..., 0] = cur01
    return np.clip((1.0 - alpha) * ref_rgb + alpha * cur_rgb, 0.0, 1.0)

def get_cart_view(d, view: str, cart_res: int, r_max: float):
    if view == "raw":
        C, _, _ = polar_to_cart_bilinear(d["I_pol_f"], r_max=r_max, cart_res=cart_res)
        return C
    if view == "clean":
        C, _, _ = polar_to_cart_bilinear(d["I_clean_f"], r_max=r_max, cart_res=cart_res)
        return C
    if view == "clean_nowater":
        C, _, _ = polar_to_cart_bilinear(d["I_clean_f"], r_max=r_max, cart_res=cart_res)
        W_pol = d["mask_water_pol"].astype(np.uint8)
        W_cart, _, _ = polar_to_cart_nearest(W_pol, r_max=r_max, cart_res=cart_res)
        W = (W_cart > 0.5).astype(np.float32)
        return C * (1.0 - W)
    raise ValueError(f"Unknown view: {view}")

def view_from_method(method: str) -> str:
    if "clean_nowater" in method:
        return "clean_nowater"
    if "clean" in method:
        return "clean"
    return "raw"

def utm_epsg_from_lonlat(lon: float, lat: float) -> str:
    zone = int(np.floor((lon + 180.0) / 6.0) + 1)
    return f"EPSG:{32600 + zone}" if lat >= 0 else f"EPSG:{32700 + zone}"

def interp_linear(t, y, tq):
    i = np.searchsorted(t, tq)
    if i <= 0: return float(y[0])
    if i >= len(t): return float(y[-1])
    t0, t1 = t[i-1], t[i]
    a = (tq - t0) / (t1 - t0 + 1e-12)
    return float((1-a)*y[i-1] + a*y[i])

def interp_yaw(t, yaw, tq):
    c = np.cos(yaw); s = np.sin(yaw)
    cq = interp_linear(t, c, tq)
    sq = interp_linear(t, s, tq)
    return float(np.arctan2(sq, cq))

def pose_at(t, x, y, yaw, tq):
    return (interp_linear(t, x, tq),
            interp_linear(t, y, tq),
            interp_yaw(t, yaw, tq))

def rel_delta(p0, p1):
    x0,y0,psi0 = p0
    x1,y1,psi1 = p1
    dxg = x1 - x0; dyg = y1 - y0
    c = np.cos(psi0); s = np.sin(psi0)
    dx =  c*dxg + s*dyg
    dy = -s*dxg + c*dyg
    dpsi = wrap_pi(psi1 - psi0)
    return float(dx), float(dy), float(dpsi)

def load_gnss_series(csv_path: Path, yaw_is_deg: bool, yaw_mode: str, lever_arm_xy: tuple[float,float]):
    df = pd.read_csv(csv_path)
    for c in ["timestamp", "latitude", "longitude", "yaw"]:
        if c not in df.columns:
            raise ValueError(f"GNSS CSV missing '{c}'. Columns={list(df.columns)}")

    t = df["timestamp"].to_numpy(float)
    lon = df["longitude"].to_numpy(float)
    lat = df["latitude"].to_numpy(float)

    epsg = utm_epsg_from_lonlat(float(np.median(lon)), float(np.median(lat)))
    tr = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    xG, yG = tr.transform(lon, lat)
    xG = np.asarray(xG, float); yG = np.asarray(yG, float)

    yaw = df["yaw"].to_numpy(float)
    if yaw_is_deg:
        yaw = np.deg2rad(yaw)
    # configurable convention
    if yaw_mode == "heading_north_cw":
        yaw = (np.pi/2.0) - yaw
    yaw = np.unwrap(yaw)

    # apply lever arm GNSS->Wband in 2D using yaw
    rx, ry = lever_arm_xy
    c = np.cos(yaw); s = np.sin(yaw)
    xW = xG + c*rx - s*ry
    yW = yG + s*rx + c*ry

    order = np.argsort(t)
    return t[order], xW[order], yW[order], yaw[order], epsg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", type=Path, required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--strides", type=int, nargs="+", default=[1,2,4,8])
    ap.add_argument("--base_idx", type=int, default=0)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--cart_res", type=int, default=512)
    ap.add_argument("--r_max", type=float, default=600.0)
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--yaw_sign", type=float, default=1.0)

    # GT GNSS
    ap.add_argument("--gnss_csv", type=Path, required=True)
    ap.add_argument("--gnss_yaw_is_deg", action="store_true")
    ap.add_argument("--gnss_yaw_mode", choices=["as_is","heading_north_cw"], default="as_is")
    ap.add_argument("--lever_arm_x", type=float, default=0.250)
    ap.add_argument("--lever_arm_y", type=float, default=0.0)
    args = ap.parse_args()

    strides = list(args.strides)
    if len(strides) != 4:
        raise SystemExit("Provide exactly 4 strides for a 2x2 grid (e.g. 1 2 4 8).")

    npzs = sorted(args.npz_dir.glob("*.npz"))
    if not npzs:
        raise SystemExit(f"No npz found in {args.npz_dir}")
    if args.base_idx + max(strides) >= len(npzs):
        raise SystemExit("base_idx+max_stride out of range.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    ref_npz = npzs[args.base_idx]
    d0 = load_clean_npz(ref_npz)
    t0 = int(ref_npz.stem) * 1e-9

    t_g, x_w, y_w, yaw_g, epsg = load_gnss_series(
        args.gnss_csv,
        yaw_is_deg=args.gnss_yaw_is_deg,
        yaw_mode=args.gnss_yaw_mode,
        lever_arm_xy=(args.lever_arm_x, args.lever_arm_y),
    )
    tmin, tmax = float(t_g[0]), float(t_g[-1])
    if not (tmin <= t0 <= tmax):
        raise SystemExit(f"Radar t0={t0} not within GNSS range [{tmin},{tmax}]")

    def gt_delta_for(t1):
        if not (tmin <= t1 <= tmax):
            return None
        p0 = pose_at(t_g, x_w, y_w, yaw_g, t0)
        p1 = pose_at(t_g, x_w, y_w, yaw_g, t1)
        return rel_delta(p0, p1)

    px = float((2.0 * args.r_max) / (args.cart_res - 1))

    for method in args.methods:
        fig, axes = plt.subplots(2,2, figsize=(12,10))
        axes = axes.ravel()

        for k, s in enumerate(strides):
            ax = axes[k]
            cur_npz = npzs[args.base_idx + s]
            d1 = load_clean_npz(cur_npz)
            t1 = int(cur_npz.stem) * 1e-9

            view = view_from_method(method)
            C0 = get_cart_view(d0, view, args.cart_res, args.r_max)
            C1 = get_cart_view(d1, view, args.cart_res, args.r_max)

            ref01 = norm01(C0)
            cur01 = norm01(C1)

            res, dbg = estimate_pair(ref_npz, cur_npz, method=method, cart_res=args.cart_res, r_max=args.r_max)
            cur_warp = warp(cur01, res.dx, res.dy, res.yaw_rad, px=px, yaw_sign=args.yaw_sign)
            img = blend(ref01, cur_warp, alpha=args.alpha)

            ax.imshow(img, origin="lower")
            ax.axis("off")

            # GT + error
            gt = gt_delta_for(t1)
            if gt is not None:
                dxg, dyg, yawg = gt
                dyg = -dyg  # align GNSS-GT y-axis to radar-odom convention (empirically verified)
                ex = res.dx - dxg
                ey = res.dy - dyg
                epsi = wrap_pi(res.yaw_rad - yawg)
                et = float(np.hypot(ex, ey))
                gt_line = f"GT: dx={dxg:.2f} dy={dyg:.2f} yaw={np.rad2deg(yawg):.2f}°"
                er_line = f"err: |t|={et:.2f} yaw={np.rad2deg(epsi):.2f}°"
            else:
                gt_line = "GT: n/a"
                er_line = ""

            title = f"stride={s}  est: dx={res.dx:.2f} dy={res.dy:.2f} yaw={np.rad2deg(res.yaw_rad):.2f}°"
            ax.set_title(title, fontsize=9)

            extra = []
            if res.psr is not None: extra.append(f"PSR={res.psr:.1f}")
            if res.icp_rmse is not None: extra.append(f"rmse={res.icp_rmse:.3f}")
            if res.icp_fitness is not None: extra.append(f"fit={res.icp_fitness:.3f}")

            box = "\n".join(extra + [gt_line] + ([er_line] if er_line else []))
            ax.text(0.02, 0.02, box,
                    transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"))

        fig.suptitle(
            f"{method} — ref idx={args.base_idx} — GNSS→Wband lever-arm ({args.lever_arm_x:.3f},{args.lever_arm_y:.3f}) m — yaw_mode={args.gnss_yaw_mode} — {epsg}",
            fontsize=11
        )
        plt.tight_layout()
        out = args.out_dir / f"method_{method}__refidx_{args.base_idx:05d}__stridegrid_GT_GNSS.png"
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print("[OK]", out)

if __name__ == "__main__":
    main()
