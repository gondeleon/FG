#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Radar polar cleaning (Script 1)

- Builds WATER mask in POLAR using AOI water polygons + manual OSM↔RADAR calibration (Sim2+Flip).
- Estimates sea-noise statistics in POLAR (range-dependent median + MAD).
- Produces a cleaned polar image suitable for downstream CFAR / point extraction.
- Debug outputs are written in CARTESIAN (PNG) for easy inspection.

Assumptions:
- Radar PNG is grayscale uint8 with shape (Naz, Nrg) = (400, 3424) (azimuth x range).
- r_max maps to last range bin: r = (idx/(Nrg-1))*r_max.
- Azimuth bins map to angle theta = (idx/(Naz-1))*2*pi (CCW, theta=0 at +X).
  This matches the polar->cart display used by your calibration GUI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
from PIL import Image

import geopandas as gpd
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.affinity import affine_transform as shp_aff

# Optional fast vectorized contains (Shapely<2 provides shapely.vectorized)
try:
    from shapely import vectorized as shp_vec  # type: ignore
    _HAS_VEC = True
except Exception:
    shp_vec = None
    _HAS_VEC = False

# Optional Wiener-like filter via SciPy
try:
    from scipy.ndimage import uniform_filter
    _HAS_SCIPY = True
except Exception:
    uniform_filter = None
    _HAS_SCIPY = False


def load_radar_polar_u8(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    I = np.asarray(img, dtype=np.uint8)
    if I.ndim != 2:
        raise ValueError(f"Expected grayscale radar PNG, got shape {I.shape}")
    return I


def load_calib_json(path: Path) -> Dict:
    d = json.loads(path.read_text())
    if "meta" not in d:
        raise ValueError("Calibration JSON missing 'meta'.")
    return d


def sim2_flip_forward_matrix(d: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    x_radar = L * x_osm_local + t
    where L = (s*R)*F, with F=diag(fx,fy)
    """
    s = float(d.get("s", 1.0))
    rot_deg = float(d.get("rot_deg", 0.0))
    fx = -1.0 if bool(d.get("flip_x", False)) else 1.0
    fy = -1.0 if bool(d.get("flip_y", False)) else 1.0
    tx = float(d.get("tx", 0.0))
    ty = float(d.get("ty", 0.0))

    th = np.deg2rad(rot_deg)
    c, sn = float(np.cos(th)), float(np.sin(th))
    R = np.array([[c, -sn],
                  [sn,  c]], dtype=np.float64)
    F = np.array([[fx, 0.0],
                  [0.0, fy]], dtype=np.float64)
    L = (s * R) @ F
    t = np.array([tx, ty], dtype=np.float64)
    return L, t


def polar_grids(Naz: int, Nrg: int, r_max: float) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0*np.pi, Naz, dtype=np.float64)
    r = np.linspace(0.0, r_max, Nrg, dtype=np.float64)
    return theta, r


def polar_to_xy(theta: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convention: theta=0 => +X, theta increases CCW, Y up.
    """
    ct = np.cos(theta)[:, None]
    st = np.sin(theta)[:, None]
    rr = r[None, :]
    X = rr * ct
    Y = rr * st
    return X, Y


def polar_to_cart_nearest(A_pol: np.ndarray, r_max: float, cart_res: int, fill: float = 0.0):
    Naz, Nrg = A_pol.shape
    xs = np.linspace(-r_max, r_max, cart_res, dtype=np.float64)
    ys = np.linspace(-r_max, r_max, cart_res, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    R = np.sqrt(X*X + Y*Y)
    TH = np.mod(np.arctan2(Y, X), 2.0*np.pi)

    r_idx = (R / r_max) * (Nrg - 1)
    t_idx = (TH / (2.0*np.pi)) * (Naz - 1)

    oor = R > r_max
    r_idx = np.clip(np.rint(r_idx), 0, Nrg - 1).astype(np.int32)
    t_idx = np.clip(np.rint(t_idx), 0, Naz - 1).astype(np.int32)

    out = A_pol[t_idx, r_idx].astype(np.float32)
    out[oor] = float(fill)
    return out.astype(np.float32), xs.astype(np.float32), ys.astype(np.float32)


def polar_to_cart_bilinear(I_pol: np.ndarray, r_max: float, cart_res: int):
    I = I_pol.astype(np.float32)
    Naz, Nrg = I.shape
    xs = np.linspace(-r_max, r_max, cart_res, dtype=np.float64)
    ys = np.linspace(-r_max, r_max, cart_res, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    R = np.sqrt(X*X + Y*Y)
    TH = np.mod(np.arctan2(Y, X), 2.0*np.pi)

    r_idx = (R / r_max) * (Nrg - 1)
    t_idx = (TH / (2.0*np.pi)) * (Naz - 1)

    oor = R > r_max
    r_idx = np.clip(r_idx, 0.0, float(Nrg - 1))
    t_idx = np.clip(t_idx, 0.0, float(Naz - 1))
    r_idx[oor] = 0.0
    t_idx[oor] = 0.0

    r0 = np.floor(r_idx).astype(np.int32)
    t0 = np.floor(t_idx).astype(np.int32)
    r1 = np.clip(r0 + 1, 0, Nrg - 1)
    t1 = np.clip(t0 + 1, 0, Naz - 1)
    fr = (r_idx - r0).astype(np.float32)
    ft = (t_idx - t0).astype(np.float32)

    v00 = I[t0, r0]
    v01 = I[t0, r1]
    v10 = I[t1, r0]
    v11 = I[t1, r1]
    out = (v00*(1-fr)*(1-ft) + v01*(fr)*(1-ft) + v10*(1-fr)*(ft) + v11*(fr)*(ft))
    out[oor] = 0.0
    return out.astype(np.float32), xs.astype(np.float32), ys.astype(np.float32)


def build_water_mask_polar(theta: np.ndarray, r: np.ndarray, L: np.ndarray, t: np.ndarray, water_geom_local):
    """
    We have OSM->RADAR: x_r = L x_o + t
    For point-in-water we need RADAR->OSM: x_o = inv(L) (x_r - t)
    """
    Naz, Nrg = theta.size, r.size
    Xr, Yr = polar_to_xy(theta, r)

    Linv = np.linalg.inv(L)
    Xo = Linv[0, 0]*(Xr - t[0]) + Linv[0, 1]*(Yr - t[1])
    Yo = Linv[1, 0]*(Xr - t[0]) + Linv[1, 1]*(Yr - t[1])

    if water_geom_local is None or water_geom_local.is_empty:
        return np.zeros((Naz, Nrg), dtype=bool)

    if _HAS_VEC:
        return shp_vec.contains(water_geom_local, Xo, Yo).astype(bool)

    # Slow fallback
    pg = prep(water_geom_local)
    mask = np.zeros((Naz, Nrg), dtype=bool)
    for a in range(Naz):
        for j in range(Nrg):
            # avoid geopandas points creation overhead in inner loop:
            from shapely.geometry import Point
            mask[a, j] = pg.contains(Point(float(Xo[a, j]), float(Yo[a, j])))
    return mask


def robust_range_stats(I_f: np.ndarray, mask_water: np.ndarray, eps: float = 1e-6):
    """
    mu_r = median_a(I[a,r] | water)
    sigma_r = 1.4826 * MAD_a(I[a,r] | water)
    """
    J = I_f.astype(np.float32).copy()
    J[~mask_water] = np.nan
    mu = np.nanmedian(J, axis=0).astype(np.float32)
    mad = np.nanmedian(np.abs(J - mu[None, :]), axis=0).astype(np.float32)
    sigma = (1.4826 * mad).astype(np.float32)

    # Fill NaNs if no water samples exist at some ranges
    nan_mu = ~np.isfinite(mu)
    if np.any(nan_mu):
        mu[nan_mu] = np.nanmedian(I_f[:, nan_mu], axis=0).astype(np.float32)
    nan_sig = ~np.isfinite(sigma) | (sigma < eps)
    if np.any(nan_sig):
        sigma[nan_sig] = float(np.nanmedian(sigma[~nan_sig])) if np.any(~nan_sig) else 1.0
    sigma = np.maximum(sigma, eps)
    return mu, sigma


def wiener_like(I: np.ndarray, win_az: int, win_rg: int, eps: float = 1e-6) -> np.ndarray:
    if not _HAS_SCIPY:
        return I.astype(np.float32)
    mu = uniform_filter(I, size=(win_az, win_rg), mode="nearest")
    mu2 = uniform_filter(I*I, size=(win_az, win_rg), mode="nearest")
    var = np.maximum(mu2 - mu*mu, 0.0)
    noise = float(np.median(var))
    out = mu + (np.maximum(var - noise, 0.0) / (var + eps)) * (I - mu)
    return out.astype(np.float32)


def clean_polar(I_f: np.ndarray,
               mask_water: np.ndarray,
               mu_r: np.ndarray,
               alpha_water: float = 0.25,
               wiener_win: Optional[Tuple[int, int]] = (5, 21)):
    """
    - Baseline remove per range: I0 = max(I - mu_r, 0)
    - Optional Wiener-like denoise
    - Attenuate water region
    """
    I0 = np.maximum(I_f - mu_r[None, :], 0.0).astype(np.float32)
    if wiener_win is not None:
        I0 = wiener_like(I0, wiener_win[0], wiener_win[1])
    I0[mask_water] *= float(alpha_water)
    return I0


def save_png_u8(path: Path, arr_u8: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_u8).save(path)


def save_png_float01(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.clip(arr, 0.0, 1.0)
    u8 = (a * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(u8).save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radar_dir", type=Path, required=True)
    ap.add_argument("--aoi_gpkg", type=Path, required=True)
    ap.add_argument("--calib_json", type=Path, required=True)

    ap.add_argument("--out_dir", type=Path, default=Path("outputs/clean"))
    ap.add_argument("--debug_dir", type=Path, default=Path("outputs/dbg_clean"))
    ap.add_argument("--max_frames", type=int, default=5)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--debug_stride", type=int, default=0, help="Write debug PNGs every N frames (0 disables).")
    ap.add_argument("--debug_timestamps", type=str, default="", help="Comma-separated list of frame stems (timestamps) to debug.")

    ap.add_argument("--Naz", type=int, default=400)
    ap.add_argument("--Nrg", type=int, default=3424)
    ap.add_argument("--r_max", type=float, default=600.0)
    ap.add_argument("--cart_res", type=int, default=512)

    ap.add_argument("--alpha_water", type=float, default=0.25)
    ap.add_argument("--wiener", type=str, default="5,21", help="win_az,win_rg or 'off'")
    ap.add_argument("--simplify_m", type=float, default=1.5)

    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    dbg_ts = set()
    if args.debug_timestamps.strip():
        dbg_ts = {s.strip() for s in args.debug_timestamps.split(",") if s.strip()}

    debug_enabled = (args.debug_stride is not None and args.debug_stride > 0) or (len(dbg_ts) > 0)
    if debug_enabled:
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    calib = load_calib_json(args.calib_json)
    L, t = sim2_flip_forward_matrix(calib)

    origin = calib["meta"].get("origin_E0N0", None)
    if origin is None or len(origin) != 2:
        raise ValueError("calib_json meta.origin_E0N0 missing; required to localize AOI.")
    E0, N0 = float(origin[0]), float(origin[1])

    # Load water polygon from AOI and localize by subtracting origin_E0N0
    water_layer = calib["meta"]["layers"].get("water", "water")
    water_gdf = gpd.read_file(args.aoi_gpkg, layer=water_layer)
    water_gdf = water_gdf[water_gdf.geometry.notna() & ~water_gdf.geometry.is_empty]
    water_geom = unary_union(water_gdf.geometry.values) if len(water_gdf) else None
    if water_geom is not None and not water_geom.is_empty:
        water_local = shp_aff(water_geom, [1, 0, 0, 1, -E0, -N0])
        if args.simplify_m > 0:
            water_local = water_local.simplify(args.simplify_m, preserve_topology=True)
    else:
        water_local = None

    theta, r = polar_grids(args.Naz, args.Nrg, args.r_max)

    frames = sorted(args.radar_dir.glob("*.png"))
    frames = frames[: args.max_frames * args.stride : args.stride]
    if not frames:
        raise FileNotFoundError(f"No PNG frames in {args.radar_dir}")

    # Wiener parse
    wiener_win = None
    if args.wiener.strip().lower() != "off":
        try:
            a, b = args.wiener.split(",")
            wiener_win = (int(a), int(b))
        except Exception:
            wiener_win = (5, 21)

    print(f"[INFO] shapely.vectorized.contains: {_HAS_VEC}")
    print(f"[INFO] scipy uniform_filter (wiener-like): {_HAS_SCIPY}")
    print(f"[INFO] frames={len(frames)} stride={args.stride}")

    for i, fp in enumerate(frames, 1):
        print(f"[INFO] [{i}/{len(frames)}] {fp.name}")
        I_u8 = load_radar_polar_u8(fp)

        stem = fp.stem
        do_dbg = False
        if debug_enabled:
            if stem in dbg_ts:
                do_dbg = True
            elif args.debug_stride > 0 and ((i - 1) % args.debug_stride == 0):
                do_dbg = True
        # En tu caso: ya viene (400,3424) => OK
        if I_u8.shape == (args.Nrg, args.Naz):
            I_u8 = I_u8.T
        if I_u8.shape != (args.Naz, args.Nrg):
            raise ValueError(f"Radar shape {I_u8.shape} != expected ({args.Naz},{args.Nrg})")

        I_f = I_u8.astype(np.float32) / 255.0

        mask_water = build_water_mask_polar(theta, r, L, t, water_local)
        mu_r, sigma_r = robust_range_stats(I_f, mask_water)
        I_clean = clean_polar(I_f, mask_water, mu_r, alpha_water=args.alpha_water, wiener_win=wiener_win)
        if do_dbg:
            # Debug in Cartesian
            raw_cart, _, _ = polar_to_cart_bilinear(I_f, r_max=args.r_max, cart_res=args.cart_res)
            clean_cart, _, _ = polar_to_cart_bilinear(I_clean, r_max=args.r_max, cart_res=args.cart_res)
            w_cart, _, _ = polar_to_cart_nearest(mask_water.astype(np.uint8), r_max=args.r_max, cart_res=args.cart_res)

            save_png_float01(args.debug_dir / f"{stem}_cart_raw.png", raw_cart)
            save_png_float01(args.debug_dir / f"{stem}_cart_clean.png", clean_cart)
            save_png_u8(args.debug_dir / f"{stem}_cart_watermask.png", (w_cart > 0.5).astype(np.uint8) * 255)

        meta = {
            "frame_png": str(fp),
            "aoi_gpkg": str(args.aoi_gpkg),
            "calib_json": str(args.calib_json),
            "Naz": args.Naz,
            "Nrg": args.Nrg,
            "r_max": args.r_max,
            "cart_res": args.cart_res,
            "alpha_water": args.alpha_water,
            "wiener_win": None if wiener_win is None else list(wiener_win),
            "simplify_m": args.simplify_m,
            "calib": {
                "s": float(calib.get("s", 1.0)),
                "rot_deg": float(calib.get("rot_deg", 0.0)),
                "tx": float(calib.get("tx", 0.0)),
                "ty": float(calib.get("ty", 0.0)),
                "flip_x": bool(calib.get("flip_x", False)),
                "flip_y": bool(calib.get("flip_y", False)),
                "origin_E0N0": [E0, N0],
            },
        }

        out_npz = args.out_dir / f"{stem}.npz"
        np.savez_compressed(
            out_npz,
            I_pol_u8=I_u8,
            I_pol_f=I_f,
            mask_water_pol=mask_water,
            mu_r=mu_r,
            sigma_r=sigma_r,
            I_clean_f=I_clean,
            meta_json=json.dumps(meta),
        )
        print(f"[OK] wrote {out_npz}")

    print("[DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
