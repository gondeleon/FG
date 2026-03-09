#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from slamboat.radar.odometry import estimate_pair
from slamboat.radar.debug import save_corr_surface_png

try:
    from scipy.ndimage import rotate, shift
    _HAS_SCIPY = True
except Exception:
    rotate = None
    shift = None
    _HAS_SCIPY = False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_npz", type=Path, required=True)
    ap.add_argument("--cur_npz", type=Path, required=True)
    ap.add_argument("--method", type=str, default="hybrid_clean",
                    choices=["icp_raw","icp_clean","hybrid_raw","hybrid_clean"])
    ap.add_argument("--out_dir", type=Path, default=Path("outputs/radar_dbg_pair"))
    ap.add_argument("--cart_res", type=int, default=512)
    ap.add_argument("--r_max", type=float, default=600.0)
    args = ap.parse_args()

    res, dbg = estimate_pair(args.ref_npz, args.cur_npz, method=args.method, cart_res=args.cart_res, r_max=args.r_max)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Save correlation surface if available
    if "corr_surface" in dbg:
        save_corr_surface_png(args.out_dir / "corr_surface.png", dbg["corr_surface"], title=f"{args.method} corr")

    # Minimal text summary
    (args.out_dir / "summary.txt").write_text(
        f"method={res.method}\n"
        f"dt_s={res.dt_s:.4f}\n"
        f"dx={res.dx:.3f}  dy={res.dy:.3f}  yaw_deg={np.rad2deg(res.yaw_rad):.2f}\n"
        f"psr={res.psr}\n"
        f"icp_fitness={res.icp_fitness}\n"
        f"icp_rmse={res.icp_rmse}\n"
    )
    print((args.out_dir / "summary.txt").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
