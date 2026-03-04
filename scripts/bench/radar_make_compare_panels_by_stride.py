#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import subprocess
import sys


def _default_base_idx(n: int, max_stride: int) -> int:
    # pick middle, but ensure base+max_stride is valid
    if n < 2 + max_stride:
        return 0
    mid = n // 2
    return int(np.clip(mid, 0, n - 1 - max_stride))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate 2x2 panels comparing the 4 methods, one panel per stride. "
                    "Pairs are fixed across strides using base index i0: (i0, i0+stride)."
    )
    ap.add_argument("--npz_dir", type=Path, required=True)
    ap.add_argument("--strides", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--base_idx", type=int, default=None,
                    help="Base index i0 in the full sorted npz list. For each stride s we use (i0, i0+s). "
                         "If omitted, chooses a safe middle index.")
    ap.add_argument("--out_dir", type=Path, default=Path("outputs/panels/compare_methods"))
    ap.add_argument("--cart_res", type=int, default=512)
    ap.add_argument("--r_max", type=float, default=600.0)
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--yaw_sign", type=float, default=1.0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_npzs = sorted(args.npz_dir.glob("*.npz"))
    if len(all_npzs) < 2:
        raise SystemExit(f"No npz files found in {args.npz_dir}")

    max_stride = max(args.strides) if args.strides else 1
    i0 = args.base_idx
    if i0 is None:
        i0 = _default_base_idx(len(all_npzs), max_stride)

    if i0 < 0 or i0 >= len(all_npzs):
        raise SystemExit(f"--base_idx out of range: {i0} not in [0,{len(all_npzs)-1}]")
    if i0 + max_stride >= len(all_npzs):
        raise SystemExit(f"--base_idx too late: i0+max_stride={i0+max_stride} >= N={len(all_npzs)}")

    for stride in args.strides:
        ref_npz = all_npzs[i0]
        cur_npz = all_npzs[i0 + stride]
        out_png = args.out_dir / f"stride_{stride:02d}__i0_{i0:05d}__compare_methods.png"

        cmd = [
            sys.executable, "scripts/radar_pair_panel_2x2.py",
            "--ref_npz", str(ref_npz),
            "--cur_npz", str(cur_npz),
            "--out_png", str(out_png),
            "--cart_res", str(args.cart_res),
            "--r_max", str(args.r_max),
            "--alpha", str(args.alpha),
            "--yaw_sign", str(args.yaw_sign),
        ]
        subprocess.check_call(cmd)
        print(f"[OK] stride {stride}: {out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
