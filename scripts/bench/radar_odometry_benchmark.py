#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import csv
import numpy as np

from slamboat.radar.odometry import estimate_pair


def load_ref_csv(path: Path) -> dict[tuple[str, str], tuple[float, float, float]]:
    """
    Expect columns:
      ref_npz, cur_npz, dx_m, dy_m, yaw_rad
    """
    m = {}
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            key = (row["ref_npz"], row["cur_npz"])
            m[key] = (float(row["dx_m"]), float(row["dy_m"]), float(row["yaw_rad"]))
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_pairs", type=int, default=200)
    ap.add_argument("--cart_res", type=int, default=512)
    ap.add_argument("--r_max", type=float, default=600.0)

    ap.add_argument("--out_csv", type=Path, default=Path("outputs/radar_benchmark.csv"))
    ap.add_argument("--ref_csv", type=Path, default=None, help="Optional reference deltas (GNSS/LiDAR)")

    args = ap.parse_args()

    npzs = sorted(args.npz_dir.glob("*.npz"))
    npzs = npzs[:: args.stride]
    pairs = list(zip(npzs[:-1], npzs[1:]))[: args.max_pairs]

    ref_map = load_ref_csv(args.ref_csv) if args.ref_csv else {}

    methods = ["icp_raw", "icp_clean_nowater", "hybrid_clean", "hybrid_clean_nowater"]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ref_npz","cur_npz","method",
            "dt_s","dx","dy","yaw_rad",
            "psr","icp_fitness","icp_rmse",
            "ref_dx","ref_dy","ref_yaw",
            "err_trans_m","err_yaw_rad"
        ])

        for ref_npz, cur_npz in pairs:
            key = (str(ref_npz), str(cur_npz))
            ref_val = ref_map.get(key, None)

            for method in methods:
                try:
                    res, _ = estimate_pair(ref_npz, cur_npz, method=method, cart_res=args.cart_res, r_max=args.r_max)
                except Exception as e:
                    w.writerow([str(ref_npz), str(cur_npz), method, "ERR", str(e)] + [""]*12)
                    continue

                if ref_val is None:
                    err_t = ""
                    err_y = ""
                    ref_dx = ref_dy = ref_yaw = ""
                else:
                    ref_dx, ref_dy, ref_yaw = ref_val
                    err_t = float(np.hypot(res.dx - ref_dx, res.dy - ref_dy))
                    # wrap yaw error to [-pi,pi]
                    dyaw = float(res.yaw_rad - ref_yaw)
                    dyaw = (dyaw + np.pi) % (2*np.pi) - np.pi
                    err_y = dyaw

                w.writerow([
                    str(ref_npz), str(cur_npz), res.method,
                    f"{res.dt_s:.6f}", f"{res.dx:.4f}", f"{res.dy:.4f}", f"{res.yaw_rad:.6f}",
                    "" if res.psr is None else f"{res.psr:.3f}",
                    "" if res.icp_fitness is None else f"{res.icp_fitness:.3f}",
                    "" if res.icp_rmse is None else f"{res.icp_rmse:.4f}",
                    ref_dx, ref_dy, ref_yaw,
                    err_t, err_y
                ])

    print(f"[OK] wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
