#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", type=Path, required=True)
    ap.add_argument("--base_idx", type=int, default=0)
    ap.add_argument("--strides", type=int, nargs="+", default=[1,2,4,8])
    ap.add_argument("--methods", type=str, nargs="+", required=True)
    ap.add_argument("--gt_csv", type=Path, required=True,
                    help="CSV with columns stride,dx_gt,dy_gt,yaw_gt_rad (from gt_from_gnss_for_radar_strides.py)")
    ap.add_argument("--out_csv", type=Path, default=Path("outputs/radar_vs_gt.csv"))
    args = ap.parse_args()

    from slamboat.radar.odometry import estimate_pair

    gt = pd.read_csv(args.gt_csv)
    gt = gt.set_index("stride")

    npzs = sorted(args.npz_dir.glob("*.npz"))
    ref = npzs[args.base_idx]

    rows = []
    for s in args.strides:
        cur = npzs[args.base_idx + s]
        for m in args.methods:
            res, dbg = estimate_pair(ref, cur, method=m)
            dx_gt = float(gt.loc[s, "dx_gt"])
            dy_gt = float(gt.loc[s, "dy_gt"])
            yaw_gt = float(gt.loc[s, "yaw_gt_rad"])

            ex = res.dx - dx_gt
            ey = res.dy - dy_gt
            epsi = wrap_pi(res.yaw_rad - yaw_gt)
            et = float(np.hypot(ex, ey))

            rows.append({
                "method": m,
                "stride": s,
                "dx": res.dx, "dy": res.dy, "yaw_rad": res.yaw_rad,
                "dx_gt": dx_gt, "dy_gt": dy_gt, "yaw_gt_rad": yaw_gt,
                "ex": ex, "ey": ey, "e_yaw": epsi, "e_trans": et,
                "psr": res.psr, "icp_rmse": res.icp_rmse, "icp_fitness": res.icp_fitness,
            })

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print("[OK]", args.out_csv)

if __name__ == "__main__":
    main()
