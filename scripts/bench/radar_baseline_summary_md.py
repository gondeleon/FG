#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np

METHODS = [
    "icp_raw",
    "icp_clean_nowater",
    "hybrid_clean",
    "hybrid_clean_nowater",
]

# Success thresholds (same spirit as radar_compare_strides.py)
F_MIN = 0.90
RMSE_MAX = 3.0
D_MAX = 50.0


def _f(x):
    try:
        return float(x)
    except Exception:
        return None


def load_rows(csv_path: Path):
    rows = []
    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("dt_s") == "ERR":
                continue
            rows.append(row)
    return rows


def summarize(rows, method: str):
    r_m = [r for r in rows if r.get("method") == method]
    if not r_m:
        return None

    ok = 0
    rmse_vals = []
    fit_vals = []
    psr_vals = []
    dt_vals = []

    for r in r_m:
        fit = _f(r.get("icp_fitness"))
        rmse = _f(r.get("icp_rmse"))
        dx = _f(r.get("dx"))
        dy = _f(r.get("dy"))
        yaw = _f(r.get("yaw_rad"))
        psr = _f(r.get("psr"))
        dt = _f(r.get("dt_s"))

        valid = (
            fit is not None and fit >= F_MIN and
            rmse is not None and rmse <= RMSE_MAX and
            dx is not None and abs(dx) <= D_MAX and
            dy is not None and abs(dy) <= D_MAX and
            yaw is not None and abs(yaw) <= np.pi
        )

        if valid:
            ok += 1
            rmse_vals.append(rmse)
            if fit is not None:
                fit_vals.append(fit)
            if psr is not None:
                psr_vals.append(psr)
            if dt is not None:
                dt_vals.append(dt)

    success = ok / len(r_m)
    rmse_med = float(np.median(rmse_vals)) if rmse_vals else float("nan")
    fit_med = float(np.median(fit_vals)) if fit_vals else float("nan")
    psr_med = float(np.median(psr_vals)) if psr_vals else float("nan")
    dt_med = float(np.median(dt_vals)) if dt_vals else float("nan")

    return dict(
        n=len(r_m),
        success=success,
        rmse_med=rmse_med,
        fit_med=fit_med,
        psr_med=psr_med,
        dt_med=dt_med,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride_csv", type=str, nargs="+", required=True,
                    help="Pairs like 1=outputs/radar_benchmark_stride_1.csv 2=... etc")
    ap.add_argument("--out_md", type=Path, default=Path("outputs/summary_baseline.md"))
    args = ap.parse_args()

    stride_to_csv = {}
    for item in args.stride_csv:
        k, v = item.split("=", 1)
        stride_to_csv[int(k)] = Path(v)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Radar odometry — baseline summary\n")
    lines.append("Success rule: fitness≥{:.2f}, rmse≤{:.1f}, |dx|,|dy|≤{:.1f} m, |yaw|≤π.\n".format(
        F_MIN, RMSE_MAX, D_MAX
    ))

    for stride in sorted(stride_to_csv.keys()):
        csv_path = stride_to_csv[stride]
        rows = load_rows(csv_path)

        lines.append(f"\n## Stride {stride}\n")
        lines.append(f"- CSV: `{csv_path}`\n")
        lines.append("\n| Method | N pairs | Success | RMSE median | Fitness median | PSR median | dt median (s) |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|\n")

        for m in METHODS:
            s = summarize(rows, m)
            if s is None:
                continue
            lines.append(
                f"| `{m}` | {s['n']} | {100*s['success']:.1f}% | "
                f"{s['rmse_med']:.3f} | {s['fit_med']:.3f} | {s['psr_med']:.3f} | {s['dt_med']:.3f} |\n"
            )

    args.out_md.write_text("".join(lines), encoding="utf-8")
    print(f"[OK] wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
