import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

CSV_FILES = {
    1: "outputs/radar_benchmark.csv",
    2: "outputs/radar_benchmark_stride_2.csv",
    4: "outputs/radar_benchmark_stride_4.csv",
    8: "outputs/radar_benchmark_stride_8.csv",
}

METHODS = [
    "icp_raw",
    "icp_clean_nowater",
    "hybrid_clean",
    "hybrid_clean_nowater",
]

# success thresholds
F_MIN = 0.90
RMSE_MAX = 3.0
D_MAX = 50.0


def f(x):
    try:
        return float(x)
    except:
        return None


def load(csv_path):
    rows = []
    with open(csv_path, newline="") as fcsv:
        r = csv.DictReader(fcsv)
        for row in r:
            if row["dt_s"] == "ERR":
                continue
            rows.append(row)
    return rows


print("\n=== RADAR ODOMETRY COMPARISON ===\n")

for stride, path in CSV_FILES.items():
    rows = load(path)
    print(f"\n--- Stride {stride}  (effective rate ≈ {4/stride:.2f} Hz) ---")

    for m in METHODS:
        r_m = [r for r in rows if r["method"] == m]
        if not r_m:
            continue

        ok = 0
        rmse_vals = []

        for r in r_m:
            fit = f(r["icp_fitness"])
            rmse = f(r["icp_rmse"])
            dx = f(r["dx"])
            dy = f(r["dy"])
            yaw = f(r["yaw_rad"])

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

        success = ok / len(r_m)
        rmse_med = np.median(rmse_vals) if rmse_vals else np.nan

        print(
            f"{m:22s} | success={success:5.1%} | rmse_med={rmse_med:5.2f}"
        )
