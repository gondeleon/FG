#!/usr/bin/env python3
"""
debug_scripts/plot_mapmatch_demo.py

Offline, deterministic demo to visualize Weak Map-Matching (Radar ↔ OSM) outputs:
- radar cartesian evidence E(x,y)
- OSM primitives overlay at best pose
- coarse search score heatmap
- JSON log for presentation

This script writes artifacts for inspection (it is not a pytest unit test).

Expected repo integration points (to be implemented in Milestone 1):
- slamboat.maps.osm_aoi.build_aoi_bundle(...)
- slamboat.radar.evidence.polar_to_cartesian_evidence(...)
- slamboat.maps.map_matching.match_radar_to_map(...)

Usage example
-------------
python debug_scripts/plot_mapmatch_demo.py \
  --radar_npz path/to/radar_scan.npz \
  --anchor_lat 55.6 --anchor_lon 12.6 \
  --radius_m 1500 \
  --out_dir ./outputs/mapmatch_demo

NPZ is assumed to contain at least:
- I: polar intensity (Naz, Nrg)
Optional:
- azimuth_rad: (Naz,) angles [rad]
- r_m: (Nrg,) range bins in meters
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import matplotlib.pyplot as plt


def load_radar_npz(path: Path) -> Dict[str, Any]:
    data = np.load(str(path), allow_pickle=True)
    out: Dict[str, Any] = {k: data[k] for k in data.files}
    if "I" not in out:
        raise ValueError(f"NPZ {path} must contain key 'I'. Keys: {list(out.keys())}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--radar_npz", type=Path, required=True)
    p.add_argument("--anchor_lat", type=float, required=True)
    p.add_argument("--anchor_lon", type=float, required=True)
    p.add_argument("--radius_m", type=float, default=1500.0)
    p.add_argument("--metric_crs", type=str, default="auto")
    p.add_argument("--out_dir", type=Path, default=Path("./outputs/mapmatch_demo"))
    # Evidence
    p.add_argument("--cart_res_m", type=float, default=0.5)
    p.add_argument("--max_range_m", type=float, default=600.0)
    p.add_argument("--use_log", action="store_true")
    # Coarse search
    p.add_argument("--search_xy_m", type=float, default=20.0)
    p.add_argument("--search_yaw_deg", type=float, default=10.0)
    p.add_argument("--coarse_step_m", type=float, default=1.0)
    p.add_argument("--coarse_step_deg", type=float, default=1.0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    radar = load_radar_npz(args.radar_npz)
    I = radar["I"]

    # Project imports (will exist after Milestone 1)
    from slamboat.maps.osm_aoi import build_aoi_bundle
    from slamboat.radar.evidence import polar_to_cartesian_evidence
    from slamboat.maps.map_matching import match_radar_to_map

    aoi = build_aoi_bundle(
        center_lat=args.anchor_lat,
        center_lon=args.anchor_lon,
        radius_m=args.radius_m,
        metric_crs=args.metric_crs,
    )

    E, grid = polar_to_cartesian_evidence(
        I=I,
        azimuth_rad=radar.get("azimuth_rad", None),
        r_m=radar.get("r_m", None),
        cart_res_m=args.cart_res_m,
        max_range_m=args.max_range_m,
        use_log=args.use_log,
    )

    # Prior for the demo: anchored at map origin with yaw=0 (your impl can use Pose3)
    X_prior = {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}
    search_cfg = dict(
        search_xy_m=args.search_xy_m,
        search_yaw_deg=args.search_yaw_deg,
        coarse_step_m=args.coarse_step_m,
        coarse_step_deg=args.coarse_step_deg,
    )

    res = match_radar_to_map(E=E, grid=grid, aoi_bundle=aoi, X_prior=X_prior, search_cfg=search_cfg)

    # ---- Plots ----
    plt.figure()
    plt.imshow(E, origin="lower")
    plt.title("Radar evidence E(x,y)")
    plt.savefig(args.out_dir / "evidence_cartesian.png", dpi=180, bbox_inches="tight")
    plt.close()

    if getattr(res, "score_surface", None) is not None:
        plt.figure()
        plt.imshow(res.score_surface, origin="lower")
        plt.title("Coarse search score surface")
        plt.savefig(args.out_dir / "score_heatmap.png", dpi=180, bbox_inches="tight")
        plt.close()

    if getattr(res, "overlay_rgb", None) is not None:
        plt.figure()
        plt.imshow(res.overlay_rgb, origin="lower")
        plt.title("OSM overlay at best pose")
        plt.savefig(args.out_dir / "map_overlay.png", dpi=180, bbox_inches="tight")
        plt.close()

    log = {
        "best_pose": getattr(res, "best_pose", None),
        "score": float(getattr(res, "score", np.nan)),
        "status": getattr(res, "status", "unknown"),
        "search_cfg": search_cfg,
        "inputs": {
            "radar_npz": str(args.radar_npz),
            "anchor_lat": args.anchor_lat,
            "anchor_lon": args.anchor_lon,
            "radius_m": args.radius_m,
        },
    }
    with open(args.out_dir / "mapmatch_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    print(f"[OK] Outputs written to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
