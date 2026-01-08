from __future__ import annotations

import argparse
import logging
from pathlib import Path

from slamboat.config import load_config
from slamboat.pipeline import run_pipeline


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> int:
    _setup_logging()
    ap = argparse.ArgumentParser(description="slamboat: GNSS+IMU online SLAM (offline replay).")
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    run_pipeline(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
