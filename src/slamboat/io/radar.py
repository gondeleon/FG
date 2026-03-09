"""
Radar reader (odometry_delta mode).

Expected columns (config-driven):
  - band: "xband" | "wband"
  - dx, dy, dz (meters)
  - droll, dpitch, dyaw (radians)

Optional:
  - quality
  - rmse_m
  - inlier_ratio
  - dt_s
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

from slamboat.config import RadarSpec
from slamboat.core.types import RadarDeltaMessage
from slamboat.io.base import SensorReaderBase
from slamboat.io.delimited import DelimitedReader

log = logging.getLogger(__name__)


class RadarDeltaReader(SensorReaderBase[RadarDeltaMessage]):
    def __init__(self, spec: RadarSpec):
        self.spec = spec
        self.r = DelimitedReader(spec.file)

    def __iter__(self) -> Iterator[RadarDeltaMessage]:
        last_t: Optional[float] = None

        for row in self.r:
            t_raw = DelimitedReader.get(row, self.spec.file.time.col)
            t = self.spec.file.time_format.to_seconds(DelimitedReader.parse_float(t_raw))

            if self.spec.file.require_monotonic_time:
                if last_t is not None and t <= last_t:
                    raise ValueError(f"Radar timestamps not monotonic: t={t} <= {last_t}")
                last_t = t

            def req_float(key: str) -> float:
                s = DelimitedReader.get(row, self.spec.file.columns[key].col)
                return float(DelimitedReader.parse_float(s))

            def opt_float(key: str) -> Optional[float]:
                if key not in self.spec.file.columns:
                    return None
                s = DelimitedReader.get(row, self.spec.file.columns[key].col)
                return float(DelimitedReader.parse_float(s))

            def req_str(key: str) -> str:
                s = DelimitedReader.get(row, self.spec.file.columns[key].col)
                return str(s).strip()

            band = req_str("band").lower()
            if band not in ("xband", "wband"):
                raise ValueError(f"Invalid radar band='{band}' at t={t}. Expected xband|wband.")

            msg = RadarDeltaMessage(
                t=float(t),
                frame_id=self.spec.frame_id,
                band=band,  # type: ignore[arg-type]
                dx=req_float("dx"),
                dy=req_float("dy"),
                dz=req_float("dz"),
                droll=req_float("droll"),
                dpitch=req_float("dpitch"),
                dyaw=req_float("dyaw"),
                quality=opt_float("quality"),
                rmse_m=opt_float("rmse_m"),
                inlier_ratio=opt_float("inlier_ratio"),
                dt_s=opt_float("dt_s"),
            )

            if self.spec.file.drop_nan_rows and DelimitedReader.any_nan(
                [msg.t, msg.dx, msg.dy, msg.dz, msg.droll, msg.dpitch, msg.dyaw]
            ):
                log.warning("Dropping Radar row with NaNs at t=%s", msg.t)
                continue

            yield msg
