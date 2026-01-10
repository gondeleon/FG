"""
LiDAR reader (odometry_delta mode).

Expected columns (config-driven):
  - dx, dy, dz (meters)
  - droll, dpitch, dyaw (radians)
Optional (for gating/debug):
  - rmse_m
  - inlier_ratio
  - fitness
  - iterations
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

from slamboat.config import LidarSpec
from slamboat.core.types import LidarDeltaMessage
from slamboat.io.base import SensorReaderBase
from slamboat.io.delimited import DelimitedReader

log = logging.getLogger(__name__)


class LidarDeltaReader(SensorReaderBase[LidarDeltaMessage]):
    def __init__(self, spec: LidarSpec):
        self.spec = spec
        self.r = DelimitedReader(spec.file)

    def __iter__(self) -> Iterator[LidarDeltaMessage]:
        last_t: Optional[float] = None

        for row in self.r:
            t_raw = DelimitedReader.get(row, self.spec.file.time.col)
            t = self.spec.file.time_format.to_seconds(DelimitedReader.parse_float(t_raw))

            if self.spec.file.require_monotonic_time:
                if last_t is not None and t <= last_t:
                    raise ValueError(f"LiDAR timestamps not monotonic: t={t} <= {last_t}")
                last_t = t

            def req_float(key: str) -> float:
                s = DelimitedReader.get(row, self.spec.file.columns[key].col)
                return float(DelimitedReader.parse_float(s))

            msg = LidarDeltaMessage(
                t=float(t),
                frame_id=self.spec.frame_id,
                dx=req_float("dx"),
                dy=req_float("dy"),
                dz=req_float("dz"),
                droll=req_float("droll"),
                dpitch=req_float("dpitch"),
                dyaw=req_float("dyaw"),
            )

            if self.spec.file.drop_nan_rows and DelimitedReader.any_nan(
                [msg.t, msg.dx, msg.dy, msg.dz, msg.droll, msg.dpitch, msg.dyaw]
            ):
                log.warning("Dropping LiDAR row with NaNs at t=%s", msg.t)
                continue

            yield msg
