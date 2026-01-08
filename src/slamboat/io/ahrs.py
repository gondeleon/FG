"""AHRS reader.

Expected columns (config-driven):
  - qx,qy,qz,qw quaternion
  - wx,wy,wz angular rates (optional)
  - ax,ay,az accelerations (optional)

Notes:
- The estimator will use AHRS orientation to correct roll/pitch (gravity direction),
  while intentionally NOT constraining yaw (magnetic vs true north mismatch).
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

from slamboat.config import AhrsSpec
from slamboat.core.types import AhrsMessage
from slamboat.io.base import SensorReaderBase
from slamboat.io.delimited import DelimitedReader

log = logging.getLogger(__name__)


class AhrsReader(SensorReaderBase[AhrsMessage]):
    def __init__(self, spec: AhrsSpec):
        self.spec = spec
        self.r = DelimitedReader(spec.file)

    def __iter__(self) -> Iterator[AhrsMessage]:
        last_t: Optional[float] = None

        for row in self.r:
            t_raw = DelimitedReader.get(row, self.spec.file.time.col)
            t = self.spec.file.time_format.to_seconds(DelimitedReader.parse_float(t_raw))

            if self.spec.file.require_monotonic_time:
                if last_t is not None and t <= last_t:
                    raise ValueError(f"AHRS timestamps not monotonic: t={t} <= {last_t}")
                last_t = t

            def opt_float(key: str) -> Optional[float]:
                if key not in self.spec.file.columns:
                    return None
                s = DelimitedReader.get(row, self.spec.file.columns[key].col)
                v = DelimitedReader.parse_float(s)
                return None if (v != v) else float(v)

            qx = opt_float("qx")
            qy = opt_float("qy")
            qz = opt_float("qz")
            qw = opt_float("qw")
            if qx is None or qy is None or qz is None or qw is None:
                log.warning("Dropping AHRS row with missing quaternion at t=%s", t)
                continue

            msg = AhrsMessage(
                t=float(t),
                frame_id=self.spec.frame_id,
                qx=float(qx),
                qy=float(qy),
                qz=float(qz),
                qw=float(qw),
                wx_rad_s=opt_float("wx"),
                wy_rad_s=opt_float("wy"),
                wz_rad_s=opt_float("wz"),
                ax_m_s2=opt_float("ax"),
                ay_m_s2=opt_float("ay"),
                az_m_s2=opt_float("az"),
            )

            if self.spec.file.drop_nan_rows and DelimitedReader.any_nan([msg.t, msg.qx, msg.qy, msg.qz, msg.qw]):
                log.warning("Dropping AHRS row with NaNs at t=%s", msg.t)
                continue

            yield msg
