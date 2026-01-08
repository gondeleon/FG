"""IMU reader (gyro + accel).

Units are configurable per column via `unit`:
  - gyro: rad/s or deg/s
  - accel: m/s^2
"""

from __future__ import annotations

import logging
import math
from typing import Iterator, Optional

from slamboat.config import ImuSpec
from slamboat.core.types import ImuMessage
from slamboat.io.base import SensorReaderBase
from slamboat.io.delimited import DelimitedReader

log = logging.getLogger(__name__)


def _gyro_to_rad_s(value: float, unit: Optional[str]) -> float:
    if unit is None or unit.lower() in {"rad/s", "rads", "rad_s"}:
        return float(value)
    if unit.lower() in {"deg/s", "degs", "deg_s"}:
        return float(value) * math.pi / 180.0
    raise ValueError(f"Unsupported gyro unit: {unit}")


class ImuReader(SensorReaderBase[ImuMessage]):
    def __init__(self, spec: ImuSpec):
        self.spec = spec
        self.r = DelimitedReader(spec.file)

    def __iter__(self) -> Iterator[ImuMessage]:
        last_t: Optional[float] = None

        u_wx = self.spec.file.columns.get("wx").unit if "wx" in self.spec.file.columns else None
        u_wy = self.spec.file.columns.get("wy").unit if "wy" in self.spec.file.columns else None
        u_wz = self.spec.file.columns.get("wz").unit if "wz" in self.spec.file.columns else None

        # Determine the minimum number of columns required for index-based parsing.
        # This protects against malformed lines (e.g., counters, truncated rows).
        required_cols: Optional[int] = None
        if not self.spec.file.has_header:
            idxs = []
            # time column
            if isinstance(self.spec.file.time.col, int):
                idxs.append(self.spec.file.time.col)
            # data columns
            for k in ["wx", "wy", "wz", "ax", "ay", "az"]:
                c = self.spec.file.columns[k].col
                if isinstance(c, int):
                    idxs.append(c)
            required_cols = (max(idxs) + 1) if idxs else None

        for row in self.r:
            # Skip malformed / truncated rows (only applies to index-based rows).
            if required_cols is not None and isinstance(row, (list, tuple)) and len(row) < required_cols:
                log.warning("Skipping malformed IMU row (len=%d < %d): %s", len(row), required_cols, row)
                continue

            t_raw = DelimitedReader.get(row, self.spec.file.time.col)
            t = self.spec.file.time_format.to_seconds(DelimitedReader.parse_float(t_raw))

            if self.spec.file.require_monotonic_time:
                if last_t is not None and t <= last_t:
                    raise ValueError(f"IMU timestamps not monotonic: t={t} <= {last_t}")
                last_t = t

            wx = DelimitedReader.parse_float(DelimitedReader.get(row, self.spec.file.columns["wx"].col))
            wy = DelimitedReader.parse_float(DelimitedReader.get(row, self.spec.file.columns["wy"].col))
            wz = DelimitedReader.parse_float(DelimitedReader.get(row, self.spec.file.columns["wz"].col))
            ax = DelimitedReader.parse_float(DelimitedReader.get(row, self.spec.file.columns["ax"].col))
            ay = DelimitedReader.parse_float(DelimitedReader.get(row, self.spec.file.columns["ay"].col))
            az = DelimitedReader.parse_float(DelimitedReader.get(row, self.spec.file.columns["az"].col))

            msg = ImuMessage(
                t=float(t),
                frame_id=self.spec.frame_id,
                wx_rad_s=_gyro_to_rad_s(float(wx), u_wx),
                wy_rad_s=_gyro_to_rad_s(float(wy), u_wy),
                wz_rad_s=_gyro_to_rad_s(float(wz), u_wz),
                ax_m_s2=float(ax),
                ay_m_s2=float(ay),
                az_m_s2=float(az),
            )

            if self.spec.file.drop_nan_rows and DelimitedReader.any_nan(
                [msg.t, msg.wx_rad_s, msg.wy_rad_s, msg.wz_rad_s, msg.ax_m_s2, msg.ay_m_s2, msg.az_m_s2]
            ):
                log.warning("Dropping IMU row with NaNs at t=%s", msg.t)
                continue

            yield msg
