"""GNSS reader.

Expected columns (config-driven):
  - lat_deg, lat_hemi (N/S)
  - lon_deg, lon_hemi (E/W)
  - heading_deg (optional)
  - geoid_height_m (optional)

Time:
  - provided as unix seconds or unix nanoseconds (configurable)
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

from slamboat.config import GnssSpec
from slamboat.core.types import GnssMessage
from slamboat.io.base import SensorReaderBase
from slamboat.io.delimited import DelimitedReader

log = logging.getLogger(__name__)


def _hemi_signed(value_deg: float, hemi: str, positive_hemi: str) -> float:
    hemi = hemi.strip().upper()
    if hemi == positive_hemi:
        return value_deg
    return -value_deg


class GnssReader(SensorReaderBase[GnssMessage]):
    def __init__(self, spec: GnssSpec):
        self.spec = spec
        self.r = DelimitedReader(spec.file)

    def __iter__(self) -> Iterator[GnssMessage]:
        last_t: Optional[float] = None

        for row in self.r:
            # time
            t_raw = DelimitedReader.get(row, self.spec.file.time.col)
            t = self.spec.file.time_format.to_seconds(DelimitedReader.parse_float(t_raw))

            if self.spec.file.require_monotonic_time:
                if last_t is not None and t <= last_t:
                    raise ValueError(f"GNSS timestamps not monotonic: t={t} <= {last_t}")
                last_t = t

            def opt_float(key: str) -> Optional[float]:
                if key not in self.spec.file.columns:
                    return None
                s = DelimitedReader.get(row, self.spec.file.columns[key].col)
                v = DelimitedReader.parse_float(s)
                return None if (v != v) else float(v)  # NaN check

            def opt_int(key: str) -> Optional[int]:
                if key not in self.spec.file.columns:
                    return None
                s = DelimitedReader.get(row, self.spec.file.columns[key].col)
                try:
                    return int(DelimitedReader.parse_int(s))
                except Exception:
                    return None

            lat = DelimitedReader.parse_float(DelimitedReader.get(row, self.spec.file.columns["lat_deg"].col))
            lat_hemi = DelimitedReader.get(row, self.spec.file.columns["lat_hemi"].col)
            lon = DelimitedReader.parse_float(DelimitedReader.get(row, self.spec.file.columns["lon_deg"].col))
            lon_hemi = DelimitedReader.get(row, self.spec.file.columns["lon_hemi"].col)

            lat_deg = _hemi_signed(float(lat), lat_hemi, "N")
            lon_deg = _hemi_signed(float(lon), lon_hemi, "E")

            heading = opt_float("heading_deg")
            geoid_h = opt_float("geoid_height_m")

            msg = GnssMessage(
                t=float(t),
                frame_id=self.spec.frame_id,
                lat_deg=float(lat_deg),
                lon_deg=float(lon_deg),
                heading_deg=None if heading is None else float(heading),
                geoid_height_m=None if geoid_h is None else float(geoid_h),
                quality=opt_int("quality"),
                n_sats=opt_int("n_sats"),
                hdop=opt_float("hdop"),
            )

            if self.spec.file.drop_nan_rows and any(v != v for v in [msg.t, msg.lat_deg, msg.lon_deg]):
                log.warning("Dropping GNSS row with NaNs at t=%s", msg.t)
                continue

            yield msg
