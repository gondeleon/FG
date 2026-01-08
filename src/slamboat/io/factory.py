"""Reader factory.

This isolates configuration parsing from concrete reader classes.
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, Tuple

from slamboat.config import SensorSpec
from slamboat.core.types import SensorKind
from slamboat.io.base import SensorReaderBase
from slamboat.io.gnss import GnssReader
from slamboat.io.imu import ImuReader
from slamboat.io.ahrs import AhrsReader


def build_readers(sensors: Dict[str, SensorSpec]) -> Dict[str, Iterator]:
    """Build iterators (streams) for each configured sensor."""
    streams: Dict[str, Iterator] = {}
    for name, spec in sensors.items():
        if spec.kind == "gnss":
            streams[name] = iter(GnssReader(spec))  # type: ignore[arg-type]
        elif spec.kind == "imu":
            streams[name] = iter(ImuReader(spec))  # type: ignore[arg-type]
        elif spec.kind == "ahrs":
            streams[name] = iter(AhrsReader(spec))  # type: ignore[arg-type]
        else:
            raise ValueError(f"Unsupported sensor kind: {spec.kind}")
    return streams


def sensor_kinds(sensors: Dict[str, SensorSpec]) -> Dict[str, SensorKind]:
    """Map sensor name to kind."""
    return {name: spec.kind for name, spec in sensors.items()}  # type: ignore[return-value]
