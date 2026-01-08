"""Output writers for estimated states."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TextIO


@dataclass(frozen=True)
class State3D:
    """Estimated 3D state snapshot."""

    t: float
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    vx: float
    vy: float
    vz: float


@dataclass
class StateWriter3D:
    """CSV writer for State3D."""

    out_csv_path: Path
    _f: Optional[TextIO] = None
    _w: Optional[csv.DictWriter] = None

    def open(self) -> None:
        self.out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.out_csv_path.open("w", encoding="utf-8", newline="")
        fieldnames = [
            "t",
            "x", "y", "z",
            "qx", "qy", "qz", "qw",
            "vx", "vy", "vz",
        ]
        self._w = csv.DictWriter(self._f, fieldnames=fieldnames)
        self._w.writeheader()

    def write(self, s: State3D) -> None:
        if self._w is None:
            raise RuntimeError("Writer not opened")
        self._w.writerow(
            dict(
                t=s.t,
                x=s.x, y=s.y, z=s.z,
                qx=s.qx, qy=s.qy, qz=s.qz, qw=s.qw,
                vx=s.vx, vy=s.vy, vz=s.vz,
            )
        )

    def close(self) -> None:
        if self._f:
            self._f.close()
        self._f = None
        self._w = None
