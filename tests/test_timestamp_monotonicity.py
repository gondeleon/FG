from __future__ import annotations

from pathlib import Path

import pytest

from slamboat.config import ColumnSpec, DelimitedFileSpec, ImuSpec, TimeSpec
from slamboat.io.imu import ImuReader


def test_monotonic_timestamps_imu(tmp_path: Path) -> None:
    p = tmp_path / "imu.tsv"
    # Non-monotonic: 2.0 then 1.0
    p.write_text(
        "2.0\t0\t0\t0\t0\t0\t0\n"
        "1.0\t0\t0\t0\t0\t0\t0\n",
        encoding="utf-8",
    )

    spec = ImuSpec(
        frame_id="imu",
        file=DelimitedFileSpec(
            path=str(p),
            delimiter="\t",
            has_header=False,
            time=ColumnSpec(col=0),
            time_format=TimeSpec(kind="unix_seconds"),
            columns={
                "wx": ColumnSpec(col=1, unit="rad/s"),
                "wy": ColumnSpec(col=2, unit="rad/s"),
                "wz": ColumnSpec(col=3, unit="rad/s"),
                "ax": ColumnSpec(col=4, unit="m/s^2"),
                "ay": ColumnSpec(col=5, unit="m/s^2"),
                "az": ColumnSpec(col=6, unit="m/s^2"),
            },
            require_monotonic_time=True,
        ),
    )

    with pytest.raises(ValueError):
        list(ImuReader(spec))
