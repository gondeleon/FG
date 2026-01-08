from __future__ import annotations

from pathlib import Path

from slamboat.config import ColumnSpec, DelimitedFileSpec, TimeSpec
from slamboat.io.delimited import DelimitedReader


def test_delimited_reader_column_mapping_no_header(tmp_path: Path) -> None:
    p = tmp_path / "imu.tsv"
    p.write_text("1.0\t10\t20\t30\t1\t2\t3\n", encoding="utf-8")

    spec = DelimitedFileSpec(
        path=str(p),
        delimiter="\t",
        has_header=False,
        time=ColumnSpec(col=0),
        time_format=TimeSpec(kind="unix_seconds"),
        columns={
            "wx": ColumnSpec(col=1),
            "wy": ColumnSpec(col=2),
            "wz": ColumnSpec(col=3),
            "ax": ColumnSpec(col=4),
            "ay": ColumnSpec(col=5),
            "az": ColumnSpec(col=6),
        },
    )

    r = DelimitedReader(spec)
    row = next(iter(r))
    assert DelimitedReader.get(row, 0) == "1.0"
    assert float(DelimitedReader.get(row, 3)) == 30.0
