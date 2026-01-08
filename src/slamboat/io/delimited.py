"""Generic delimited file reader.

Supports:
- CSV / TSV / space-separated
- optional header
- optional comment prefix
- monotonic timestamp validation (handled by sensor readers)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import numpy as np

from slamboat.config import DelimitedFileSpec


def _normalize_delim(d: str) -> str:
    # YAML may contain "\t" as literal tab
    return "\t" if d == "\t" else d


@dataclass(frozen=True)
class Row:
    """A raw row from file with both list and dict access."""
    raw: List[str]
    by_name: Optional[Dict[str, str]] = None


class DelimitedReader:
    """Iterates rows from a delimited file."""

    def __init__(self, spec: DelimitedFileSpec):
        self.spec = spec
        self.path = Path(spec.path)

    def __iter__(self) -> Iterator[Row]:
        if not self.path.exists():
            raise FileNotFoundError(f"Missing file: {self.path}")

        delim = _normalize_delim(self.spec.delimiter)
        with self.path.open("r", encoding="utf-8", newline="") as f:
            if self.spec.has_header:
                reader = csv.DictReader(f, delimiter=delim)
                for r in reader:
                    if self._is_comment(r):
                        continue
                    # DictReader gives dict of str->str (values can be None)
                    # Keep insertion order by converting to list (not needed now).
                    yield Row(raw=[], by_name={k: (v if v is not None else "") for k, v in r.items()})
            else:
                reader = csv.reader(f, delimiter=delim)
                for r in reader:
                    if not r:
                        continue
                    if self.spec.comment_prefix and r[0].lstrip().startswith(self.spec.comment_prefix):
                        continue
                    yield Row(raw=[c for c in r], by_name=None)

    def _is_comment(self, r: Dict[str, str]) -> bool:
        if not self.spec.comment_prefix:
            return False
        # If the first field starts with comment_prefix treat as comment row
        first = next(iter(r.values()), "")
        return str(first).lstrip().startswith(self.spec.comment_prefix)

    @staticmethod
    def get(row: Row, col: Union[int, str]) -> str:
        if isinstance(col, int):
            return row.raw[col]
        assert row.by_name is not None
        return row.by_name[col]

    @staticmethod
    def parse_float(s: str) -> float:
        if s is None:
            return float("nan")
        s2 = str(s).strip()
        if s2 == "" or s2.lower() in {"nan", "na", "none"}:
            return float("nan")
        return float(s2)

    @staticmethod
    def parse_int(s: str) -> int:
        s2 = str(s).strip()
        return int(float(s2))

    @staticmethod
    def any_nan(values: List[float]) -> bool:
        return any(np.isnan(v) for v in values)
