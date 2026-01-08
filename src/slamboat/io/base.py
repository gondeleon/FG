"""I/O contracts (Reader interface).

Readers are intentionally **dumb**:
- Parse files/streams
- Convert units
- Validate monotonic timestamps
- Produce typed messages

Readers must NOT import GTSAM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Generic, TypeVar

from slamboat.core.types import MessageBase


TMsg = TypeVar("TMsg", bound=MessageBase)


class SensorReaderBase(ABC, Generic[TMsg]):
    """Base class for sensor readers."""

    @abstractmethod
    def __iter__(self) -> Iterator[TMsg]:
        raise NotImplementedError
