"""Timestamp-based dispatcher for offline replay.

The dispatcher merges multiple monotonic sensor streams into a single time-ordered stream
of events. This is the key component that allows swapping:

  Offline files  -> Live streaming (ROS2, sockets, etc.)

without touching the estimator core.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

from slamboat.core.types import MessageBase, SensorKind


@dataclass(frozen=True)
class Event:
    """Single sensor event."""
    t: float
    sensor_name: str
    kind: SensorKind
    msg: MessageBase


class Dispatcher:
    """Merge N sensor iterators into a single time-sorted iterator."""

    def __init__(
        self,
        streams: Dict[str, Iterator[MessageBase]],
        kinds: Dict[str, SensorKind],
        realtime: bool = False,
        speedup: float = 20.0,
        max_steps: Optional[int] = None,
    ):
        self.streams = streams
        self.kinds = kinds
        self.realtime = realtime
        self.speedup = float(speedup)
        self.max_steps = max_steps

    def run(self) -> Iterator[Event]:
        # Prime heap with first element from each stream
        heap: list[Tuple[float, str, MessageBase]] = []
        for name, it in self.streams.items():
            try:
                msg = next(it)
                heapq.heappush(heap, (msg.t, name, msg))
            except StopIteration:
                continue

        if not heap:
            return

        t0_wall = time.time()
        t0_data = heap[0][0]

        steps = 0
        while heap:
            t, name, msg = heapq.heappop(heap)
            kind = self.kinds[name]

            if self.realtime:
                # Sleep to emulate real-time at given speedup.
                dt_data = t - t0_data
                target_wall = t0_wall + dt_data / max(self.speedup, 1e-9)
                now = time.time()
                if target_wall > now:
                    time.sleep(target_wall - now)

            yield Event(t=float(t), sensor_name=name, kind=kind, msg=msg)

            # push next from the same stream
            try:
                nxt = next(self.streams[name])
                heapq.heappush(heap, (nxt.t, name, nxt))
            except StopIteration:
                pass

            steps += 1
            if self.max_steps is not None and steps >= self.max_steps:
                break
