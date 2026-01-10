from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class IcpMetrics:
    rmse_m: Optional[float] = None
    inlier_ratio: Optional[float] = None
    fitness: Optional[float] = None
    iterations: Optional[int] = None
    correspondences: Optional[int] = None
    dt_s: Optional[float] = None


@dataclass
class IcpResult:
    """Output of ICP: relative transform + quality metrics."""
    delta_T: np.ndarray
    metrics: IcpMetrics = field(default_factory=IcpMetrics)

    def __post_init__(self) -> None:
        T = np.asarray(self.delta_T, dtype=float)
        if T.shape != (4, 4):
            raise ValueError(f"delta_T must be 4x4, got {T.shape}")
        self.delta_T = T
