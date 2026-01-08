"""Geodesy helpers.

For this project we use a simple local tangent plane approximation (ENU) around a reference
latitude/longitude.

This is sufficient for harbor-scale operations (few km). If you later need higher accuracy,
swap this module with a proper geodesy library (pyproj) without touching the estimator core.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class LocalENU:
    """Local ENU converter around a reference LLH point."""

    ref_lat_deg: float
    ref_lon_deg: float
    ref_alt_m: float = 0.0

    def ll_to_enu_m(self, lat_deg: float, lon_deg: float) -> Tuple[float, float]:
        """Convert latitude/longitude to local ENU (meters), ignoring altitude."""
        # WGS84 equatorial radius (meters)
        R = 6378137.0
        lat0 = math.radians(self.ref_lat_deg)

        dlat = math.radians(lat_deg - self.ref_lat_deg)
        dlon = math.radians(lon_deg - self.ref_lon_deg)

        east = R * math.cos(lat0) * dlon
        north = R * dlat
        return east, north

    def llh_to_enu_m(self, lat_deg: float, lon_deg: float, alt_m: float) -> Tuple[float, float, float]:
        """Convert latitude/longitude/altitude to local ENU (meters)."""
        e, n = self.ll_to_enu_m(lat_deg, lon_deg)
        u = float(alt_m - self.ref_alt_m)
        return e, n, u
