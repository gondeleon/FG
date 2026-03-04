"""Core message types used across the pipeline.

Design rule:
- I/O layer produces these messages (pure Python, no GTSAM imports).
- Core estimator consumes these messages to build factor graphs.

All messages carry:
  - `t` timestamp in seconds (float, unix time converted to seconds)
  - `frame_id` string identifying the sensor frame in the extrinsics tree
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


SensorKind = Literal["gnss", "imu", "ahrs", "lidar", "radar"]


@dataclass(frozen=True)
class MessageBase:
    """Base message with timestamp."""
    t: float  # seconds


@dataclass(frozen=True)
class GnssMessage(MessageBase):
    """GNSS measurement (raw lat/lon + optional heading).

    `heading_deg` is typically true-north referenced if computed from dual-antenna RTK.
    """
    frame_id: str
    lat_deg: float
    lon_deg: float
    heading_deg: Optional[float] = None
    geoid_height_m: Optional[float] = None

    quality: Optional[int] = None
    n_sats: Optional[int] = None
    hdop: Optional[float] = None


@dataclass(frozen=True)
class ImuMessage(MessageBase):
    """IMU sample in the IMU/body frame."""
    frame_id: str
    wx_rad_s: float
    wy_rad_s: float
    wz_rad_s: float
    ax_m_s2: float
    ay_m_s2: float
    az_m_s2: float


@dataclass(frozen=True)
class AhrsMessage(MessageBase):
    """AHRS sample.

    The AHRS provides an orientation estimate (quaternion) plus often gyro/accel streams.
    We will use orientation mostly for roll/pitch correction (gravity direction), and ignore yaw
    (magnetic north vs true north mismatch).
    """
    frame_id: str
    qx: float
    qy: float
    qz: float
    qw: float

    # Optional raw streams (kept for completeness / future use)
    wx_rad_s: Optional[float] = None
    wy_rad_s: Optional[float] = None
    wz_rad_s: Optional[float] = None
    ax_m_s2: Optional[float] = None
    ay_m_s2: Optional[float] = None
    az_m_s2: Optional[float] = None
    mx_uT: float | None = None
    my_uT: float | None = None
    mz_uT: float | None = None

@dataclass(frozen=True)
class LidarDeltaMessage(MessageBase):
    """Relative motion from lidar keyframe i->j.

    Convention:
      - Translation in meters.
      - Angles in radians.
      - Expressed in the *lidar_i* frame (i.e., i_T_j).
    """
    frame_id: str
    dx: float
    dy: float
    dz: float
    droll: float
    dpitch: float
    dyaw: float


    # Optional ICP quality metrics (from offline ICP or online scan-matching)
    rmse_m: Optional[float] = None
    fitness: Optional[float] = None
    inlier_ratio: Optional[float] = None
    iterations: Optional[int] = None
    correspondences: Optional[int] = None
    dt_s: Optional[float] = None

@dataclass(frozen=True)
class RadarDeltaMessage(MessageBase):
    """Relative motion from radar keyframe i->j.

    Convention (same as LiDAR delta):
        - Translation in meters.
        - Angles in radians.
        - Expressed in the *radar_i* frame (i.e., i_T_j).

    Notes:
        - `band` distinguishes X-band / W-band pipelines.
        - `quality` is optional and can be used to scale covariance (worse quality -> larger sigmas).
    """
    frame_id: str
    band: Literal["xband", "wband"]

    dx: float
    dy: float
    dz: float
    droll: float
    dpitch: float
    dyaw: float

    # Optional metrics
    quality: Optional[float] = None
    rmse_m: Optional[float] = None
    inlier_ratio: Optional[float] = None
    dt_s: Optional[float] = None