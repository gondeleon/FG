"""Configuration models (Pydantic).

The goal is that *rig changes* (different sensor placement / frames) require only:
  - `configs/config.yaml` edits for data paths and column mapping
  - `configs/extrinsics.json` edits for rigid transforms

The estimator should not need code changes when the physical setup changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional, Union, Mapping

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator


class TimeSpec(BaseModel):
    """Time conversion for log timestamps."""

    kind: Literal["unix_seconds", "unix_nanoseconds"] = "unix_seconds"

    def to_seconds(self, raw: Union[int, float]) -> float:
        if self.kind == "unix_seconds":
            return float(raw)
        if self.kind == "unix_nanoseconds":
            return float(raw) * 1e-9
        raise ValueError(f"Unsupported time kind: {self.kind}")


class ColumnSpec(BaseModel):
    """Column selection for delimited files.

    If `has_header=true`, `col` must be a column name (str).
    If `has_header=false`, `col` must be a column index (int).
    """

    col: Union[str, int]
    unit: Optional[str] = None


class DelimitedFileSpec(BaseModel):
    """Generic delimited file specification."""

    path: str
    delimiter: Literal[",", "\t", " ", ";"] = "\t"
    has_header: bool = False
    comment_prefix: Optional[str] = None

    time: ColumnSpec
    columns: Dict[str, ColumnSpec]

    require_monotonic_time: bool = True
    drop_nan_rows: bool = True

    time_format: TimeSpec = Field(default_factory=TimeSpec)

    @model_validator(mode="after")
    def _check_header_usage(self) -> "DelimitedFileSpec":
        for k, v in {"time": self.time, **self.columns}.items():
            if self.has_header and isinstance(v.col, int):
                raise ValueError(
                    f"{k}.col is int but has_header=true. Use column names or set has_header=false."
                )
            if (not self.has_header) and isinstance(v.col, str):
                raise ValueError(
                    f"{k}.col is str but has_header=false. Use indices or set has_header=true."
                )
        return self


class GnssSpec(BaseModel):
    kind: Literal["gnss"] = "gnss"
    frame_id: str = "gps"
    file: DelimitedFileSpec

    # Local tangent plane reference for ENU conversion.
    ref_lat_deg: float
    ref_lon_deg: float
    ref_alt_m: float = 0.0

    use_geoid_height_as_z: bool = False


class ImuSpec(BaseModel):
    kind: Literal["imu"] = "imu"
    frame_id: str = "imu"
    file: DelimitedFileSpec


class AhrsSpec(BaseModel):
    kind: Literal["ahrs"] = "ahrs"
    frame_id: str = "ahrs"
    file: DelimitedFileSpec

    # Orientation convention of the logged quaternion:
    # - world_T_sensor: quaternion rotates vectors from sensor frame to world frame
    # - sensor_T_world: quaternion rotates vectors from world frame to sensor frame
    orientation_convention: Literal["world_T_sensor", "sensor_T_world"] = "world_T_sensor"

    # Quaternion order in the log file (default matches: qx,qy,qz,qw)
    quaternion_order: Literal["xyzw", "wxyz"] = "xyzw"

    file: DelimitedFileSpec

class LidarSpec(BaseModel):
    kind: Literal["lidar"] = "lidar"
    frame_id: str = "lidar"

    # For now: CSV of relative pose constraints (odometry / ICP output)
    mode: Literal["odometry_delta"] = "odometry_delta"
    file: DelimitedFileSpec

class RadarSpec(BaseModel):
    kind: Literal["radar"] = "radar"
    frame_id: str = "radar"

    # For now: CSV of relative pose constraints (odometry output)
    mode: Literal["odometry_delta"] = "odometry_delta"
    file: DelimitedFileSpec


SensorSpec = Union[GnssSpec, ImuSpec, AhrsSpec, LidarSpec, RadarSpec]


class RadarGatingSpec(BaseModel):
    """Kinematic gating for radar deltas before inserting a factor."""
    v_max_mps: float = 12.0
    yaw_rate_max_dps: float = 25.0


class RadarQualityScalingSpec(BaseModel):
    """Scale covariance based on an optional [0..1] quality score."""
    enabled: bool = True
    q_low: float = 0.2
    q_high: float = 0.9
    alpha_low: float = 6.0   # worse quality -> larger alpha
    alpha_high: float = 1.0  # best quality -> alpha ~ 1


class RadarOdomNoiseSpec(BaseModel):
    """Base (diagonal) noise for Radar odometry BetweenFactorPose3."""
    sigma_x_m: float = 0.50
    sigma_y_m: float = 0.50
    sigma_z_m: float = 1.50
    sigma_roll_deg: float = 5.0
    sigma_pitch_deg: float = 5.0
    sigma_yaw_deg: float = 3.0


class RadarOdomRobustSpec(BaseModel):
    """Optional robust kernel for Radar odometry factors."""
    enabled: bool = False
    kernel: Literal["none", "huber", "cauchy"] = "huber"
    param: float = 1.345


class RadarBandSpec(BaseModel):
    """Per-band radar odometry configuration."""
    noise: RadarOdomNoiseSpec = Field(default_factory=RadarOdomNoiseSpec)
    gating: RadarGatingSpec = Field(default_factory=RadarGatingSpec)
    quality_scaling: RadarQualityScalingSpec = Field(default_factory=RadarQualityScalingSpec)
    robust: RadarOdomRobustSpec = Field(default_factory=RadarOdomRobustSpec)


class RadarOdometrySpec(BaseModel):
    """Radar odometry configuration (per band)."""
    enabled: bool = False
    max_age_s: float = 0.25

    # Expect keys: "xband", "wband"
    bands: Dict[str, RadarBandSpec] = Field(default_factory=dict)

class RobustKernelSpec(BaseModel):
    kind: Literal["none", "huber", "cauchy"] = "huber"
    param: float = 1.0


class NoiseSpec(BaseModel):
    # GNSS measurement noise (position in meters)
    sigma_xy_m: float = 2.0
    sigma_z_m: float = 5.0

    # GNSS heading noise (degrees)
    sigma_yaw_deg: float = 30.0

    # Initial priors
    sigma_v0_m_s: float = 1.0
    sigma_bias_acc_m_s2: float = 0.2
    sigma_bias_gyro_rad_s: float = 0.05

    # Bias random walk between keyframes
    sigma_bias_acc_rw_m_s2: float = 0.02
    sigma_bias_gyro_rw_rad_s: float = 0.005

    # AHRS roll/pitch correction (degrees). Yaw is intentionally not used.
    sigma_roll_pitch_deg: float = 3.0


class ImuPreintegrationSpec(BaseModel):
    gravity_m_s2: float = 9.81
    accel_noise_sigma: float = 0.1
    gyro_noise_sigma: float = 0.01
    integration_sigma: float = 1e-4
    bias_acc_cov_sigma: float = 0.1
    bias_gyro_cov_sigma: float = 0.01
    max_dt_s: float = 0.05


class AhrsFusionSpec(BaseModel):
    enable: bool = True
    # We will attach the AHRS factor to the nearest GNSS keyframe (latest sample).
    max_age_s: float = 0.2
    # Yaw handling:
    # - "off": do not constrain yaw from AHRS (paper-like default)
    # - "fallback_if_no_gnss_heading": use AHRS yaw only when GNSS heading is missing
    yaw_mode: str = "off"

    # Optional constant offset (deg) to align AHRS yaw (magnetic) to navigation yaw (true north).
    # Only used when yaw_mode allows yaw.
    yaw_offset_deg: float = 0.0

    # Only used when yaw_mode == "fallback_if_no_gnss_heading"
    sigma_yaw_deg: float = 30.0


class ReplaySpec(BaseModel):
    realtime: bool = False
    speedup: float = 20.0
    max_steps: Optional[int] = None


class OutputSpec(BaseModel):
    out_dir: str = "outputs"
    state_csv: str = "state_trace.csv"


class FramesSpec(BaseModel):
    """Names of frames used by the estimator.

    These must exist in `extrinsics.json` (or be the declared parent/root frame).
    """

    state_frame: str = "imu"
    imu_frame: str = "imu"
    gnss_frame: str = "gps"
    ahrs_frame: str = "ahrs"

class GnssGatingSpec(BaseModel):
    """
    GNSS gating rules to reject low-quality fixes before adding factors.

    Typical fields available in the dataset:
      - quality: integer indicator (RTK/fix/float/etc. depending on receiver)
      - hdop: horizontal dilution of precision
      - n_sats: number of satellites used
    """
    enable: bool = True
    min_quality: int = 1
    max_hdop: float = 2.5
    min_sats: int = 8

    # Optional: reject sudden jumps in measured GNSS position (meters) between accepted fixes.
    enable_jump_check: bool = True
    max_jump_m: float = 25.0

class LidarOdomKeyframesSpec(BaseModel):
    """How LiDAR keyframes are selected/associated to estimator keyframes."""

    mode: Literal["gnss", "fixed_period"] = "gnss"
    period_s: float = 1.0
    max_age_s: float = 0.25


class LidarIcpSpec(BaseModel):
    """ICP backend configuration (scan matching)."""

    backend: Literal["open3d", "none"] = "open3d"
    voxel_size_m: float = 0.25
    max_corr_dist_m: float = 1.5
    max_iters: int = 50


class LidarGatingSpec(BaseModel):
    """Quality gates for ICP results before inserting a factor."""

    max_translation_m: Optional[float] = 3.0
    max_rotation_deg: Optional[float] = 15.0
    max_rmse_m: Optional[float] = 0.8
    min_inlier_ratio: Optional[float] = 0.30
    dt_min_s: Optional[float] = 0.0
    dt_max_s: Optional[float] = 5.0


class LidarOdomNoiseSpec(BaseModel):
    """Base (diagonal) noise for LiDAR odometry BetweenFactorPose3."""

    sigma_x_m: float = 0.30
    sigma_y_m: float = 0.30
    sigma_z_m: float = 0.50
    sigma_roll_deg: float = 5.0
    sigma_pitch_deg: float = 5.0
    sigma_yaw_deg: float = 3.0
    scale_with_rmse: bool = False


class LidarOdomRobustSpec(BaseModel):
    """Optional robust kernel for LiDAR odometry factors."""

    enabled: bool = False
    kernel: Literal["none", "huber", "cauchy"] = "huber"
    param: float = 1.345


class LidarOdomDebugSpec(BaseModel):
    dump_csv: bool = True
    csv_path: str = "outputs/lidar_icp_pairs.csv"


class LidarOdometrySpec(BaseModel):
    """LiDAR odometry / scan matching configuration."""

    enabled: bool = False
    keyframes: LidarOdomKeyframesSpec = Field(default_factory=LidarOdomKeyframesSpec)
    icp: LidarIcpSpec = Field(default_factory=LidarIcpSpec)
    gating: LidarGatingSpec = Field(default_factory=LidarGatingSpec)
    noise: LidarOdomNoiseSpec = Field(default_factory=LidarOdomNoiseSpec)
    robust: LidarOdomRobustSpec = Field(default_factory=LidarOdomRobustSpec)
    debug: LidarOdomDebugSpec = Field(default_factory=LidarOdomDebugSpec)


class EstimatorSpec(BaseModel):
    mode: Literal["pose3_imu_preint"] = "pose3_imu_preint"
    robust_kernel: RobustKernelSpec = Field(default_factory=RobustKernelSpec)
    noise: NoiseSpec = Field(default_factory=NoiseSpec)
    imu_preintegration: ImuPreintegrationSpec = Field(default_factory=ImuPreintegrationSpec)
    gnss_gating: GnssGatingSpec = Field(default_factory=GnssGatingSpec)
    ahrs_fusion: AhrsFusionSpec = Field(default_factory=AhrsFusionSpec)
    lidar_odometry: LidarOdometrySpec = Field(default_factory=LidarOdometrySpec)
    radar_odometry: RadarOdometrySpec = Field(default_factory=RadarOdometrySpec)


class AppConfig(BaseModel):
    sensors: Dict[str, SensorSpec]
    extrinsics_path: str = "configs/extrinsics.json"
    frames: FramesSpec = Field(default_factory=FramesSpec)

    replay: ReplaySpec = Field(default_factory=ReplaySpec)
    estimator: EstimatorSpec = Field(default_factory=EstimatorSpec)
    outputs: OutputSpec = Field(default_factory=OutputSpec)

    @model_validator(mode="after")
    def _check_required_sensors(self) -> "AppConfig":
        kinds = {name: spec.kind for name, spec in self.sensors.items()}
        if "gnss" not in kinds.values():
            raise ValueError("At least one GNSS sensor is required.")
        if "imu" not in kinds.values():
            raise ValueError("At least one IMU sensor is required.")
        return self


def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    try:
        return AppConfig.model_validate(data)
    except ValidationError as e:
        raise SystemExit(f"Config validation error:\n{e}") from e
