# Factor Catalog

This file documents the factor graph used in `slamboat`, including the measurement model, intent, and tuning knobs.

Notation:
- `i` = previous keyframe, `j` = next keyframe
- `X(i)` = Pose3 of state frame in world (ENU)
- `V(i)` = velocity in world
- `B(i)` = IMU bias
- `G(i)` = Pose3 of GNSS antenna frame in world

## 1) IMU preintegration factor

### Purpose
Propagate motion between keyframes using high-rate IMU.

### Variables
- `X(i), V(i), B(i), X(j), V(j)`

### Measurement
Preintegrated IMU increment computed from IMU samples between `t_i` and `t_j`.

### Factor
- `ImuFactor(Xi, Vi, Xj, Vj, Bi, pim)`

### Tuning knobs
- `estimator.imu_preintegration.*`:
  - `gravity_m_s2`
  - `accel_noise_sigma`, `gyro_noise_sigma`
  - `integration_sigma`
  - `max_dt_s` (drop/reset integration on large dt)

### Failure modes
- Timestamp gaps / resets → reset preintegration
- Wrong gravity sign / frame → roll/pitch drift

## 2) Bias random-walk factor

### Purpose
Constrain bias evolution (random walk) between keyframes.

### Variables
- `B(i), B(j)`

### Factor
- `BetweenFactorConstantBias(Bi, Bj, 0, bias_rw)`

### Tuning knobs
- `noise.sigma_bias_*_rw_*`

## 3) GNSS prior factor (on G(i))

### Purpose
Anchor the trajectory using GNSS position and (optional) heading.

### Variables
- `G(i)`

### Measurement
- Position: from lat/lon converted to ENU
- Yaw: from RTK baseline heading (true-north), if available

### Factor
- `PriorFactorPose3(G(i), z_gnss, Σ_gnss)`

### Tuning knobs
- `noise.sigma_xy_m`, `noise.sigma_z_m`, `noise.sigma_yaw_deg`
- gating: `estimator.gnss_gating.*`

### Notes
- If heading is missing/unreliable, yaw sigma should be large.

## 4) Rigid link factor between state and GNSS

### Purpose
Express the GNSS antenna pose as the state pose composed with the lever arm.

### Variables
- `X(i), G(i)`

### Measurement
- `T_state_gps` derived from extrinsics.

### Factor
- `BetweenFactorPose3(X(i), G(i), T_state_gps, Σ_rigid)` with very tight Σ.

### Notes
- This is the mechanism that allows putting GNSS priors on `G(i)` while estimating `X(i)`.

## 5) AHRS orientation prior (roll/pitch)

### Purpose
Correct roll/pitch using gravity direction from AHRS (without trusting magnetic yaw).

### Variables
- `X(i)`

### Measurement
- AHRS quaternion at/near keyframe time.

### Factor
- `PriorFactorPose3(X(i), z_ahrs, Σ_ahrs)` with:
  - small σ roll/pitch
  - huge σ yaw by default
  - huge σ translation

### Tuning knobs
- `noise.sigma_roll_pitch_deg`
- `estimator.ahrs_fusion.*`:
  - `yaw_mode` (off / fallback_if_no_gnss_heading)
  - `max_age_s`
  - optional `yaw_offset_deg` and `sigma_yaw_deg`

## 6) Planned: LiDAR odometry / scan-matching

### Intended purpose
Provide relative motion constraints when GNSS degrades.

### Candidate factors
- `BetweenFactorPose3(X(i), X(j), ΔT_lidar)`
- robust kernels (Huber/Cauchy) with outlier rejection

### Tuning knobs (planned)
- LiDAR odom covariance model
- keyframe selection / downsampling
- gating by ICP fitness
