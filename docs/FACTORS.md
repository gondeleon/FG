# Factor Catalog

This file documents the factor graph used in `slamboat`, including the measurement model, intent, tuning knobs, and failure modes.

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
  - `max_dt_s`

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

## 4) Rigid link factor between state and GNSS

### Purpose
Express the GNSS antenna pose as the state pose composed with the lever arm.

### Variables
- `X(i), G(i)`

### Measurement
- `T_state_gps` derived from extrinsics.

### Factor
- `BetweenFactorPose3(X(i), G(i), T_state_gps, Σ_rigid)` with very tight Σ.

## 5) AHRS orientation prior (roll/pitch)

### Purpose
Correct roll/pitch using gravity direction from AHRS (without trusting magnetic yaw by default).

### Variables
- `X(i)`

### Factor
- `PriorFactorPose3(X(i), z_ahrs, Σ_ahrs)` with:
  - small σ roll/pitch
  - huge σ yaw by default
  - huge σ translation

### Tuning knobs
- `estimator.ahrs_fusion.*`:
  - `yaw_mode` (off / fallback_if_no_gnss_heading)
  - `max_age_s`
  - optional `yaw_offset_deg`, `sigma_yaw_deg`

## 6) LiDAR odometry / scan-matching (implemented - vertical slice)

### Purpose
Provide relative motion constraints that remain available when GNSS is degraded or absent.

### Variables
- `X(i), X(j)`

### Measurement
- Relative transform `ΔT_lidar = i_T_j` obtained from ICP between LiDAR keyframes (scan or submap).
- Convention:
  - translation in meters
  - angles in radians
  - `ΔT_lidar` expressed in the LiDAR frame at keyframe `i` (consistent with `LidarDeltaMessage`).

### Factor
- `BetweenFactorPose3(X(i), X(j), ΔT_lidar, Σ_lidar)`
- Optional robust kernel: Huber/Cauchy wrapping the base diagonal noise.

### Noise model
- Start with fixed diagonal sigmas:
  - `sigma_{x,y,z}_m`
  - `sigma_{roll,pitch,yaw}_deg`
- Future (optional): scale by ICP rmse/fitness.

### Gating (outlier rejection)
Config-driven thresholds:
- `max_translation_m`
- `max_rotation_deg`
- `max_rmse_m`
- `min_inlier_ratio`

Notes:
- When using Open3D ICP, `fitness` is treated as a proxy for inlier ratio if explicit inlier_ratio is unavailable.

### Failure modes / gotchas
- Wrong frame convention (using j_T_i instead of i_T_j) → catastrophic constraints.
- Over-inserting highly correlated LiDAR factors (e.g., many per second without intermediate states) → overconfidence.
- Large dt without sufficient overlap → ICP degeneracy (reject by gating).
