# Project Bible — slamboat (FG-SLAM for a harbor boat)

This document is the single source of truth for the project goals, conventions, and architectural decisions.

## 1) Goal

Estimate the real-time state of a boat in an urban harbor using a factor graph in GTSAM.

Primary outputs at discrete keyframes `i`:
- Position `(x, y, z)` in ENU
- Orientation quaternion of the state frame in world
- Velocity `(vx, vy, vz)` in world

Roadmap (incremental):
1. GNSS + IMU preintegration (implemented)
2. AHRS roll/pitch correction with yaw policy (implemented)
3. LiDAR odometry / scan-matching factors (implemented: vertical slice with offline deltas + BetweenFactor insertion)
4. ROS2 streaming source (planned)
5. Fixed-lag smoother / sliding window (planned)

## 2) Frames, conventions, and units

World frame:
- ENU (x East, y North, z Up)
- Origin defined by fixed reference lat/lon/alt in `configs/config.yaml`

State frame:
- `frames.state_frame` (current: `imu`)
- Rig convention: IMU is co-located with LiDAR (`imu` same_as `lidar`)

Orientation:
- Yaw is rotation around +Z (ENU)
- Quaternions are xyzw unless specified

IMU units:
- gyro rad/s
- accel m/s^2
- gravity magnitude from config

## 3) I/O contract (decoupling)

Canonical data path:
1) Readers parse logs (CSV/TXT) by YAML column mapping
2) Dispatcher merges streams by timestamp
3) Optional WatermarkScheduler reorders late arrivals
4) Estimator consumes `Event(kind, msg)` and updates the factor graph
5) Writer exports state traces

## 4) Extrinsics convention

Extrinsics live in `configs/extrinsics.json` and store `T_ahrs_sensor` (AHRS → sensor), with:
- transform is `T_parent_child`
- translation expressed in parent
- quaternion order xyzw

The estimator derives required transforms (e.g., `T_state_gps`) from that canonical representation.

## 5) Estimator: variables and factor graph

Variables at keyframe i:
- `X(i)`: Pose3 of state frame in world (ENU)
- `V(i)`: velocity in world
- `B(i)`: imuBias
- `G(i)`: Pose3 of GNSS antenna frame in world (aux variable)

Current factors:
- IMU preintegration between keyframes + bias RW
- GNSS prior on `G(i)` + rigid `Between(X(i), G(i))`
- AHRS roll/pitch orientation prior on `X(i)` (yaw policy configurable)
- LiDAR odometry `Between(X(i), X(j), ΔT_lidar)` with gating + optional robust kernel

Solver:
- incremental optimization via `gtsam.ISAM2`

## 6) LiDAR odometry (vertical slice implemented)

### What is implemented
- Offline ICP tool:
  - Input: `data/points/*.bin` (timestamp in filename in ns)
  - Output: `outputs/lidar_delta.csv` with delta pose + metrics
- Reader:
  - `src/slamboat/io/lidar.py::LidarDeltaReader` parses `odometry_delta` CSV into `LidarDeltaMessage`
- Estimator integration:
  - consumes `kind="lidar"` deltas and inserts:
    - `BetweenFactorPose3(X(i), X(j), ΔT_lidar, Σ_lidar)`
  - gating rejects outliers using:
    - translation/rotation
    - rmse
    - (proxy) inlier ratio from Open3D fitness
  - optional robust kernel (Huber/Cauchy)

### Repro commands
Generate deltas:
```bash
python scripts/lidar_icp_to_delta_csv.py \
  --points_dir data/points \
  --out_csv outputs/lidar_delta.csv \
  --step_s 1.0 \
  --max_range_m 80
