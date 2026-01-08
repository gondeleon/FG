# Project Bible — slamboat (FG-SLAM for a harbor boat)

This document is the **single source of truth** for the project goals, conventions, and architectural decisions.
Use it as the compact context block when starting a new development thread (LiDAR, ROS2, fixed-lag, etc.).

## 1) Goal

Estimate the real-time state of a boat in an urban harbor using a **factor graph in GTSAM**.

Primary outputs at discrete keyframes `i`:

- Position: `(x, y, z)` in **ENU** (local tangent plane)
- Orientation: quaternion `(qx, qy, qz, qw)` representing the **state frame** in world
- Velocity: `(vx, vy, vz)` in world

Roadmap (incremental):

1. GNSS + IMU preintegration (implemented)
2. AHRS roll/pitch correction with yaw policy (implemented)
3. LiDAR odometry / scan-matching factors (planned)
4. ROS2 streaming source (planned)
5. Fixed-lag smoother / sliding window (planned)

## 2) Frames, conventions, and units

### World frame
- World = **ENU** (x East, y North, z Up)
- Local origin is defined by a **fixed reference lat/lon/alt** in `configs/config.yaml`.

### State frame
- **State frame** is configurable: `frames.state_frame`.
- Current configuration: `state_frame = imu`.
  - In this rig, the IMU is **co-located with LiDAR** (`imu` is defined as `same_as: lidar` in `configs/extrinsics.json`).

### Orientation
- Quaternions use `xyzw` order unless explicitly specified.
- Yaw is rotation around +Z (ENU).

### IMU units
- Gyro: rad/s
- Accel: m/s^2
- Gravity magnitude is set by config (`estimator.imu_preintegration.gravity_m_s2`).

## 3) I/O contract (decoupling)

The estimator core must not depend on the input backend.

### Canonical data path

1. **Readers** parse logs (CSV/TXT) based on YAML column mapping (no GTSAM import)
2. **Dispatcher** merges streams by timestamp (supports simulated realtime)
3. **Streaming layer** optionally reorders by **watermark** (late/out-of-order tolerance)
4. **Estimator** consumes `Event(kind, msg)` and updates the factor graph
5. **Writer** exports state traces

### Event model
- Events are typed by `kind` (e.g., `gnss`, `imu`, `ahrs`, later `lidar`).
- Each event has a sensor timestamp `t` in seconds.

## 4) Extrinsics convention

Extrinsics are stored in `configs/extrinsics.json`.

Current convention (declared inside the JSON):

- `parent_frame`: `ahrs`
- Transform stored is `T_parent_child`.
- Translation is expressed in the **parent** frame.

So, for each sensor frame `S` (gps/lidar/radar/imu), JSON provides **AHRS → S**.

The estimator works in the **state frame** (currently IMU/LiDAR), and derives the required transforms internally, e.g.:

- `T_state_gps`
- `T_state_ahrs`

## 5) Estimator: variables and factor graph

### Variables at keyframe i

- `X(i)`: `Pose3` of the **state frame** in world (ENU)
- `V(i)`: `Vector3` velocity in world
- `B(i)`: `imuBias::ConstantBias`
- `G(i)`: `Pose3` of the GNSS antenna frame in world (aux variable)

### Factors (current)

1) **IMU preintegration** between consecutive keyframes
- `ImuFactor(Xi, Vi, Xj, Vj, Bi, pim)`
- `BetweenFactorConstantBias(Bi, Bj, 0, bias_rw)`

2) **GNSS prior** on `G(i)`
- Translation always
- Yaw optionally (from RTK baseline heading)

3) **Rigid link** between state and GNSS
- `BetweenFactorPose3(X(i), G(i), T_state_gps)` with very tight noise

4) **AHRS orientation prior** on `X(i)`
- Roll/pitch only (gravity direction)
- Yaw policy is configurable (see ADR-0002)

### Solver
- Incremental optimization via `gtsam.ISAM2`.

## 6) Quality gating / robustness

GNSS gating (config-driven):

- `min_quality`, `max_hdop`, `min_sats`
- optional `max_jump_m` (ENU XY jump check)

Robust kernels may be added later per factor type.

## 7) Online mode (streaming)

Two independent problems:

1) **Data scheduling** (timestamps, latency, out-of-order)
2) **Incremental estimation** (iSAM2 / fixed-lag)

This repo currently supports:

- Offline replay with optional realtime sleeping (`--realtime`)
- Watermark reordering (`--watermark-delay`) to tolerate late arrivals

ROS2 integration will plug into the same event stream without touching the estimator.

## 8) How to start a new work thread

Use this minimal context block:

- Repo: `slamboat`
- World: ENU
- State frame: configurable, currently `imu` (co-located with LiDAR)
- Extrinsics: JSON stores `T_ahrs_sensor`
- Variables: `X,V,B,G`
- Factors: IMU preint, GNSS prior on `G`, rigid `X↔G`, AHRS roll/pitch prior
- Solver: iSAM2
- I/O: Readers → Dispatcher → (Watermark) → Estimator

Then specify:

- Task
- Acceptance criteria (tests/metrics)
- Branch name
