# slamboat — Online factor-graph state estimation (Pose3 + IMU preintegration)

This repository is a **config-driven** Python project to estimate the real-time state of a boat in a harbor
using **GTSAM**. It is designed so that changing the sensor rig requires editing only:

- `configs/config.yaml` (paths, column mapping, units)
- `configs/extrinsics.json` (rig geometry / frames)

The estimator core is **independent** from the input backend:
offline file replay today, ROS2/socket streaming later.

## What is implemented in this iteration

State at keyframes `i` (currently: each GNSS sample creates a keyframe):

- `X(i)`: `Pose3` of the **state frame** in ENU world (default: state frame = IMU)
- `V(i)`: velocity in world frame
- `B(i)`: IMU bias
- `G(i)`: `Pose3` of the **GNSS antenna frame** in world frame

Factors:

- IMU preintegration between keyframes:
  - `ImuFactor(Xi, Vi, Xj, Vj, Bi, preintegrated)`
  - `BetweenFactorConstantBias(Bi, Bj, 0, bias_rw)`

- GPS (GNSS) factor **as in the referenced approach**:
  - GNSS provides position (ENU/UTM-like local coordinates) + heading (true north)
  - We put a prior on the GPS pose variable `G(i)` (translation + yaw; roll/pitch are left free)

- Rigid connection between the state frame (IMU) and the GPS antenna:
  - `BetweenFactorPose3(X(i), G(i), T_imu_gps)` with *very tight* noise
  - `T_imu_gps` is derived from `configs/extrinsics.json`

- Optional AHRS factor:
  - AHRS provides orientation but yaw is magnetic-north referenced
  - We **only use roll/pitch** (gravity direction) by applying a Pose3 prior on `X(i)` with:
    - small roll/pitch sigma
    - extremely large yaw sigma (yaw effectively ignored)
    - extremely large translation sigmas (no position from AHRS)

## Install

Create a virtual environment and install in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

### GTSAM is mandatory

If `gtsam` is missing, the pipeline **stops** (fail-fast):

```bash
python -c "import gtsam; print('gtsam OK')"
```

## Run

Edit `configs/config.yaml` to point to your logs and then:

```bash
python -m slamboat.cli.run --config configs/config.yaml
```

Output:

- `outputs/state_trace.csv` (t, x, y, z, qx, qy, qz, qw, vx, vy, vz)

## How to add a new sensor

1. Add a new `*Spec` model in `src/slamboat/config.py` (including `frame_id`).
2. Implement a reader in `src/slamboat/io/<sensor>.py` producing a typed message in `src/slamboat/core/types.py`.
3. Register the reader in `src/slamboat/io/factory.py`.
4. Extend the estimator to consume the new message type (or store it for later fusion).

## Architecture overview (high level)

- `slamboat/io/*`:
  file parsing, unit conversion, timestamp validation (NO GTSAM import)

- `slamboat/dispatch/*`:
  merges streams by timestamp; can be swapped for streaming sources

- `slamboat/core/*`:
  estimation logic (GTSAM), state output

- `configs/*`:
  everything you need to change between rigs/datasets

## Tests

```bash
pytest
```

The end-to-end test will be skipped if `gtsam` is not installed.
