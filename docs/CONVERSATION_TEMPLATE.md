# Conversation template (for new chat threads)

Copy/paste this into a new ChatGPT conversation when you start a focused task.

---

## Context (stable)

- Repo: `slamboat`
- Goal: Real-time state estimation for a harbor boat using GTSAM factor graphs
- World: ENU
- State frame: configurable, currently `imu` (co-located with LiDAR)
- Extrinsics: `configs/extrinsics.json` stores `T_ahrs_sensor` (AHRS → sensor)
- Variables: `X(i)=Pose3`, `V(i)=Vel3`, `B(i)=Bias`, `G(i)=GPS Pose3`
- Factors: IMU preint, GNSS prior on `G(i)`, rigid `Between(X(i),G(i))`, AHRS roll/pitch prior
- Solver: iSAM2
- I/O: Readers → Dispatcher → (WatermarkScheduler) → Estimator

## Task

<Describe exactly one deliverable, e.g., "Add ROS2 source that pushes Events".>

## Acceptance criteria

- [ ] Unit tests added/updated
- [ ] CLI command to run the new functionality
- [ ] Example config snippet
- [ ] Metrics/plots if relevant

## Branch

- `feature/<name>`

## Files to touch

- List expected files

---
