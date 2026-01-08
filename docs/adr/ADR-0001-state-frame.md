# ADR-0001: State frame is IMU/LiDAR

## Context
We need a consistent frame for state variables in the factor graph. The rig contains GNSS, AHRS, LiDAR and an IMU co-located with LiDAR.

## Decision
Define `X(i)` as the pose of the **state frame** in world (ENU). The default state frame is `imu`, and the rig defines `imu` as `same_as: lidar`.

## Consequences
- Extrinsics are used to map measurements into the state frame.
- Changing rigs should require only editing `configs/extrinsics.json` and `configs/config.yaml`.

## Alternatives
- Use a separate base_link frame not tied to a physical sensor.
- Use GNSS frame as the state (rejected: lever arm causes inconsistency; GNSS dropouts).
