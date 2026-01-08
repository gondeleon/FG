# ADR-0003: Extrinsics JSON convention (AHRS → sensor)

## Context
The dataset calibration provides extrinsics between AHRS and other sensors, obtained via hand-eye calibration and blueprint translations.

## Decision
Store all extrinsics as `T_ahrs_sensor` in `configs/extrinsics.json`, explicitly declaring:
- parent_frame = ahrs
- transform = T_parent_child
- quaternion order = xyzw
- translation expressed in parent

The estimator derives required transforms (e.g., `T_state_gps`) from this canonical representation.

## Consequences
- Rig portability: new rigs replace a single JSON.
- Clear directionality avoids frame-mixing bugs.

## Alternatives
- Store all transforms w.r.t. state frame directly (rejected: harder to maintain across calibrations).
