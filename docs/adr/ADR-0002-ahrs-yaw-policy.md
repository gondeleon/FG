# ADR-0002: AHRS yaw policy (magnetic vs true north)

## Context
AHRS yaw is typically referenced to **magnetic north**, while GNSS heading (RTK baseline) is referenced to **true north**.
Mixing them without correction introduces systematic yaw bias.

## Decision
Default policy: **use AHRS only for roll/pitch** (gravity direction), and ignore yaw by assigning a very large yaw sigma in the AHRS Pose3 prior.

Optional policy: `fallback_if_no_gnss_heading`
- Only constrain yaw from AHRS when GNSS heading is missing.
- Allow a constant `yaw_offset_deg` to align magnetic to navigation yaw.

## Consequences
- Stable roll/pitch in mild sea states.
- Avoids magnetic declination/systematic yaw conflicts.

## Alternatives
- Full magnetometer fusion with declination model (planned, but requires validated magnetic calibration).
