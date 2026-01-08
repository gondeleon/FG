# ADR-0004: Watermark scheduler for online event ordering

## Context
Online streams (ROS2) can deliver messages out of order and with variable latency.
A strict timestamp order requirement would block the pipeline or cause inconsistent fusion.

## Decision
Introduce a watermark-based scheduler:
- Maintain `max_seen_t`
- Emit events when `t <= max_seen_t - watermark_delay_s`

Expose `--watermark-delay` to tune late-arrival tolerance.

## Consequences
- Stable fusion under mild out-of-order conditions.
- Adds bounded latency of `watermark_delay_s`.

## Alternatives
- Hard reorder with full buffering (unbounded latency)
- No reorder (risk of negative dt in IMU integration)
