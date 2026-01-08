# Architecture (Pose3 + IMU preintegration)

```text
configs/config.yaml  +  configs/extrinsics.json
         |
         v
 slamboat.config (Pydantic validation)
         |
         v
I/O readers (CSV/TXT config-driven)
  - GnssReader -> GnssMessage
  - ImuReader  -> ImuMessage
         |
         v
Dispatcher (merge streams by timestamp)
         |
         v
Core Estimator (GTSAM iSAM2)
  Variables: X(i)=Pose3, V(i)=Vel3, B(i)=Bias
  Factors:
    - ImuFactor (preintegrated IMU between GNSS keyframes)
    - BetweenFactorConstantBias (bias random walk)
    - PriorFactorPose3 (GNSS position + optional yaw)
         |
         v
outputs/state_trace.csv
```
