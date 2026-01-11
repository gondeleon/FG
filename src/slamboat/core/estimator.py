"""GTSAM estimator (Pose3 + IMU preintegration).

This version is designed for:
  - state X(i) in the IMU frame (co-located with LiDAR)
  - GNSS provides position + heading (true north), but at the GPS antenna frame
  - AHRS provides orientation, but yaw is referenced to magnetic north

We implement the following strategy (matching your quoted calibration approach):

  * GPS factor:
      - use GNSS position (ENU) and heading as a prior on the GPS pose variable G(i)

  * AHRS factor:
      - use AHRS orientation only to correct roll/pitch (gravity direction)
      - yaw is NOT used (set yaw sigma extremely large)

To keep the factor graph modular and adaptable:
  - we introduce a GPS pose variable G(i)
  - connect X(i) -> G(i) via a known rigid transform T_imu_gps from extrinsics.json

This avoids custom factors and keeps the system portable across rigs.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np

from slamboat.config import AhrsSpec, EstimatorSpec, NoiseSpec
from slamboat.core.geo import LocalENU
from slamboat.core.output import State3D
from slamboat.core.types import AhrsMessage, GnssMessage, ImuMessage, MessageBase, SensorKind, LidarDeltaMessage
from slamboat.extrinsics import ExtrinsicsDB
from slamboat.lidar.gating import gate_icp_result
from slamboat.lidar.icp_types import IcpMetrics, IcpResult

log = logging.getLogger(__name__)


def require_gtsam() -> None:
    try:
        import gtsam  # noqa: F401
    except Exception as e:  # pragma: no cover
        log.warning("GTSAM is required but not available: %s", e)
        raise SystemExit(2) from e


def _deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0

def _finite(x) -> bool:
    try:
        return x is not None and float(x) == float(x) and abs(float(x)) != float("inf")
    except Exception:
        return False

def _get_any(obj: object, names: list[str]):
    """
    Return the first existing attribute value from `names` on `obj`, else None.
    This makes the code robust to different GnssMessage field naming conventions.
    """
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None

def _sigmas_pose6(roll: float, pitch: float, yaw: float, x: float, y: float, z: float) -> np.ndarray:
    # GTSAM Pose3 noise order: [roll, pitch, yaw, x, y, z]
    return np.array([roll, pitch, yaw, x, y, z], dtype=float)

def _navstate_pose(nav) -> object:
    """
    Return Pose3 from a GTSAM NavState in a version-robust way.
    Some bindings expose pose() as a method; keep a fallback.
    """
    if hasattr(nav, "pose"):
        return nav.pose()  # common
    if hasattr(nav, "getPose"):
        return nav.getPose()
    raise AttributeError("NavState has no pose()/getPose() in this GTSAM build.")


def _navstate_vel(nav) -> object:
    """
    Return velocity (Vector3) from a GTSAM NavState in a version-robust way.
    Different GTSAM Python builds expose velocity as v() or velocity().
    """
    if hasattr(nav, "v"):
        return nav.v()  # some builds
    if hasattr(nav, "velocity"):
        return nav.velocity()  # other builds
    if hasattr(nav, "getV"):
        return nav.getV()
    raise AttributeError("NavState has no v()/velocity()/getV() in this GTSAM build.")

class OnlineEstimatorPose3ImuPreint:
    """Pose3 estimator with GTSAM IMU preintegration (ImuFactor)."""
    # Design note:
    # In a GNSS+IMU setup without direct velocity measurements, the velocity
    # component of the NavState is weakly observable. We therefore treat V(i)
    # as an internal variable used by the IMU factor, while exporting
    # finite-difference velocity derived from the estimated poses.

    def __init__(
        self,
        cfg: EstimatorSpec,
        noise: NoiseSpec,
        enu: LocalENU,
        extr: ExtrinsicsDB,
        state_frame: str,
        gnss_frame: str,
        ahrs_frame: str,
        ahrs_specs: Optional[list[AhrsSpec]] = None,
        last_keyframe_t: Optional[float] = None,
        lidar_pending_delta: Optional[LidarDeltaMessage] = None,
        lidar_last_used_k: Optional[int] = None,

    ):
        require_gtsam()
        import gtsam  # type: ignore

        self.gtsam = gtsam
        self.cfg = cfg
        self.noise = noise
        self.enu = enu

        self.extr = extr
        self.state_frame = state_frame
        self.gnss_frame = gnss_frame
        self.ahrs_frame = ahrs_frame
        self.ahrs_specs = ahrs_specs or []

        self.last_keyframe_t = last_keyframe_t
        self.lidar_pending_delta = lidar_pending_delta
        self.lidar_last_used_k = lidar_last_used_k
        # Precompute rigid transform between IMU(state) and GPS antenna frames
        # We need imu_T_gps so that: world_T_gps = world_T_imu * imu_T_gps
        self.imu_T_gps = self.extr.pose3(self.state_frame, self.gnss_frame)

        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.isam = gtsam.ISAM2()

        self.i = 0  # keyframe index
        self._bootstrapped = False
        
        # For GNSS gating (jump detection between accepted fixes).
        self._last_good_gnss_xy: Optional[Tuple[float, float]] = None

        self.last_imu: Optional[ImuMessage] = None
        self.last_ahrs: Optional[AhrsMessage] = None

        # Last accepted GNSS position in ENU (for jump gating).
        self._last_good_gnss_xy_m: Optional[np.ndarray] = None

        self.bias0 = gtsam.imuBias.ConstantBias()
        self.preint_params = self._make_preint_params(cfg)
        self.pim = gtsam.PreintegratedImuMeasurements(self.preint_params, self.bias0)
        # Keep track of the bias currently used by the preintegrator so that we can reset safely on IMU gaps.
        self._pim_bias = self.bias0

    # --- Symbols ---
    def _X(self, k: int):
        return self.gtsam.symbol("x", k)

    def _V(self, k: int):
        return self.gtsam.symbol("v", k)

    def _B(self, k: int):
        return self.gtsam.symbol("b", k)

    def _G(self, k: int):
        # GPS pose variable
        return self.gtsam.symbol("g", k)

    # --- Noise models ---
    def _noise_pose(self, sigmas6: np.ndarray):
        return self.gtsam.noiseModel.Diagonal.Sigmas(sigmas6)

    def _noise_vel(self, sigma_v: float):
        return self.gtsam.noiseModel.Isotropic.Sigma(3, float(sigma_v))

    def _noise_bias6(self, sig_acc: float, sig_gyro: float):
        sig = np.array([sig_acc, sig_acc, sig_acc, sig_gyro, sig_gyro, sig_gyro], dtype=float)
        return self.gtsam.noiseModel.Diagonal.Sigmas(sig)

    def _noise_bias_rw6(self, sig_acc_rw: float, sig_gyro_rw: float):
        sig = np.array([sig_acc_rw, sig_acc_rw, sig_acc_rw, sig_gyro_rw, sig_gyro_rw, sig_gyro_rw], dtype=float)
        return self.gtsam.noiseModel.Diagonal.Sigmas(sig)

    def _tight_noise_pose(self):
        # Extrinsics are treated as deterministic in this iteration.
        sig = _sigmas_pose6(1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6)
        return self._noise_pose(sig)

    # --- Preintegration params ---
    def _make_preint_params(self, cfg: EstimatorSpec):
        import gtsam  # type: ignore

        p = gtsam.PreintegrationParams.MakeSharedU(cfg.imu_preintegration.gravity_m_s2)
        I3 = np.eye(3, dtype=float)

        p.setAccelerometerCovariance(I3 * (cfg.imu_preintegration.accel_noise_sigma**2))
        p.setGyroscopeCovariance(I3 * (cfg.imu_preintegration.gyro_noise_sigma**2))
        p.setIntegrationCovariance(I3 * (cfg.imu_preintegration.integration_sigma**2))
        
        #--------------------------------------------------------------
        #                               Note
        #--------------------------------------------------------------
        # The official ImuFactor workflow sets ONLY the measurement covariances:
        #   - GyroscopeCovariance
        #   - AccelerometerCovariance
        #   - IntegrationCovariance
        #
        # Bias handling is modeled in the factor graph via:
        #   - a prior on the initial bias (B0), and/or
        #   - a random-walk constraint BetweenFactorConstantBias between biases at keyframes.
        # If we need a bias covariances inside the preintegration params, we need to use 
        # CombinedImuFactor / PreintegrationCombinedParams workflow.

        # If the state/body frame is not coincident with the IMU sensor frame, set body_P_sensor here.
        # Example:
        # p.setBodyPSensor(gtsam.Pose3(gtsam.Rot3.Quaternion(w, x, y, z), gtsam.Point3(tx, ty, tz)))


        return p

    # --- Helpers ---
    def _pose_from_gnss(self, gnss: GnssMessage, use_alt: bool):
        """Build a Pose3 measurement in the GNSS frame, expressed in world (ENU)."""
        import gtsam  # type: ignore

        if use_alt and gnss.geoid_height_m is not None:
            x, y, z = self.enu.llh_to_enu_m(gnss.lat_deg, gnss.lon_deg, gnss.geoid_height_m)
        else:
            x, y = self.enu.ll_to_enu_m(gnss.lat_deg, gnss.lon_deg)
            z = 0.0

        yaw = 0.0 if gnss.heading_deg is None else _deg_to_rad(gnss.heading_deg)
        R = gtsam.Rot3.Yaw(yaw)  # roll/pitch not observed by GNSS
        t = gtsam.Point3(float(x), float(y), float(z))
        return gtsam.Pose3(R, t)
    
    def _gnss_passes_gating(self, gnss: GnssMessage, use_alt: bool) -> bool:
        """
        GNSS gating using config-driven thresholds:
          - min_quality
          - max_hdop
          - min_sats
          - max_jump_m (optional jump check in ENU)

        If a sample is rejected, we do NOT update the last_good_gnss position.
        """
        g = self.cfg.gnss_gating
        if not g.enable:
            return True

        # --- scalar gates (robust to naming differences in GnssMessage) ---
        quality = _get_any(gnss, ["quality_indicator", "quality", "gps_quality", "fix_quality"])
        hdop = _get_any(gnss, ["hdop", "h_dop", "horizontal_dop"])
        n_sats = _get_any(gnss, ["num_sats", "n_sats", "satellites", "sats_used"])

        if quality is not None and int(quality) < int(g.min_quality):
            log.warning("GNSS gated: quality=%s < min_quality=%s (t=%.3f)", quality, g.min_quality, gnss.t)
            return False

        if hdop is not None and float(hdop) > float(g.max_hdop):
            log.warning("GNSS gated: hdop=%.3f > max_hdop=%.3f (t=%.3f)", float(hdop), float(g.max_hdop), gnss.t)
            return False

        if n_sats is not None and int(n_sats) < int(g.min_sats):
            log.warning("GNSS gated: sats=%s < min_sats=%s (t=%.3f)", n_sats, g.min_sats, gnss.t)
            return False


        # --- jump check in ENU (XY) ---
        if g.enable_jump_check:
            x_m, y_m = self.enu.ll_to_enu_m(gnss.lat_deg, gnss.lon_deg)
            xy = np.array([float(x_m), float(y_m)], dtype=float)

            if self._last_good_gnss_xy_m is not None:
                d = float(np.linalg.norm(xy - self._last_good_gnss_xy_m))
                if d > float(g.max_jump_m):
                    log.warning(
                        "GNSS gated: jump=%.2f m > max_jump_m=%.2f m (t=%.3f)",
                        d,
                        float(g.max_jump_m),
                        gnss.t,
                    )
                    return False

            # Accept and update last good position
            self._last_good_gnss_xy_m = xy

        return True

    def _pose_from_ahrs_to_state(self, ahrs: AhrsMessage):
        """Convert AHRS quaternion into a world->state Rot3 measurement.

        We assume the logged quaternion describes world<->AHRS depending on config.
        Then map AHRS rotation to state frame using extrinsics.
        """
        import gtsam  # type: ignore

        # Find matching AHRS spec (by frame_id) to interpret convention.
        spec: Optional[AhrsSpec] = None
        for s in self.ahrs_specs:
            if s.frame_id == ahrs.frame_id:
                spec = s
                break

        # Default: world_T_sensor
        convention = spec.orientation_convention if spec is not None else "world_T_sensor"
        q_order = spec.quaternion_order if spec is not None else "xyzw"

        if q_order == "xyzw":
            qx, qy, qz, qw = ahrs.qx, ahrs.qy, ahrs.qz, ahrs.qw
            Rot_w_a = gtsam.Rot3.Quaternion(float(qw), float(qx), float(qy), float(qz))  # (w,x,y,z)
        else:  # wxyz in log
            qw, qx, qy, qz = ahrs.qw, ahrs.qx, ahrs.qy, ahrs.qz
            Rot_w_a = gtsam.Rot3.Quaternion(float(qw), float(qx), float(qy), float(qz))

        if convention == "sensor_T_world":
            Rot_w_a = Rot_w_a.inverse()

        # Convert AHRS -> state using extrinsics.
        # extr.pose3(ahrs_frame, state_frame) returns T(ahrs, state): maps state -> ahrs
        T_a_s = self.extr.pose3(self.ahrs_frame, self.state_frame)
        Rot_a_s = T_a_s.rotation()  # maps state vectors -> ahrs vectors

        # state->world rotation:
        #   state -> ahrs -> world  ==>  R_w_s = R_w_a * R_a_s
        Rot_w_s = Rot_w_a.compose(Rot_a_s)
        return Rot_w_s

    def _bootstrap(self, gnss: GnssMessage, use_alt: bool) -> None:
        import gtsam  # type: ignore

        # Initialize IMU pose by "pulling back" the GNSS pose using extrinsics:
        # world_T_imu ≈ world_T_gps ∘ inv(imu_T_gps)
        gps_pose = self._pose_from_gnss(gnss, use_alt=use_alt)
        imu_pose0 = gps_pose.compose(self.imu_T_gps.inverse())

        v0 = np.zeros(3, dtype=float)
        b0 = self.bias0

        # Priors: loose roll/pitch/yaw for IMU initial pose
        sig_pose0 = _sigmas_pose6(_deg_to_rad(30.0), _deg_to_rad(30.0), _deg_to_rad(60.0),
                                 5.0, 5.0, 10.0)
        self.graph.add(gtsam.PriorFactorPose3(self._X(0), imu_pose0, self._noise_pose(sig_pose0)))
        self.graph.add(gtsam.PriorFactorVector(self._V(0), v0, self._noise_vel(self.noise.sigma_v0_m_s)))
        self.graph.add(
            gtsam.PriorFactorConstantBias(
                self._B(0),
                b0,
                self._noise_bias6(self.noise.sigma_bias_acc_m_s2, self.noise.sigma_bias_gyro_rad_s),
            )
        )

        # Add GPS pose variable + rigid connection + measurement prior
        self._add_gps_factors(k=0, gps_pose=gps_pose, heading_available=(gnss.heading_deg is not None))

        # Initial values
        self.initial.insert(self._X(0), imu_pose0)
        self.initial.insert(self._V(0), v0)
        self.initial.insert(self._B(0), b0)

        # For G(0), a reasonable initial is gps_pose
        self.initial.insert(self._G(0), gps_pose)

        self.isam.update(self.graph, self.initial)
        self.graph.resize(0)
        self.initial.clear()

        self.i = 0
        self.pim.resetIntegrationAndSetBias(b0)
        self._pim_bias = b0
        self._bootstrapped = True
        self.last_keyframe_t = gnss.t

    def _add_gps_factors(self, k: int, gps_pose, heading_available: bool) -> None:
        import gtsam  # type: ignore

        # Hard constraint: G(k) = X(k) ∘ imu_T_gps
        self.graph.add(gtsam.BetweenFactorPose3(self._X(k), self._G(k), self.imu_T_gps, self._tight_noise_pose()))

        # GNSS prior on GPS pose:
        # - translation: sigma_xy, sigma_z
        # - yaw: sigma_yaw if heading exists, else very weak
        # - roll/pitch: extremely weak (GNSS doesn't measure gravity)
        yaw_sig = _deg_to_rad(self.noise.sigma_yaw_deg) if heading_available else _deg_to_rad(179.0)

        sig = _sigmas_pose6(
            _deg_to_rad(179.0),
            _deg_to_rad(179.0),
            yaw_sig,
            self.noise.sigma_xy_m,
            self.noise.sigma_xy_m,
            self.noise.sigma_z_m,
        )
        self.graph.add(gtsam.PriorFactorPose3(self._G(k), gps_pose, self._noise_pose(sig)))

    def _add_ahrs_factor_if_available(self, k: int, t_k: float, heading_available: bool) -> None:
        """
        Attach an AHRS orientation prior on X(k).

        Strategy (rig-adaptable, matches your paper quote):
        - Always use AHRS to constrain roll/pitch (gravity direction).
        - Yaw is magnetic-north referenced, so by default we do NOT constrain yaw.
        - Optionally, allow yaw only when GNSS heading is missing (fallback mode).

        Config:
        estimator.ahrs_fusion.yaw_mode:
            - "off"
            - "fallback_if_no_gnss_heading"
        """
        if not self.cfg.ahrs_fusion.enable:
            return
        if self.last_ahrs is None:
            return
        if abs(t_k - self.last_ahrs.t) > float(self.cfg.ahrs_fusion.max_age_s):
            return

        import gtsam  # type: ignore

        Rot_w_state_meas = self._pose_from_ahrs_to_state(self.last_ahrs)

        # Decide whether to use yaw.
        yaw_mode = str(self.cfg.ahrs_fusion.yaw_mode).lower()
        yaw_sig = 1e6  # default: ignore yaw

        if yaw_mode == "off":
            yaw_sig = 1e6

        elif yaw_mode == "fallback_if_no_gnss_heading":
            # Only constrain yaw when GNSS heading is not available.
            if not heading_available:
                yaw_sig = _deg_to_rad(float(self.cfg.ahrs_fusion.sigma_yaw_deg))

                # Optional yaw offset to align magnetic yaw -> navigation yaw (true north).
                # Applied as a world-frame Z rotation.
                yaw_off_deg = float(self.cfg.ahrs_fusion.yaw_offset_deg)
                if abs(yaw_off_deg) > 1e-12:
                    Rot_w_state_meas = gtsam.Rot3.Yaw(_deg_to_rad(yaw_off_deg)).compose(Rot_w_state_meas)
            else:
                yaw_sig = 1e6

        else:
            raise ValueError(f"Unsupported ahrs_fusion.yaw_mode: {self.cfg.ahrs_fusion.yaw_mode}")

        # Pose3 prior noise:
        # - roll/pitch: tight-ish (gravity)
        # - yaw: either huge (ignored) or configured (fallback)
        # - translation: huge (AHRS has no position)
        sig_rp = _deg_to_rad(self.noise.sigma_roll_pitch_deg)
        sig = _sigmas_pose6(sig_rp, sig_rp, yaw_sig, 1e6, 1e6, 1e6)

        meas_pose = gtsam.Pose3(Rot_w_state_meas, gtsam.Point3(0.0, 0.0, 0.0))
        self.graph.add(gtsam.PriorFactorPose3(self._X(k), meas_pose, self._noise_pose(sig)))


    def _integrate_imu(self, imu: ImuMessage) -> None:
        import gtsam  # type: ignore

        if self.last_imu is None:
            self.last_imu = imu
            return

        dt = imu.t - self.last_imu.t
        if dt <= 0:
            self.last_imu = imu
            return

        # Guard against large IMU time gaps to avoid injecting unrealistic impulses into preintegration.
        # Typical symptoms: |v| spikes (e.g., 50+ m/s) on a vessel dataset.
        max_dt = float(getattr(self.cfg.imu_preintegration, "max_dt_s", 0.05))
        if dt > max_dt:
            log.warning("Large IMU gap detected (dt=%.6f s > %.6f s). Resetting preintegration.", dt, max_dt)
            self.pim.resetIntegrationAndSetBias(self._pim_bias)
            self.last_imu = imu
            return


        acc = np.array([self.last_imu.ax_m_s2, self.last_imu.ay_m_s2, self.last_imu.az_m_s2], dtype=float)
        omg = np.array([self.last_imu.wx_rad_s, self.last_imu.wy_rad_s, self.last_imu.wz_rad_s], dtype=float)

        self.pim.integrateMeasurement(acc, omg, float(dt))
        self.last_imu = imu

    def _current_estimate(self):
        res = self.isam.calculateEstimate()
        pose = res.atPose3(self._X(self.i))
        vel = res.atVector(self._V(self.i))
        bias = res.atConstantBias(self._B(self.i))
        return pose, vel, bias

    def _pose_to_state(self, t: float, pose, vel) -> State3D:
        """
        Convert a GTSAM Pose3 + velocity vector into a serializable State3D.

        Note: In GTSAM Python bindings, Pose3.translation() may return either:
        - gtsam.Point3 (with .x(), .y(), .z())
        - numpy.ndarray([x, y, z])
        depending on the build. We support both.
        """
        import numpy as _np  # local import to keep module import-time light

        # --- translation ---
        trans = pose.translation()
        if hasattr(trans, "x"):
            x, y, z = float(trans.x()), float(trans.y()), float(trans.z())
        else:
            # assume array-like length 3
            x, y, z = float(trans[0]), float(trans[1]), float(trans[2])

        # --- quaternion ---
        # toQuaternion() can return a gtsam.Quaternion (w,x,y,z accessors) or array-like [w,x,y,z]
        q = pose.rotation().toQuaternion()
        if hasattr(q, "w"):
            qw, qx, qy, qz = float(q.w()), float(q.x()), float(q.y()), float(q.z())
        else:
            # assume array-like [w, x, y, z]
            qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])

        # --- velocity ---
        # vel may be numpy array, list, or gtsam Vector
        if isinstance(vel, _np.ndarray):
            vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])
        else:
            vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])
        
        # The NavState velocity V(i) is weakly observable in a GNSS+IMU setup
        # without direct velocity measurements (e.g., Doppler GNSS).
        # For output purposes, we therefore provide a finite-difference velocity
        # computed from consecutive poses, which is physically more reliable.
        #
        # The graph velocity V(i) is still used internally by the IMU factor.

        if not hasattr(self, "_prev_state_for_fd"):
            vx_fd, vy_fd, vz_fd = 0.0, 0.0, 0.0
        else:
            dt_fd = float(t - self._prev_state_for_fd.t)
            if dt_fd > 0:
                vx_fd = (x - self._prev_state_for_fd.x) / dt_fd
                vy_fd = (y - self._prev_state_for_fd.y) / dt_fd
                vz_fd = (z - self._prev_state_for_fd.z) / dt_fd
            else:
                vx_fd, vy_fd, vz_fd = 0.0, 0.0, 0.0

        state = State3D(
            t=float(t),
            x=x,
            y=y,
            z=z,
            qx=qx,
            qy=qy,
            qz=qz,
            qw=qw,
            vx=vx_fd,
            vy=vy_fd,
            vz=vz_fd,
        )

        self._prev_state_for_fd = state
        return state



    # --- Public handler ---
    def handle(self, kind: SensorKind, msg: MessageBase, use_alt: bool) -> Optional[State3D]:
        if kind == "imu":
            self._integrate_imu(msg)  # type: ignore[arg-type]
            return None

        if kind == "ahrs":
            self.last_ahrs = msg  # type: ignore[assignment]
            return None

        if kind == "gnss":
            return self._handle_gnss(msg, use_alt=use_alt)  # type: ignore[arg-type]

        if kind == "lidar":
            # Buffer the latest lidar delta; it will be consumed when the next GNSS keyframe is created.
            self.lidar_pending_delta = msg  # type: ignore[assignment]
            return None


        raise ValueError(f"Unsupported kind: {kind}")

    def _handle_gnss(self, gnss: GnssMessage, use_alt: bool) -> Optional[State3D]:
        import gtsam  # type: ignore

        # -------------------------
        # GNSS gating (basic)
        # -------------------------
        # [Atencion ToDo]
        # Esto no es fatal, pero puede producir comportamiento inconsistente (y logs confusos).
        # Lo dejo para un commit posterior de limpieza, porque ahora estamos enfocados en LiDAR.
        if not self._gnss_passes_gating(gnss, use_alt=use_alt):
            # Reject this GNSS update: keep integrating IMU, do not create a keyframe.
            return None

        # -------------------------
        # GNSS gating (quality/hdop/sats + optional jump check)
        # -------------------------
        gate = getattr(self.cfg, "gnss_gating", None)
        if gate is not None and getattr(gate, "enable", True):
            q_ok = (gnss.quality is not None) and (int(gnss.quality) >= int(gate.min_quality))
            s_ok = (gnss.n_sats is not None) and (int(gnss.n_sats) >= int(gate.min_sats))
            h_ok = (gnss.hdop is not None) and _finite(gnss.hdop) and (float(gnss.hdop) <= float(gate.max_hdop))

            if not (q_ok and s_ok and h_ok):
                log.warning(
                    "GNSS rejected by gating: quality=%s hdop=%s n_sats=%s",
                    gnss.quality, gnss.hdop, gnss.n_sats
                )
                return None

            # Jump check in ENU (computed from GNSS lat/lon)
            if getattr(gate, "enable_jump_check", True) and self._last_good_gnss_xy is not None:
                # Use the same ENU conversion used for GNSS pose creation
                x_tmp, y_tmp = self.enu.ll_to_enu_m(gnss.lat_deg, gnss.lon_deg)
                dx = x_tmp - self._last_good_gnss_xy[0]
                dy = y_tmp - self._last_good_gnss_xy[1]
                jump = float((dx * dx + dy * dy) ** 0.5)
                if jump > float(gate.max_jump_m):
                    log.warning("GNSS rejected by jump check: jump=%.2f m > %.2f m", jump, gate.max_jump_m)
                    return None


        if not self._bootstrapped:
            self._bootstrap(gnss, use_alt=use_alt)
            pose, vel, _ = self._current_estimate()
            return self._pose_to_state(gnss.t, pose, vel)

        j = self.i + 1
        pose_i, vel_i, bias_i = self._current_estimate()

        nav_i = gtsam.NavState(pose_i, vel_i)
        nav_j_pred = self.pim.predict(nav_i, bias_i)

        # IMU factor between states
        self.graph.add(
            gtsam.ImuFactor(self._X(self.i), self._V(self.i), self._X(j), self._V(j), self._B(self.i), self.pim)
        )

        # Bias random walk
        self.graph.add(
            gtsam.BetweenFactorConstantBias(
                self._B(self.i),
                self._B(j),
                gtsam.imuBias.ConstantBias(),
                self._noise_bias_rw6(self.noise.sigma_bias_acc_rw_m_s2, self.noise.sigma_bias_gyro_rw_rad_s),
            )
        )

        # GPS pose variable + GNSS prior
        gps_pose = self._pose_from_gnss(gnss, use_alt=use_alt)
        # Mark GNSS as accepted (for jump gating).
        x_ok, y_ok = self.enu.ll_to_enu_m(gnss.lat_deg, gnss.lon_deg)
        self._last_good_gnss_xy = (float(x_ok), float(y_ok))
        
        self._add_gps_factors(k=j, gps_pose=gps_pose, heading_available=(gnss.heading_deg is not None))

        # AHRS roll/pitch factor on X(j) if available
        self._add_ahrs_factor_if_available(
            k=j,
            t_k=gnss.t,
            heading_available=(gnss.heading_deg is not None),
        )

        # Initial guesses
        self.initial.insert(self._X(j), _navstate_pose(nav_j_pred))
        self.initial.insert(self._V(j), _navstate_vel(nav_j_pred))
        self.initial.insert(self._B(j), bias_i)

        # G(j) initial guess: measured gps pose
        self.initial.insert(self._G(j), gps_pose)

        # --- LiDAR odometry factor (optional) ---
        lodom = getattr(self.cfg, "lidar_odometry", None)
        if lodom is not None and getattr(lodom, "enabled", False):
            if self.lidar_pending_delta is not None:
                # Only use if close to this keyframe time
                max_age = float(getattr(getattr(lodom, "keyframes", object()), "max_age_s", 0.25))
                if abs(float(gnss.t) - float(self.lidar_pending_delta.t)) <= max_age:
                    # Build 4x4 from delta (i_T_j in lidar_i frame)
                    import gtsam  # type: ignore
                    dx = float(self.lidar_pending_delta.dx)
                    dy = float(self.lidar_pending_delta.dy)
                    dz = float(self.lidar_pending_delta.dz)
                    droll = float(self.lidar_pending_delta.droll)
                    dpitch = float(self.lidar_pending_delta.dpitch)
                    dyaw = float(self.lidar_pending_delta.dyaw)

                    R = gtsam.Rot3.RzRyRx(droll, dpitch, dyaw)
                    t = gtsam.Point3(dx, dy, dz)
                    delta_pose = gtsam.Pose3(R, t)

                    # Convert Pose3 -> 4x4 for gating
                    T = delta_pose.matrix()
                    # Metrics: use CSV-provided dt/rmse/fitness when available
                    dt_s = None
                    if getattr(self.lidar_pending_delta, "dt_s", None) is not None:
                        dt_s = float(self.lidar_pending_delta.dt_s)  # type: ignore[arg-type]
                    elif self.last_keyframe_t  is not None:
                        dt_s = float(gnss.t - self.last_keyframe_t )

                    rmse_m = getattr(self.lidar_pending_delta, "rmse_m", None)
                    fitness = getattr(self.lidar_pending_delta, "fitness", None)
                    inlier_ratio = getattr(self.lidar_pending_delta, "inlier_ratio", None)

                    # If no explicit inlier_ratio, treat Open3D fitness as proxy
                    if inlier_ratio is None and fitness is not None:
                        inlier_ratio = fitness

                    metrics = IcpMetrics(
                        rmse_m=float(rmse_m) if rmse_m is not None else None,
                        fitness=float(fitness) if fitness is not None else None,
                        inlier_ratio=float(inlier_ratio) if inlier_ratio is not None else None,
                        correspondences=getattr(self.lidar_pending_delta, "correspondences", None),
                        iterations=getattr(self.lidar_pending_delta, "iterations", None),
                        dt_s=dt_s,
                    )
                    icp_res = IcpResult(delta_T=T, metrics=metrics)

                    ok, reasons, summary = gate_icp_result(icp_res, lodom.gating)

                    if ok:
                        # Noise model (Pose3 order: [roll,pitch,yaw,x,y,z])
                        def _d2r(d): return float(d) * math.pi / 180.0
                        sig = np.array([
                            _d2r(lodom.noise.sigma_roll_deg),
                            _d2r(lodom.noise.sigma_pitch_deg),
                            _d2r(lodom.noise.sigma_yaw_deg),
                            float(lodom.noise.sigma_x_m),
                            float(lodom.noise.sigma_y_m),
                            float(lodom.noise.sigma_z_m),
                        ], dtype=float)
                        base = self.gtsam.noiseModel.Diagonal.Sigmas(sig)

                        # Optional robust kernel
                        model = base
                        if getattr(lodom, "robust", None) is not None and lodom.robust.enabled:
                            k = str(lodom.robust.kernel).lower()
                            if k == "huber":
                                ker = self.gtsam.noiseModel.mEstimator.Huber(float(lodom.robust.param))
                                model = self.gtsam.noiseModel.Robust.Create(ker, base)
                            elif k == "cauchy":
                                ker = self.gtsam.noiseModel.mEstimator.Cauchy(float(lodom.robust.param))
                                model = self.gtsam.noiseModel.Robust.Create(ker, base)

                        self.graph.add(self.gtsam.BetweenFactorPose3(self._X(self.i), self._X(j), delta_pose, model))
                        log.info("LiDAR odom accepted: i=%d j=%d trans=%.3f rot=%.3f", self.i, j, summary["trans_m"], summary["rot_deg"])
                    else:
                        log.warning("LiDAR odom rejected: i=%d j=%d reasons=%s summary=%s", self.i, j, reasons, summary)

                    # Consume the delta (do not reuse)
                    self.lidar_pending_delta = None

        self.isam.update(self.graph, self.initial)
        self.graph.resize(0)
        self.initial.clear()

        # advance, reset preintegration with newest bias
        self.i = j
        pose_j, vel_j, bias_j = self._current_estimate()
        self.pim.resetIntegrationAndSetBias(bias_j)
        self._pim_bias = bias_j
        self.last_keyframe_t = gnss.t

        return self._pose_to_state(gnss.t, pose_j, vel_j)
