"""Pipeline runner (wires config -> readers -> dispatcher -> estimator)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from slamboat.config import AppConfig, AhrsSpec
from slamboat.core.estimator import OnlineEstimatorPose3ImuPreint, require_gtsam
from slamboat.core.geo import LocalENU
from slamboat.core.output import StateWriter3D
from slamboat.dispatch.dispatcher import Dispatcher
from slamboat.extrinsics import load_extrinsics
from slamboat.io.factory import build_readers, sensor_kinds

log = logging.getLogger(__name__)


def run_pipeline(cfg: AppConfig) -> Path:
    # Fail fast if GTSAM is not available.
    require_gtsam()

    # Load extrinsics (rig geometry).
    extr = load_extrinsics(cfg.extrinsics_path)

    readers = build_readers(cfg.sensors)
    kinds = sensor_kinds(cfg.sensors)

    # Use first GNSS spec as ENU reference.
    gnss_specs = [s for s in cfg.sensors.values() if s.kind == "gnss"]
    ref = gnss_specs[0]

    enu = LocalENU(
        ref_lat_deg=ref.ref_lat_deg,  # type: ignore[attr-defined]
        ref_lon_deg=ref.ref_lon_deg,  # type: ignore[attr-defined]
        ref_alt_m=getattr(ref, "ref_alt_m", 0.0),
    )

    ahrs_specs = [s for s in cfg.sensors.values() if s.kind == "ahrs"]  # type: ignore[assignment]

    est = OnlineEstimatorPose3ImuPreint(
        cfg=cfg.estimator,
        noise=cfg.estimator.noise,
        enu=enu,
        extr=extr,
        state_frame=cfg.frames.state_frame,
        gnss_frame=cfg.frames.gnss_frame,
        ahrs_frame=cfg.frames.ahrs_frame,
        ahrs_specs=ahrs_specs,  # type: ignore[arg-type]
    )

    dispatcher = Dispatcher(
        streams=readers,
        kinds=kinds,
        realtime=cfg.replay.realtime,
        speedup=cfg.replay.speedup,
        max_steps=cfg.replay.max_steps,
    )

    out_dir = Path(cfg.outputs.out_dir)
    writer = StateWriter3D(out_csv_path=out_dir / cfg.outputs.state_csv)
    writer.open()

    n_events = 0
    n_states = 0
    last_state_t: Optional[float] = None

    use_alt = bool(getattr(ref, "use_geoid_height_as_z", False))

    for ev in dispatcher.run():
        n_events += 1
        state = est.handle(ev.kind, ev.msg, use_alt=use_alt)
        if state is not None:
            if last_state_t is None or state.t != last_state_t:
                writer.write(state)
                n_states += 1
                last_state_t = state.t

    writer.close()
    log.info("Done. events=%d states=%d output=%s", n_events, n_states, writer.out_csv_path)
    return writer.out_csv_path
