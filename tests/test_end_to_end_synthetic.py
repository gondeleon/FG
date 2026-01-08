from __future__ import annotations

from pathlib import Path

import pytest

from slamboat.config import AppConfig, ColumnSpec, DelimitedFileSpec, GnssSpec, ImuSpec, TimeSpec
from slamboat.pipeline import run_pipeline


def _gtsam_available() -> bool:
    try:
        import gtsam  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _gtsam_available(), reason="gtsam not installed")
def test_pipeline_end_to_end_synthetic(tmp_path: Path) -> None:
    # Synthetic IMU: stationary (acc measures gravity-ish, but we don't enforce it here)
    imu_p = tmp_path / "imu.tsv"
    imu_rows = []
    t0 = 1000.0
    dt = 0.02
    for k in range(200):
        t = t0 + k * dt
        imu_rows.append(f"{t}\t0\t0\t0\t0\t0\t0\n")
    imu_p.write_text("".join(imu_rows), encoding="utf-8")

    # Synthetic GNSS: small motion in lat/lon (very small, local approximation)
    gnss_p = tmp_path / "gnss.tsv"
    gnss_rows = []
    for k in range(5):
        t = t0 + k * 1.0
        # lat/lon with hemispheres and heading
        lat = 36.0 + 1e-6 * k
        lon = 129.0 + 1e-6 * k
        gnss_rows.append(f"{t}\t0\t{lat}\tN\t{lon}\tE\t0\t1\t10\t0.8\t0\n")
    gnss_p.write_text("".join(gnss_rows), encoding="utf-8")

    # Minimal extrinsics: hub is ahrs, imu and gps coincide for the test
    extr_p = tmp_path / "extrinsics.json"
    extr_p.write_text(
        """{
          "convention": {
            "parent_frame": "ahrs",
            "transform": "T_parent_child",
            "quaternion_order": "xyzw",
            "translation_expressed_in": "parent"
          },
          "frames": {
            "gps":  { "quaternion": [0,0,0,1], "translation": [0,0,0] },
            "imu":  { "quaternion": [0,0,0,1], "translation": [0,0,0] }
          }
        }""",
        encoding="utf-8",
    )

    cfg = AppConfig.model_validate(
        {
            "extrinsics_path": str(extr_p),
            "frames": {"state_frame": "imu", "imu_frame": "imu", "gnss_frame": "gps", "ahrs_frame": "ahrs"},
            "sensors": {
                "gnss0": {
                    "kind": "gnss",
                    "frame_id": "gps",
                    "ref_lat_deg": 36.0,
                    "ref_lon_deg": 129.0,
                    "ref_alt_m": 0.0,
                    "use_geoid_height_as_z": False,
                    "file": {
                        "path": str(gnss_p),
                        "delimiter": "\t",
                        "has_header": False,
                        "time": {"col": 0},
                        "time_format": {"kind": "unix_seconds"},
                        "columns": {
                            "gps_time_s": {"col": 1},
                            "lat_deg": {"col": 2},
                            "lat_hemi": {"col": 3},
                            "lon_deg": {"col": 4},
                            "lon_hemi": {"col": 5},
                            "heading_deg": {"col": 6, "unit": "deg"},
                            "quality": {"col": 7},
                            "n_sats": {"col": 8},
                            "hdop": {"col": 9},
                            "geoid_height_m": {"col": 10},
                        },
                    },
                },
                "imu0": {
                    "kind": "imu",
                    "frame_id": "imu",
                    "file": {
                        "path": str(imu_p),
                        "delimiter": "\t",
                        "has_header": False,
                        "time": {"col": 0},
                        "time_format": {"kind": "unix_seconds"},
                        "columns": {
                            "wx": {"col": 1, "unit": "rad/s"},
                            "wy": {"col": 2, "unit": "rad/s"},
                            "wz": {"col": 3, "unit": "rad/s"},
                            "ax": {"col": 4, "unit": "m/s^2"},
                            "ay": {"col": 5, "unit": "m/s^2"},
                            "az": {"col": 6, "unit": "m/s^2"},
                        },
                    },
                },
            },
            "replay": {"realtime": False, "speedup": 100.0, "max_steps": None},
            "outputs": {"out_dir": str(tmp_path / "out"), "state_csv": "state.csv"},
        }
    )

    out_csv = run_pipeline(cfg)
    assert out_csv.exists()
    content = out_csv.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) >= 2  # header + at least one state
