"""Extrinsics loader and transform utilities.

Assumptions for this project (explicit in the JSON):
- A single hub/root frame (e.g., AHRS) to which all sensor frames are related
- Each entry provides a rigid transform T_parent_child

Convention used here:
  T(a, b) returns a transform that maps coordinates in frame b into frame a.
  (i.e., p_a = R_ab * p_b + t_ab)

With a hub `H` and stored transforms T(H, S) for each sensor S, we can compute:
  T(A, B) = inv(T(H, A)) ∘ T(H, B)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Transform3:
    """Rigid transform in SE(3): p_parent = R * p_child + t."""
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)

    def inverse(self) -> "Transform3":
        Rinv = self.R.T
        tinv = -Rinv @ self.t
        return Transform3(R=Rinv, t=tinv)

    def compose(self, other: "Transform3") -> "Transform3":
        # self ∘ other : p_self = R1 (R2 p + t2) + t1
        R = self.R @ other.R
        t = self.R @ other.t + self.t
        return Transform3(R=R, t=t)

    def to_gtsam_pose3(self):
        """Convert to gtsam.Pose3 (requires gtsam at runtime)."""
        import gtsam  # type: ignore

        Rot = gtsam.Rot3(self.R)
        P = gtsam.Point3(float(self.t[0]), float(self.t[1]), float(self.t[2]))
        return gtsam.Pose3(Rot, P)


def _quat_xyzw_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion (x,y,z,w) to rotation matrix."""
    # Normalize
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n <= 0:
        raise ValueError("Invalid quaternion norm")
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n

    # Standard formula
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz

    R = np.array(
        [
            [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
            [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
            [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)],
        ],
        dtype=float,
    )
    return R


class ExtrinsicsDB:
    """Extrinsics database with composition utilities."""

    def __init__(
        self,
        parent_frame: str,
        transforms_parent_child: Dict[str, Transform3],
    ):
        self.parent_frame = parent_frame
        self._T_parent_child = transforms_parent_child

    def frames(self) -> Tuple[str, ...]:
        return (self.parent_frame, *tuple(self._T_parent_child.keys()))

    def T(self, a: str, b: str) -> Transform3:
        """Return transform T(a,b): maps coordinates in frame b into frame a."""
        H = self.parent_frame

        if a == b:
            return Transform3(R=np.eye(3), t=np.zeros(3))

        if a == H:
            return self._T_parent_child[b]
        if b == H:
            return self._T_parent_child[a].inverse()

        # General case with hub
        return self._T_parent_child[a].inverse().compose(self._T_parent_child[b])

    def pose3(self, a: str, b: str):
        return self.T(a, b).to_gtsam_pose3()


def load_extrinsics(path: str | Path) -> ExtrinsicsDB:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))

    conv = data.get("convention", {})
    parent_frame = conv.get("parent_frame", "ahrs")
    quat_order = conv.get("quaternion_order", "xyzw")
    t_expressed_in = conv.get("translation_expressed_in", "parent")
    transform_dir = conv.get("transform", "T_parent_child")

    if quat_order != "xyzw":
        raise ValueError("This loader currently expects quaternion_order=xyzw in extrinsics.json")
    if t_expressed_in != "parent":
        raise ValueError("This loader currently expects translation_expressed_in=parent")
    if transform_dir != "T_parent_child":
        raise ValueError("This loader currently expects transform=T_parent_child")

    frames = data.get("frames", data)  # backward compatible

    # Resolve "same_as" aliases
    resolved: Dict[str, Dict] = {}
    for name, item in frames.items():
        if name == "convention":
            continue
        if isinstance(item, dict) and "same_as" in item:
            resolved[name] = {"_alias": item["same_as"]}
        else:
            resolved[name] = item

    # second pass: expand aliases
    expanded: Dict[str, Dict] = {}
    for name, item in resolved.items():
        if "_alias" in item:
            alias = item["_alias"]
            if alias not in resolved or "_alias" in resolved[alias]:
                raise ValueError(f"Invalid same_as alias: {name} -> {alias}")
            expanded[name] = resolved[alias]
        else:
            expanded[name] = item

    transforms: Dict[str, Transform3] = {}
    for name, item in expanded.items():
        if name == parent_frame:
            # Parent frame itself can appear; skip
            continue
        q = item["quaternion"]
        t = item["translation"]
        R = _quat_xyzw_to_R(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        tt = np.array([float(t[0]), float(t[1]), float(t[2])], dtype=float)
        transforms[name] = Transform3(R=R, t=tt)

    return ExtrinsicsDB(parent_frame=parent_frame, transforms_parent_child=transforms)
