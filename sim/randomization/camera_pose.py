"""Domain randomization: front-camera pose jitter.

Adjusts model.cam_pos / model.cam_quat for the "front" camera.
"""
from __future__ import annotations

import mujoco
import numpy as np


_DEFAULT_FRONT_POS = np.array([0.35, 0.0, 0.30])


def randomize_front_camera(
    model: mujoco.MjModel,
    rng: np.random.Generator,
    *,
    xy_jitter: float = 0.03,
    z_jitter: float = 0.02,
    angle_jitter_deg: float = 5.0,
) -> None:
    try:
        cam_id = model.camera("front").id
    except KeyError:
        return

    dx, dy = rng.uniform(-xy_jitter, xy_jitter, size=2)
    dz = rng.uniform(-z_jitter, z_jitter)
    model.cam_pos[cam_id] = _DEFAULT_FRONT_POS + np.array([dx, dy, dz])

    # Tiny rotation around vertical axis (yaw)
    yaw = np.deg2rad(rng.uniform(-angle_jitter_deg, angle_jitter_deg))
    # Compose with the original orientation: original xyaxes "0 -1 0  -0.5 0 0.866"
    # gives a quat we'll approximate as small yaw perturbation of identity.
    base_quat = model.cam_quat[cam_id].copy()
    dq = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
    model.cam_quat[cam_id] = _quat_mul(dq, base_quat)


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
