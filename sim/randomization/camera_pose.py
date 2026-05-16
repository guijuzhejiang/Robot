"""Domain randomization: front-camera pose jitter.

Adjusts model.cam_pos / model.cam_quat for the "front" camera.
"""
from __future__ import annotations

import mujoco
import numpy as np


# Captured at first call from the MJCF so DR jitters around the actual
# scene-designer default instead of a stale hardcoded value, and so we don't
# accumulate yaw drift across resets.
_DEFAULT_FRONT_POS: np.ndarray | None = None
_DEFAULT_FRONT_QUAT: np.ndarray | None = None


def randomize_front_camera(
    model: mujoco.MjModel,
    rng: np.random.Generator,
    *,
    xy_jitter: float = 0.02,
    z_jitter: float = 0.02,
    angle_jitter_deg: float = 4.0,
) -> None:
    global _DEFAULT_FRONT_POS, _DEFAULT_FRONT_QUAT
    try:
        cam_id = model.camera("front").id
    except KeyError:
        return

    if _DEFAULT_FRONT_POS is None:
        _DEFAULT_FRONT_POS = model.cam_pos[cam_id].copy()
        _DEFAULT_FRONT_QUAT = model.cam_quat[cam_id].copy()

    dx, dy = rng.uniform(-xy_jitter, xy_jitter, size=2)
    dz = rng.uniform(-z_jitter, z_jitter)
    model.cam_pos[cam_id] = _DEFAULT_FRONT_POS + np.array([dx, dy, dz])

    # Small yaw perturbation around the ORIGINAL MJCF orientation (not the
    # previous-reset orientation — that would accumulate drift).
    yaw = np.deg2rad(rng.uniform(-angle_jitter_deg, angle_jitter_deg))
    dq = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
    model.cam_quat[cam_id] = _quat_mul(dq, _DEFAULT_FRONT_QUAT)


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
