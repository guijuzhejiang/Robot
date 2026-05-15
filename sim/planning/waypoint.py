"""Waypoint motion planner for SO-ARM101.

Given a start joint config and a target ee position, this module generates a
joint-space trajectory by:
  1. Interpolating in Cartesian space between current and target (N waypoints)
  2. Solving IK at each waypoint (warmstart from previous solution)
  3. Optionally checking joint limits

Output is a list of (5-arm-joint) qpos arrays. Caller appends gripper command.
"""
from __future__ import annotations

import mujoco
import numpy as np

from sim.controllers.ik import EeIkController, SO101_ARM_JOINTS


def plan_cartesian_path(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: np.ndarray,
    *,
    n_waypoints: int = 20,
    ik: EeIkController | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Plan a joint-space trajectory to reach target_pos.

    Returns:
        traj: (n_waypoints, 5) array of arm-joint angles.
        warnings: list of human-readable warnings (e.g. "joint X near limit").

    Note: This is best-effort planning — IK solutions are local. For collision
    avoidance, use the env's `_check_done()` after replay rather than baking it
    here.
    """
    ik = ik or EeIkController(model)
    warnings: list[str] = []

    start_pos = data.site("gripperframe").xpos.copy()
    target_pos = np.asarray(target_pos, dtype=np.float64)

    # Linear interpolation of ee position
    alphas = np.linspace(0.0, 1.0, n_waypoints)
    traj = np.zeros((n_waypoints, 5), dtype=np.float64)

    # Snapshot current full qpos for stepping IK forward
    saved_qpos = data.qpos.copy()
    for i, a in enumerate(alphas):
        wp = (1 - a) * start_pos + a * target_pos
        q = ik.solve(data, wp)
        # write to data so next IK warmstarts from here
        for j, name in enumerate(SO101_ARM_JOINTS):
            qid = model.joint(name).qposadr[0]
            data.qpos[qid] = q[j]
        mujoco.mj_forward(model, data)
        traj[i] = q

    # Joint-limit check (last waypoint)
    for j, name in enumerate(SO101_ARM_JOINTS):
        lo, hi = model.actuator_ctrlrange[j]
        if not (lo + 1e-3 < traj[-1, j] < hi - 1e-3):
            warnings.append(f"{name} target near limit: {traj[-1, j]:.3f} ∉ ({lo:.3f}, {hi:.3f})")

    # Restore data
    data.qpos[:] = saved_qpos
    mujoco.mj_forward(model, data)

    return traj, warnings
