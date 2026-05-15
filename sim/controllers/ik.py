"""Differential IK for SO-ARM101 using mink.

We only solve for the 5 arm joints (shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll). The gripper joint is excluded from IK and controlled
separately.
"""
from __future__ import annotations

import mink
import mujoco
import numpy as np


SO101_ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)


class EeIkController:
    """Position-only end-effector IK on a 5-DoF SO-ARM101.

    Reuses the previous frame's q as warmstart. Orientation is left free
    (only position task) because the SO101 is 5-DoF — 6-DoF position+orientation
    targets are generally not reachable.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        ee_site_name: str = "gripperframe",
        position_cost: float = 1.0,
        posture_cost: float = 1e-3,
        damping: float = 1e-3,
        max_iters: int = 5,
        dt: float = 0.02,
    ):
        self.model = model
        self.ee_site_name = ee_site_name
        self.dt = dt
        self.max_iters = max_iters
        self.damping = damping

        self._config = mink.Configuration(model)

        # Position-only frame task (orientation_cost=0)
        self._frame_task = mink.FrameTask(
            frame_name=ee_site_name,
            frame_type="site",
            position_cost=position_cost,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        # Light posture regularization to keep solutions stable
        self._posture_task = mink.PostureTask(model=model, cost=posture_cost)
        self._tasks = [self._frame_task, self._posture_task]
        self._limits = [mink.ConfigurationLimit(model=model)]

        # Index of the 5 arm joints in qpos
        self._arm_qids = np.array(
            [model.joint(name).qposadr[0] for name in SO101_ARM_JOINTS]
        )

    def solve(self, data: mujoco.MjData, target_pos: np.ndarray) -> np.ndarray:
        """Return target qpos for the 5 arm joints, given world-frame ee target."""
        # Sync mink config with current sim state (warmstart)
        self._config.update(data.qpos.copy())
        self._posture_task.set_target_from_configuration(self._config)

        # Build SE3 target (use current orientation as identity-equivalent)
        import mink as _mink  # local alias

        target = _mink.SE3.from_translation(np.asarray(target_pos, dtype=np.float64))
        self._frame_task.set_target(target)

        for _ in range(self.max_iters):
            dq = mink.solve_ik(
                self._config,
                self._tasks,
                self.dt,
                solver="quadprog",
                damping=self.damping,
                limits=self._limits,
            )
            self._config.integrate_inplace(dq, self.dt)

        full_q = self._config.q
        return np.array([full_q[i] for i in self._arm_qids], dtype=np.float64)
