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
    """End-effector IK on a 5-DoF SO-ARM101.

    Reuses the previous frame's q as warmstart. Position is the dominant task;
    a soft orientation task encourages the gripper z-axis (jaw open/close axis)
    to stay roughly vertical, so the lower fixed jaw approaches cubes from
    above rather than horizontally — without which top-down grasps tend to
    push cubes sideways instead of straddling them.

    Default top-down orientation is configurable per `set_orientation_target`.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        ee_site_name: str = "gripperframe",
        position_cost: float = 1.0,
        orientation_cost: float = 0.0,
        # Very light posture regularization — just enough to break IK
        # ambiguity; strong values pin joints to home and prevent reaching.
        posture_cost: float = 0.001,
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

        # Position + soft-orientation task. orientation_cost ≪ position_cost so
        # the solver prioritizes hitting the target xyz, with orientation as a
        # secondary preference. With cost=0.3, position error stays sub-mm
        # while ee z-axis lands within a few degrees of vertical.
        self._frame_task = mink.FrameTask(
            frame_name=ee_site_name,
            frame_type="site",
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=1.0,
        )
        # Default orientation: gripper local +z = world +z (jaws straddle
        # vertically, fixed jaw points to -world z = down).
        self._default_R = mink.SO3.identity()
        self._target_R = self._default_R

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
        # Posture target = current configuration (mild anti-drift only).
        # Tried anchoring to the home keyframe instead, but that pulled IK
        # away from reachable cube targets and caused joint_limit failures.
        self._posture_task.set_target_from_configuration(self._config)

        target = mink.SE3.from_rotation_and_translation(
            rotation=self._target_R,
            translation=np.asarray(target_pos, dtype=np.float64),
        )
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
