"""PickPlaceBlue env: pick the blue cube, place it on the plate.

Scene layout (workspace = table top at z=0):
  - Left half  (x ∈ [0.05, 0.22]):  red + blue cubes (random pos, min spacing)
  - Right half (x ∈ [0.24, 0.32]):  plate

Success (all 4 must hold):
  1. blue_cube xy distance to plate center < plate radius
  2. blue_cube bottom z within ±1 cm of plate top surface
  3. red_cube total displacement < 2 cm (not knocked away)
  4. gripper opened in the final state (ctrl[5] > threshold)

Failure modes are recorded in info["failure_mode"]:
  - "color_confusion": blue still near start AND red moved (likely grabbed red)
  - "transport_collision_red": red moved > 2 cm
  - "place_miss": blue moved, but not on plate
  - "lift_drop": blue z stayed near 0 (never lifted)
  - "joint_limit": any joint hit ctrlrange edge for >30 consecutive steps
  - "timeout": max_episode_steps reached without success
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sim.envs.base import BaseSoArmEnv

REPO_ROOT = Path(__file__).resolve().parents[2]


class PickPlaceBlueEnv(BaseSoArmEnv):
    SCENE_PATH = str(REPO_ROOT / "assets" / "scenes" / "pick_place_blue.xml")
    DEFAULT_CAMERA = "front"

    # Geometry constants (must match MJCF)
    CUBE_HALF = 0.02
    PLATE_RADIUS = 0.06
    PLATE_TOP_Z = 0.01   # plate sits with bottom at z=0, half-height 0.005, top at z=0.01

    # Success thresholds
    DISTRACTOR_MOVE_TOL = 0.02
    PLACE_Z_TOL = 0.015
    GRIPPER_OPEN_THRESHOLD = 0.3  # ctrl[5] above this counts as "open enough"

    # Reset randomization regions (xy in workspace frame)
    CUBE_X_RANGE = (0.10, 0.22)
    CUBE_Y_RANGE = (-0.12, 0.12)
    CUBE_MIN_SEPARATION = 0.08
    PLATE_X_RANGE = (0.24, 0.30)
    PLATE_Y_RANGE = (-0.06, 0.06)

    def __init__(self, *, randomize_domain: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.randomize_domain = randomize_domain
        self._red_init_xy = np.zeros(2)
        self._blue_init_xy = np.zeros(2)
        self._joint_limit_streak = 0
        # qpos addresses for free joints (3 pos + 4 quat each)
        self._red_qadr = self.model.joint("red_cube_freejoint").qposadr[0]
        self._blue_qadr = self.model.joint("blue_cube_freejoint").qposadr[0]
        self._plate_qadr = self.model.joint("plate_freejoint").qposadr[0]

    # ---------------- observations ----------------
    def _build_observation_space(self):
        spaces = dict(super()._build_observation_space().spaces)
        spaces["red_cube_pos"] = self._box((3,))
        spaces["blue_cube_pos"] = self._box((3,))
        spaces["plate_pos"] = self._box((3,))
        spaces["gripper_qpos"] = self._box((1,))
        import gymnasium as gym
        return gym.spaces.Dict(spaces)

    @staticmethod
    def _box(shape):
        import gymnasium as gym
        return gym.spaces.Box(-np.inf, np.inf, shape, np.float32)

    def _compute_obs(self):
        obs = super()._compute_obs()
        obs["red_cube_pos"] = self.body_pos("red_cube").astype(np.float32)
        obs["blue_cube_pos"] = self.body_pos("blue_cube").astype(np.float32)
        obs["plate_pos"] = self.body_pos("plate").astype(np.float32)
        obs["gripper_qpos"] = np.array(
            [self.data.qpos[self.model.joint("gripper").qposadr[0]]], dtype=np.float32
        )
        return obs

    # ---------------- reset / randomization ----------------
    def _post_reset(self, rng: np.random.Generator) -> None:
        # Sample two non-overlapping cube xy positions in left half
        for _ in range(50):
            red_xy = self._sample_xy(self.CUBE_X_RANGE, self.CUBE_Y_RANGE, rng)
            blue_xy = self._sample_xy(self.CUBE_X_RANGE, self.CUBE_Y_RANGE, rng)
            if np.linalg.norm(red_xy - blue_xy) >= self.CUBE_MIN_SEPARATION:
                break
        # Plate position (right half)
        plate_xy = self._sample_xy(self.PLATE_X_RANGE, self.PLATE_Y_RANGE, rng)

        self._write_free_joint(self._red_qadr,   red_xy,  z=self.CUBE_HALF, yaw=rng.uniform(-np.pi, np.pi))
        self._write_free_joint(self._blue_qadr,  blue_xy, z=self.CUBE_HALF, yaw=rng.uniform(-np.pi, np.pi))
        self._write_free_joint(self._plate_qadr, plate_xy, z=0.005, yaw=rng.uniform(-np.pi, np.pi))

        self._red_init_xy = red_xy.copy()
        self._blue_init_xy = blue_xy.copy()
        self._joint_limit_streak = 0

        if self.randomize_domain:
            self._apply_domain_randomization(rng)

    def _apply_domain_randomization(self, rng: np.random.Generator) -> None:
        """Light DR hooks; subclasses or external modules can extend.
        Phase 1 keeps this minimal — full DR modules live in sim/randomization/.
        """
        from sim.randomization.lighting import randomize_lights
        from sim.randomization.camera_pose import randomize_front_camera

        randomize_lights(self.model, rng)
        randomize_front_camera(self.model, rng)

    @staticmethod
    def _sample_xy(x_range, y_range, rng):
        return np.array([rng.uniform(*x_range), rng.uniform(*y_range)])

    def _write_free_joint(self, qadr: int, xy: np.ndarray, *, z: float, yaw: float):
        self.data.qpos[qadr + 0] = xy[0]
        self.data.qpos[qadr + 1] = xy[1]
        self.data.qpos[qadr + 2] = z
        # quat (w, x, y, z) for yaw rotation about z
        self.data.qpos[qadr + 3] = np.cos(yaw / 2.0)
        self.data.qpos[qadr + 4] = 0.0
        self.data.qpos[qadr + 5] = 0.0
        self.data.qpos[qadr + 6] = np.sin(yaw / 2.0)

    # ---------------- termination / success ----------------
    def _check_done(self) -> tuple[bool, bool, dict]:
        success, failure_mode = self.evaluate_success()
        info = self._info()
        info["is_success"] = success
        info["failure_mode"] = failure_mode

        # joint-limit safety check
        ctrl = self.data.ctrl[:5]
        lo, hi = self.ctrl_limits[:5, 0], self.ctrl_limits[:5, 1]
        at_limit = np.any((ctrl < lo + 1e-3) | (ctrl > hi - 1e-3))
        self._joint_limit_streak = self._joint_limit_streak + 1 if at_limit else 0

        terminated = success
        truncated = False
        if self._joint_limit_streak >= 30:
            terminated = True
            info["failure_mode"] = "joint_limit"
        return terminated, truncated, info

    def evaluate_success(self) -> tuple[bool, str | None]:
        blue = self.body_pos("blue_cube")
        red = self.body_pos("red_cube")
        plate = self.body_pos("plate")

        # 1) blue xy in plate radius?
        in_plate_xy = np.linalg.norm(blue[:2] - plate[:2]) < self.PLATE_RADIUS - self.CUBE_HALF
        # 2) blue at plate-surface height?
        blue_bottom_z = blue[2] - self.CUBE_HALF
        on_plate_z = abs(blue_bottom_z - self.PLATE_TOP_Z) < self.PLACE_Z_TOL
        # 3) red untouched?
        red_moved = np.linalg.norm(red[:2] - self._red_init_xy) > self.DISTRACTOR_MOVE_TOL
        # 4) gripper open?
        gripper_open = self.data.ctrl[5] > self.GRIPPER_OPEN_THRESHOLD

        if in_plate_xy and on_plate_z and not red_moved and gripper_open:
            return True, None

        # Failure-mode triage (only when not yet succeeded)
        blue_moved = np.linalg.norm(blue[:2] - self._blue_init_xy) > 0.02
        if red_moved and not blue_moved:
            return False, "color_confusion"
        if red_moved:
            return False, "transport_collision_red"
        if not blue_moved and self._step_count >= 10:
            return False, "grasp_fail"
        if blue_moved and not in_plate_xy:
            return False, "place_miss"
        if blue[2] < self.CUBE_HALF + 0.02 and blue_moved:
            return False, "lift_drop"
        return False, None
