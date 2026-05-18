"""PickPlace env (rebuilt on pick-101 SO-101 model).

Task simplification (vs. legacy 2-cube version): a SINGLE 3 cm red cube is
the pick target. A white cylindrical plate (freejoint, can be nudged) is the
place target. The class/file name is color-agnostic (`PickPlace`) so it
remains valid if the cube color changes in future variants.

Why this rewrite — the menagerie SO-101 grasp tuning capped at ~26 % success
because that model lacks finger pads, graspframe, and per-fingertip sites,
and our keyframe started the wrist OUT of the top-down configuration.
pick-101's `so101_new_calib.xml` ships all three plus a calibrated TCP, and
its `test_topdown_pick.py` reaches ~100 % grasp success by:

  1. Initialising the wrist already pointing straight down
     (wrist_flex = wrist_roll = pi/2).
  2. Locking joints [3, 4] and letting DLS IK adjust only base/shoulder/
     elbow for the XYZ target — so the wrist never drifts.
  3. Using ABSOLUTE gripper commands so contact-detection tightening can
     ramp ctrl[5] smoothly instead of saturating in one tick.

This env wires those three behaviours up as the DEFAULT runtime mode for
the scripted oracle. RL training can still override `use_dls_ik`,
`locked_joints`, and `gripper_action_mode` if needed.

Success (all 3 must hold):
  1. red_cube xy distance to plate center < (plate_radius - cube_half)
  2. red_cube bottom z within +/-1.5 cm of plate top surface
  3. gripper opened (ctrl[5] above threshold)

Failure modes recorded in info["failure_mode"]:
  - "grasp_fail":   cube never moved
  - "lift_drop":    cube moved but never lifted clear of table
  - "place_miss":   cube moved but not over plate
  - "plate_off":    plate displaced > PLATE_MOVE_TOL (knocked away)
  - "joint_limit":  any joint pinned at ctrlrange edge for >30 steps
  - "timeout":      max_episode_steps reached without success
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sim.envs.base import BaseSoArmEnv

REPO_ROOT = Path(__file__).resolve().parents[2]


class PickPlaceEnv(BaseSoArmEnv):
    SCENE_PATH = str(REPO_ROOT / "assets" / "scenes" / "pick_place.xml")
    DEFAULT_CAMERA = "front"

    # Geometry constants (must match the MJCF).
    CUBE_HALF = 0.015                       # 3 cm cube
    PLATE_RADIUS = 0.06
    PLATE_HALF_HEIGHT = 0.002
    PLATE_TOP_Z = 2 * PLATE_HALF_HEIGHT     # plate bottom at z=0, top at z=0.004

    # Success thresholds
    PLACE_Z_TOL = 0.015
    PLATE_MOVE_TOL = 0.05                   # plate may shift slightly under impact
    GRIPPER_OPEN_THRESHOLD = 0.3            # ctrl[5] above this counts as "open"

    # Reset randomisation regions (xy in workspace frame, table z=0).
    #
    # Wider workspace, enabled by per-episode wrist_roll alignment (see
    # `_post_reset`): instead of locking wrist_roll at pi/2 in WORLD frame,
    # we set wrist_roll = pi/2 - atan2(cube.y, cube.x) at reset so the
    # finger-spread axis returns to world +y once shoulder_pan swings the
    # arm to face the cube. This makes the locked-wrist DLS-IK reach
    # the full ~16 cm wide / 4 cm deep band reliably instead of just the
    # narrow centre. Plate sits on the +x side so transport sweeps cube
    # away from the arm base.
    CUBE_X_RANGE = (0.16, 0.22)
    CUBE_Y_RANGE = (-0.08, 0.08)
    PLATE_X_RANGE = (0.24, 0.30)
    PLATE_Y_RANGE = (-0.06, 0.06)
    CUBE_PLATE_MIN_SEPARATION = 0.12        # keep cube outside plate disc

    # After success is FIRST detected, keep stepping this many control ticks
    # so the recorded video shows the cube settling and the gripper retract.
    SUCCESS_HOLD_STEPS = 40

    def __init__(self, *, randomize_domain: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.randomize_domain = randomize_domain
        self._red_init_xy = np.zeros(2)
        self._plate_init_xy = np.zeros(2)
        self._red_max_z = 0.0
        self._joint_limit_streak = 0
        self._success_streak = 0
        # wrist_roll value chosen at each reset so the locked-wrist gripper's
        # finger-spread axis lands along world +y once shoulder_pan has
        # swung to face the cube. Read by the scripted oracle so it can
        # apply FINGER_WIDTH_OFFSET along the right world axis.
        self.wrist_roll_alignment: float = float(np.pi / 2)

        # Pick-101 runtime defaults for the scripted oracle.
        self.use_dls_ik = True
        self.locked_joints = [3, 4]
        self.gripper_action_mode = "absolute"

        # qpos addresses for free joints (3 pos + 4 quat each).
        self._red_qadr = self.model.joint("red_cube_freejoint").qposadr[0]
        self._plate_qadr = self.model.joint("plate_freejoint").qposadr[0]

    # ---------------- observations ----------------
    def _build_observation_space(self):
        import gymnasium as gym
        spaces = dict(super()._build_observation_space().spaces)
        spaces["red_cube_pos"] = self._box((3,))
        spaces["plate_pos"] = self._box((3,))
        spaces["gripper_qpos"] = self._box((1,))
        return gym.spaces.Dict(spaces)

    @staticmethod
    def _box(shape):
        import gymnasium as gym
        return gym.spaces.Box(-np.inf, np.inf, shape, np.float32)

    def _compute_obs(self):
        obs = super()._compute_obs()
        obs["red_cube_pos"] = self.body_pos("red_cube").astype(np.float32)
        obs["plate_pos"] = self.body_pos("plate").astype(np.float32)
        obs["gripper_qpos"] = np.array(
            [self.data.qpos[self.model.joint("gripper").qposadr[0]]],
            dtype=np.float32,
        )
        return obs

    # ---------------- reset / randomisation ----------------
    def _post_reset(self, rng: np.random.Generator) -> None:
        # Sample cube + plate such that cube is outside the plate disc.
        for _ in range(50):
            cube_xy = self._sample_xy(self.CUBE_X_RANGE, self.CUBE_Y_RANGE, rng)
            plate_xy = self._sample_xy(self.PLATE_X_RANGE, self.PLATE_Y_RANGE, rng)
            if np.linalg.norm(cube_xy - plate_xy) >= self.CUBE_PLATE_MIN_SEPARATION:
                break

        # Cube yaw quantized to {0, pi/2, pi, 3pi/2}: 3 cm cube is 4-fold
        # symmetric about z, so this still looks "randomly rotated" to the
        # camera but the gripper's parallel jaws always see a flat face
        # (no 45-deg diagonal seat that would slide out of the grasp).
        cube_yaw = float(rng.choice([0.0, np.pi / 2, np.pi, 3 * np.pi / 2]))
        self._write_free_joint(
            self._red_qadr, cube_xy, z=self.CUBE_HALF, yaw=cube_yaw,
        )
        self._write_free_joint(
            self._plate_qadr, plate_xy, z=self.PLATE_HALF_HEIGHT,
            yaw=rng.uniform(-np.pi, np.pi),
        )

        # Wrist-roll alignment: with wrist_flex pinned at pi/2, wrist_roll
        # rotates the gripper about world +z. The finger-spread axis is
        # world +y when shoulder_pan + wrist_roll = pi/2. Once the IK
        # has swung shoulder_pan to atan2(cube.y, cube.x) (so the arm
        # points at the cube), we want wrist_roll = pi/2 - that angle
        # — and we set it BEFORE the episode starts so the keyframe-
        # initialised gripper already has the right roll for this cube.
        # Clamped to the wrist_roll ctrl range so cubes at extreme
        # angles do not push the actuator into saturation.
        alpha = float(np.arctan2(cube_xy[1], cube_xy[0]))
        wrist_roll = float(np.pi / 2 - alpha)
        wrist_roll_lo, wrist_roll_hi = self.ctrl_limits[4]
        wrist_roll = float(np.clip(wrist_roll, wrist_roll_lo, wrist_roll_hi))
        self.wrist_roll_alignment = wrist_roll
        wrist_roll_qadr = self.model.joint("wrist_roll").qposadr[0]
        self.data.qpos[wrist_roll_qadr] = wrist_roll
        self.data.ctrl[4] = wrist_roll

        self._red_init_xy = cube_xy.copy()
        self._plate_init_xy = plate_xy.copy()
        self._red_max_z = self.CUBE_HALF
        self._joint_limit_streak = 0
        self._success_streak = 0

        if self.randomize_domain:
            self._apply_domain_randomization(rng)

    def _apply_domain_randomization(self, rng: np.random.Generator) -> None:
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
        self.data.qpos[qadr + 3] = np.cos(yaw / 2.0)
        self.data.qpos[qadr + 4] = 0.0
        self.data.qpos[qadr + 5] = 0.0
        self.data.qpos[qadr + 6] = np.sin(yaw / 2.0)

    # ---------------- termination / success ----------------
    def _check_done(self) -> tuple[bool, bool, dict]:
        # Track max lift height seen so far for lift_drop diagnosis.
        red_z = self.body_pos("red_cube")[2]
        if red_z > self._red_max_z:
            self._red_max_z = red_z

        success, failure_mode = self.evaluate_success()
        info = self._info()
        info["is_success"] = success
        info["failure_mode"] = failure_mode

        ctrl = self.data.ctrl[:5]
        lo, hi = self.ctrl_limits[:5, 0], self.ctrl_limits[:5, 1]
        at_limit = np.any((ctrl < lo + 1e-3) | (ctrl > hi - 1e-3))
        self._joint_limit_streak = self._joint_limit_streak + 1 if at_limit else 0

        if success:
            self._success_streak += 1
        else:
            self._success_streak = 0

        terminated = self._success_streak >= self.SUCCESS_HOLD_STEPS
        truncated = False
        if self._joint_limit_streak >= 30:
            terminated = True
            info["failure_mode"] = "joint_limit"
        return terminated, truncated, info

    def evaluate_success(self) -> tuple[bool, str | None]:
        red = self.body_pos("red_cube")
        plate = self.body_pos("plate")

        in_plate_xy = np.linalg.norm(red[:2] - plate[:2]) < (
            self.PLATE_RADIUS - self.CUBE_HALF
        )
        red_bottom_z = red[2] - self.CUBE_HALF
        on_plate_z = abs(red_bottom_z - self.PLATE_TOP_Z) < self.PLACE_Z_TOL
        gripper_open = self.data.ctrl[5] > self.GRIPPER_OPEN_THRESHOLD
        plate_moved = (
            np.linalg.norm(plate[:2] - self._plate_init_xy) > self.PLATE_MOVE_TOL
        )

        if in_plate_xy and on_plate_z and gripper_open and not plate_moved:
            return True, None

        # Failure-mode triage (only relevant pre-success).
        red_moved = np.linalg.norm(red[:2] - self._red_init_xy) > 0.02
        if plate_moved:
            return False, "plate_off"
        if not red_moved and self._step_count >= 10:
            return False, "grasp_fail"
        if self._red_max_z < self.CUBE_HALF + 0.04 and red_moved:
            return False, "lift_drop"
        if red_moved and not in_plate_xy:
            return False, "place_miss"
        return False, None
