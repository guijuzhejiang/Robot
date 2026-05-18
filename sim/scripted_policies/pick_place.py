"""Scripted oracle policy for PickPlace (rebuilt on pick-101 pattern).

Task: pick the single 3 cm red cube and place it on the white plate.
This policy mirrors ggand0/pick-101 `test_topdown_pick.py` for the pick
half (steps 1-4) and adds 4 place phases (transport/descend/release/retract).

Why pick-101's pattern works:
  - Initial wrist pose has wrist_flex = wrist_roll = pi/2 (set by the home
    keyframe in pick_place.xml), so the fixed jaw points straight DOWN.
  - The env's DlsIkController runs with `locked_joints = [3, 4]`, so IK only
    moves base/shoulder/elbow for XYZ — the wrist NEVER drifts.
  - Gripper actions are absolute (env.gripper_action_mode = "absolute"),
    matching pick-101's IKController mapping. This lets the GRASP phase
    do a smooth gradual close + post-contact tighten without saturating
    the actuator in one tick.

Phases:
   1. APPROACH        ee → above cube (cube_top + 0.035 m), gripper at 0.3
   2. DESCEND         ee → cube grasp height (z = 0.020 m), gripper at 0.3
   3. GRASP           gradual close (0.3 → -0.8) with contact detection;
                      after both finger pads touch the cube, tighten by 0.4
   4. LIFT            raise cube to grasp_pos.z + 0.05 m, hold grasp action
   5. TRANSPORT       move xy over plate at TRANSPORT_Z, hold grasp action
   6. PLACE_DESCEND   lower to plate_top + 0.025 m
   7. PLACE_OPEN      open gripper to 0.4, hold position so cube drops
   8. PLACE_RETRACT   lift to TRANSPORT_Z (gripper stays open)
   9. DONE

This policy is an *oracle*: reads obs["red_cube_pos"] / obs["plate_pos"]
and the live MuJoCo contact buffer directly. It is the data-generation
reference, NOT the VLA target policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np


Phase = Literal[
    "APPROACH", "DESCEND", "GRASP", "LIFT",
    "TRANSPORT", "PLACE_DESCEND", "PLACE_OPEN", "PLACE_RETRACT", "DONE",
]


@dataclass
class _Constants:
    # FINGER_WIDTH_OFFSET: gripperframe is NOT centered between the two
    # finger pads — it sits ~1 cm offset toward the static finger along the
    # finger-spread axis (world +y at wrist_roll=pi/2). To centre a cube
    # between the jaws, the IK target y must be shifted by approximately
    # -cube_half_width. Copied verbatim from pick-101 test_topdown_pick.py
    # `finger_width_offset = -0.015`. Without this offset the static fingertip
    # crashes into the cube top and DESCEND jams ~5 mm above the cube top.
    FINGER_WIDTH_OFFSET: float = -0.015
    # APPROACH_Z_OFFSET: gripperframe above cube TOP. pick-101 uses
    # cube_top + 30 mm; we mirror.
    APPROACH_Z_OFFSET: float = 0.035
    # DESCEND_Z_ABS: gripperframe target z when grasping. pick-101 uses
    # cube_center + 5 mm so the finger pads straddle the upper half of the
    # cube (best geometry for the SO-101 jaw-closing arc).
    DESCEND_Z_ABS: float = 0.020
    LIFT_Z_OFFSET: float = 0.05            # cube z + this after lifting
    # Cruise height matched to LIFT end so the arm only translates xy during
    # transport — at TRANSPORT_Z = 0.15 the elbow has to flex past its lower
    # ctrl limit on cubes that require a large wrist_roll alignment.
    TRANSPORT_Z: float = 0.10
    PLACE_OFFSET_Z: float = 0.025          # gripperframe above plate top before release

    REACH_THR: float = 0.008               # 8 mm position tolerance
    # Minimum env-steps each motion phase HOLDS the target before advancing.
    # The SO-101 sts3215 servo model is under-damped (zeta ~ 0.26), so the ee
    # overshoots its target by 3-4 mm and takes ~30-40 ms to settle. If the
    # policy advances to GRASP the moment _reached is True (typically after
    # ~15 env steps), the fingertips are still oscillating around cube
    # height — the static finger dips below the cube bottom on the
    # undershoot, sliding under the cube and launching it sideways before
    # the gripper has a chance to close. Holding at the descend target for
    # an extra 15 env steps gives the actuator time to settle.
    PHASE_HOLD_STEPS: int = 15

    GRASP_RAMP_STEPS: int = 60             # pre-contact gradual close
    GRASP_HOLD_STEPS: int = 30             # post-contact tighten
    OPEN_HOLD_STEPS: int = 25              # let cube settle on plate
    SETTLE_STEPS: int = 15                 # extra settle steps before retract
    PHASE_TIMEOUT: int = 150

    MAX_SPEED: float = 0.04                # m / env step for free motion
    DESCEND_SPEED: float = 0.02            # slower for the two descend phases

    GRIPPER_OPEN: float = 0.3              # absolute action, slightly open
    GRIPPER_CLOSED: float = -0.8           # absolute action, firm close
    GRIPPER_RELEASE: float = 0.4           # absolute action, release (> open threshold)
    TIGHTEN_AMOUNT: float = 0.4            # extra close after contact


class PickPlacePolicy:
    """Scripted oracle. Call `policy(env, obs)` each env step.

    Color-agnostic name (`PickPlacePolicy`); the underlying task is currently
    red-cube only but the policy logic doesn't depend on color.
    """

    def __init__(self, *, max_speed: float | None = None, descend_speed: float | None = None):
        self.c = _Constants()
        if max_speed is not None:
            self.c.MAX_SPEED = max_speed
        if descend_speed is not None:
            self.c.DESCEND_SPEED = descend_speed

        self.phase: Phase = "APPROACH"
        self.phase_steps = 0

        # Pick-101 starts CLOSED on XY transit then opens; in our setup the
        # arm reaches the workspace via the home keyframe (no transit), so
        # we can start at GRIPPER_OPEN.
        self.gripper: float = self.c.GRIPPER_OPEN

        # Filled during GRASP / LIFT.
        self.grasp_action: float = self.c.GRIPPER_CLOSED
        self.contact_step: int | None = None
        self.contact_action: float | None = None

        # Snapshots taken on the FIRST __call__ — pick-101's test_topdown_pick.py
        # reads `actual_cube_pos` once after the settle phase and uses that
        # value for both APPROACH and DESCEND so the ee doesn't chase a cube
        # that has been nudged sideways by a fingertip brush. Same idea here.
        self.cube_anchor: np.ndarray | None = None
        self.plate_anchor: np.ndarray | None = None
        # Anchored ee XYZ at the moment GRASP starts — held during GRASP, used
        # as the lift-from origin in LIFT.
        self.cube_grasp_pos: np.ndarray | None = None

        # Geom ids — cached on first call (env-dependent).
        self._cube_gid: int | None = None
        self._pad_gids: tuple[int, int] | None = None

    # ---------------- public ----------------
    def reset(self):
        self.__init__()

    def __call__(self, env, obs) -> np.ndarray:
        """Drive `env` for one ctrl tick.

        The policy uses the env's `ee_target_override` + `gripper_action_override`
        side-channel so the IK substep loop sees a FIXED absolute target each
        env step (and decays toward it via gain=0.5). This mirrors pick-101's
        `test_topdown_pick.py`, where `move_to_position` keeps target_pos
        constant for hundreds of mj_steps.

        Returns a zero-action 4-vector so the env's gym contract is satisfied;
        the overrides are what actually drive the actuators.
        """
        if self._cube_gid is None:
            self._cache_ids(env.model)

        # Snapshot cube / plate on the first call — pick-101 reads
        # `actual_cube_pos` once after settle and never re-reads.
        if self.cube_anchor is None:
            self.cube_anchor = obs["red_cube_pos"].copy()
            self.plate_anchor = obs["plate_pos"].copy()

        cube = self.cube_anchor
        plate = self.plate_anchor
        self.phase_steps += 1
        timed_out = self.phase_steps >= self.c.PHASE_TIMEOUT

        # Default: hold last target + gripper. Phases below override.
        target = None

        # ---------------- pick half (pick-101 mirror) ----------------
        if self.phase == "APPROACH":
            target = np.array([
                cube[0],
                cube[1] + self.c.FINGER_WIDTH_OFFSET,
                cube[2] + 0.015 + self.c.APPROACH_Z_OFFSET,
            ])
            self.gripper = self.c.GRIPPER_OPEN
            if self._reached(env, target) or timed_out:
                self._advance("DESCEND")

        elif self.phase == "DESCEND":
            target = np.array([
                cube[0],
                cube[1] + self.c.FINGER_WIDTH_OFFSET,
                self.c.DESCEND_Z_ABS,
            ])
            self.gripper = self.c.GRIPPER_OPEN
            if self._reached(env, target) or timed_out:
                # Anchor GRASP / LIFT origin to the IK TARGET (not current ee),
                # so the transition is target-continuous: ee keeps converging
                # toward the same xyz while the gripper starts closing. If we
                # snapshotted env.ee_pos() here, GRASP would yank the ee back
                # up by REACH_THR mm at the moment the fingers should be
                # straddling the cube.
                self.cube_grasp_pos = target.copy()
                self._advance("GRASP")

        elif self.phase == "GRASP":
            # Hold XY at anchored grasp position; only the gripper ramps.
            target = self.cube_grasp_pos
            grasping = self._is_grasping(env)
            if self.contact_step is None:
                t = min(self.phase_steps / self.c.GRASP_RAMP_STEPS, 1.0)
                self.gripper = self.c.GRIPPER_OPEN + (
                    self.c.GRIPPER_CLOSED - self.c.GRIPPER_OPEN
                ) * t
                if grasping:
                    self.contact_step = self.phase_steps
                    self.contact_action = self.gripper
            else:
                steps_since = self.phase_steps - self.contact_step
                tgt_act = max(self.contact_action - self.c.TIGHTEN_AMOUNT, -1.0)
                t_slow = min(steps_since / self.c.GRASP_HOLD_STEPS, 1.0)
                self.gripper = self.contact_action + (tgt_act - self.contact_action) * t_slow
                if t_slow >= 1.0:
                    self.grasp_action = self.gripper
                    self._advance("LIFT")

            # Safety advance if no contact detected within the ramp window.
            if self.contact_step is None and self.phase_steps >= self.c.GRASP_RAMP_STEPS + 10:
                self.grasp_action = self.gripper
                self._advance("LIFT")
            if timed_out and self.phase == "GRASP":
                self.grasp_action = self.gripper
                self._advance("LIFT")

        elif self.phase == "LIFT":
            assert self.cube_grasp_pos is not None
            target = self.cube_grasp_pos.copy()
            target[2] = self.cube_grasp_pos[2] + self.c.LIFT_Z_OFFSET
            self.gripper = self.grasp_action
            if self._reached(env, target) or timed_out:
                self._advance("TRANSPORT")

        # ---------------- place half ----------------
        # For TRANSPORT / PLACE_DESCEND the cube is held in the gripper, so
        # the jaw-spread axis (and therefore the cube's offset from
        # gripperframe) is rotated by (shoulder_pan_at_plate - shoulder_pan
        # _at_cube) about world z relative to PICK time. Rather than
        # predict that rotation analytically, we close the loop on the
        # live cube observation: ee_target = ee_current + (desired_cube -
        # cube_obs). When the cube is already at the desired pose the
        # delta vanishes, so this is a stable controller.
        elif self.phase == "TRANSPORT":
            # Drive ee directly to plate xy at the cruise z. We do NOT
            # close the loop on cube_obs here: at the start of TRANSPORT
            # ee is still over the cube while plate is ~10 cm away, so a
            # cube-tracked target would jump the IK by a full delta in one
            # tick and shoulder_pan/elbow_flex can saturate. PLACE_DESCEND
            # below takes care of the final xy correction.
            target = np.array([plate[0], plate[1], self.c.TRANSPORT_Z])
            self.gripper = self.grasp_action
            if self._reached(env, target) or timed_out:
                self._advance("PLACE_DESCEND")

        elif self.phase == "PLACE_DESCEND":
            cube_obs = obs["red_cube_pos"]
            ee_cur = env.ee_pos()
            desired_cube = np.array([
                plate[0], plate[1],
                env.PLATE_TOP_Z + env.CUBE_HALF + self.c.PLACE_OFFSET_Z,
            ])
            target = ee_cur + (desired_cube - cube_obs)
            self.gripper = self.grasp_action
            if (
                abs(cube_obs[2] - desired_cube[2]) < 0.01
                and self._reached_cube(env, obs, desired_cube[:2])
            ) or timed_out:
                self._advance("PLACE_OPEN")

        elif self.phase == "PLACE_OPEN":
            # Hold ee where it is — the cube falls a few mm to plate.
            target = env.ee_pos()
            self.gripper = self.c.GRIPPER_RELEASE
            if self.phase_steps >= self.c.OPEN_HOLD_STEPS + self.c.SETTLE_STEPS:
                self._advance("PLACE_RETRACT")

        elif self.phase == "PLACE_RETRACT":
            ee_cur = env.ee_pos()
            target = np.array([ee_cur[0], ee_cur[1], self.c.TRANSPORT_Z])
            self.gripper = self.c.GRIPPER_RELEASE
            if self._reached(env, target) or timed_out:
                self._advance("DONE")

        else:  # DONE — hold position, gripper open
            target = env.ee_pos()
            self.gripper = self.c.GRIPPER_RELEASE

        # Publish to env side-channel so the substep IK loop sees a fixed
        # absolute target and the gripper ctrl picks up our absolute action.
        env.ee_target_override = target
        env.gripper_action_override = self.gripper

        # Action returned for gym compliance / logging; not used by env
        # while overrides are set.
        return np.array([0.0, 0.0, 0.0, self.gripper], dtype=np.float32)

    def _reached_cube(self, env, obs, desired_cube_xy: np.ndarray) -> bool:
        """For place-side phases: phase-advance condition driven by the
        cube's xy, not the gripperframe's. The cube tracks the gripper
        through a rotated offset that depends on shoulder_pan / wrist_roll,
        so checking ee against a fixed plate target would advance early
        when the cube is still ~1-2 cm off-plate.
        """
        if self.phase_steps < self.c.PHASE_HOLD_STEPS:
            return False
        cube = obs["red_cube_pos"]
        return float(np.linalg.norm(cube[:2] - desired_cube_xy)) < self.c.REACH_THR

    def _reached(self, env, target: np.ndarray) -> bool:
        """True when the ee is within REACH_THR of `target` AND the phase has
        been holding the target for at least PHASE_HOLD_STEPS env steps.

        The hold requirement lets the under-damped sts3215 actuator settle
        out of its ~3 mm overshoot before the next phase starts — without
        it, the fingertips oscillate through cube space and launch the cube
        sideways during the DESCEND→GRASP transition.
        """
        if self.phase_steps < self.c.PHASE_HOLD_STEPS:
            return False
        return float(np.linalg.norm(env.ee_pos() - target)) < self.c.REACH_THR

    # ---------------- internals ----------------
    def _advance(self, name: Phase):
        self.phase = name
        self.phase_steps = 0

    def _cache_ids(self, model: mujoco.MjModel) -> None:
        self._cube_gid = model.geom("red_cube_geom").id
        self._pad_gids = (
            model.geom("static_finger_pad").id,
            model.geom("moving_finger_pad").id,
        )

    def _is_grasping(self, env) -> bool:
        """Both finger pads in contact with the cube."""
        assert self._pad_gids is not None and self._cube_gid is not None
        static_id, moving_id = self._pad_gids
        cube_id = self._cube_gid
        seen_static = False
        seen_moving = False
        d = env.data
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            if g1 == cube_id or g2 == cube_id:
                other = g2 if g1 == cube_id else g1
                if other == static_id:
                    seen_static = True
                elif other == moving_id:
                    seen_moving = True
        return seen_static and seen_moving


# Backwards-compat: leave _Constants accessible since the legacy policy
# instance exposed `policy.c.*` and the parallel_runner reads
# `obs["red_cube_pos"]` directly.
