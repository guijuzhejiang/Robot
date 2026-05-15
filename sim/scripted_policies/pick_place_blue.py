"""6-stage scripted oracle policy for PickPlaceBlue.

Phases:
  1. APPROACH:  ee → 10 cm above blue cube
  2. DESCEND:   ee → grasp height above blue cube
  3. GRASP:     close gripper for N steps
  4. LIFT:      raise ee by 12 cm
  5. TRANSPORT: ee → 10 cm above plate (avoid red cube via high lift)
  6. PLACE_RELEASE: descend to plate top + 3 cm → open gripper → retract

This policy is an *oracle*: it reads `obs["blue_cube_pos"]` / `obs["plate_pos"]`
directly. It is the data-generation reference, NOT the VLA target policy.
The VLA in Phase 4 must learn to do the same task from images + language only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Phase = Literal[
    "APPROACH", "DESCEND", "GRASP", "LIFT", "TRANSPORT", "PLACE_RELEASE", "DONE"
]


@dataclass
class _Constants:
    APPROACH_HEIGHT: float = 0.10
    # gripperframe is between-fingertips; to grip a 4cm cube (center z=0.02),
    # descend so fingertips reach near the table — z ≈ 0.005 sits jaws around cube.
    DESCEND_Z_ABS: float = 0.008
    LIFT_HEIGHT: float = 0.14
    TRANSPORT_HEIGHT: float = 0.16
    PLACE_OFFSET_Z: float = 0.04    # release height above plate top
    REACH_THR: float = 0.015        # m
    GRASP_STEPS: int = 25
    RELEASE_STEPS: int = 15
    PHASE_TIMEOUT: int = 80


class PickPlaceBluePolicy:
    """Scripted oracle. Call `policy(env, obs)` each step."""

    def __init__(self, *, max_speed: float = 0.04, descend_speed: float = 0.02):
        self.c = _Constants()
        self.phase: Phase = "APPROACH"
        self.phase_steps = 0
        self.gripper = 1.0  # +1 open, -1 close (env's ee mode uses delta)
        self.blue_pickup_pos: np.ndarray | None = None
        self.plate_target_xy: np.ndarray | None = None
        self.max_speed = max_speed
        self.descend_speed = descend_speed

    def reset(self):
        self.__init__(max_speed=self.max_speed, descend_speed=self.descend_speed)

    def _step_to(self, env, target_xyz: np.ndarray, max_speed: float | None = None):
        """Return (action_xyz_normalized, reached)."""
        cur = env.ee_pos()
        delta = np.asarray(target_xyz) - cur
        dist = np.linalg.norm(delta)
        if dist < self.c.REACH_THR:
            return np.zeros(3), True
        speed = max_speed if max_speed is not None else self.max_speed
        delta = delta / dist * min(dist, speed)
        # env converts action[:3] into delta = action * 0.05
        return np.clip(delta / 0.05, -1.0, 1.0), False

    def _advance(self, name: Phase):
        self.phase = name
        self.phase_steps = 0

    # ---------------- main call ----------------
    def __call__(self, env, obs) -> np.ndarray:
        blue = obs["blue_cube_pos"]
        plate = obs["plate_pos"]
        self.phase_steps += 1
        timed_out = self.phase_steps >= self.c.PHASE_TIMEOUT

        if self.phase == "APPROACH":
            target = blue + np.array([0.0, 0.0, self.c.APPROACH_HEIGHT])
            d, reached = self._step_to(env, target)
            self.gripper = 1.0
            if reached or timed_out:
                self._advance("DESCEND")

        elif self.phase == "DESCEND":
            # gripperframe is fingertip-between point. Bring it down to ~cube mid
            # so jaws straddle the cube from above.
            target = np.array([blue[0], blue[1], self.c.DESCEND_Z_ABS])
            d, reached = self._step_to(env, target, max_speed=self.descend_speed)
            self.gripper = 0.8  # keep open while descending
            if reached or timed_out:
                self.blue_pickup_pos = blue.copy()
                self._advance("GRASP")

        elif self.phase == "GRASP":
            d = np.zeros(3)
            self.gripper = -1.0  # close
            if self.phase_steps >= self.c.GRASP_STEPS:
                self._advance("LIFT")

        elif self.phase == "LIFT":
            assert self.blue_pickup_pos is not None
            target = self.blue_pickup_pos + np.array([0.0, 0.0, self.c.LIFT_HEIGHT])
            d, reached = self._step_to(env, target)
            self.gripper = -1.0
            if reached or timed_out:
                self._advance("TRANSPORT")

        elif self.phase == "TRANSPORT":
            # Move ee horizontally to above-plate at TRANSPORT_HEIGHT.
            # High z to avoid hitting the red cube along the way.
            self.plate_target_xy = plate[:2].copy()
            target = np.array([plate[0], plate[1], self.c.TRANSPORT_HEIGHT])
            d, reached = self._step_to(env, target)
            self.gripper = -1.0
            if reached or timed_out:
                self._advance("PLACE_RELEASE")

        elif self.phase == "PLACE_RELEASE":
            # Phase substeps: descend → release → retract
            descend_target = np.array([
                plate[0], plate[1],
                env.PLATE_TOP_Z + self.c.PLACE_OFFSET_Z,
            ])
            ee = env.ee_pos()
            if ee[2] > descend_target[2] + self.c.REACH_THR:
                d, _ = self._step_to(env, descend_target, max_speed=self.descend_speed)
                self.gripper = -1.0
            elif self.phase_steps < self.c.GRASP_STEPS + self.c.RELEASE_STEPS:
                d = np.zeros(3)
                self.gripper = 1.0  # open
            else:
                # Retract upward
                retract = np.array([plate[0], plate[1], self.c.TRANSPORT_HEIGHT])
                d, reached = self._step_to(env, retract)
                self.gripper = 1.0
                if reached or self.phase_steps >= self.c.PHASE_TIMEOUT * 2:
                    self._advance("DONE")

        else:  # DONE
            d = np.zeros(3)
            self.gripper = 1.0

        return np.concatenate([d, [self.gripper]]).astype(np.float32)
