"""Antipodal grasp sampler for axis-aligned-ish cuboids (cubes).

For PickPlace we treat the target as a small cube (4cm), approximate it as
an oriented bounding box (OBB), and generate candidate top-down grasps along
the two horizontal faces. Each candidate is a 6-DoF gripper pose:
  - position = cube center (top-down approach)
  - orientation = yaw aligned to one of cube's horizontal axes

Scoring penalizes:
  - Proximity to distractor (red cube)
  - Proximity to walls of plate
  - Yaw rotations that put the gripper near joint limits (approximated: |yaw|)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GraspCandidate:
    position: np.ndarray  # (3,) world frame, fingertip target
    yaw: float            # gripper rotation about world-z, radians
    score: float          # higher is better


def sample_cube_grasps(
    cube_pos: np.ndarray,
    cube_yaw: float = 0.0,
    *,
    cube_half_extent: float = 0.015,
    obstacle_positions: list[np.ndarray] | None = None,
    obstacle_radii: list[float] | None = None,
    grasp_height: float = 0.008,
    n_per_axis: int = 4,
    rng: np.random.Generator | None = None,
) -> list[GraspCandidate]:
    """Sample top-down antipodal grasps on a cube.

    Args:
        cube_pos: cube center xyz in world frame.
        cube_yaw: cube yaw rotation in radians.
        cube_half_extent: half edge length.
        obstacle_positions: optional list of (3,) world positions to avoid.
        obstacle_radii: per-obstacle exclusion radius (default 5cm each).
        grasp_height: target gripperframe z (fingertip near table).
        n_per_axis: candidates per cube-aligned grasp axis (jitter).
        rng: optional rng for jitter.

    Returns: list of GraspCandidate sorted by score (descending).
    """
    rng = rng or np.random.default_rng()
    obstacle_positions = obstacle_positions or []
    if obstacle_radii is None:
        obstacle_radii = [0.05] * len(obstacle_positions)

    candidates: list[GraspCandidate] = []

    # Two grasp axes: aligned with cube +x and cube +y (rotated by yaw)
    for axis_yaw_offset in (0.0, np.pi / 2):
        base_yaw = cube_yaw + axis_yaw_offset
        for _ in range(n_per_axis):
            yaw_jitter = rng.uniform(-0.1, 0.1)
            yaw = _wrap(base_yaw + yaw_jitter)

            # Position: cube center, with small xy jitter
            xy_jitter = rng.uniform(-0.005, 0.005, size=2)
            pos = np.array([
                cube_pos[0] + xy_jitter[0],
                cube_pos[1] + xy_jitter[1],
                grasp_height,
            ])

            # Score: start positive, penalize obstacle proximity and large yaws
            score = 1.0
            for obs_pos, obs_r in zip(obstacle_positions, obstacle_radii):
                d = np.linalg.norm(pos[:2] - obs_pos[:2])
                if d < obs_r:
                    score -= (obs_r - d) / obs_r
            score -= 0.1 * abs(yaw) / np.pi

            candidates.append(GraspCandidate(position=pos, yaw=yaw, score=score))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def _wrap(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi
