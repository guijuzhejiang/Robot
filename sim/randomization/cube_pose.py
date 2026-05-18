"""Cube-pose randomization utilities.

The env (pick_place.py) does its own cube + plate placement inside
`_post_reset`. This module exposes a standalone helper for tests / external
pipelines that need sampled positions without rolling through the env.

NOTE: cube COLOR is intentionally NOT randomized — it's the language anchor.
"""
from __future__ import annotations

import numpy as np


def sample_cube_and_plate_xy(
    rng: np.random.Generator,
    *,
    cube_x_range=(0.16, 0.22),
    cube_y_range=(-0.08, 0.08),
    plate_x_range=(0.24, 0.30),
    plate_y_range=(-0.05, 0.05),
    min_separation: float = 0.12,
    max_tries: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (cube_xy, plate_xy) with at least min_separation between them."""
    for _ in range(max_tries):
        cube = np.array([rng.uniform(*cube_x_range), rng.uniform(*cube_y_range)])
        plate = np.array([rng.uniform(*plate_x_range), rng.uniform(*plate_y_range)])
        if np.linalg.norm(cube - plate) >= min_separation:
            return cube, plate
    # Fallback: spread along x
    return (
        np.array([cube_x_range[0], 0.0]),
        np.array([plate_x_range[1], 0.0]),
    )
