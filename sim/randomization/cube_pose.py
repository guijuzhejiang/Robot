"""Cube-pose randomization utilities.

The env (pick_place_blue.py) already does cube placement inside `_post_reset`.
This module exposes a standalone helper for tests / external pipelines that need
sampled positions without rolling through the env.

NOTE: cube COLOR is intentionally NOT randomized — it's the language anchor.
"""
from __future__ import annotations

import numpy as np


def sample_two_cube_xy(
    rng: np.random.Generator,
    *,
    x_range=(0.10, 0.22),
    y_range=(-0.12, 0.12),
    min_separation: float = 0.08,
    max_tries: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (red_xy, blue_xy) with at least min_separation between them."""
    for _ in range(max_tries):
        red = np.array([rng.uniform(*x_range), rng.uniform(*y_range)])
        blue = np.array([rng.uniform(*x_range), rng.uniform(*y_range)])
        if np.linalg.norm(red - blue) >= min_separation:
            return red, blue
    # Fallback: spread along x
    return np.array([x_range[0], 0.0]), np.array([x_range[1], 0.0])
