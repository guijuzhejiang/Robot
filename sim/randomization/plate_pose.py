"""Plate pose randomization helper (analogous to cube_pose)."""
from __future__ import annotations

import numpy as np


def sample_plate_xy(
    rng: np.random.Generator,
    *,
    x_range=(0.24, 0.30),
    y_range=(-0.06, 0.06),
) -> np.ndarray:
    return np.array([rng.uniform(*x_range), rng.uniform(*y_range)])
