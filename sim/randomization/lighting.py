"""Domain randomization: light positions, directions, and intensities.

Mutates `model.light_*` arrays in place. Call before env.reset() settles.
Lights `light0`, `light1`, `light2` are defined in pick_place.xml.
"""
from __future__ import annotations

import mujoco
import numpy as np


def randomize_lights(model: mujoco.MjModel, rng: np.random.Generator) -> None:
    """Randomize positions + diffuse intensity of all named lights."""
    for i in range(model.nlight):
        # Random hemisphere position above the table
        x = rng.uniform(-0.4, 0.4)
        y = rng.uniform(-0.4, 0.4)
        z = rng.uniform(0.6, 1.2)
        model.light_pos[i] = (x, y, z)

        # Point roughly down toward workspace center
        target = np.array([rng.uniform(-0.1, 0.3), rng.uniform(-0.1, 0.1), 0.0])
        direction = target - model.light_pos[i]
        direction /= np.linalg.norm(direction) + 1e-9
        model.light_dir[i] = direction

        # Slight color temperature shift via diffuse RGB
        intensity = rng.uniform(0.3, 0.8)
        warmth = rng.uniform(-0.1, 0.1)
        model.light_diffuse[i] = (
            intensity + warmth,
            intensity,
            intensity - warmth,
        )
