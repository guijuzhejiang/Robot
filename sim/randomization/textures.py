"""Texture randomization for tablecloth.

MuJoCo doesn't let you swap a `<texture file=...>` at runtime cleanly, so this
module's strategy is to mutate the `rgba` of the `tablecloth` material, which is
visually equivalent to swapping a procedural texture's base colors.

For richer texture diversity, swap the tablecloth `<texture>` to a `file` mode
and pre-load several texture variants — see assets/textures/ directory for
where to drop PBR images (e.g., from ambientCG).
"""
from __future__ import annotations

import mujoco
import numpy as np


# A small palette of "tablecloth-like" base colors (RGB in [0,1])
_PRESETS = np.array([
    [0.85, 0.85, 0.85],  # white
    [0.70, 0.65, 0.55],  # beige
    [0.55, 0.50, 0.45],  # tan
    [0.40, 0.45, 0.55],  # cool grey
    [0.65, 0.55, 0.45],  # warm grey
    [0.30, 0.35, 0.40],  # dark grey
])


def randomize_tablecloth(model: mujoco.MjModel, rng: np.random.Generator) -> None:
    try:
        mat_id = model.material("tablecloth").id
    except KeyError:
        return
    idx = rng.integers(0, len(_PRESETS))
    rgb = _PRESETS[idx] + rng.uniform(-0.05, 0.05, size=3)
    rgb = np.clip(rgb, 0.0, 1.0)
    model.mat_rgba[mat_id, :3] = rgb
