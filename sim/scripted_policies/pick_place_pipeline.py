"""Task-level trajectory pipeline.

Wraps the scripted oracle into an episode generator that returns an Episode
(frames + metadata + success flag) or None on env-init failure.

Public entry point:
    generate_pickplace_episode(env, seed, *, instruction_pool=None, rng=None)
        -> Episode | None

History: an antipodal grasp sampler used to choose the best XY grasp pose
around the cube to avoid the (now removed) blue distractor. With the task
simplified to a single red cube + plate, the sampler is no longer needed —
the pick-101-style locked-wrist DLS IK descends straight down on the cube
center and the success rate is dominated by the contact-detection grasp.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.scripted_policies.pick_place_blue import PickPlaceBluePolicy


@dataclass
class Episode:
    frames: list[dict] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    task: str = ""
    success: bool = False
    failure_mode: str | None = None
    seed: int = 0
    n_steps: int = 0
    extra: dict = field(default_factory=dict)


def generate_pickplace_episode(
    env,
    seed: int,
    *,
    instruction_pool: list[str] | None = None,
    rng: np.random.Generator | None = None,
    max_steps: int | None = None,
) -> Episode | None:
    """Run one PickPlace episode in `env` using the oracle policy.

    `instruction_pool`: if provided, randomly samples a `task` string from it.
    Returns Episode (success flag set); caller decides save/discard.
    Returns None only if env.reset() itself fails.
    """
    rng = rng or np.random.default_rng(seed)
    task = "put the red cube on the plate"
    if instruction_pool:
        task = rng.choice(instruction_pool)

    obs, _ = env.reset(seed=seed)

    policy = PickPlaceBluePolicy()
    policy.reset()

    ep = Episode(task=task, seed=seed)
    steps = max_steps or env.max_episode_steps
    done = False
    info: dict = {}
    for _ in range(steps):
        action = policy(env, obs)
        # The pick-101 oracle snapshots cube/plate once on first call and
        # applies a FINGER_WIDTH_OFFSET to center the cube between the jaws,
        # so no per-step xy override here (overriding would wipe the offset).
        next_obs, _, term, trunc, info = env.step(action)
        ep.frames.append(
            {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
        )
        ep.actions.append(action.copy())
        obs = next_obs
        if term or trunc:
            done = True
            break
    ep.n_steps = len(ep.actions)
    ep.success = bool(info.get("is_success", False))
    ep.failure_mode = info.get("failure_mode")
    if not done:
        ep.failure_mode = ep.failure_mode or "timeout"
    return ep
