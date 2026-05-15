"""Phase 2: task-level trajectory pipeline.

This wraps the Phase 1 oracle policy into a reusable episode generator that:
  - Uses antipodal grasp sampler to pick the best grasp pose
  - Adds extra-high transport waypoint that adapts to red cube position
  - Returns an Episode (frames + metadata + success flag) or None on failure

Public entry point:
    generate_pickplace_episode(env, seed, *, instruction_pool=None, rng=None)
        -> Episode | None
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.grasp.antipodal import sample_cube_grasps
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
    """Run one PickPlaceBlue episode in `env` using oracle policy. Returns Episode.

    `instruction_pool`: if provided, randomly samples a `task` string from it.
    Returns Episode (success flag set); caller decides save/discard.
    Returns None if env.reset() itself fails (very rare).
    """
    rng = rng or np.random.default_rng(seed)
    task = "put the blue cube on the plate"
    if instruction_pool:
        task = rng.choice(instruction_pool)

    obs, _ = env.reset(seed=seed)

    # Use grasp sampler to pick the best grasp position (could be off-center)
    # This isn't strictly used by the simple oracle below — kept for richer policies.
    grasps = sample_cube_grasps(
        cube_pos=obs["blue_cube_pos"],
        obstacle_positions=[obs["red_cube_pos"], obs["plate_pos"]],
        obstacle_radii=[0.06, 0.08],
        rng=rng,
    )
    if not grasps:
        return None

    policy = PickPlaceBluePolicy()
    policy.reset()
    # Override DESCEND target to the highest-scoring grasp position
    target_xy = grasps[0].position[:2]

    ep = Episode(task=task, seed=seed)

    steps = max_steps or env.max_episode_steps
    done = False
    for _ in range(steps):
        action = policy(env, obs)
        # In DESCEND phase, redirect target xy to the chosen grasp position.
        if policy.phase == "DESCEND":
            # Override stale obs values used inside policy via a small nudge of
            # action[:2] toward target_xy
            cur_xy = env.ee_pos()[:2]
            desired_xy = target_xy
            dx, dy = (desired_xy - cur_xy) / 0.05
            action[0] = float(np.clip(dx, -1.0, 1.0))
            action[1] = float(np.clip(dy, -1.0, 1.0))

        next_obs, _, term, trunc, info = env.step(action)
        ep.frames.append({k: (v.copy() if isinstance(v, np.ndarray) else v)
                          for k, v in obs.items()})
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
