"""End-to-end MimicGen-style augmentation orchestrator.

Pipeline:
    for each real demo:
        segments = segment(demo)
        for trial in range(N_per_demo):
            new_scene = randomize_scene(rng)
            result = replay_segmented_demo(env, segmented_demo, new_scene)
            if result.success:
                save_to_lerobot(result, task=demo.task)

Usage (CLI):
    python -m data.mimicgen_adapter.augment \\
        --source-repo-id local/so101_real_pickplace_v0 \\
        --output-repo-id local/so101_sim_mimicgen_v1 \\
        --n-per-demo 50

If `--from-sim-seeds N` is passed instead of `--source-repo-id`, the script
generates N SYNTHETIC source demos by running the scripted oracle policy
(useful for testing the pipeline before real demos exist).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.converters.sim_to_lerobot import make_or_resume_dataset
from data.mimicgen_adapter.replayer import replay_segmented_demo
from data.mimicgen_adapter.segmenter import episode_from_ee_actions, segment
from data.mimicgen_adapter.types import ObjectPose, SegmentedDemo


def randomize_scene(rng: np.random.Generator, *, env) -> dict[str, ObjectPose]:
    """Sample a fresh red/plate configuration consistent with env workspace.

    Yaw is fixed to 0 for both cube and plate: both are rotationally
    symmetric for this task (4-fold cube, circular plate), so randomising
    yaw would only rotate the replayed ee trajectory about the new anchor
    by a meaningless angle — which actually breaks replay because source
    demos record yaw=0 while a random new-anchor yaw produces a random
    rotation in ``_transform_xy_under_anchor``.
    """
    red_xy = np.array([rng.uniform(*env.CUBE_X_RANGE), rng.uniform(*env.CUBE_Y_RANGE)])
    plate_xy = np.array([rng.uniform(*env.PLATE_X_RANGE), rng.uniform(*env.PLATE_Y_RANGE)])
    return {
        "red":  ObjectPose("red",  red_xy,  yaw=0.0, z=env.CUBE_HALF),
        "plate": ObjectPose("plate", plate_xy, yaw=0.0, z=env.PLATE_HALF_HEIGHT),
    }


def synthesize_source_demos(n: int, *, env, instruction_pool: list[str],
                            rng: np.random.Generator,
                            require_success: bool = False) -> list[SegmentedDemo]:
    """Generate N synthetic source demos using the scripted oracle policy.

    Used when no real-demo dataset is available yet — produces SegmentedDemo
    objects directly without going through LeRobotDataset on-disk format.

    Args:
        require_success: if True, only keep demos where env.evaluate_success()
            returned True. Default False so segmenter-pipeline tests work even
            when the scripted policy has low success rate (current ~0–20%).
    """
    from sim.scripted_policies.pick_place import PickPlacePolicy

    demos: list[SegmentedDemo] = []
    seed = 0
    attempts = 0
    while len(demos) < n and attempts < n * 10:
        attempts += 1
        seed += 1
        obs, _ = env.reset(seed=seed)
        policy = PickPlacePolicy()
        ee_positions: list[np.ndarray] = []
        gripper_states: list[float] = []
        # NOTE: `actions` here is structurally needed (episode_from_ee_actions
        # uses its length to size the Frame list) but its values are NOT used
        # downstream. The scripted oracle returns [0,0,0,gripper] (it drives
        # the arm via env.ee_target_override), so this list is essentially
        # [0,0,0,*]. Real ee motion is reconstructed by the replayer from
        # `ee_positions` (per-frame TCP world coords) — do NOT add this list
        # to any LeRobot dataset directly.
        actions: list[np.ndarray] = []
        done = False
        while not done:
            action = policy(env, obs)
            ee_positions.append(env.ee_pos())
            # Use ctrl[5] AFTER step for gripper state; but record pre-step to align
            gripper_states.append(float(env.data.ctrl[5]))
            obs_next, _, term, trunc, info = env.step(action)
            actions.append(action.copy())
            obs = obs_next
            done = term or trunc
        if require_success and not info.get("is_success"):
            continue
        # Build Frames from arrays — actions only used for length, see note above.
        frames = episode_from_ee_actions(
            np.stack(actions), np.stack(ee_positions), np.array(gripper_states),
            initial_objects={
                "red":  ObjectPose("red", env._red_init_xy.copy(), 0.0, env.CUBE_HALF),
                "plate": ObjectPose("plate", env.body_pos("plate")[:2].copy(), 0.0,
                                    env.PLATE_HALF_HEIGHT),
            },
        )
        task = rng.choice(instruction_pool) if instruction_pool else "put the red cube on the plate"
        sd = segment(frames, task=task)
        if sd is None:
            continue
        demos.append(sd)
    return demos


def augment(
    *,
    source_demos: list[SegmentedDemo],
    output_repo_id: str,
    n_per_demo: int,
    seed: int = 0,
) -> dict:
    """Replay each source demo into N_per_demo new randomized scenes."""
    from sim.envs.pick_place import PickPlaceEnv

    env = PickPlaceEnv(observation_mode="both", action_mode="ee",
                       max_episode_steps=400)
    writer = make_or_resume_dataset(repo_id=output_repo_id, fps=30, reset=True)
    rng = np.random.default_rng(seed)

    stats = {"attempted": 0, "succeeded": 0, "failed_by_mode": {}}

    for demo_idx, demo in enumerate(source_demos):
        for trial in range(n_per_demo):
            stats["attempted"] += 1
            new_scene = randomize_scene(rng, env=env)
            result = replay_segmented_demo(env, demo, new_scene,
                                           seed=stats["attempted"])
            if result.success:
                # Write the recorded actions back into a fresh episode by
                # replaying them step-by-step from the same reset state.
                # `result.actions` is the joint label (primary), but env
                # still consumes ee gym actions; pass both so _write_episode
                # can step with ee and record joint.
                _write_episode(env, writer, demo.task, new_scene,
                               actions=result.actions,
                               ee_actions=result.ee_actions,
                               seed=stats["attempted"])
                stats["succeeded"] += 1
            else:
                mode = result.failure_mode or "unknown"
                stats["failed_by_mode"][mode] = stats["failed_by_mode"].get(mode, 0) + 1

    env.close()
    stats["success_rate"] = stats["succeeded"] / max(1, stats["attempted"])
    stats["output_repo_id"] = output_repo_id
    stats["output_episodes"] = writer.num_episodes
    stats["output_frames"] = writer.num_frames
    return stats


def _write_episode(env, writer, task: str, new_scene: dict,
                   *,
                   actions: list[np.ndarray],
                   ee_actions: list[np.ndarray],
                   seed: int) -> None:
    """Reset env, place objects, replay, stream frames into writer.

    `actions`     — joint ctrl labels (6,), written to the `action` column.
    `ee_actions`  — ee-delta gym actions (4,), consumed by env.step (since
                    env.action_mode='ee') and written to `ee_action` column.
    """
    from data.mimicgen_adapter.replayer import _place_objects

    obs, _ = env.reset(seed=seed)
    _place_objects(env, new_scene)
    obs = env._compute_obs()
    for action, ee_action in zip(actions, ee_actions):
        writer.add_frame_from_obs(obs, action, task=task,
                                  ee_action=ee_action)
        # env still consumes ee gym actions to drive the IK; the recorded
        # `action` column above is the joint label, not the gym input.
        next_obs, _, term, trunc, _ = env.step(ee_action)
        obs = next_obs
        if term or trunc:
            break
    writer.save_episode()


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-repo-id", help="LeRobot repo_id of real demos")
    group.add_argument("--from-sim-seeds", type=int,
                       help="Generate N synthetic source demos via scripted policy")
    ap.add_argument("--output-repo-id", required=True)
    ap.add_argument("--n-per-demo", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instructions",
                    default="data/instructions/pick_place.txt")
    args = ap.parse_args()

    instructions = []
    p = Path(args.instructions)
    if p.exists():
        instructions = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]

    if args.from_sim_seeds:
        from sim.envs.pick_place import PickPlaceEnv

        env = PickPlaceEnv(observation_mode="state", action_mode="ee",
                           max_episode_steps=400)
        rng = np.random.default_rng(args.seed)
        demos = synthesize_source_demos(
            args.from_sim_seeds, env=env, instruction_pool=instructions, rng=rng,
        )
        env.close()
        print(f"[augment] generated {len(demos)} synthetic source demos")
    else:
        raise NotImplementedError(
            "Loading real demos from LeRobot dataset requires the FK helper "
            "noted in segmenter.py. Use --from-sim-seeds for now."
        )

    stats = augment(source_demos=demos, output_repo_id=args.output_repo_id,
                    n_per_demo=args.n_per_demo, seed=args.seed)
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
