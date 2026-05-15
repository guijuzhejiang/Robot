"""Parallel data collection runner.

Spawns N worker processes, each running one env in isolation. Each worker
writes frames into its OWN LeRobot dataset (under a shard repo_id), and the
master process merges shards at the end.

Usage:
    python -m sim.collectors.parallel_runner \\
        --num-episodes 1000 --num-workers 8 \\
        --repo-id local/so101_pickplace_blue_v1

NOTE: For each shard, LeRobotDataset.create() is called; per-process MuJoCo
rendering uses each process's own EGL context (no shared state). On GPU
machines with limited VRAM, reduce num-workers.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np


def _worker(args: tuple) -> dict:
    """Worker process: collect N episodes into its own shard repo_id."""
    worker_id, n_episodes, seed_offset, repo_id, instruction_pool, randomize_domain = args

    # Suppress noisy ffmpeg / pynvml warnings from each child
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("SVT_LOG", "0")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from sim.envs.pick_place_blue import PickPlaceBlueEnv
    from sim.scripted_policies.pick_place_pipeline import generate_pickplace_episode
    from data.converters.sim_to_lerobot import make_or_resume_dataset

    env = PickPlaceBlueEnv(
        observation_mode="both",
        action_mode="ee",
        max_episode_steps=300,
        randomize_domain=randomize_domain,
    )
    writer = make_or_resume_dataset(repo_id=repo_id, fps=30, reset=True)
    rng = np.random.default_rng(seed_offset)

    counts = {"success": 0, "fail": 0, "by_mode": {}}
    for i in range(n_episodes):
        seed = seed_offset + i
        ep = generate_pickplace_episode(
            env, seed=seed, instruction_pool=instruction_pool, rng=rng
        )
        if ep is None:
            counts["fail"] += 1
            continue
        # Replay frames into writer
        # NOTE: simpler approach for now — generate ep already stepped env,
        # but didn't write frames. Use a one-shot replay pattern: actions only
        # mode is impractical because we can't re-run determinism trivially.
        # So we rerun the env+policy here and stream into writer directly.
        obs, _ = env.reset(seed=seed)
        from sim.scripted_policies.pick_place_blue import PickPlaceBluePolicy
        from sim.grasp.antipodal import sample_cube_grasps
        policy = PickPlaceBluePolicy()
        policy.reset()
        grasps = sample_cube_grasps(
            cube_pos=obs["blue_cube_pos"],
            obstacle_positions=[obs["red_cube_pos"], obs["plate_pos"]],
            obstacle_radii=[0.06, 0.08],
            rng=rng,
        )
        target_xy = grasps[0].position[:2] if grasps else obs["blue_cube_pos"][:2]
        done = False
        info: dict = {}
        while not done:
            action = policy(env, obs)
            if policy.phase == "DESCEND":
                cur_xy = env.ee_pos()[:2]
                action[:2] = np.clip((target_xy - cur_xy) / 0.05, -1.0, 1.0)
            next_obs, _, term, trunc, info = env.step(action)
            writer.add_frame_from_obs(obs, action, task=ep.task)
            obs = next_obs
            done = term or trunc
        if info.get("is_success"):
            writer.save_episode()
            counts["success"] += 1
        else:
            writer.discard_episode()
            counts["fail"] += 1
            mode = info.get("failure_mode") or "unknown"
            counts["by_mode"][mode] = counts["by_mode"].get(mode, 0) + 1
    env.close()
    return {"worker_id": worker_id, **counts, "repo_id": repo_id,
            "episodes_saved": writer.num_episodes}


def collect(
    num_episodes: int,
    num_workers: int,
    repo_id: str,
    *,
    instruction_pool: list[str] | None = None,
    randomize_domain: bool = True,
) -> dict:
    """Top-level coordinator. Distributes work across workers; returns aggregate."""
    per_worker = num_episodes // num_workers
    remainder = num_episodes - per_worker * num_workers

    tasks = []
    seed_offset = 0
    for w in range(num_workers):
        n = per_worker + (1 if w < remainder else 0)
        shard_id = f"{repo_id}_shard{w:02d}"
        tasks.append((w, n, seed_offset, shard_id, instruction_pool, randomize_domain))
        seed_offset += n

    t0 = time.time()
    if num_workers == 1:
        results = [_worker(tasks[0])]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(num_workers) as pool:
            results = pool.map(_worker, tasks)
    elapsed = time.time() - t0

    total_success = sum(r["success"] for r in results)
    total_fail = sum(r["fail"] for r in results)
    by_mode: dict[str, int] = {}
    for r in results:
        for m, c in r["by_mode"].items():
            by_mode[m] = by_mode.get(m, 0) + c

    return {
        "elapsed_sec": elapsed,
        "total_success": total_success,
        "total_fail": total_fail,
        "success_rate": total_success / max(1, total_success + total_fail),
        "by_failure_mode": by_mode,
        "shards": [r["repo_id"] for r in results],
        "shard_episode_counts": [r["episodes_saved"] for r in results],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-episodes", type=int, default=20)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--repo-id", type=str, default="local/so101_pickplace_blue_v1")
    ap.add_argument("--instructions", type=str, default="data/instructions/pick_place_blue.txt")
    ap.add_argument("--no-dr", action="store_true", help="Disable domain randomization")
    args = ap.parse_args()

    pool = None
    p = Path(args.instructions)
    if p.exists():
        pool = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]

    summary = collect(
        num_episodes=args.num_episodes,
        num_workers=args.num_workers,
        repo_id=args.repo_id,
        instruction_pool=pool,
        randomize_domain=not args.no_dr,
    )
    import json
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
