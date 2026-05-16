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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def _worker(args: tuple) -> dict:
    """Worker process: collect N episodes into its own shard repo_id."""
    (worker_id, n_episodes, seed_offset, repo_id,
     instruction_pool, randomize_domain, keep_failures,
     img_height, img_width, max_episode_steps) = args

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
        max_episode_steps=max_episode_steps,
        randomize_domain=randomize_domain,
        img_height=img_height,
        img_width=img_width,
    )
    # Register the dataset's image features at the SAME resolution we render
    # at — LeRobot validates every add_frame's shape against the schema and
    # would otherwise reject 480x640 frames against a 240x320 declaration.
    writer = make_or_resume_dataset(
        repo_id=repo_id, fps=30, reset=True,
        img_height=env.IMG_HEIGHT, img_width=env.IMG_WIDTH,
    )
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
        done = False
        info: dict = {}
        # Buffer task per-frame; we may need to relabel on failure when
        # keep_failures=True (frame-level task field is the only place we
        # can stash the failure mode in LeRobot 0.5.2).
        frame_actions: list[np.ndarray] = []
        frame_obs: list[dict] = []
        while not done:
            action = policy(env, obs)
            # NOTE: the legacy xy-tracking override (drive ee_xy to current
            # cube_xy each tick) was removed. The pick-101 oracle now
            # snapshots the cube position once on policy.__call__ and
            # applies a FINGER_WIDTH_OFFSET to centre the cube between the
            # jaws — overriding ee_xy here would erase that offset and the
            # static fingertip would crash into the cube top during DESCEND.
            next_obs, _, term, trunc, info = env.step(action)
            frame_obs.append(obs)
            frame_actions.append(action.copy())
            obs = next_obs
            done = term or trunc

        succeeded = bool(info.get("is_success"))
        if succeeded:
            task_label = ep.task
        else:
            mode = info.get("failure_mode") or "unknown"
            task_label = f"{ep.task} [FAIL:{mode}]"

        if succeeded or keep_failures:
            for fo, fa in zip(frame_obs, frame_actions):
                writer.add_frame_from_obs(fo, fa, task=task_label)
            writer.save_episode()
        # else: drop — nothing was buffered, so no discard needed.

        if succeeded:
            counts["success"] += 1
        else:
            counts["fail"] += 1
            mode = info.get("failure_mode") or "unknown"
            counts["by_mode"][mode] = counts["by_mode"].get(mode, 0) + 1
    env.close()
    return {"worker_id": worker_id, **counts, "repo_id": repo_id,
            "episodes_saved": writer.num_episodes}


def _cleanup_shards_to_videos(
    repo_id: str, shard_ids: list[str]
) -> tuple[Path, dict]:
    """Split each shard's bundled mp4 into per-episode mp4s, delete shards.

    LeRobot 0.5 packs all episodes of a chunk into a single mp4
    (`videos/observation.images.<cam>/chunk-NNN/file-NNN.mp4`); each episode
    occupies a `[from_timestamp, to_timestamp]` range listed in
    `meta/episodes/chunk-NNN/file-NNN.parquet`. We use ffmpeg to slice each
    range into its own mp4 so the user can browse one file per episode.

    Layout produced (under ~/.cache/huggingface/lerobot/):
        local/<repo_id>_videos/
            front/
                shard00_ep0000_FAIL_grasp_fail.mp4
                shard00_ep0001_SUCCESS.mp4
                ...
            wrist/
                shard00_ep0000_FAIL_grasp_fail.mp4
                ...

    Returns (output_dir, stats).
    """
    import re
    import subprocess

    import pandas as pd

    base = Path.home() / ".cache" / "huggingface" / "lerobot"
    out_dir = base / f"{repo_id}_videos"
    front_dir = out_dir / "front"
    wrist_dir = out_dir / "wrist"
    front_dir.mkdir(parents=True, exist_ok=True)
    wrist_dir.mkdir(parents=True, exist_ok=True)

    stats = {"front_mp4": 0, "wrist_mp4": 0, "shards_removed": 0,
             "split_errors": 0}

    fail_tag_re = re.compile(r"\[FAIL:([^\]]+)\]")

    for shard_id in shard_ids:
        shard_root = base / shard_id
        if not shard_root.exists():
            continue
        shard_tag = shard_id.rsplit("_shard", 1)[-1] if "_shard" in shard_id else "00"

        ep_meta_files = sorted((shard_root / "meta" / "episodes").rglob("*.parquet"))
        for meta_pf in ep_meta_files:
            df = pd.read_parquet(meta_pf)
            for _, row in df.iterrows():
                ep_idx = int(row["episode_index"])
                task = row["tasks"][0] if hasattr(row["tasks"], "__len__") else str(row["tasks"])
                m = fail_tag_re.search(str(task))
                status_tag = f"FAIL_{m.group(1)}" if m else "SUCCESS"

                for cam, out_subdir, stat_key in [
                    ("front", front_dir, "front_mp4"),
                    ("wrist", wrist_dir, "wrist_mp4"),
                ]:
                    t_from = float(row[f"videos/observation.images.{cam}/from_timestamp"])
                    t_to = float(row[f"videos/observation.images.{cam}/to_timestamp"])
                    chunk_idx = int(row[f"videos/observation.images.{cam}/chunk_index"])
                    file_idx = int(row[f"videos/observation.images.{cam}/file_index"])
                    src = (shard_root / "videos" / f"observation.images.{cam}"
                           / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4")
                    if not src.exists():
                        stats["split_errors"] += 1
                        continue
                    dst = out_subdir / f"shard{shard_tag}_ep{ep_idx:04d}_{status_tag}.mp4"
                    duration = max(0.0, t_to - t_from)
                    # -c copy keeps the original encoding (fast, no re-encode).
                    proc = subprocess.run(
                        ["ffmpeg", "-y", "-loglevel", "error",
                         "-ss", f"{t_from:.3f}", "-t", f"{duration:.3f}",
                         "-i", str(src), "-c", "copy", str(dst)],
                        capture_output=True,
                    )
                    if proc.returncode != 0 or not dst.exists():
                        stats["split_errors"] += 1
                    else:
                        stats[stat_key] += 1

        shutil.rmtree(shard_root)
        stats["shards_removed"] += 1

    return out_dir, stats


def collect(
    num_episodes: int,
    num_workers: int,
    repo_id: str,
    *,
    instruction_pool: list[str] | None = None,
    randomize_domain: bool = True,
    keep_failures: bool = False,
    cleanup_after_collect: bool = False,
    img_height: int = 480,
    img_width: int = 640,
    max_episode_steps: int = 500,
) -> dict:
    """Top-level coordinator. Distributes work across workers; returns aggregate."""
    per_worker = num_episodes // num_workers
    remainder = num_episodes - per_worker * num_workers

    tasks = []
    seed_offset = 0
    for w in range(num_workers):
        n = per_worker + (1 if w < remainder else 0)
        shard_id = f"{repo_id}_shard{w:02d}"
        tasks.append((w, n, seed_offset, shard_id, instruction_pool,
                      randomize_domain, keep_failures,
                      img_height, img_width, max_episode_steps))
        seed_offset += n

    t0 = time.time()
    if num_workers == 1:
        results = [_worker(tasks[0])]
    else:
        # ProcessPoolExecutor with mp_context=spawn — workers are NOT daemonic,
        # so LeRobot's internal video-encoding subprocesses can fork freely.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool:
            results = list(pool.map(_worker, tasks))
    elapsed = time.time() - t0

    total_success = sum(r["success"] for r in results)
    total_fail = sum(r["fail"] for r in results)
    by_mode: dict[str, int] = {}
    for r in results:
        for m, c in r["by_mode"].items():
            by_mode[m] = by_mode.get(m, 0) + c

    summary = {
        "elapsed_sec": elapsed,
        "total_success": total_success,
        "total_fail": total_fail,
        "success_rate": total_success / max(1, total_success + total_fail),
        "by_failure_mode": by_mode,
        "keep_failures": keep_failures,
        "shards": [r["repo_id"] for r in results],
        "shard_episode_counts": [r["episodes_saved"] for r in results],
    }

    if cleanup_after_collect:
        out_dir, cleanup_stats = _cleanup_shards_to_videos(
            repo_id, [r["repo_id"] for r in results]
        )
        summary["cleanup"] = {
            "videos_dir": str(out_dir),
            **cleanup_stats,
            "note": "Shard datasets removed; audit_dataset will no longer find this repo_id.",
        }

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-episodes", type=int, default=20)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--repo-id", type=str, default="local/so101_pickplace_blue_v1")
    ap.add_argument("--instructions", type=str, default="data/instructions/pick_place_blue.txt")
    ap.add_argument("--no-dr", action="store_true", help="Disable domain randomization")
    ap.add_argument(
        "--keep-failures", action="store_true",
        help="Save failed episodes too. Their `task` field is suffixed with "
             "[FAIL:<mode>] so audit/training can filter them out. Useful for "
             "debugging the scripted policy by watching the videos.",
    )
    ap.add_argument(
        "--cleanup-after-collect", action="store_true",
        help="After collection, move all mp4 videos to "
             "~/.cache/huggingface/lerobot/<repo_id>_videos/{front,wrist}/ "
             "and DELETE the shard datasets (parquet + meta). "
             "Saves disk space; audit_dataset will no longer work on this run.",
    )
    ap.add_argument(
        "--img-height", type=int, default=480,
        help="Per-camera render height. Default 480 (native front-cam size). "
             "Scene supports up to 1080; larger values will need an "
             "<global offheight=...> bump in the XML.",
    )
    ap.add_argument(
        "--img-width", type=int, default=640,
        help="Per-camera render width. Default 640. Scene supports up to 1920.",
    )
    ap.add_argument(
        "--max-episode-steps", type=int, default=500,
        help="Per-episode ctrl-tick cap. Default 500 (5 s @ 100 Hz).",
    )
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
        keep_failures=args.keep_failures,
        cleanup_after_collect=args.cleanup_after_collect,
        img_height=args.img_height,
        img_width=args.img_width,
        max_episode_steps=args.max_episode_steps,
    )
    import json
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
