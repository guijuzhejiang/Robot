"""Parallel data collection runner.

Spawns N worker processes, each running one env in isolation. Each worker
writes frames into its OWN LeRobot dataset (under a shard repo_id), and the
master process merges shards at the end.

Usage:
    python -m sim.collectors.parallel_runner \\
        --num-episodes 1000 --num-workers 8 \\
        --repo-id local/so101_pickplace_v1

NOTE: For each shard, LeRobotDataset.create() is called; per-process MuJoCo
rendering uses each process's own EGL context (no shared state). On GPU
machines with limited VRAM, reduce num-workers.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue as _queue
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
     img_height, img_width, max_episode_steps,
     progress_queue) = args

    # Suppress noisy ffmpeg / pynvml warnings from each child
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("SVT_LOG", "0")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from sim.envs.pick_place import PickPlaceEnv
    from sim.scripted_policies.pick_place_pipeline import generate_pickplace_episode
    from data.converters.sim_to_lerobot import make_or_resume_dataset

    env = PickPlaceEnv(
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
        from sim.scripted_policies.pick_place import PickPlacePolicy
        from sim.grasp.antipodal import sample_cube_grasps
        policy = PickPlacePolicy()
        policy.reset()
        done = False
        info: dict = {}
        # Buffer task per-frame; we may need to relabel on failure when
        # keep_failures=True (frame-level task field is the only place we
        # can stash the failure mode in LeRobot 0.5.2).
        # PRIMARY label = joint ctrl (LeRobot SO-100/101 convention, what
        # SmolVLA / ACT / lerobot-record real teleop all use).
        # SIM-ONLY auxiliary = ee-delta (the gym action_mode='ee' produces
        # this; useful for sim debugging and future Stage-2 ee-derive tool).
        frame_actions: list[np.ndarray] = []      # joint ctrl, (6,) radians+qpos
        frame_ee_actions: list[np.ndarray] = []   # ee-delta, (4,) normalized [-1, 1]
        frame_obs: list[dict] = []
        while not done:
            # Capture TCP pose BEFORE step so we can re-encode the
            # ee-delta auxiliary as the real world-frame displacement.
            # The scripted oracle returns [0, 0, 0, gripper] because it
            # drives the arm via env.ee_target_override; recording that
            # raw would give a dataset where the ee_action label is zero
            # everywhere. See BaseSoArmEnv.encode_ee_delta_action.
            ee_before = env.ee_pos()
            gym_action = policy(env, obs)
            # NOTE: the legacy xy-tracking override (drive ee_xy to current
            # cube_xy each tick) was removed. The pick-101 oracle now
            # snapshots the cube position once on policy.__call__ and
            # applies a FINGER_WIDTH_OFFSET to centre the cube between the
            # jaws — overriding ee_xy here would erase that offset and the
            # static fingertip would crash into the cube top during DESCEND.
            next_obs, _, term, trunc, info = env.step(gym_action)
            # PRIMARY action: snapshot the actuator command IK + gripper
            # logic just resolved. Absolute joint ctrl targets in physical
            # units (radians for arm, qpos for gripper) — the cleanest
            # joint label, matches what lerobot-record SO-101 produces.
            joint_action = env.data.ctrl[:6].astype(np.float32).copy()
            # Auxiliary ee-delta: re-encoded real world-frame TCP delta
            # (decoupled from the ee_target_override side-channel).
            ee_action = env.encode_ee_delta_action(
                ee_before, env.ee_pos(), gripper_norm=float(gym_action[3])
            )
            frame_obs.append(obs)
            frame_actions.append(joint_action)
            frame_ee_actions.append(ee_action)
            obs = next_obs
            done = term or trunc

        succeeded = bool(info.get("is_success"))
        if succeeded:
            task_label = ep.task
        else:
            mode = info.get("failure_mode") or "unknown"
            task_label = f"{ep.task} [FAIL:{mode}]"

        if succeeded or keep_failures:
            for fo, fa, fea in zip(frame_obs, frame_actions, frame_ee_actions):
                writer.add_frame_from_obs(
                    fo, fa, task=task_label, ee_action=fea
                )
            writer.save_episode()
        # else: drop — nothing was buffered, so no discard needed.

        if succeeded:
            counts["success"] += 1
        else:
            counts["fail"] += 1
            mode = info.get("failure_mode") or "unknown"
            counts["by_mode"][mode] = counts["by_mode"].get(mode, 0) + 1

        # Live progress report — best-effort; never let queue hiccups kill
        # the worker mid-collection (drained progress is cosmetic, the
        # final per-shard `counts` returned below is authoritative).
        if progress_queue is not None:
            try:
                progress_queue.put({
                    "worker_id": worker_id,
                    "succeeded": succeeded,
                    "failure_mode": info.get("failure_mode") if not succeeded else None,
                })
            except Exception:
                pass
    env.close()
    return {"worker_id": worker_id, **counts, "repo_id": repo_id,
            "episodes_saved": writer.num_episodes}


def _prune_staging_images(shard_ids: list[str]) -> dict:
    """Delete the `images/` staging directory inside each shard.

    LeRobot writes per-frame PNG frames to `<shard>/images/` while it
    builds the chunked video files under `<shard>/videos/`. Once the
    mp4s are encoded those PNGs are redundant, but LeRobot occasionally
    leaves some behind (e.g. on the last episode of a chunk). This
    function removes only `images/` from each shard, leaving the
    training-essential `data/` + `videos/` + `meta/` untouched.

    Returns a stats dict counting how many shards were pruned and how
    many bytes were reclaimed.
    """
    base = Path.home() / ".cache" / "huggingface" / "lerobot"
    stats = {"shards_pruned": 0, "shards_skipped_no_images": 0,
             "bytes_freed": 0}

    for shard_id in shard_ids:
        images_dir = base / shard_id / "images"
        if not images_dir.exists():
            stats["shards_skipped_no_images"] += 1
            continue
        freed = 0
        for p in images_dir.rglob("*"):
            if p.is_file():
                try:
                    freed += p.stat().st_size
                except OSError:
                    pass
        shutil.rmtree(images_dir)
        stats["bytes_freed"] += freed
        stats["shards_pruned"] += 1

    return stats


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

    # ProcessPoolExecutor with mp_context=spawn — workers are NOT daemonic,
    # so LeRobot's internal video-encoding subprocesses can fork freely.
    ctx = mp.get_context("spawn")
    # Manager.Queue() so workers can post live progress events that the
    # main process drains into a tqdm bar. Drop to None and the workers
    # simply skip reporting (everything still works, just no live UI).
    manager = ctx.Manager()
    progress_q = manager.Queue()

    tasks = []
    seed_offset = 0
    for w in range(num_workers):
        n = per_worker + (1 if w < remainder else 0)
        shard_id = f"{repo_id}_shard{w:02d}"
        tasks.append((w, n, seed_offset, shard_id, instruction_pool,
                      randomize_domain, keep_failures,
                      img_height, img_width, max_episode_steps,
                      progress_q))
        seed_offset += n

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    t0 = time.time()
    succ_live = 0
    fail_live = 0
    by_mode_live: dict[str, int] = {}

    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool:
        futures = [pool.submit(_worker, t) for t in tasks]

        bar = None
        if tqdm is not None:
            bar = tqdm(total=num_episodes,
                       desc=f"Collecting {repo_id}",
                       unit="ep", smoothing=0.05, dynamic_ncols=True)

        done_eps = 0
        try:
            while done_eps < num_episodes:
                try:
                    evt = progress_q.get(timeout=1.0)
                except _queue.Empty:
                    # Re-raise any worker exception immediately so the user
                    # sees the traceback instead of hanging on a dead pool.
                    for f in futures:
                        if f.done():
                            exc = f.exception()
                            if exc is not None:
                                raise exc
                    if all(f.done() for f in futures):
                        # Pool drained; if total reported < expected just exit.
                        break
                    continue

                done_eps += 1
                if evt.get("succeeded"):
                    succ_live += 1
                else:
                    fail_live += 1
                    mode = evt.get("failure_mode") or "unknown"
                    by_mode_live[mode] = by_mode_live.get(mode, 0) + 1

                if bar is not None:
                    bar.update(1)
                    bar.set_postfix(
                        succ=succ_live, fail=fail_live,
                        rate=f"{succ_live * 100 / max(done_eps, 1):.0f}%",
                    )
                elif done_eps % 5 == 0 or done_eps == num_episodes:
                    el = time.time() - t0
                    ep_rate = done_eps / max(el, 1e-6)
                    eta = (num_episodes - done_eps) / max(ep_rate, 1e-6)
                    print(f"[{done_eps}/{num_episodes}] succ={succ_live} "
                          f"fail={fail_live} ({ep_rate:.2f} ep/s, ETA {eta:.0f}s)",
                          flush=True)
        finally:
            if bar is not None:
                bar.close()

        results = [f.result() for f in futures]

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
        cleanup_stats = _prune_staging_images([r["repo_id"] for r in results])
        cleanup_stats["mb_freed"] = round(cleanup_stats["bytes_freed"] / (1024 * 1024), 1)
        summary["cleanup"] = {
            **cleanup_stats,
            "note": ("Removed each shard's images/ staging dir. "
                     "Shards still hold the training-essential data/ + videos/ + meta/. "
                     "Next step: python -m data.converters.merge_shards "
                     f"--shard-glob '{repo_id}_shard*' --output-repo {repo_id}"),
        }

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-episodes", type=int, default=20)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--repo-id", type=str, default="local/so101_pickplace_v1")
    ap.add_argument("--instructions", type=str, default="data/instructions/pick_place.txt")
    ap.add_argument("--no-dr", action="store_true", help="Disable domain randomization")
    ap.add_argument(
        "--keep-failures", action="store_true",
        help="Save failed episodes too. Their `task` field is suffixed with "
             "[FAIL:<mode>] so audit/training can filter them out. Useful for "
             "debugging the scripted policy by watching the videos.",
    )
    ap.add_argument(
        "--cleanup-after-collect", action="store_true",
        help="After collection, delete the `images/` staging PNG dir from "
             "each shard (LeRobot leaves these as encoder leftovers). "
             "Keeps the training-essential data/ + videos/ + meta/ intact, "
             "so audit_dataset and `data.converters.merge_shards` still work. "
             "Typical disk savings: 100s of MB to several GB on large runs.",
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
