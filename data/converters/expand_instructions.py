"""Augment a LeRobot dataset by adding language-variant copies of each episode.

Strategy: for each existing episode in the source dataset, write K duplicate
episodes into the output dataset, each with a different `task` field sampled
from the instruction pool. Frame data is identical (no re-rendering needed).

This is a CHEAP way to multiply language diversity without re-running sim.

CLI:
    python -m data.converters.expand_instructions \\
        --source-repo-id local/so101_real_pickplace_v0 \\
        --output-repo-id local/so101_real_pickplace_v0_langx3 \\
        --instructions data/instructions/pick_place.txt \\
        --copies 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _root(repo_id: str) -> Path:
    return Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id


def expand_instructions(
    source_repo_id: str,
    output_repo_id: str,
    *,
    instructions: list[str],
    copies: int = 3,
    seed: int = 0,
) -> dict:
    import pandas as pd

    from data.converters.sim_to_lerobot import make_or_resume_dataset

    src_root = _root(source_repo_id)
    info = json.loads((src_root / "meta" / "info.json").read_text())
    features = info["features"]
    fps = info["fps"]
    user_features = {}
    for k, v in features.items():
        if k in {"timestamp", "frame_index", "episode_index", "index",
                 "task_index", "task"}:
            continue
        # info.json stores shape as list; LeRobot validator compares against
        # array.shape which is a tuple — coerce to tuple here.
        spec = dict(v)
        if "shape" in spec:
            spec["shape"] = tuple(spec["shape"])
        user_features[k] = spec

    has_video = any(f.get("dtype") == "video" for f in user_features.values())
    if has_video:
        raise NotImplementedError(
            "expand_instructions cannot duplicate video features in-place. "
            "Either (a) downgrade videos to image features before merging, or "
            "(b) only run this on state/action datasets, or "
            "(c) extend this helper to symlink underlying mp4 chunks."
        )

    writer = make_or_resume_dataset(repo_id=output_repo_id, fps=fps,
                                    features=user_features, reset=True)
    rng = np.random.default_rng(seed)
    eps_written = 0

    for pf in sorted((src_root / "data").rglob("*.parquet")):
        df = pd.read_parquet(pf)
        if "episode_index" not in df.columns:
            continue
        for ep_idx, ep_df in df.groupby("episode_index"):
            chosen_tasks = rng.choice(instructions, size=copies, replace=False
                                      if copies <= len(instructions) else True)
            for new_task in chosen_tasks:
                for _, row in ep_df.iterrows():
                    frame = {}
                    for fname, fspec in user_features.items():
                        val = row[fname]
                        # Parquet may store arrays as Python lists; coerce to np
                        # with the feature's declared dtype.
                        dtype = fspec.get("dtype", "float32")
                        if dtype in {"float32", "float64", "int32", "int64"}:
                            frame[fname] = np.asarray(val, dtype=dtype)
                        else:
                            frame[fname] = val
                    frame["task"] = str(new_task)
                    writer.ds.add_frame(frame)
                writer.save_episode()
                eps_written += 1

    writer.ds.finalize()

    return {
        "output_repo_id": output_repo_id,
        "input_episodes": info.get("total_episodes"),
        "output_episodes": eps_written,
        "copies_per_input_episode": copies,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-repo-id", required=True)
    ap.add_argument("--output-repo-id", required=True)
    ap.add_argument("--instructions", default="data/instructions/pick_place.txt")
    ap.add_argument("--copies", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    instr = [ln.strip() for ln in Path(args.instructions).read_text().splitlines() if ln.strip()]
    result = expand_instructions(args.source_repo_id, args.output_repo_id,
                                 instructions=instr, copies=args.copies, seed=args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
