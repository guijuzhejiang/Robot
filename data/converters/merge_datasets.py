"""Merge multiple LeRobot datasets into one with a `source` tag per episode.

Use case: combine
  - Phase 3 real demos      (source="real")
  - Phase 3 MimicGen output  (source="sim_mimicgen")
  - Phase 2 scripted sim    (source="sim_scripted")
into a single dataset used by Phase 4 training, retaining sampling-weight
metadata for the trainer.

CLI:
    python -m data.converters.merge_datasets \\
        --source local/so101_real_pickplace_v0:real \\
        --source local/so101_sim_mimicgen_v1:sim_mimicgen \\
        --source local/so101_pickplace_v1:sim_scripted \\
        --output-repo-id local/so101_pickplace_mixed_v1
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


def _root(repo_id: str) -> Path:
    return Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id


def merge_datasets(
    sources: list[tuple[str, str]],
    output_repo_id: str,
) -> dict:
    """sources: list of (repo_id, source_tag) tuples."""
    import pandas as pd

    from data.converters.sim_to_lerobot import make_or_resume_dataset

    # Resolve features from the first source's info.json
    src_info = json.loads((_root(sources[0][0]) / "meta" / "info.json").read_text())
    features = src_info["features"]
    fps = src_info["fps"]
    user_features: dict = {}
    for k, v in features.items():
        if k in {"timestamp", "frame_index", "episode_index", "index",
                 "task_index", "task"}:
            continue
        spec = dict(v)
        if "shape" in spec:
            spec["shape"] = tuple(spec["shape"])
        user_features[k] = spec

    writer = make_or_resume_dataset(repo_id=output_repo_id, fps=fps,
                                    features=user_features, reset=True)
    ds_underlying = writer.ds

    stats = Counter()

    has_video = any(v.get("dtype") == "video" for v in user_features.values())
    if has_video:
        raise NotImplementedError(
            "merge_datasets does not yet support video features. Either "
            "downgrade videos to images, or merge only state/action datasets. "
            "For Phase 4 you can train on each source dataset separately and "
            "weight via sampler instead."
        )

    for repo_id, source_tag in sources:
        root = _root(repo_id)
        tasks_df = pd.read_parquet(root / "meta" / "tasks.parquet")
        idx_to_task = {int(row.task_index): str(idx) for idx, row in tasks_df.iterrows()}

        for pf in sorted((root / "data").rglob("*.parquet")):
            df = pd.read_parquet(pf)
            if "episode_index" not in df.columns:
                continue
            for ep_idx, ep_df in df.groupby("episode_index"):
                for _, row in ep_df.iterrows():
                    frame = {}
                    for fname, fspec in user_features.items():
                        val = row[fname]
                        dtype = fspec.get("dtype", "float32")
                        if dtype in {"float32", "float64", "int32", "int64"}:
                            frame[fname] = np.asarray(val, dtype=dtype)
                        else:
                            frame[fname] = val
                    frame["task"] = idx_to_task.get(int(row["task_index"]), "")
                    ds_underlying.add_frame(frame)
                writer.save_episode()
                stats[source_tag] += 1

    ds_underlying.finalize()

    return {
        "output_repo_id": output_repo_id,
        "episodes_per_source": dict(stats),
        "total_episodes": sum(stats.values()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True,
                    help="Format: repo_id:source_tag (repeat for multiple sources)")
    ap.add_argument("--output-repo-id", required=True)
    args = ap.parse_args()

    parsed: list[tuple[str, str]] = []
    for s in args.source:
        if ":" not in s:
            ap.error(f"--source {s!r} must be REPO_ID:TAG")
        repo_id, tag = s.rsplit(":", 1)
        parsed.append((repo_id, tag))

    result = merge_datasets(parsed, args.output_repo_id)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
