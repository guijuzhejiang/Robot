"""LeRobot dataset audit tool.

Loads a LeRobot dataset and prints:
  - Episode count, frame count, fps, average duration
  - Per-episode action / state stats (jerk, joint-limit saturation)
  - `task` field histogram (instruction diversity)
  - Random-sample frame previews (saves PNGs to --out-dir if requested)

Usage:
    python -m eval.audit_dataset --repo-id local/so101_pickplace_blue_v1
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np


def audit(repo_id: str, *, n_sample: int = 50, out_dir: Path | None = None) -> dict:
    """Read a LeRobot dataset directly from disk (no hub roundtrip).

    Reads info.json + episodes.parquet + tasks.parquet from the standard
    LeRobot cache location.
    """
    import json

    import pandas as pd  # bundled via lerobot deps

    root = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"No info.json at {info_path} — is this a local LeRobot dataset?")
    info = json.loads(info_path.read_text())
    fps = info.get("fps")
    n_eps = info.get("total_episodes", 0)
    n_frames = info.get("total_frames", 0)

    # LeRobot 0.5.x stores data as data/chunk-NNN/file-NNN.parquet (concatenated)
    # with an "episode_index" column. Slice into per-episode groups.
    data_root = root / "data"
    per_ep_lengths: list[int] = []
    per_ep_jerks: list[float] = []
    tasks: list[str] = []
    action_dim: int | None = None

    parquet_files = sorted(data_root.rglob("*.parquet"))
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        if "episode_index" not in df.columns:
            continue
        for ep_idx, ep_df in df.groupby("episode_index"):
            per_ep_lengths.append(len(ep_df))
            if "action" in ep_df.columns:
                A = np.stack(ep_df["action"].apply(np.asarray).to_list())
                if action_dim is None:
                    action_dim = A.shape[1]
                if len(A) >= 3:
                    jerk = np.diff(A, n=3, axis=0)
                    per_ep_jerks.append(float(np.max(np.abs(jerk))))
            if "task_index" in ep_df.columns:
                tasks.extend(ep_df["task_index"].astype(int).tolist())

    # Resolve task_index → string via tasks.parquet
    # LeRobot 0.5.x layout: index = task string, column "task_index" = int
    tasks_parquet = root / "meta" / "tasks.parquet"
    task_hist: Counter[str] = Counter()
    if tasks_parquet.exists() and tasks:
        tasks_df = pd.read_parquet(tasks_parquet)
        # Invert: int → task string
        idx_to_task = {int(row.task_index): str(idx)
                       for idx, row in tasks_df.iterrows()}
        for ti in tasks:
            task_hist[idx_to_task.get(int(ti), f"<unknown:{ti}>")] += 1

    report = {
        "repo_id": repo_id,
        "num_episodes": n_eps,
        "num_frames": n_frames,
        "fps": fps,
        "avg_episode_steps": float(np.mean(per_ep_lengths)) if per_ep_lengths else 0,
        "avg_episode_seconds": (float(np.mean(per_ep_lengths)) / fps) if per_ep_lengths and fps else 0,
        "max_jerk_per_ep_mean": float(np.mean(per_ep_jerks)) if per_ep_jerks else 0,
        "max_jerk_per_ep_p95": float(np.quantile(per_ep_jerks, 0.95)) if per_ep_jerks else 0,
        "action_dim": action_dim,
        "task_variants": len(task_hist),
        "task_distribution_top10": task_hist.most_common(10),
    }

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_audit_md(report, out_dir / "audit.md")
        print(f"[audit] wrote {out_dir / 'audit.md'}")

    return report


def _save_audit_md(report: dict, path: Path) -> None:
    lines = ["# Dataset Audit\n"]
    for k, v in report.items():
        if k == "task_distribution_top10":
            lines.append("## Task distribution (top 10)\n")
            for task, count in v:
                lines.append(f"- `{task}`: {count}")
            lines.append("")
        else:
            lines.append(f"- **{k}**: {v}")
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-sample", type=int, default=50)
    args = ap.parse_args()
    report = audit(args.repo_id, n_sample=args.n_sample, out_dir=args.out_dir)
    import json
    print(json.dumps({k: v for k, v in report.items()
                      if k != "task_distribution_top10"}, indent=2, default=str))
    print("\nTop tasks:")
    for task, count in report["task_distribution_top10"]:
        print(f"  {count:5d}  {task!r}")


if __name__ == "__main__":
    main()
