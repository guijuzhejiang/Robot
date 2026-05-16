"""Sim → LeRobot dataset converter.

Usage:
    writer = make_or_resume_dataset(repo_id="local/so101_pickplace_blue_v0", fps=30)
    for ep in range(N):
        obs, _ = env.reset(seed=ep)
        policy.reset()
        done = False
        while not done:
            action = policy(env, obs)
            next_obs, _, term, trunc, info = env.step(action)
            writer.add_frame_from_obs(obs, action, task="put the blue cube on the plate")
            obs = next_obs
            done = term or trunc
        if info.get("is_success"):
            writer.save_episode()
        else:
            writer.discard_episode()
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def build_features(*, img_height: int = 480, img_width: int = 640) -> dict:
    """Default LeRobot feature schema for PickPlaceBlue (action_mode='ee').

    Camera height/width are parameterised so callers can request 720p / 1080p
    captures without forking the schema; LeRobot validates frame shape on
    every add_frame, so this MUST match what env._compute_obs produces.
    """
    return {
        "observation.images.front": {
            "dtype": "video",
            "shape": (img_height, img_width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (img_height, img_width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": [f"q{i}" for i in range(6)],
        },
        "action": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["dx", "dy", "dz", "gripper"],
        },
    }


# Back-compat alias — prefer build_features() so resolution travels with the call.
DEFAULT_FEATURES = build_features()


def make_or_resume_dataset(
    repo_id: str,
    fps: int = 30,
    features: dict | None = None,
    *,
    reset: bool = False,
    img_height: int = 480,
    img_width: int = 640,
) -> "DatasetWriter":
    """Open or resume a LeRobot dataset on disk.

    If `reset=True`, removes existing directory and recreates.
    Detects corrupt resumes (info.json without tasks.parquet) and rebuilds.

    ``img_height`` / ``img_width`` only matter when CREATING a new dataset;
    they're ignored when resuming an existing one (the existing meta wins).
    """
    if features is None:
        features = build_features(img_height=img_height, img_width=img_width)
    root = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
    meta_tasks = root / "meta" / "tasks.parquet"

    if reset and root.exists():
        print(f"[sim_to_lerobot] RESET=True, removing {root}")
        shutil.rmtree(root)
    elif root.exists() and not meta_tasks.exists():
        print(f"[sim_to_lerobot] WARN: corrupted dataset at {root} (no tasks.parquet). Removing.")
        shutil.rmtree(root)

    if root.exists():
        ds = LeRobotDataset(repo_id)
        print(f"[sim_to_lerobot] Resumed: {ds.num_episodes} episodes, {ds.num_frames} frames")
    else:
        ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features)
        print(f"[sim_to_lerobot] Created new dataset at {root}")
    return DatasetWriter(ds)


class DatasetWriter:
    """Thin wrapper around LeRobotDataset for ergonomic per-frame writing.

    LeRobot 0.5.2's add_frame() requires a flat dict including the `task` field;
    save_episode() commits the buffered frames; clear_episode_buffer() drops them.
    """

    def __init__(self, dataset: LeRobotDataset):
        self.ds = dataset

    def add_frame_from_obs(
        self,
        obs: dict,
        action: np.ndarray,
        *,
        task: str,
    ) -> None:
        self.ds.add_frame({
            "observation.images.front": obs["image_front"],
            "observation.images.wrist": obs["image_wrist"],
            "observation.state": obs["arm_qpos"].astype("float32"),
            "action": np.asarray(action, dtype=np.float32),
            "task": task,
        })

    def save_episode(self) -> None:
        self.ds.save_episode()

    def discard_episode(self) -> None:
        self.ds.clear_episode_buffer()

    @property
    def num_episodes(self) -> int:
        return self.ds.num_episodes

    @property
    def num_frames(self) -> int:
        return self.ds.num_frames
