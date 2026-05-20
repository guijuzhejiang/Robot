"""Sim → LeRobot dataset converter.

Usage:
    writer = make_or_resume_dataset(repo_id="local/so101_pickplace_v0", fps=30)
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


_SO101_MOTOR_NAMES = [
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
]


def build_features(
    *,
    img_height: int = 480,
    img_width: int = 640,
    extra: dict[str, tuple[int, ...]] | None = None,
) -> dict:
    """Default LeRobot feature schema for SO-101 PickPlaceRed.

    Conforms to the LeRobot SO-100/101 community convention used by SmolVLA
    pretraining and by `lerobot-record` real-robot teleop: the `action`
    column holds 6-dim joint position targets (radians for arm + qpos for
    gripper), motor names suffixed with `.pos` to match the leader-arm
    `get_action()` output format.

    The auxiliary `ee_action` column (4-dim ee-delta + gripper, normalized
    [-1, 1]) is a sim-only debugging artifact — it has no real-teleop
    counterpart and is NOT used by mainstream pretrained VLAs. It is
    retained for: (a) sim trajectory audit / visualization, (b) future
    derive-tool that converts to ee-mode formats (OpenVLA / X-VLA style)
    via FK from `observation.state`.

    Camera height/width are parameterised so callers can request 720p /
    1080p captures without forking the schema; LeRobot validates frame
    shape on every add_frame, so this MUST match what env._compute_obs
    produces.

    ``extra`` registers caller-supplied float32 columns by name → shape, e.g.
    ``{"red_pose": (7,), "plate_pose": (7,)}``. Used by sim_teleop to ship
    per-frame object poses so MimicGen ``get_object_poses()`` can pull them
    out at conversion time. These columns are NOT consumed by mainstream
    VLAs (which only see ``observation.*`` and ``action``) so they cost
    storage but not training compatibility.
    """
    feats = {
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
            "names": _SO101_MOTOR_NAMES,
        },
        # PRIMARY action label. 6-dim joint position targets — directly
        # compatible with SmolVLA / ACT / Diffusion Policy / Pi-0 family
        # pretraining; matches what `lerobot-record` SO-101 teleop produces.
        # Recorded from `env.data.ctrl[:6]` snapshot post-step. Units:
        # radians for 5 arm joints + qpos for gripper. NOT normalized
        # (LeRobot stats.json computes mean/std automatically).
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": _SO101_MOTOR_NAMES,
        },
        # SIM-ONLY auxiliary label. 4-dim ee-delta + gripper, normalized
        # [-1, 1] (×EE_DELTA_SCALE=0.05m → meters). Not produced by real
        # teleop and not directly compatible with any major pretrained VLA.
        # Useful for: (a) sim audit, (b) Stage-2 derive tool that expands
        # to ee-pose / ee-delta-with-rotation formats for OpenVLA / X-VLA
        # experiments via FK from observation.state.
        "ee_action": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["dx", "dy", "dz", "gripper"],
        },
    }
    if extra:
        for name, shape in extra.items():
            feats[name] = {
                "dtype": "float32",
                "shape": tuple(shape),
                "names": [f"{name}_{i}" for i in range(int(np.prod(shape)))],
            }
    return feats


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
    extra_features: dict[str, tuple[int, ...]] | None = None,
) -> "DatasetWriter":
    """Open or resume a LeRobot dataset on disk.

    If `reset=True`, removes existing directory and recreates.
    Detects corrupt resumes (info.json without tasks.parquet) and rebuilds.

    ``img_height`` / ``img_width`` only matter when CREATING a new dataset;
    they're ignored when resuming an existing one (the existing meta wins).
    """
    if features is None:
        features = build_features(img_height=img_height, img_width=img_width,
                                  extra=extra_features)
    elif extra_features:
        for name, shape in extra_features.items():
            features.setdefault(name, {
                "dtype": "float32",
                "shape": tuple(shape),
                "names": [f"{name}_{i}" for i in range(int(np.prod(shape)))],
            })
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
        ee_action: np.ndarray | None = None,
        extra: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Write one frame.

        `action`     — PRIMARY label, shape (6,), joint position targets
                       (radians for 5 arm joints + qpos for gripper). This
                       is the column SmolVLA / ACT / Diffusion Policy and
                       LeRobot real-robot teleop all expect.
        `ee_action`  — SIM-ONLY auxiliary, shape (4,), ee-delta normalized
                       [-1, 1] + gripper. If None, falls back to zeros —
                       acceptable for any caller that doesn't compute it.
        `extra`      — caller-supplied columns matching the names registered
                       via ``extra_features`` at dataset creation time (e.g.
                       per-frame red/plate pose for downstream MimicGen).
        """
        if ee_action is None:
            ee_action = np.zeros(4, dtype=np.float32)
        frame = {
            "observation.images.front": obs["image_front"],
            "observation.images.wrist": obs["image_wrist"],
            "observation.state": obs["arm_qpos"].astype("float32"),
            "action": np.asarray(action, dtype=np.float32),
            "ee_action": np.asarray(ee_action, dtype=np.float32),
            "task": task,
        }
        if extra:
            for name, value in extra.items():
                frame[name] = np.asarray(value, dtype=np.float32)
        self.ds.add_frame(frame)

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
