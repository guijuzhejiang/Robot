"""Gripper-event–based subtask segmenter.

Splits a PickPlace episode into 4 segments:

    approach      : t=0  →  gripper_close_t-1     (anchor: red)
    grasp         : gripper_close_t  →  lift_start_t-1  (anchor: red)
    transport     : lift_start_t  →  gripper_open_t-1   (no anchor)
    place_release : gripper_open_t  →  T-1              (anchor: plate)

Boundaries:
  - gripper_close_t: first frame where gripper monotonically crosses below
    the "closed" threshold and stays there for ≥ CLOSE_HOLD frames.
  - lift_start_t   : first frame after grasp where ee_z derivative > LIFT_VEL
    or ee_z exceeds (grasp_z + LIFT_MIN_DELTA).
  - gripper_open_t : first frame in transport-or-later where gripper crosses
    back above OPEN threshold.

Input: list[Frame] (see types.py). Object poses can be missing per-frame (we
use initial_objects from t=0 for anchor mapping).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from data.mimicgen_adapter.types import Frame, Segment, SegmentedDemo


CLOSE_THR = 0.3     # gripper ctrl < this = "closed". The scripted policy's
                    # absolute-gripper mapping ramps env.data.ctrl[5] from ~1.07
                    # (open) down to ~0.02 (closed); 0.3 is the midpoint with
                    # margin so brief overshoots don't false-trigger.
OPEN_THR = 0.8      # > this = "open" (open state holds at ~1.07–1.17)
CLOSE_HOLD = 3      # consecutive frames required to confirm close event
LIFT_VEL = 0.005    # m/frame
LIFT_MIN_DELTA = 0.02  # m above grasp height


def find_gripper_close(frames: list[Frame]) -> int | None:
    """Return frame index of first sustained gripper close."""
    streak = 0
    for i, f in enumerate(frames):
        if f.gripper < CLOSE_THR:
            streak += 1
            if streak >= CLOSE_HOLD:
                return i - CLOSE_HOLD + 1
        else:
            streak = 0
    return None


def find_gripper_open(frames: list[Frame], after: int) -> int | None:
    """First frame >= `after` where gripper opens (> OPEN_THR)."""
    for i in range(after, len(frames)):
        if frames[i].gripper > OPEN_THR:
            return i
    return None


def find_lift_start(frames: list[Frame], grasp_t: int) -> int | None:
    """First frame after grasp_t where ee rises significantly."""
    grasp_z = frames[grasp_t].ee_pos[2]
    for i in range(grasp_t + 1, len(frames)):
        dz = frames[i].ee_pos[2] - grasp_z
        if dz > LIFT_MIN_DELTA:
            return i
        # also accept large +z velocity
        if i >= grasp_t + 2:
            vz = (frames[i].ee_pos[2] - frames[i - 2].ee_pos[2]) / 2.0
            if vz > LIFT_VEL:
                return i
    return None


def segment(
    frames: list[Frame],
    *,
    task: str = "put the red cube on the plate",
    fps: int = 30,
) -> SegmentedDemo | None:
    """Run the 4-way segmentation. Returns None if any boundary not found.

    `frames[0].objects` must contain at least 'red' and 'plate' poses for
    anchor resolution; if not, the caller should run object_tracker first to
    fill them in.
    """
    if len(frames) < 10:
        return None

    grasp_t = find_gripper_close(frames)
    if grasp_t is None:
        return None
    lift_t = find_lift_start(frames, grasp_t)
    if lift_t is None:
        return None
    open_t = find_gripper_open(frames, after=lift_t)
    if open_t is None:
        return None
    T = len(frames)

    if not (0 < grasp_t < lift_t < open_t < T):
        return None

    segs: list[Segment] = [
        Segment(name="approach", anchor="red",
                frames=frames[:grasp_t], t_start=0, t_end=grasp_t - 1),
        Segment(name="grasp", anchor="red",
                frames=frames[grasp_t:lift_t], t_start=grasp_t, t_end=lift_t - 1),
        Segment(name="transport", anchor=None,
                frames=frames[lift_t:open_t], t_start=lift_t, t_end=open_t - 1),
        Segment(name="place_release", anchor="plate",
                frames=frames[open_t:], t_start=open_t, t_end=T - 1),
    ]
    initial = dict(frames[0].objects) if frames[0].objects else {}
    return SegmentedDemo(task=task, fps=fps, segments=segs, initial_objects=initial)


def episode_from_lerobot_dataset(repo_id: str, episode_index: int) -> list[Frame] | None:
    """Load a single episode from a LeRobotDataset (local cache) into list[Frame].

    Reads ee position from `observation.state` (assumes joint qpos was logged;
    for ee mode we'd need a forward-kinematics pass — here we approximate by
    using the action's xyz delta integrated, plus initial ee from observation).
    For datasets logged in ee-action mode, use `episode_from_ee_actions` instead.
    """
    raise NotImplementedError(
        "For LeRobotDataset → Frame conversion, prefer `episode_from_ee_actions`. "
        "Direct loading from joint qpos requires a FK helper we haven't shipped."
    )


def episode_from_ee_actions(
    actions: np.ndarray,
    ee_positions: np.ndarray,
    gripper_states: np.ndarray,
    *,
    initial_objects: dict | None = None,
) -> list[Frame]:
    """Build a list[Frame] from per-step arrays (used by smoke tests + sim
    replays). Useful for unit-testing the segmenter."""
    T = len(actions)
    assert ee_positions.shape == (T, 3), ee_positions.shape
    assert gripper_states.shape == (T,), gripper_states.shape
    identity_quat = np.array([1.0, 0.0, 0.0, 0.0])
    frames = []
    for t in range(T):
        frames.append(Frame(
            t=t,
            ee_pos=ee_positions[t].copy(),
            ee_quat=identity_quat.copy(),
            gripper=float(gripper_states[t]),
            objects=initial_objects if t == 0 and initial_objects else {},
        ))
    return frames
