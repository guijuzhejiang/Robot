"""Shared dataclasses for the MimicGen adapter."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

SegmentName = Literal["approach_blue", "grasp", "transport", "place_release"]
AnchorName = Literal["blue", "plate", None]


@dataclass
class ObjectPose:
    """2D + yaw object pose in robot frame (z assumed = object half-height on table)."""
    name: str               # "red", "blue", "plate"
    xy: np.ndarray          # shape (2,)
    yaw: float = 0.0        # radians, rotation about world z
    z: float | None = None  # explicit z if known (else assumed = half-extent)

    def to_xyz(self) -> np.ndarray:
        return np.array([self.xy[0], self.xy[1], self.z if self.z is not None else 0.0])


@dataclass
class Frame:
    """One time-step worth of demo data, world-frame ee + obj poses."""
    t: int                   # frame index within the episode
    ee_pos: np.ndarray       # (3,) world frame
    ee_quat: np.ndarray      # (4,) wxyz (often identity for SO101 5-DoF position-only)
    gripper: float           # ctrl value or qpos value (low = closed)
    objects: dict[str, ObjectPose] = field(default_factory=dict)


@dataclass
class Segment:
    """A contiguous slice of frames with an associated anchor object."""
    name: SegmentName
    anchor: AnchorName      # which object's pose the segment is "expressed relative to"
    frames: list[Frame]
    t_start: int
    t_end: int              # inclusive

    def __len__(self) -> int:
        return len(self.frames)

    def first(self) -> Frame:
        return self.frames[0]

    def last(self) -> Frame:
        return self.frames[-1]


@dataclass
class SegmentedDemo:
    """A real demo split into [approach_blue, grasp, transport, place_release]."""
    task: str
    fps: int
    segments: list[Segment]
    initial_objects: dict[str, ObjectPose]  # snapshot at t=0

    def segment_named(self, name: SegmentName) -> Segment | None:
        for s in self.segments:
            if s.name == name:
                return s
        return None
