"""Multi-anchor replay of a SegmentedDemo in the PickPlaceBlue sim.

For each segment we transform the segment's ee waypoints according to the
anchor object's new pose, then drive the env via ee-mode action to track each
target. Outputs a list of (obs, action) tuples plus a success flag.

Transformations:

  approach_blue / grasp  (anchor=blue):
      T_new = T_new_blue_xy_yaw  @  inv(T_old_blue_xy_yaw)
      apply to ee xy (z preserved); yaw of blue rotates ee xy about blue center.

  transport (no anchor):
      interpolate between (last frame of grasp under new transform) and
      (first frame of place_release under new transform). Lift to TRANSPORT_Z
      to clear red cube + table.

  place_release (anchor=plate):
      T_new = T_new_plate_xy_yaw  @  inv(T_old_plate_xy_yaw)
      apply to ee xy.

Red cube avoidance: during transport, if any interpolated ee waypoint passes
within RED_AVOID_RADIUS of new red position, raise that waypoint's z by
RED_AVOID_LIFT (or fail the segment after MAX_RAISES).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.mimicgen_adapter.types import ObjectPose, Segment, SegmentedDemo


TRANSPORT_Z = 0.17        # cruise altitude over transport
RED_AVOID_RADIUS = 0.06   # m
RED_AVOID_LIFT = 0.04     # extra lift per raise
MAX_TRANSPORT_INTERP = 30 # max interpolation steps for transport bridge


@dataclass
class ReplayResult:
    actions: list[np.ndarray]
    success: bool
    failure_mode: str | None = None
    final_obs: dict | None = None


def _yaw_to_R(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _transform_xy_under_anchor(
    xy: np.ndarray,
    old_anchor: ObjectPose,
    new_anchor: ObjectPose,
) -> np.ndarray:
    """Map a point's xy under (translation, yaw) transform of the anchor."""
    # local = R_old^-1 (xy - old_anchor.xy)
    R_old_inv = _yaw_to_R(-old_anchor.yaw)
    local = R_old_inv @ (xy - old_anchor.xy)
    # global = R_new local + new_anchor.xy
    R_new = _yaw_to_R(new_anchor.yaw)
    return R_new @ local + new_anchor.xy


def _segment_ee_targets(
    seg: Segment,
    old_anchor: ObjectPose | None,
    new_anchor: ObjectPose | None,
) -> np.ndarray:
    """Return (T, 3) array of transformed ee positions for this segment."""
    pts = np.stack([f.ee_pos for f in seg.frames])  # (T, 3)
    if old_anchor is None or new_anchor is None:
        return pts  # transport — keep raw
    out = pts.copy()
    for i in range(len(pts)):
        out[i, :2] = _transform_xy_under_anchor(pts[i, :2], old_anchor, new_anchor)
    return out


def _avoid_red(
    points: np.ndarray,
    red_xy: np.ndarray,
    *,
    radius: float = RED_AVOID_RADIUS,
    lift: float = RED_AVOID_LIFT,
) -> np.ndarray:
    """If any point xy is within `radius` of red, raise that point's z by `lift`.
    Returns a new array (does not mutate)."""
    out = points.copy()
    for i in range(len(out)):
        d = np.linalg.norm(out[i, :2] - red_xy)
        if d < radius:
            out[i, 2] += lift * (1.0 - d / radius)
    return out


def _interp_transport(
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    n: int,
    *,
    cruise_z: float = TRANSPORT_Z,
    red_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Build N waypoints between start and end with a cruise-z plateau in the
    middle (so transport flies above the table)."""
    n = max(3, min(n, MAX_TRANSPORT_INTERP))
    ramp = max(1, n // 4)
    wps = np.zeros((n, 3))
    # First ramp: rise to cruise z
    for i in range(ramp):
        a = (i + 1) / ramp
        wps[i] = start_xyz * (1 - a) + np.array([start_xyz[0], start_xyz[1], cruise_z]) * a
    # Middle cruise: interpolate xy at cruise z
    mid_n = n - 2 * ramp
    for j in range(mid_n):
        a = (j + 1) / (mid_n + 1)
        x = (1 - a) * start_xyz[0] + a * end_xyz[0]
        y = (1 - a) * start_xyz[1] + a * end_xyz[1]
        wps[ramp + j] = (x, y, cruise_z)
    # Final ramp: descend to end
    for k in range(ramp):
        a = (k + 1) / ramp
        wps[n - ramp + k] = (
            np.array([end_xyz[0], end_xyz[1], cruise_z]) * (1 - a) + end_xyz * a
        )
    if red_xy is not None:
        wps = _avoid_red(wps, red_xy)
    return wps


def replay_segmented_demo(
    env,
    demo: SegmentedDemo,
    new_scene: dict[str, ObjectPose],
    *,
    max_steps_per_segment: int = 200,
    seed: int = 0,
) -> ReplayResult:
    """Replay a SegmentedDemo into a NEW scene defined by `new_scene`.

    `new_scene` must contain "blue", "plate", and "red" ObjectPoses.
    `env` must be a PickPlaceBlueEnv with action_mode="ee".

    Process:
      1. Reset env, override cube/plate positions with new_scene values.
      2. For each segment: build transformed ee targets, walk env toward each.
      3. Track gripper open/close from demo gripper signal.
      4. Check env's `evaluate_success()` at the end.
    """
    obs, _ = env.reset(seed=seed)
    # Override scene placement
    _place_objects(env, new_scene)
    obs = env._compute_obs()  # refresh after placement

    actions: list[np.ndarray] = []
    info: dict = {}
    failure_mode: str | None = None

    # Resolve anchors
    old_blue = demo.initial_objects.get("blue")
    old_plate = demo.initial_objects.get("plate")
    new_blue = new_scene["blue"]
    new_plate = new_scene["plate"]
    new_red = new_scene["red"]

    # Pre-compute end of grasp (last grasped position) and start of place
    # for transport interpolation
    grasp_seg = demo.segment_named("grasp")
    place_seg = demo.segment_named("place_release")
    if grasp_seg is None or place_seg is None:
        return ReplayResult(actions=[], success=False,
                            failure_mode="missing_segment")
    grasp_end_xyz = _segment_ee_targets(grasp_seg, old_blue, new_blue)[-1]
    place_start_xyz = _segment_ee_targets(place_seg, old_plate, new_plate)[0]

    for seg in demo.segments:
        if seg.name == "transport":
            targets = _interp_transport(
                start_xyz=grasp_end_xyz,
                end_xyz=place_start_xyz,
                n=len(seg.frames),
                red_xy=new_red.xy,
            )
            seg_gripper = np.array([f.gripper for f in seg.frames])
        else:
            if seg.name in {"approach_blue", "grasp"}:
                targets = _segment_ee_targets(seg, old_blue, new_blue)
            else:
                targets = _segment_ee_targets(seg, old_plate, new_plate)
            seg_gripper = np.array([f.gripper for f in seg.frames])

        # Drive env toward each target
        for tgt, g in zip(targets, seg_gripper):
            for _ in range(3):  # up to 3 sub-steps per waypoint
                ee = env.ee_pos()
                delta = tgt - ee
                if np.linalg.norm(delta) < 0.01:
                    break
                action = np.zeros(4, dtype=np.float32)
                action[:3] = np.clip(delta / 0.05, -1.0, 1.0)
                action[3] = 1.0 if g > 0.5 else (-1.0 if g < 0.0 else 0.0)
                obs, _, term, trunc, info = env.step(action)
                actions.append(action.copy())
                if term or trunc:
                    failure_mode = info.get("failure_mode") or "early_term"
                    break
            else:
                continue
            break
        else:
            continue
        break

    success = bool(info.get("is_success", False))
    if not success and failure_mode is None:
        failure_mode = info.get("failure_mode") or "replay_failed"
    return ReplayResult(actions=actions, success=success,
                        failure_mode=None if success else failure_mode,
                        final_obs=obs)


def _place_objects(env, new_scene: dict[str, ObjectPose]) -> None:
    """Forcibly write red/blue/plate poses into env.data.qpos."""
    import mujoco

    for color in ("red", "blue", "plate"):
        if color not in new_scene:
            continue
        pose = new_scene[color]
        if color == "red":
            qadr = env._red_qadr
            z = env.CUBE_HALF
        elif color == "blue":
            qadr = env._blue_qadr
            z = env.CUBE_HALF
        else:
            qadr = env._plate_qadr
            z = env.PLATE_HALF_HEIGHT
        env._write_free_joint(qadr, pose.xy, z=z, yaw=pose.yaw)
        if color == "red":
            env._red_init_xy = pose.xy.copy()
        elif color == "blue":
            env._blue_init_xy = pose.xy.copy()
    mujoco.mj_forward(env.model, env.data)
    # Settle a few steps so cubes/plate rest on table
    for _ in range(20):
        mujoco.mj_step(env.model, env.data)
