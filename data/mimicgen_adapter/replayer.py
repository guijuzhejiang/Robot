"""Multi-anchor replay of a SegmentedDemo in the PickPlace sim.

For each segment we transform the segment's ee waypoints according to the
anchor object's new pose, then drive the env via ee-mode action to track each
target. Outputs a list of (obs, action) tuples plus a success flag.

Transformations:

  approach / grasp  (anchor=red):
      T_new = T_new_red_xy_yaw  @  inv(T_old_red_xy_yaw)
      apply to ee xy (z preserved); yaw of red rotates ee xy about red center.

  transport (no anchor):
      interpolate between (last frame of grasp under new transform) and
      (first frame of place_release under new transform). Lift to TRANSPORT_Z
      to clear the table.

  place_release (anchor=plate):
      T_new = T_new_plate_xy_yaw  @  inv(T_old_plate_xy_yaw)
      apply to ee xy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.mimicgen_adapter.types import ObjectPose, Segment, SegmentedDemo


TRANSPORT_Z = 0.17        # cruise altitude over transport
MAX_TRANSPORT_INTERP = 30 # max interpolation steps for transport bridge


@dataclass
class ReplayResult:
    # PRIMARY action: 6-dim joint ctrl target per step (env.data.ctrl[:6]
    # post-step). Matches LeRobot SO-100/101 convention.
    actions: list[np.ndarray] = None  # type: ignore[assignment]
    # SIM-ONLY auxiliary: 4-dim ee-delta normalized [-1, 1] + gripper.
    # The gym action consumed by env.step (env.action_mode='ee').
    ee_actions: list[np.ndarray] = None  # type: ignore[assignment]
    success: bool = False
    failure_mode: str | None = None
    final_obs: dict | None = None

    def __post_init__(self):
        if self.actions is None:
            self.actions = []
        if self.ee_actions is None:
            self.ee_actions = []


def _ctrl_to_action_gripper(ctrl_val: float, g_lo: float, g_hi: float) -> float:
    """Invert the env's absolute-gripper mapping.

    Env applies ``ctrl[5] = (a3 + 1) / 2 * (g_hi - g_lo) + g_lo`` when
    ``gripper_action_mode == 'absolute'``. We invert so a recorded ctrl[5]
    from a source demo replays as the original gripper command:
    ``a3 = 2 * (ctrl - g_lo) / (g_hi - g_lo) - 1``.
    """
    return float(np.clip(2.0 * (ctrl_val - g_lo) / (g_hi - g_lo) - 1.0, -1.0, 1.0))


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


def _interp_transport(
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    n: int,
    *,
    cruise_z: float = TRANSPORT_Z,
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

    `new_scene` must contain "red" and "plate" ObjectPoses.
    `env` must be a PickPlaceEnv with action_mode="ee".

    Process:
      1. Reset env, override cube/plate positions with new_scene values.
      2. For each segment: build transformed ee targets, issue one env.step
         per transformed source frame so the replay timing matches source.
      3. Map recorded gripper ctrl back through env's absolute-gripper
         mapping so action[3] reproduces ctrl[5] each step.
      4. Stop early if env terminates (success or failure).
    """
    obs, _ = env.reset(seed=seed)
    # Override scene placement
    _place_objects(env, new_scene)
    obs = env._compute_obs()  # refresh after placement

    # Source demos are collected with absolute gripper ctrl; mirror that here
    # so action[3] inversion below produces the same ctrl[5] in replay.
    env.gripper_action_mode = "absolute"
    g_lo, g_hi = float(env.ctrl_limits[5, 0]), float(env.ctrl_limits[5, 1])

    actions: list[np.ndarray] = []        # primary: joint ctrl
    ee_actions: list[np.ndarray] = []     # auxiliary: ee-delta gym action
    info: dict = {}
    failure_mode: str | None = None

    # Resolve anchors
    old_red = demo.initial_objects.get("red")
    old_plate = demo.initial_objects.get("plate")
    new_red = new_scene["red"]
    new_plate = new_scene["plate"]

    # Pre-compute end of grasp (last grasped position) and start of place
    # for transport interpolation
    grasp_seg = demo.segment_named("grasp")
    place_seg = demo.segment_named("place_release")
    if grasp_seg is None or place_seg is None:
        return ReplayResult(actions=[], success=False,
                            failure_mode="missing_segment")
    grasp_end_xyz = _segment_ee_targets(grasp_seg, old_red, new_red)[-1]
    place_start_xyz = _segment_ee_targets(place_seg, old_plate, new_plate)[0]

    early_term = False
    for seg in demo.segments:
        if seg.name == "transport":
            targets = _interp_transport(
                start_xyz=grasp_end_xyz,
                end_xyz=place_start_xyz,
                n=len(seg.frames),
            )
        elif seg.name in {"approach", "grasp"}:
            targets = _segment_ee_targets(seg, old_red, new_red)
        else:  # place_release
            targets = _segment_ee_targets(seg, old_plate, new_plate)
        seg_gripper = np.array([f.gripper for f in seg.frames])

        # One env.step per source frame: keeps replay timing aligned with
        # the source policy and lets the gripper ctrl command apply on
        # every frame (the previous "skip-if-delta-small" path silently
        # dropped the entire GRASP phase, where ee barely moves but the
        # gripper is ramping closed).
        for tgt, g in zip(targets, seg_gripper):
            ee = env.ee_pos()
            delta = tgt - ee
            # Build the gym ee action (env.action_mode='ee' consumes this).
            # Normalize against the env's ee-delta scale so it can be
            # re-applied later by `env.step()` (delta-mode IK) exactly as
            # the scripted-policy collector records it via
            # `BaseSoArmEnv.encode_ee_delta_action`.
            ee_action = np.zeros(4, dtype=np.float32)
            ee_action[:3] = np.clip(delta / env.EE_DELTA_SCALE, -1.0, 1.0)
            ee_action[3] = _ctrl_to_action_gripper(float(g), g_lo, g_hi)
            obs, _, term, trunc, info = env.step(ee_action)
            # PRIMARY action: joint ctrl snapshot post-step.
            actions.append(env.data.ctrl[:6].astype(np.float32).copy())
            ee_actions.append(ee_action.copy())
            if term or trunc:
                if not info.get("is_success"):
                    failure_mode = info.get("failure_mode") or "early_term"
                early_term = True
                break
        if early_term:
            break

    success = bool(info.get("is_success", False))
    if not success and failure_mode is None:
        failure_mode = info.get("failure_mode") or "replay_failed"
    return ReplayResult(actions=actions, ee_actions=ee_actions,
                        success=success,
                        failure_mode=None if success else failure_mode,
                        final_obs=obs)


def _place_objects(env, new_scene: dict[str, ObjectPose]) -> None:
    """Forcibly write red/plate poses into env.data.qpos.

    Also re-aligns ``wrist_roll`` to the new cube position. The env's
    ``_post_reset`` sets ``wrist_roll = pi/2 - atan2(cube.y, cube.x)`` so
    the gripper's finger-spread axis lands along world +y once
    shoulder_pan swings to face the cube. After we override the cube
    pose here, that alignment becomes stale (still pointing at the
    sampled cube), which is the dominant cause of ``lift_drop`` failures
    in mimicgen replay — the gripper closes on a cube edge instead of
    straddling two parallel faces.
    """
    import mujoco

    for name in ("red", "plate"):
        if name not in new_scene:
            continue
        pose = new_scene[name]
        if name == "red":
            qadr = env._red_qadr
            z = env.CUBE_HALF
        else:
            qadr = env._plate_qadr
            z = env.PLATE_HALF_HEIGHT
        env._write_free_joint(qadr, pose.xy, z=z, yaw=pose.yaw)
        if name == "red":
            env._red_init_xy = pose.xy.copy()
        else:
            env._plate_init_xy = pose.xy.copy()

    if "red" in new_scene:
        red_xy = new_scene["red"].xy
        alpha = float(np.arctan2(red_xy[1], red_xy[0]))
        wrist_roll = float(np.pi / 2 - alpha)
        wrist_roll_lo, wrist_roll_hi = env.ctrl_limits[4]
        wrist_roll = float(np.clip(wrist_roll, wrist_roll_lo, wrist_roll_hi))
        env.wrist_roll_alignment = wrist_roll
        wrist_roll_qadr = env.model.joint("wrist_roll").qposadr[0]
        env.data.qpos[wrist_roll_qadr] = wrist_roll
        env.data.ctrl[4] = wrist_roll

    mujoco.mj_forward(env.model, env.data)
    # Settle a few steps so cube/plate rest on table
    for _ in range(20):
        mujoco.mj_step(env.model, env.data)
