"""Thin Gymnasium-compatible base for SO-ARM MuJoCo envs.

Subclasses override:
  - SCENE_PATH (str)
  - _build_observation_space() / _build_action_space()
  - _post_reset(self.np_random)   # randomize objects, set keyframe pose
  - _compute_obs() -> dict
  - _apply_action(action)         # write to data.ctrl
  - _check_done() -> (terminated, truncated, info)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np


class BaseSoArmEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    SCENE_PATH: str = ""  # subclass must set
    HOME_KEYFRAME: str = "home"
    DEFAULT_CAMERA: str = "front"
    # Default render resolution. Front camera is defined at 640x480 in
    # the scene XML; rendering at that native size avoids downscaling
    # blur. Override per-instance via the constructor kwargs.
    IMG_HEIGHT: int = 480
    IMG_WIDTH: int = 640
    SIM_STEPS_PER_CTRL: int = 5  # 0.002s * 5 = 10ms per ctrl step (100 Hz)
    # World-frame meters per unit of normalized ee-delta action. action[:3] ∈
    # [-1, 1] maps to ±EE_DELTA_SCALE m of gripperframe (TCP) translation per
    # env step. Used by `_apply_ee_action` to drive the IK target AND by
    # `encode_ee_delta_action` to record real ee motion as a training label
    # under the same scale (so the dataset's action column round-trips
    # through `_apply_ee_action` at deploy time).
    EE_DELTA_SCALE: float = 0.05

    def __init__(
        self,
        *,
        observation_mode: str = "both",
        action_mode: str = "ee",
        render_mode: str | None = None,
        max_episode_steps: int = 200,
        seed: int | None = None,
        img_height: int | None = None,
        img_width: int | None = None,
    ):
        if not self.SCENE_PATH:
            raise ValueError("Subclass must set SCENE_PATH")
        if observation_mode not in {"state", "image", "both"}:
            raise ValueError(observation_mode)
        if action_mode not in {"joint", "ee"}:
            raise ValueError(action_mode)

        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        # Per-instance render resolution overrides the class-level default
        # (so a single env class can produce SD videos for fast iteration
        # and HD videos for the final batch without subclassing).
        if img_height is not None:
            self.IMG_HEIGHT = int(img_height)
        if img_width is not None:
            self.IMG_WIDTH = int(img_width)

        self.model = mujoco.MjModel.from_xml_path(self.SCENE_PATH)
        self.data = mujoco.MjData(self.model)
        self.np_random = np.random.default_rng(seed)
        self._step_count = 0
        self._renderer: mujoco.Renderer | None = None
        self._viewer = None
        # Joints (0-4) the scripted policy wants the IK to LEAVE ALONE. The
        # canonical use is `locked_joints=[4]` so wrist_roll stays at its
        # home value while DLS IK adjusts the other four joints. Mirrors
        # github.com/ggand0/pick-101 `test_topdown_pick.py`.
        self.locked_joints: list[int] | None = None
        # If >0, the IK adds an auxiliary task pulling ee+z toward world-z so
        # the gripper stays pointed at the table during manipulation. 0 means
        # position-only (legacy mink-like behavior).
        self.down_orientation_weight: float = 0.0
        # Gripper interpretation for ee-mode `action[3]`:
        #   "delta"    — action[3] * 0.2 added to ctrl[5] each step (legacy,
        #                smooth for RL).
        #   "absolute" — action[3] in [-1, 1] linearly maps to ctrl[5]'s
        #                full ctrlrange every step (pick-101 IKController
        #                style; required by the scripted oracle's gradual
        #                close + contact-detection tightening).
        self.gripper_action_mode: str = "delta"
        # Cached IK target + gripper ctrl set by `_apply_ee_action` and
        # re-applied every mj_step inside `step()` (mirrors pick-101's 500 Hz
        # IK loop).
        self._ee_target: np.ndarray | None = None
        self._ee_gripper_ctrl: float | None = None
        # Optional ABSOLUTE world-frame target for the gripperframe site.
        # When set (e.g. by a scripted oracle that knows the full goal pos),
        # env.step() ignores action[:3] and holds this target fixed across
        # all SIM_STEPS_PER_CTRL substeps — letting the IK gain decay it
        # exponentially toward the target, exactly the way pick-101's
        # test_topdown_pick.py drives its 500 Hz IK loop. Stays in effect
        # until cleared (None).
        self.ee_target_override: np.ndarray | None = None
        # Optional ABSOLUTE gripper action ([-1, 1], pick-101 IKController
        # mapping). When set, ignores action[3].
        self.gripper_action_override: float | None = None

        # cache joint indices for the 6 actuators
        self.n_arm_joints = self.model.nu  # 6 = 5 arm + 1 gripper
        self.ctrl_limits = self.model.actuator_ctrlrange.copy()  # (6, 2)

        # action / observation spaces
        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()

    # ---------------- spaces ----------------
    def _build_action_space(self) -> gym.spaces.Box:
        if self.action_mode == "joint":
            # 6-dim: normalized [-1, 1] mapped to ctrl_limits
            return gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        # ee mode: [dx, dy, dz, gripper] in normalized [-1, 1]
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def _build_observation_space(self) -> gym.spaces.Dict:
        spaces: dict[str, gym.Space] = {}
        if self.observation_mode in {"state", "both"}:
            spaces["arm_qpos"] = gym.spaces.Box(-np.inf, np.inf, (6,), np.float32)
            spaces["arm_qvel"] = gym.spaces.Box(-np.inf, np.inf, (6,), np.float32)
        if self.observation_mode in {"image", "both"}:
            for cam in ("front", "wrist"):
                spaces[f"image_{cam}"] = gym.spaces.Box(
                    0, 255, (self.IMG_HEIGHT, self.IMG_WIDTH, 3), np.uint8
                )
        return gym.spaces.Dict(spaces)

    # ---------------- core API ----------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        # Apply home keyframe if present
        try:
            kf_id = self.model.key(self.HOME_KEYFRAME).id
            mujoco.mj_resetDataKeyframe(self.model, self.data, kf_id)
        except KeyError:
            pass

        # Clear any cached IK target from the previous episode so the first
        # substep loop after reset doesn't yank ctrl back toward stale ee_pos.
        self._ee_target = None
        self._ee_gripper_ctrl = None
        self.ee_target_override = None
        self.gripper_action_override = None

        self._post_reset(self.np_random)
        # Settle physics so cubes/plate rest on table. 50 mj_steps matches
        # pick-101's test_topdown_pick.py settle period.
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        self._step_count = 0
        return self._compute_obs(), self._info()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32).flatten()
        self._apply_action(action)
        for _ in range(self.SIM_STEPS_PER_CTRL):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs = self._compute_obs()
        terminated, truncated, info = self._check_done()
        if self._step_count >= self.max_episode_steps:
            truncated = True
        reward = 0.0
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "human":
            self._render_human()
            return None
        return self._render_camera(self.DEFAULT_CAMERA)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ---------------- rendering ----------------
    def _render_camera(self, camera_name: str) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.IMG_HEIGHT, width=self.IMG_WIDTH
            )
        self._renderer.update_scene(self.data, camera=camera_name)
        return self._renderer.render()

    def _render_human(self):
        # Lazy import to avoid pulling glfw on headless setups
        import mujoco.viewer  # type: ignore

        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    # ---------------- helpers for subclasses ----------------
    def _apply_action(self, action: np.ndarray) -> None:
        """Default action handler:
        - 'joint' mode: action in [-1, 1] → linearly mapped to ctrlrange
        - 'ee' mode: action = [dx, dy, dz, gripper_delta] in [-1, 1]
            dx,dy,dz in meters/step scaled to 5cm; gripper delta added.
            Uses mink IK if available (lazy import); else falls back to
            simple Jacobian damped pseudoinverse (configurable).
        """
        if self.action_mode == "joint":
            lo, hi = self.ctrl_limits[:, 0], self.ctrl_limits[:, 1]
            mid = (lo + hi) / 2.0
            half = (hi - lo) / 2.0
            target = mid + np.clip(action, -1, 1) * half
            self.data.ctrl[:] = target
        else:
            self._apply_ee_action(action)

    def _apply_ee_action(self, action: np.ndarray) -> None:
        """Cache the EE target + gripper ctrl for this env step.

        Two modes:
          - Override mode: when `ee_target_override` is set (scripted oracle
            path), use it directly as the absolute world-frame target. The
            IK substep loop in `step()` refreshes ctrl against this fixed
            target each mj_step, so motion follows a smooth exponential
            decay (pick-101 behaviour).
          - Delta mode (default, RL-friendly): action[:3] scaled by 0.05 m
            and added to the current ee position to form a 1-shot target.
            IK is solved once per ctrl tick.

        Likewise for the gripper: `gripper_action_override` wins over
        action[3] when set.
        """
        if self.ee_target_override is not None:
            self._ee_target = np.asarray(self.ee_target_override, dtype=np.float64).copy()
        else:
            delta_xyz = np.clip(action[:3], -1, 1) * self.EE_DELTA_SCALE
            self._ee_target = self.data.site("gripperframe").xpos.copy() + delta_xyz

        g_lo, g_hi = self.ctrl_limits[5]
        if self.gripper_action_override is not None:
            a3 = float(np.clip(self.gripper_action_override, -1, 1))
            self._ee_gripper_ctrl = float((a3 + 1.0) * 0.5 * (g_hi - g_lo) + g_lo)
        elif self.gripper_action_mode == "absolute":
            a3 = float(np.clip(action[3], -1, 1))
            self._ee_gripper_ctrl = float((a3 + 1.0) * 0.5 * (g_hi - g_lo) + g_lo)
        else:
            a3 = float(np.clip(action[3], -1, 1))
            self._ee_gripper_ctrl = float(
                np.clip(self.data.ctrl[5] + a3 * 0.2, g_lo, g_hi)
            )
        # Apply once now so mj_forward / observers see a consistent ctrl.
        self._refresh_ee_ctrl()

    def _refresh_ee_ctrl(self) -> None:
        """Resolve IK against the cached EE target and write ctrl[:6]."""
        if self._ee_target is None:
            return
        if getattr(self, "use_dls_ik", False):
            from sim.controllers.ik_dls import DlsIkController  # lazy
            if not isinstance(getattr(self, "_ik", None), DlsIkController):
                self._ik = DlsIkController(self.model, ee_site_name="gripperframe")
            q_arm = self._ik.step(
                self.data, self._ee_target, gain=0.5,
                locked_joints=self.locked_joints,
                down_orientation_weight=self.down_orientation_weight,
            )
        else:
            from sim.controllers.ik import EeIkController  # lazy
            if not isinstance(getattr(self, "_ik", None), EeIkController):
                self._ik = EeIkController(self.model, ee_site_name="gripperframe")
            q_arm = self._ik.solve(self.data, self._ee_target)
        self.data.ctrl[:5] = q_arm
        if self._ee_gripper_ctrl is not None:
            self.data.ctrl[5] = self._ee_gripper_ctrl

    # ---------------- subclass hooks ----------------
    def _post_reset(self, rng: np.random.Generator) -> None:
        """Hook for randomization. Default: no-op."""
        return None

    def _compute_obs(self) -> dict:
        obs: dict[str, np.ndarray] = {}
        if self.observation_mode in {"state", "both"}:
            obs["arm_qpos"] = self.data.qpos[: self.n_arm_joints].astype(np.float32).copy()
            obs["arm_qvel"] = self.data.qvel[: self.n_arm_joints].astype(np.float32).copy()
        if self.observation_mode in {"image", "both"}:
            obs["image_front"] = self._render_camera("front")
            obs["image_wrist"] = self._render_camera("wrist_cam")
        return obs

    def _check_done(self) -> tuple[bool, bool, dict]:
        return False, False, self._info()

    def _info(self) -> dict:
        return {}

    # ---------------- accessors ----------------
    def ee_pos(self) -> np.ndarray:
        return self.data.site("gripperframe").xpos.copy()

    def body_pos(self, name: str) -> np.ndarray:
        return self.data.body(name).xpos.copy()

    def encode_ee_delta_action(
        self,
        ee_before: np.ndarray,
        ee_after: np.ndarray,
        gripper_norm: float,
    ) -> np.ndarray:
        """Re-encode a recorded env step as a normalized ee-mode action.

        Scripted oracles drive the arm via `ee_target_override`, which makes
        env.step ignore the gym action's `[:3]` channel. Datasets recorded
        through that path therefore see action[:3] = 0 — useless as a VLA
        training label. This helper produces the action that *would have*
        produced the actual world-frame motion that occurred:

            action[:3] = clip((ee_after - ee_before) / EE_DELTA_SCALE, -1, 1)
            action[ 3] = gripper_norm  (already in [-1, 1])

        Clipping is needed because the override path can move the TCP more
        than EE_DELTA_SCALE m in one ctrl step (the IK gain is exponential
        toward a fixed absolute target), while the delta-mode action cap is
        ±EE_DELTA_SCALE. Clipping keeps the action schema consistent at the
        cost of saturating during the fastest transport ticks.
        """
        delta_world = np.asarray(ee_after) - np.asarray(ee_before)
        dxyz = np.clip(delta_world / self.EE_DELTA_SCALE, -1.0, 1.0)
        return np.array(
            [dxyz[0], dxyz[1], dxyz[2], float(gripper_norm)], dtype=np.float32
        )
