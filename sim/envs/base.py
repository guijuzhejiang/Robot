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
    IMG_HEIGHT: int = 240
    IMG_WIDTH: int = 320
    SIM_STEPS_PER_CTRL: int = 5  # 0.002s * 5 = 10ms per ctrl step (100 Hz)

    def __init__(
        self,
        *,
        observation_mode: str = "both",
        action_mode: str = "ee",
        render_mode: str | None = None,
        max_episode_steps: int = 200,
        seed: int | None = None,
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

        self.model = mujoco.MjModel.from_xml_path(self.SCENE_PATH)
        self.data = mujoco.MjData(self.model)
        self.np_random = np.random.default_rng(seed)
        self._step_count = 0
        self._renderer: mujoco.Renderer | None = None
        self._viewer = None

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

        self._post_reset(self.np_random)
        # Settle physics so cubes/plate rest on table
        for _ in range(20):
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
        from sim.controllers.ik import EeIkController  # lazy

        if not hasattr(self, "_ik"):
            self._ik = EeIkController(self.model, ee_site_name="gripperframe")
        delta_xyz = np.clip(action[:3], -1, 1) * 0.05  # 5cm/step max
        gripper_delta = float(np.clip(action[3], -1, 1)) * 0.2  # accumulated

        target_pos = self.data.site("gripperframe").xpos.copy() + delta_xyz
        # keep current orientation by reusing site xmat → quat
        q_arm = self._ik.solve(self.data, target_pos)
        # Map first 5 actuators to IK joints
        self.data.ctrl[:5] = q_arm
        # Gripper accumulates
        cur_g = self.data.ctrl[5]
        g_lo, g_hi = self.ctrl_limits[5]
        self.data.ctrl[5] = np.clip(cur_g + gripper_delta, g_lo, g_hi)

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
