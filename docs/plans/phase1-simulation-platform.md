# Phase 1：仿真平台搭建

**周期**：1–2 周
**前置依赖**：Phase 0 完成
**目标**：在 MuJoCo 中跑通 SO-ARM101 模型；实现最小可用的 **`PickPlaceRed`** 任务环境（reset / step / success / 域随机化），手写脚本策略在仿真里完成"抓红 cube → 放进 plate"全流程

> 核心任务详细规约见 [README.md](README.md)。当前脚本策略成功率 **93.8%**（v21 配置：pick-101 风格俯视顶抓 + cube yaw 4 等分量化 + wrist_roll 按 cube 位置同步）。详见 [docs/lessons-learned-so101-grasp.md](../lessons-learned-so101-grasp.md)。

---

## 代码入口（快速开始）

```bash
# 视觉自检
python -m mujoco.viewer --mjcf=assets/scenes/pick_place.xml

# 跑 20 条 episode 看视频（保留失败，清理临时 PNG）
python -m sim.collectors.parallel_runner --num-episodes 20 --num-workers 4 \
    --repo-id local/so101_debug --keep-failures --cleanup-after-collect

# 关 DR 做 debug
python -m sim.collectors.parallel_runner --num-episodes 20 --num-workers 4 \
    --repo-id local/so101_debug_nodr --no-dr
```

**两个 flag**：
- `--keep-failures`：失败 episode 也写入，`task` 字段加 `[FAIL:<mode>]` 后缀（mode ∈ `grasp_fail` / `lift_drop` / `place_miss` / `plate_off` / `joint_limit` / `timeout`）
- `--cleanup-after-collect`：每个 shard 落盘后删 `images/` 临时 PNG，保留 `data/` + `videos/` + `meta/`

**入口与任务清单的对应关系**：

| Task | 实现文件 |
|------|---------|
| T1.1–T1.2 MuJoCo + SO101 模型 | `assets/scenes/pick_place.xml` + `assets/so101_pick101/` |
| T1.3 BaseSoArmEnv 基类 | [`sim/envs/base.py`](../../sim/envs/base.py) |
| T1.4 PickPlaceRed 环境 | [`sim/envs/pick_place.py`](../../sim/envs/pick_place.py) |
| T1.5 双相机渲染 | `assets/scenes/pick_place.xml` + `sim/envs/base.py::_render_cameras` |
| T1.6 微分 IK（DLS） | [`sim/controllers/ik_dls.py`](../../sim/controllers/ik_dls.py) |
| T1.7 9 阶段脚本策略 | [`sim/scripted_policies/pick_place.py`](../../sim/scripted_policies/pick_place.py) |
| T1.8 域随机化 | [`sim/randomization/`](../../sim/randomization/) |
| T1.9 LeRobot writer | [`data/converters/sim_to_lerobot.py`](../../data/converters/sim_to_lerobot.py) |

---

## 任务清单

### T1.1 安装并验证 MuJoCo

**步骤**：
- [ ] `pip install mujoco`
- [ ] `python -m mujoco.viewer` 弹出 GUI
- [ ] 用官方 humanoid.xml 验证物理仿真

**验证**：viewer 中可手动拖动物体

### T1.2 导入 SO-ARM100 模型

**步骤**：
- [ ] `git clone https://github.com/google-deepmind/mujoco_menagerie assets/menagerie`
- [ ] 找到 `assets/menagerie/trs_so_arm100/so_arm100.xml`
- [ ] 写最小加载脚本 `sim/scripts/load_so101.py`

**验证**：viewer 显示完整机械臂，关节滑块响应正确

---

### T1.3 BaseSoArmEnv 基类

**步骤**：设计 `sim/envs/base.py` 暴露 gymnasium 兼容接口：
- `reset(seed)`：初始化场景、采样物体姿态
- `step(action)`：写入关节目标 → 物理仿真 → 返回 obs
- `get_observation()`：`{joint_pos, joint_vel, ee_pose, images}`
- `evaluate_success()`：抽象方法
- `render()`：返回 RGB + depth

**验证**：基类可被子类继承并实例化

---

### T1.4 实现 PickPlaceRed 环境

**场景**：SO-ARM101（pick-101 移植的 `assets/so101_pick101/so101_new_calib.xml`，带 graspframe + finger pad）+ 桌面 + 红 cube（3cm）+ 白 plate（cylinder，6cm 半径 × 0.2cm 半高，freejoint，质量 0.15kg）。

**MJCF 约定**：cube/plate body origin 放在 geom 中心，避免 IK 计算偏移。

**`reset()` 工作区（v21）**：
- `CUBE_X_RANGE=(0.16, 0.22)`、`CUBE_Y_RANGE=(-0.08, 0.08)`
- `PLATE_X_RANGE=(0.24, 0.30)`、`PLATE_Y_RANGE=(-0.06, 0.06)`
- cube yaw 量化到 `{0, π/2, π, 3π/2}`（利用 cube 4 重对称避开 45° 夹爪失败）
- **wrist_roll 按 cube 位置同步**：`wrist_roll = clip(π/2 - atan2(cube.y, cube.x), 限位)`，保证总 z 旋转 = π/2

**`step()`**：`action_mode="ee"` 时 action 4 维 (dx, dy, dz, gripper)；`action_mode="joint"` 时 7 维关节目标。

**`evaluate_success()`**：cube xy 距 plate < 半径 AND cube 底面 z 在 plate 表面 ±1cm AND 夹爪已松开。

**`evaluate_failure_mode()`**：`grasp_fail` / `lift_drop` / `place_miss` / `plate_off` / `joint_limit` / `timeout`。

**关键文件**：`sim/envs/pick_place.py`、`assets/scenes/pick_place.xml`、`assets/so101_pick101/`

**参考**：[pick-101 仓库](https://github.com/ggand0/pick-101)（俯视顶抓 oracle 与模型来源）

**验证**：`env.reset() + env.step(zero_action)` 跑 100 步不崩；cube/plate 在 wrist+front 相机里清晰可见

---

### T1.5 双相机渲染

**步骤**：MJCF 加两个 camera site：`wrist_cam`（吸附在 ee，俯视）+ `front_cam`（固定桌前 30cm，俯角 30°）。`get_observation()` 渲染两路 640×480@30fps RGB。

**验证**：`obs.images.wrist` 与 `obs.images.front` 都是 640×480×3 uint8

---

### T1.6 微分 IK（DLS）

用 mink 在 `sim/controllers/ik.py` 实现 `ik(target_pose, current_q) -> q_target`。DLS 阻尼 + `locked_joints` 锁 wrist。**关键**：`locked_joints` 读 `data.ctrl[j]` 而不是 `data.qpos`，防止物理漂移污染 ctrl。

**验证**：给定桌面上方 10cm 目标位姿，200 step 后 ee 距目标 < 5mm

---

### T1.7 9 阶段脚本策略

在 `sim/scripted_policies/pick_place.py` 写 9 阶段状态机：

1. **HOME**：等待复位
2. **APPROACH**：移到 cube 上方 3.5cm（含 `FINGER_WIDTH_OFFSET=-0.015`）
3. **DESCEND**：下降到 z=0.020（finger_pad 触 cube 顶）
4. **GRASP**：渐进闭合（60 步斜坡）+ 接触检测后保压
5. **LIFT**：抬起到 `TRANSPORT_Z=0.10`
6. **TRANSPORT**：水平移到 plate 上方
7. **PLACE_DESCEND**：闭环 `target = ee_cur + (desired_cube - cube_obs)` 下降到 plate 表面上方
8. **RELEASE**：松开夹爪
9. **RETRACT**：抬起撤离

**关键技巧**：
- `cube_anchor`/`plate_anchor` 首次 snapshot，PICK 阶段不读 obs（防 chase loop）
- 跨 substep 用 `env.ee_target_override` + `env.gripper_action_override` 保持稳定目标
- 用 `env.evaluate_success()` 判定

**验证**：100 次随机 reset，成功率 ≥ 90%（v21 实测 93.8%）

---

### T1.8 域随机化（基础）

在 `sim/randomization/` 实现：
- `lighting.py`：3 个点光源位置/强度
- `textures.py`：桌面纹理库 10+ 张
- `cube_pose.py`：xy + yaw（量化 4 等分）+ plate 位置 ±3cm
- `camera_pose.py`：±5cm / ±5° front cam

**铁律**：**不要随机化 cube 颜色**——red 是语言锚点，破坏即破坏语义对齐。plate 颜色可随机（不影响指令）。

`env.reset()` 按概率调用；留 `--no-dr` 开关。

**验证**：连续 reset 10 次截图，视觉差异明显

---

### T1.9 LeRobot 数据格式 dummy 导出

在 `data/converters/sim_to_lerobot.py` 写 writer，用 T1.7 脚本策略跑 5 条 episode 存 LeRobot 格式；`task` 字段从 `data/instructions/pick_place.txt` 随机抽。

**验证**：`lerobot-dataset-viz` 能正常播放

---

## 验收标准

- [ ] SO-ARM101 在 MuJoCo 中可控、可渲染
- [ ] PickPlaceRed 环境完整（reset/step/success/失败归因/双相机）
- [ ] 脚本策略 100 次成功率 ≥ 90%（实测 93.8%）
- [ ] 域随机化可独立开关（cube 颜色不参与）
- [ ] 5 条 episode 已导出 LeRobot 格式并可视化

---

## 风险与陷阱

| 风险 | 应对 |
|------|------|
| pick-101 模型与真机几何差异 | 用真机实测的关节限位 / 长度反向校准 MJCF；finger_pad 1.25mm 是 sim 假设，真机换硅胶垫 |
| IK 解奇异时跳变 | DLS 阻尼 + locked_joints 锁 wrist；目标限制在工作空间内 |
| 渲染太慢拖累采集 | `MUJOCO_GL=egl` 离屏；scene XML `offwidth/offheight` 设到 1920×1080 上限 |
| DR 太激进导致策略不收敛 | 先 `--no-dr` 验证策略，再加 DR；维度逐个加入 |
| 仿真"抓住"但物理不稳定 | cube friction `0.5 0.05 0.001` + condim=4 + priority=1，让 finger_pad 继承 cube 摩擦 |
| **500Hz IK 让成功率反而变差** | sts3215 actuator 严重欠阻尼（ζ≈0.26），IK 比 actuator 还快引发共振；用 100Hz IK + 跨 substep 锁定目标 |
