---
title: SO-101 Top-Down Grasp 调参与踩坑全记录
date: 2026-05-16
context: Phase 1 L-1 修复（0% → 15%）+ Phase 2 完整 pick-101 迁移（15% → 93.8%）
---

# SO-101 Top-Down Grasp 调参与踩坑全记录

> 本文档记录在 PickPlaceBlue 任务中，把脚本 oracle policy 从 0% 抓取成功率提升到可用基线（含夹爪真正垂直桌面）的整个调试过程。**写给未来的自己 / 任何接手 SO-101 manipulation 的人**——避免我们走过的弯路。

## TL;DR — 两句话教训

> **(Phase 1)** SO-101 这种 5-DoF 小臂做 top-down 抓取，**不能让 IK 自由解全 5 个关节**——必须先选定让夹爪朝下的 wrist 配置（wrist_flex + wrist_roll），**lock 这两个关节**，IK 只解前 3 个（shoulder_pan / shoulder_lift / elbow_flex）来满足 XYZ 位置。
>
> **(Phase 2)** 光抄方法不够，**模型本身也要抄**。pick-101 的 `so101_new_calib.xml` 多了 graspframe / 双 fingertip 站点 / 1.25 mm 指垫 collision geom + `cone="elliptic"` 摩擦——这些是稳定夹持的物理基础。menagerie 的 `so101.xml` 没有这些，再怎么调 IK 也只能到 26%。把整套模型搬过来 + finger_width_offset + yaw 量化 + 腕部按 cube 位置同步对齐，从 15% 一路推到 93.8%。

---

## 我们到底错在哪里

### 错误 1：把 mink 的 5-DoF position-only IK 当万能解

**做的事**：在 `sim/controllers/ik.py` 用 mink 的 `FrameTask(orientation_cost=0)`，让 IK 自由解所有 5 个关节满足 ee 位置 target。

**为什么错**：position-only IK 在 5-DoF 上有 2-DoF 的"零空间"——可以满足 position 的同时让 ee 朝向任意。posture cost 极小（0.001）时 mink 会选择"对当前 q 修改最小"的解，而这个解的 ee 朝向**没有任何保证**。结果就是夹爪斜着、横着、随机方向地朝向 cube，闭合时只能"擦过" cube 而不是垂直夹住。

**正确做法**（pick-101 模式）：先 hardcode 让夹爪朝下的 wrist 关节配置，再 lock 它们，IK 只在剩下 3 个关节空间里解 3-D position（well-determined）。

### 错误 2：用 force_wrist 在 IK 之后覆盖 ctrl

**做的事**：让 mink IK 解完 5 个关节，然后在 `_apply_ee_action` 末尾用 `data.ctrl[3] = force_wrist_flex; data.ctrl[4] = force_wrist_roll` 直接覆盖。

**为什么错**：IK 在内部以为它能用全 5 个关节满足 ee position，所以会把 shoulder/elbow 解到一个特定值（假设 wrist 也会按它给的值动）。结果 wrist 被外部覆盖后，shoulder/elbow 的值不再对 — ee 实际位置和 IK 期望位置差很多，整个解失效。**8/8 episode 全部 joint_limit**。

**教训**：约束必须**喂进 IK solver 里面**，不能在 IK 输出之后覆盖。pick-101 的 `locked_joints=[3,4]` 参数把约束传到 Jacobian 计算阶段（只用 Jp 的 3 列），这是正确姿势。

### 错误 3：以为"ee +z 朝 world +z (dot = +1)"就是夹爪朝下

**做的事**：我在 menagerie SO101 上扫了 wrist 配置，找到 `q=[0, 0.3, 1.3, -1.5, 0]` 让 `ee+z · world+z = +0.994`，以为这就是"夹爪朝下"姿态，就把它当 home keyframe。

**为什么错**：在我们的 menagerie SO101 中，jaws 开合方向是 ee 局部 z 轴。`ee+z · world+z = +1` 意味着 jaws 张开口朝 **world+z（上方）**——cube 在桌面上**无法**进入这种张开口。正确的应该是 `ee+z · world-z = +1`（即 ee+z 朝下，张开口朝下，cube 在 jaws 之间）。

用户看视频反馈"夹爪朝天"——一针见血指出我搞反了。

**教训**：选好"朝下"配置后**先用 viewer / 渲染一帧确认**，不要依赖 dot product 的符号。

### 错误 4：把 home keyframe 的 ctrl 写超出 ctrlrange

**做的事**：原 home keyframe 的 `shoulder_lift = -1.8`，但 `shoulder_lift` 的 ctrlrange 是 `[-1.745, 1.745]`。

**为什么错**：每个 control step 一开始 ctrl 就被 mujoco clip 到 -1.745，立刻触发 `at_limit`（因为我们的 `joint_limit` 检测阈值 1e-3 比 0.055 大得多）。30 步后 episode 被 `joint_limit_streak` 提前终止。所有 episode 都"出师未捷"。

**教训**：写 keyframe 一定**对照每个 actuator 的 ctrlrange**，留至少 0.05 rad 余量。menagerie 的 `so101.xml` 没有自带 keyframe，自己写时容易撞这个坑。

### 错误 5：DR 默认值偷偷推翻 MJCF 设计

**做的事**：`sim/randomization/camera_pose.py` 有一个 `_DEFAULT_FRONT_POS = (0.35, 0, 0.30)` 模块级常量。每次 `env.reset()` 时被覆盖到 `cam_pos`。

**为什么错**：MJCF 里 camera 实际是 `(0.55, -0.35, 0.5)`，DR 模块的"默认"是过时的旧值。每次 reset 都把相机偷偷拉到 stale 位置——录出来的视频视角完全错。MJCF 的设计意图被 DR 模块默默推翻。

**教训**：DR 模块应该**从 MJCF 读初始值**（如 `model.cam_pos[id]` 的首次值缓存），不要 hardcode "default"。否则 MJCF 和 DR 各自演化容易脱节。

### 错误 6：试图扩展 mink 加 orientation_cost 来强行让 ee 朝下

**做的事**：在 mink FrameTask 里加 `orientation_cost = 0.05/0.1/0.3`，target = "ee +z 朝 world -z"。

**为什么错**：SO-101 是 5-DoF，IK 无法同时满足 3-DoF position + 3-DoF orientation。orientation_cost 不为 0 时，mink 的 QP solver 在两者之间妥协，但 trade-off 总让某个关节被 LM-solver 推到 ctrlrange 边缘——所有 episode 立刻 `joint_limit`。

**教训**：5-DoF 不能给"完整 6-DoF target"。要么 position-only（orientation 失控），要么 lock 一部分关节让 IK 维度匹配。前者不可用，后者是 pick-101 路线。

---

## 关键的"啊哈"时刻

1. **用户上传的 SO-101 静态图**：清楚显示机械臂顶端夹爪朝下，证明 5-DoF **能**实现这个姿态。这彻底推翻了我"5-DoF 不够"的判断——根本不是关节自由度问题，是 IK 使用方式问题。

2. **找到 [ggando.com/blog/so101-rl-lift](https://ggando.com/blog/so101-rl-lift/) + 仓库 [github.com/ggand0/pick-101](https://github.com/ggand0/pick-101)**：
   > "To do top-down grasping, I needed the gripper pointing down with fingers horizontal, then move to an XYZ target."

   `tests/test_topdown_pick.py:116` 一行 `locked_joints = [3, 4]` 道出了核心。

3. **DLS Jacobian IK vs mink QP**：pick-101 用 ~70 行 DLS Jacobian IK，原生支持 `locked_joints`；mink 是个完整 IK 框架但 freezing joints 需要绕路。**简单方案胜出**——把那 70 行抄过来比硬撸 mink API 快多了。

4. **不同 SO-101 MJCF 的关节约定不通用**：pick-101 用 `wrist_flex=π/2, wrist_roll=π/2` 让 ee 朝下；同样的值在 menagerie SO-101 上让 ee 朝 world +y（横向）。原因是两个模型的 mesh / joint axis convention 不同。**抄数字不行，要抄方法**。我们最终在 menagerie 上用 `wrist_flex=0, wrist_roll=-2.74` 实现 ee 朝下（dot=0.94）。

---

## 正确的修复（最终配置）

### A. IK 模块：DLS Jacobian with locked joints

文件：[`sim/controllers/ik_dls.py`](../sim/controllers/ik_dls.py)

```python
class DlsIkController:
    def step(self, data, target_pos, *, gain=0.5, locked_joints=None):
        # mj_jacSite 拿 ee Jacobian
        # 只取 active joint 的列：Jp[:, active]
        # DLS: dq = (J^T J + λ²I)^-1 J^T err
        # locked joint 位置 dq=0
```

70 行，无第三方 IK 依赖，原生支持 `locked_joints=[3,4]`。

### B. Home keyframe：夹爪朝下的姿态

文件：[`assets/scenes/pick_place_blue.xml`](../assets/scenes/pick_place_blue.xml)

```xml
<key name="home"
     qpos="0 -1.4 1.4 0 -2.74 -0.17 ..."
     ctrl="0 -1.4 1.4 0 -2.74 -0.17"/>
```

`q=[0, -1.4, 1.4, 0, -2.74]` 给 `ee+z · world-z = 0.94`（夹爪基本垂直朝下），所有关节远离 ctrlrange 极限。

### C. Env hook：暴露 locked_joints 给策略

文件：[`sim/envs/base.py`](../sim/envs/base.py)

```python
class BaseSoArmEnv:
    self.locked_joints: list[int] | None = None  # 默认不锁

    def _apply_ee_action(self, action):
        ...
        q_arm = self._ik.step(self.data, target_pos,
                              gain=0.5, locked_joints=self.locked_joints)
        self.data.ctrl[:5] = q_arm
```

### D. Policy：grasp 阶段锁 wrist

文件：[`sim/scripted_policies/pick_place_blue.py`](../sim/scripted_policies/pick_place_blue.py)

```python
def __call__(self, env, obs):
    if self.phase == "DONE":
        env.locked_joints = None
    else:
        env.locked_joints = [3, 4]  # 锁 wrist_flex, wrist_roll
    ...
```

### E. 工作区：基于 reach envelope 调整

`PickPlaceBlueEnv.CUBE_X_RANGE / CUBE_Y_RANGE` 需要根据 down-facing wrist 下的实际 reach envelope 重调——5-DoF + locked wrist 下 reach 范围比 free-wrist 小，y 方向尤其窄。当前用 `cube_x ∈ [0.18, 0.25], y ∈ [-0.05, 0.05]`。

---

# Phase 2: 完整 pick-101 迁移 (15% → 93.8%)

> Phase 1 把抓取方法论搞对了（locked-wrist + DLS IK），但成功率卡在 15-26%。Phase 2 才发现：**方法对了模型不对也不行**——把 pick-101 整套自包含模型 + 一系列容易忽视的细节都搬过来，才一路推到 93.8%。

## 又错在哪里（Phase 2）

### 错误 7：以为 menagerie SO101 = pick-101 SO101，只抄方法不抄模型

**做的事**：把 pick-101 的 `IKController + locked_joints` 方法搬过来用在 menagerie 的 `so101.xml` 上。

**为什么错**：模型本身不一样：

| 项 | menagerie `so101.xml` | pick-101 `so101_new_calib.xml` |
|---|---|---|
| `gripperframe` 位置 | `(0.012, ~0, -0.098)` 偏 X | `(0, 0, -0.098)` 正中心 |
| `graspframe` 站点 | ❌ 无 | ✅ 用于 reach reward |
| `static_fingertip` / `moving_fingertip` 站点 | ❌ 无 | ✅ 计算指尖中点 |
| `static_finger_pad` / `moving_finger_pad` geom | ❌ 无 | ✅ 1.25 mm 接触贴片，**多点接触稳定夹持** |
| 摩擦/接触 | `friction="1 0.005 0.0001" condim=4` 默认 | `friction="0.5 0.05 0.001"` + `cone="elliptic"` + `noslip_iterations="3"` + `solref="0.001 1"` |

没有指垫 + elliptic 摩擦锥 → 夹爪闭合时 cube 直接从指缝滑出去；没有 fingertip 站点 → 没法做指尖中点计算或 contact-detection 抓取检测。

**正确做法**：把 pick-101 整个 `models/so101/` 目录原封不动搬到 `assets/so101_pick101/`（含 STL + `so101_new_calib.xml`），scene 文件 `<include file="../so101_pick101/so101_new_calib.xml"/>` 直接复用。**别只抄数字、别只抄方法、整套抄过来**。

### 错误 8：以为 gripperframe 就是两指中心

**做的事**：把 IK target 设到 `(cube.x, cube.y, cube_top + offset)`，沿世界 +y 直接对准 cube。

**为什么错**：pick-101 的 `gripperframe` 站点是 TCP（10cm forward 的虚拟点），**不在两指中心**——它沿 jaw spread 方向偏向 static finger ~12mm。如果不补偿，DESCEND 时 static fingertip 直接撞上 cube 顶面，把 cube 推飞。pick-101 的 `test_topdown_pick.py:113` 用一个不起眼的常量：

```python
finger_width_offset = -0.015  # offset along Y to center grip
```

把这条加上之前——**单测都跑不通**。

**正确做法**：descend 目标 y 加 `FINGER_WIDTH_OFFSET = -0.015`（cube_half_width 的负值）。读他人代码时**每个魔法数字都要查清来由**，尤其是看起来"显然多余"的偏移。

### 错误 9：每个 tick 都从 obs 重读 cube 位置去追

**做的事**：APPROACH/DESCEND 阶段每个 ctrl tick 重新计算 `target = (cube_obs.x, cube_obs.y - 0.015, z)`。

**为什么错**：cube 被 finger 轻轻一碰会移位 1-2 mm，policy 立刻把目标跟着挪过去，gripper 跟着追，又碰到 cube ... **追逐环**让 cube 在 DESCEND 期间被推走 2-3 cm。

**正确做法**：pick-101 的 `test_topdown_pick.py:139` 在 settle 之后**一次性快照** `actual_cube_pos`，整个 PICK 流程都用这个固定值，不再重读 obs。我们在 policy 的 `__call__` 第一次被调时 snapshot `cube_anchor` / `plate_anchor`，整个 pick 用这两个常量。

### 错误 10：gripper action 用 delta 而非 absolute

**做的事**：保留旧 env 的"action[3] * 0.2 加到 ctrl[5]"delta 接口（RL 友好）。

**为什么错**：pick-101 的 grasp 阶段是**渐进闭合 + 接触后再紧握**——需要绝对 gripper 值平滑斜坡（0.3 → -0.8，250 个 mj_step）。delta 模式下 action[3]=-1 → ctrl 每步减 0.2 → 5 步就饱和到 ctrlrange 下限，没法做接触检测后的精细收紧。

**正确做法**：env 加 `gripper_action_mode = "absolute"`，pick-101 的 IKController 那种 `(a+1)/2 * range + lo` 映射。RL 路径保持 delta 默认。

### 错误 11：把 500 Hz IK 当成万能解

**做的事**：观察到 pick-101 裸 mj_step loop 在固定 cube 位置 100% 成功，就把 env.step 改成"每个 mj_step 都跑 IK"（500 Hz），以为这能让 ee 不振荡。

**为什么错**：sts3215 伺服模型严重欠阻尼（ζ ≈ 0.26）。500 Hz IK 意味着 actuator 每 2 ms 收到新 ctrl 目标——比 actuator 自身的设定时间（~80 ms）快 40 倍。actuator 跟不上、来不及衰减振荡，反而抖得比 100 Hz 单次 IK 还厉害。**16 seed 实测从 3/16 (100 Hz) 掉到 1/16 (500 Hz)**。

**正确做法**：保留 env.step 单次 IK + actuator 自然 settle 100 Hz；用 ee_target_override 旁路让 scripted oracle 提供**绝对世界系目标**，IK gain=0.5 自然产生指数衰减（vs delta 模式下的 step-clamped 等速运动）。

### 错误 12：DESCEND 一够 REACH_THR 就立刻进 GRASP

**做的事**：REACH_THR=8mm，ee 距 target 8mm 内就 _advance("GRASP")。

**为什么错**：欠阻尼 actuator 在抵达 target 时仍有 3-4 mm 过冲并需要 30-40 ms 衰减。policy 在到达阈值瞬间切到 GRASP，闭合 jaw 时 ee 还在振荡——static fingertip 短暂插进 cube 下沿、把 cube 顶飞。pick-101 是固定 200 个 mj_step 给 DESCEND，给了 400 ms 让 actuator 充分稳定。

**正确做法**：加 `PHASE_HOLD_STEPS = 15`，每个 phase 至少持有 15 个 ctrl tick（150 ms）才允许切换，**即使 _reached 早已 True**。让欠阻尼的执行器稳定下来。

### 错误 13：cube yaw 随机 ±π 然后纳闷为什么 grasp 时好时坏

**做的事**：`_post_reset` 里 `yaw=rng.uniform(-π, π)`。

**为什么错**：locked-wrist 的 jaws 永远沿世界 +y 方向闭合；cube 转 45° 后呈现的是对角线面，**两个平行 jaw 根本咬不住对角**——闭合时 cube 从指间滑出。同样位置不同 yaw 时成时败，看起来像随机故障，其实是确定性的。

**正确做法**：3 cm cube 4-fold 对称，把 yaw 量化到 `{0, π/2, π, 3π/2}` 4 个值——前置相机看起来仍像"随机旋转"，但 jaws 永远咬在 cube 平面。**这一项把成功率从 ~60% 提到 ~92%**。

### 错误 14：以为 locked wrist + 中心工作空间就行了

**做的事**：把 cube y 范围从 (-0.08, 0.08) 收到 (-0.04, 0.04) 拿到 92%，以为搞定了。

**为什么错**：90 度向下夹爪意味着 wrist_roll 沿世界 Z 把 jaws 固定到世界 Y 方向。但当 shoulder_pan 旋转去够 cube 时，整个 wrist 跟着旋转——jaws 不再沿世界 Y 而是沿"arm-cube 连线的垂直方向"。`FINGER_WIDTH_OFFSET=-0.015` 沿世界 Y 的假设在 cube 偏离 y=0 时不准。

**正确做法**：reset 时按 cube 位置**预先调整 wrist_roll**：

```python
α = atan2(cube.y, cube.x)
wrist_roll = π/2 - α            # 让 shoulder_pan + wrist_roll 总是 π/2
qpos[4] = ctrl[4] = wrist_roll  # 一次设到位、整个 episode 锁住
```

shoulder_pan IK 解出来约 = α 时，总 z 旋转 = π/2，jaws 沿世界 Y。然后 `FINGER_WIDTH_OFFSET` 公式继续生效。工作空间这才能拉回到完整 (-0.08, 0.08)。

### 错误 15：PLACE_DESCEND 假设腕部仍沿世界 Y

**做的事**：用 `target = (plate.x, plate.y + FINGER_WIDTH_OFFSET, plate_top_z)`。

**为什么错**：PICK 时 wrist_roll = π/2 - α_cube。TRANSPORT 时 shoulder_pan 换到指向 plate（α_plate），但 wrist_roll 锁着不变。总 z 旋转变成 π/2 + (α_plate - α_cube)——jaws 已经不沿世界 Y。`FINGER_WIDTH_OFFSET` 沿世界 Y 加进去就偏到盘子外了。

**正确做法**：PLACE_DESCEND 用**闭环反馈**：

```python
desired_cube = (plate.x, plate.y, plate_top + cube_half + offset)
target_ee = ee_current + (desired_cube - cube_obs)  # 每 tick 重算
```

无需知道腕部转了多少度——cube 离目标还差多少，ee 就移多少，自洽稳定。同样的形式 TRANSPORT 不能用（cube 离 plate 太远 → 单 tick 跳一个完整 delta → IK 被推过关节限位），TRANSPORT 走直接 plate xy 目标即可，靠 PLACE_DESCEND 做最后精修。

### 错误 16：LeRobot dataset schema 写死 240×320

**做的事**：把 env IMG_HEIGHT/IMG_WIDTH 从 240×320 升到 480×640，结果 `parallel_runner` 在第一帧就崩：

```
ValueError: The feature 'observation.images.front' of shape '(480, 640, 3)'
does not have the expected shape '(240, 320, 3)' or '(320, 3, 240)'.
```

**为什么错**：`data/converters/sim_to_lerobot.py` 里 `DEFAULT_FEATURES` 把分辨率硬编进 schema。LeRobot 创建 dataset 时按 schema 注册，之后每 `add_frame` 都做 shape 校验。env 渲染分辨率改了、schema 没改、就炸。

**正确做法**：`make_or_resume_dataset(img_height=, img_width=)` 参数化；`build_features()` 工厂动态拼 shape；`parallel_runner._worker` 把 `env.IMG_HEIGHT/IMG_WIDTH` 透传过去——**渲染什么尺寸，schema 就声明什么尺寸**。MJCF `<visual><global offwidth="1920" offheight="1080"/></visual>` 把 offscreen framebuffer 上限设到 1080p，env 构造时 `img_height=720/1080` 就能直接出 HD。

---

## Phase 2 修复（最终配置）

### F. 整套 pick-101 模型搬过来

```
assets/so101_pick101/
├── so101_new_calib.xml      # 完整复制，未改动
├── assets/                  # 14 个 STL meshes
│   └── *.stl
└── ...                      # 其它备用 scene xml
```

scene 文件 `assets/scenes/pick_place_blue.xml` 用 `<include file="../so101_pick101/so101_new_calib.xml"/>` 引用，加上单红 cube + freejoint 白盘 + 三盏可 DR 光源 + front 相机 + home keyframe。

### G. env 加三条 scripted-oracle 旁路

[`sim/envs/base.py`](../sim/envs/base.py)：

```python
self.ee_target_override: np.ndarray | None = None  # 绝对世界系 IK 目标
self.gripper_action_override: float | None = None  # 绝对 gripper action
self.gripper_action_mode: str = "delta"            # "absolute" 给 oracle 用
```

`_apply_ee_action` 检测到 override 时跳过 action[:3] 的 delta 解释，直接用 override 作 IK target。

### H. wrist_roll 按 cube 位置同步对齐

[`sim/envs/pick_place_blue.py`](../sim/envs/pick_place_blue.py) `_post_reset`：

```python
alpha = float(np.arctan2(cube_xy[1], cube_xy[0]))
wrist_roll = float(np.pi / 2 - alpha)
wrist_roll = np.clip(wrist_roll, *self.ctrl_limits[4])
self.wrist_roll_alignment = wrist_roll        # 暴露给 policy
self.data.qpos[wrist_roll_qadr] = wrist_roll
self.data.ctrl[4] = wrist_roll
```

`locked_joints = [3, 4]` 保留——`wrist_roll` 在 episode 内仍然锁死，只是初始值不再是常数 π/2。

### I. policy 用 cube_anchor + finger_width_offset + 闭环 PLACE

[`sim/scripted_policies/pick_place_blue.py`](../sim/scripted_policies/pick_place_blue.py)：

```python
FINGER_WIDTH_OFFSET = -0.015     # gripperframe 偏向 static finger，沿世界 Y 补偿
PHASE_HOLD_STEPS = 15            # 让欠阻尼 actuator 稳定下来

# 首次调用快照（pick-101 actual_cube_pos 同款）
if self.cube_anchor is None:
    self.cube_anchor = obs["red_cube_pos"].copy()
    self.plate_anchor = obs["plate_pos"].copy()

# APPROACH/DESCEND: 沿世界 Y 加 offset
target = (cube_anchor.x, cube_anchor.y + FINGER_WIDTH_OFFSET, ...)

# PLACE_DESCEND: 闭环跟踪 cube 实际位置（不依赖任何 jaw 朝向假设）
desired_cube = (plate.x, plate.y, plate_top_z + cube_half + offset)
target = ee_current + (desired_cube - cube_obs)
```

cube yaw 在 env 端量化到 `{0, π/2, π, 3π/2}`。

### J. 分辨率参数化

[`sim/envs/base.py`](../sim/envs/base.py)：默认 `IMG_HEIGHT=480, IMG_WIDTH=640`（front 相机 native）；构造参数 `img_height=` / `img_width=` 可覆盖。

[`assets/scenes/pick_place_blue.xml`](../assets/scenes/pick_place_blue.xml)：`<global offwidth="1920" offheight="1080"/>` 支持到 1080p。

[`data/converters/sim_to_lerobot.py`](../data/converters/sim_to_lerobot.py)：`build_features(img_height, img_width)` 工厂；`make_or_resume_dataset` 同名 kwargs。

[`sim/collectors/parallel_runner.py`](../sim/collectors/parallel_runner.py)：CLI `--img-height` / `--img-width` / `--max-episode-steps`，worker 用 `env.IMG_HEIGHT/IMG_WIDTH` 注册 dataset schema。

---

## 一个表格总结：进步轨迹

| 版本 | 改动 | success rate | 夹爪垂直? | 工作空间 (y) |
|---|---|---:|---|---|
| 原版 | 4cm cube, 旧策略, mink position-only IK | 0%/50 | 否（随机斜向） | (-0.12, +0.12) |
| v5 | + 3cm cube, closed-start gripper, tight xy tracking | 5%/50 | 否 | (-0.10, +0.10) |
| v6 | + DESCEND_Z=0.020, workspace shrunk, 修 home keyframe 越界 bug | 26%/50 | 否（但有抓到 cube） | (-0.10, +0.10) |
| v16 (pick-101 风格 IK) | DLS Jacobian IK + locked wrist + down-facing home keyframe | 15%/50 | **是** | (-0.05, +0.05) |
| **v17 (整套 pick-101 模型)** | so101_new_calib.xml + 指垫 + elliptic 摩擦 + finger_width_offset | 3/8 = 38% | 是 | (-0.08, +0.08) |
| v18 | + cube/plate snapshot 锚定（停止追逐环）+ absolute gripper mode | 7/16 = 44% | 是 | (-0.08, +0.08) |
| v19 | + PHASE_HOLD_STEPS 让欠阻尼 actuator 稳定 | 19/32 = 59% | 是 | (-0.08, +0.08) |
| v20 (方案 A) | + cube yaw 量化 {0, π/2, π, 3π/2} + 工作空间收窄到 ±0.04 | **59/64 = 92.2%** | 是 | (-0.04, +0.04) |
| **v21 (方案 B)** | + wrist_roll = π/2 − atan2(cube.y, cube.x) 每 episode 对齐 + 闭环 PLACE_DESCEND | **60/64 = 93.8%** | 是 | **(-0.08, +0.08)** ← 恢复全范围 |

关键拐点：
- **v16 → v17**：从只抄方法升级到整套模型搬过来，单步 ~23 个百分点。
- **v19 → v20**：cube yaw 量化（一行代码），~33 个百分点。
- **v20 → v21**：方案 B 把工作空间放大 4 倍 (±0.04 → ±0.08) 的同时保住 ~94%。

---

## 给未来的 checklist

下次在 SO-101（或任何小型 5-DoF 臂）实现 grasp 时，按这个 checklist 走，能少走 80% 的弯路：

### Phase 1: 把方法论搞对

1. [ ] 找一个**社区已经验证 work 的参考实现**（gym-lowcostrobot, pick-101, 等），先读 README + grasp test 脚本
2. [ ] 在 MuJoCo viewer 里**手动滑动每个关节**，确认每个关节怎么影响 ee 朝向；记下 "ee 朝下"对应的 (shoulder_lift, elbow_flex, wrist_flex, wrist_roll) 组合
3. [ ] 在 home keyframe 里**直接设到那个姿态**（不指望 IK 帮你转过去）
4. [ ] IK 用 **locked_joints 模式**，只在剩下的关节空间解 position；DLS Jacobian 70 行够用，不需要 mink
5. [ ] 验证 home keyframe 的每个 ctrl 都**至少离 ctrlrange 边界 0.05 rad**
6. [ ] 用 viewer 或单帧渲染**眼睛确认**夹爪真的朝下，不要只看 dot product
7. [ ] 用 cube 尺寸跟随社区标准（gym-lowcostrobot 3cm），不要自己随手定 4cm 这种"看起来挺大的"数字
8. [ ] DR 模块**不要 hardcode 默认值**——首次调用从 model 读快照存起来
9. [ ] 第一次能跑通后**先跑 50 episode** 看稳定 success rate，不要只看 4 episode 就下结论
10. [ ] success 后**hold 至少 1 秒**再 terminate，让视频录到完整 place + retract 过程

### Phase 2: 把成功率从 15% 推到 90%+

11. [ ] **整套模型一起搬**，不只搬 IK 方法：参考库的 `models/` 目录（meshes + 校准过的 XML）直接拷过来。menagerie 缺指垫 / fingertip 站点 / 摩擦锥配置，再调 IK 也只能到 ~25%
12. [ ] 找到参考库的**每一个"魔法常量"**并搬过来——`FINGER_WIDTH_OFFSET = -0.015` 这种看起来多余的偏移往往是关键
13. [ ] **快照** cube/plate 起始位置，整个 pick 流程只用快照值，**绝不**每个 tick 重读 obs（追逐环会把 cube 推走）
14. [ ] gripper action 用 **absolute 模式**（不要 delta）；scripted oracle 的 contact-detection 抓紧需要平滑斜坡
15. [ ] **不要盲目提高 IK 频率**：欠阻尼 actuator（ζ < 0.5）下 500 Hz IK 比 100 Hz IK 更糟糕——actuator 跟不上目标抖动。先实测 100 Hz baseline 再考虑
16. [ ] 每个 phase 加 `PHASE_HOLD_STEPS`（≥ 15 个 ctrl tick），让 actuator 完成过冲衰减再切下一阶段
17. [ ] cube yaw **量化到对称角度**（3 cm cube → `{0, π/2, π, 3π/2}`），不要全 `[-π, π]` 随机——locked-wrist 的平行 jaw 咬不住对角面。这一项往往是 30+ 个百分点
18. [ ] 工作空间想拉大时，给 wrist_roll **按 cube 位置同步对齐**：`wrist_roll = π/2 − atan2(cube.y, cube.x)`，把"shoulder_pan 摆过去后 jaw 仍沿世界 Y"的不变量保住
19. [ ] PLACE 阶段**不要假设腕部朝向**——cube 被夹住后跟着 jaw 转，用 `target_ee = ee + (desired_cube − cube_obs)` 闭环反馈才稳健
20. [ ] LeRobot dataset 的图像 feature shape **要和 env 渲染分辨率联动**：用工厂函数 `build_features(img_height, img_width)`，别在模块顶部写死 `(240, 320, 3)`
21. [ ] MJCF 的 `<global offwidth="..." offheight="..."/>` 决定 offscreen renderer 的上限——想录 720p/1080p 视频时记得提前设
22. [ ] 跑诊断时**实测 64+ seeds**，不要 8/16 就下结论——cube yaw 这种确定性 bug 在小样本里看起来像随机故障

---

## 参考资料

- ggando.com 博客（核心）：<https://ggando.com/blog/so101-rl-lift/>
- pick-101 仓库：<https://github.com/ggand0/pick-101>
  - `tests/test_topdown_pick.py` — 标准 4 阶段 pick 流程
  - `src/controllers/ik_controller.py` — DLS Jacobian IK with locked_joints
- gym-lowcostrobot：<https://github.com/perezjln/gym-lowcostrobot>
  - `gym_lowcostrobot/assets/low_cost_robot_6dof/pick_place_cube.xml` — 3cm cube 标准
- ECE 4560 SO-101 作业：<https://maegantucker.com/ECE4560/assignment6-so101/>
- mujoco_menagerie SO-101：<https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101>

---

## 后续待办

Phase 2 已处理完的：
- [x] ~~进一步缩 workspace 或调 DLS IK gain 降低 joint_limit 占比~~（PHASE_HOLD_STEPS + yaw 量化解决了根因，joint_limit 已基本消失）
- [x] ~~考虑用 wrist_roll align cube yaw~~（v21 方案 B：用 cube xy 而非 yaw 算 wrist_roll，效果更好）
- [x] ~~PLACE_DESCEND 的精度还可以再调让 place_miss → success~~（闭环反馈方案搞定）

仍可继续打磨：
- [ ] 剩余 ~6% 失败是 `joint_limit`，下游表象其实是 GRASP 没抓住——可以加 LIFT 后的 grasp-check（cube_z 是否真的离地），失败立刻 abort 走 retract，避免错误归类
- [ ] cube 摩擦从 `0.5 0.05 0.001` 调到 `1 0.05 0.001` 可能减少滑出指间的 case
- [ ] `wrist_roll_alignment` 在 cube y 极端时被 clip 到 ctrlrange 边界——可考虑提前过滤这些 cube 位置或留死区
- [ ] 接入实际机械臂时验证 sim → real 的腕部 alignment 是否能照搬（实际伺服阻尼可能不同，过冲特征会变）
