# Phase 1：仿真平台搭建

**周期**：1–2 周
**前置依赖**：Phase 0 完成
**目标**：在 MuJoCo 中跑通 SO-ARM100 模型；实现一个最小可用的 **`PickPlaceBlue`** 任务环境（含 reset / step / success / 域随机化），手写脚本策略能在仿真里完成"抓蓝 cube → 放进 plate"全流程

> **核心任务**：桌面上随机摆放 1 红 cube + 1 蓝 cube + 1 plate，把蓝 cube 放进 plate。详细规约见 [README.md](README.md) 顶部"核心任务定义"。

---

## 代码入口（快速开始）

> 所有命令在仓库根目录 `/home/zzg/workspace/pycharm/Robot` 下、`conda activate py312_cu121` 后执行。

> ⚠️ **已知限制 L-1**：当前 6 阶段脚本策略实际抓取成功率约 **0–20%**（[docs/implementation-status.md](../implementation-status.md#已知限制)），默认 collector 只保存成功 episode，所以"正式批量"命令会输出 0 个 mp4。要先看视频/做调试，**强烈推荐加 `--keep-failures --cleanup-after-collect`**，失败 episode 也会被切成单独 mp4 集中到一个目录、shard 临时文件自动删除。

| 想做的事 | 一行命令 | 产出 |
|---------|---------|------|
| 用 MuJoCo viewer 打开仿真场景，手动拖动 SO101 / cube / plate（T1.1–T1.2、T1.5 视觉自检） | `python -m mujoco.viewer --mjcf=assets/scenes/pick_place_blue.xml` | GUI 窗口 |
| **看视频 / debug L-1**：跑 N 条 episode、不管成败都切成 mp4、删 dataset 缓存 | `python -m sim.collectors.parallel_runner --num-episodes 20 --num-workers 4 --repo-id local/so101_debug --keep-failures --cleanup-after-collect` | `~/.cache/huggingface/lerobot/local/so101_debug_videos/{front,wrist}/shardXX_epNNNN_<SUCCESS\|FAIL_mode>.mp4`（每条 episode 一个 10s mp4） |
| 关闭域随机化做 debug | 上一条命令追加 `--no-dr` | 同上 |
| **正式批量采集（L-1 修好后）**：只保存成功 episode 到 LeRobot dataset，供训练 | `python -m sim.collectors.parallel_runner --num-episodes 1000 --num-workers 4 --repo-id local/so101_pickplace_blue_v0` | `~/.cache/huggingface/lerobot/local/so101_pickplace_blue_v0_shard{00..03}/`（保留完整 dataset 含 mp4 + parquet） |
| 审计采集到的数据集 | `python -m eval.audit_dataset --repo-id local/so101_pickplace_blue_v0_shard00 --n-sample 50` | 控制台报告 + 可选 `--out-dir` 输出 markdown |
| 清理本地 cache（占空间时用） | `rm -rf ~/.cache/huggingface/lerobot/local/so101_pickplace_blue_v0_shard*` | — |

**两个新 flag 详解**：

- `--keep-failures`：失败 episode 也写入 dataset；它们的 `task` 字段会被打上 `[FAIL:<mode>]` 后缀（mode ∈ `grasp_fail` / `place_miss` / `distractor_hit` / `color_confusion` / `timeout`），下游 audit / training 可以按此过滤剔除。
- `--cleanup-after-collect`：跑完后用 ffmpeg 按 `episodes.parquet` 里记的 `[from_timestamp, to_timestamp]` 把每个 episode 从 LeRobot 合并 mp4 中切出来，写到 `<repo_id>_videos/{front,wrist}/shardXX_epNNNN_<status>.mp4`，然后 **删除整个 shard dataset 目录**（含 parquet、meta、原 chunked mp4）。代价：删完后 `audit_dataset` 找不到这个 repo_id，但视频文件保留。

**入口与任务清单的对应关系**：

| Task | 实现文件（库） | 由哪个 CLI 串起来 |
|------|--------------|------------------|
| T1.1–T1.2 验证 MuJoCo + 模型 | `assets/scenes/pick_place_blue.xml` + `assets/menagerie/robotstudio_so101/` | `python -m mujoco.viewer` |
| T1.3 BaseSoArmEnv 基类 | [`sim/envs/base.py`](../../sim/envs/base.py) | 被 `PickPlaceBlueEnv` 继承 |
| T1.4 PickPlaceBlue 环境 | [`sim/envs/pick_place_blue.py`](../../sim/envs/pick_place_blue.py) | `parallel_runner` / `pick_place_pipeline` |
| T1.5 双相机渲染 | `assets/scenes/pick_place_blue.xml`（`front` + `wrist` camera）+ `sim/envs/base.py::_render_cameras` | 同上 |
| T1.6 微分 IK | [`sim/controllers/ik.py`](../../sim/controllers/ik.py)（mink） | 被脚本策略调用 |
| T1.7 6 阶段脚本策略 | [`sim/scripted_policies/pick_place_blue.py`](../../sim/scripted_policies/pick_place_blue.py) | `parallel_runner` / `pick_place_pipeline` |
| T1.8 域随机化 | [`sim/randomization/`](../../sim/randomization/) | 默认开启，`--no-dr` 关闭 |
| T1.9 LeRobot writer | [`data/converters/sim_to_lerobot.py`](../../data/converters/sim_to_lerobot.py) | 被 `parallel_runner` 调用 |

> **TL;DR**：当下 L-1 未修，想看抓取视频请用表中第 2 行（`--keep-failures --cleanup-after-collect`）。输出在 `~/.cache/huggingface/lerobot/local/<repo_id>_videos/front/shardXX_epNNNN_FAIL_grasp_fail.mp4`（每条 episode 一个 10s mp4，文件名含成败状态）。

---

## 关键技术与工具

| 工具 | 用途 | 获取方式 |
|------|------|---------|
| MuJoCo | 物理引擎 | `pip install mujoco` |
| mujoco_menagerie | 官方机器人模型库（含 `trs_so_arm100/`） | `git clone https://github.com/google-deepmind/mujoco_menagerie` |
| mink | MuJoCo 上的微分 IK 求解器 | `pip install mink` |
| dm_control | DeepMind 仿真组件（observation/reward 辅助） | `pip install dm_control` |
| gymnasium | 标准 RL/IL 环境接口 | `pip install gymnasium` |
| imageio + ffmpeg | 录制仿真视频 | `pip install imageio[ffmpeg]` |

### 选型说明：为什么主路径是 MuJoCo + Menagerie

先澄清一个常见误读：**RoboCasa / MimicGen 与 gym-lowcostrobot 并没有从主项目里被淘汰**。它们和 MuJoCo + Menagerie 的关系是：

- **MimicGen** 是数据扩增框架（algorithm），底层仍是 MuJoCo（通过 robosuite）。我们在 **Phase 3 仍然使用 MimicGen 的算法思路**，只是不依赖 robosuite 整个栈。
- **gym-lowcostrobot** 是 LeRobot 生态对 SO-ARM 系列的封装，底层模型本身就是 mujoco_menagerie 的 SO-ARM100。所以它和我们主路径**底层一致**，区别在 env API 与任务库。
- **RoboCasa** 是 robosuite 之上的厨房场景库，原生不支持 SO-ARM100。

| 平台 | SO-ARM 原生 | 任务库 | 数据扩增 | 入门难度 | 灵活度 |
|------|------------|--------|----------|---------|--------|
| **MuJoCo + Menagerie**（主路径） | ✅ 官方 MJCF | 自写 | 配合 MimicGen 思路 | 中 | ⭐⭐⭐⭐⭐ |
| **gym-lowcostrobot** | ✅ 内置 | PickPlace / Push / Stack 等已有 | LeRobot 数据原生 | 低 | ⭐⭐⭐ |
| **RoboCasa + MimicGen** | ❌（Franka / UR5 等） | 100+ 厨房 / 操作任务 | ⭐ MimicGen 原生 | 高 | ⭐⭐⭐⭐ |

**主路径定为 MuJoCo + Menagerie 的三个理由**：

1. **SO-ARM100 的官方 MJCF 在 mujoco_menagerie**（DeepMind 维护，与真机几何最接近），免去 URDF → MJCF 的工作
2. Phase 2 要写的所有自定义化（自己的 grasp sampler、planner、自定义任务、自定义 DR）都在这一层最直接，没有 robosuite / RoboCasa 等中间抽象层
3. 主路径输出 LeRobot 格式，与 Phase 3 真机 demo / Phase 4 训练对齐

**不直接用 RoboCasa/MimicGen 作为主平台的原因**：SO-ARM100 不是 robosuite 原生，需要 URDF → MJCF + 控制器接入；RoboCasa 的厨房资产对 SO101 这种小臂桌面任务也用不上。真正有价值的是 **MimicGen 的算法**，这部分我们 Phase 3 移植。

**不直接用 gym-lowcostrobot 作为主平台的原因**：作为社区项目，任务定义和自定义 DR 接口不够灵活；但它是**最快起步**的选项，下面列出搭建方法，你可以把它当作"快速验证 + 找 bug 工具"。

如果你想用 gym-lowcostrobot 先起步、后期切换到自建 MuJoCo + Menagerie，是合理路径：底层模型一致 + LeRobot 数据格式天然兼容，切换成本低。两个备选平台的具体搭建方法见本文末尾「备选平台搭建」一节。

---

## 任务清单

### T1.1 安装并验证 MuJoCo

**目标**：能加载并渲染基本场景

**步骤**：
- [ ] `pip install mujoco`
- [ ] 运行 `python -m mujoco.viewer`，确认 GUI 弹出
- [ ] 用官方示例 humanoid.xml 验证物理仿真

**关键文件**：无

**参考**：
- `https://mujoco.readthedocs.io/en/stable/python.html`

**验证**：viewer 中可手动拖动物体

---

### T1.2 导入 SO-ARM100 模型

**目标**：在 MuJoCo 里加载机械臂

**步骤**：
- [ ] `git clone https://github.com/google-deepmind/mujoco_menagerie assets/menagerie`
- [ ] 找到 `assets/menagerie/trs_so_arm100/so_arm100.xml`
- [ ] 写一个最小加载脚本 `sim/scripts/load_so101.py`
- [ ] 用 viewer 打开，确认关节滑块能控制 6 自由度 + 1 夹爪

**关键文件**：
- `sim/scripts/load_so101.py`：最小加载器
- `assets/menagerie/trs_so_arm100/`：模型资产

**参考**：
- Menagerie SO-ARM100 README

**验证**：viewer 显示完整机械臂，关节滑块响应正确

---

### T1.3 搭建任务基类

**目标**：定义一个可复用的 BaseEnv 抽象

**步骤**：
- [ ] 设计 `sim/envs/base.py`，包含：
  - `reset(seed)`：初始化场景、采样物体姿态
  - `step(action)`：写入关节目标 → 物理仿真 → 返回 obs
  - `get_observation()`：返回 `{joint_pos, joint_vel, ee_pose, images}`
  - `evaluate_success()`：抽象方法
  - `render()`：返回 RGB + depth
- [ ] 暴露 gymnasium 兼容接口

**关键文件**：
- `sim/envs/base.py`

**参考**：
- LeRobot 仿真环境实现：`https://github.com/perezjln/gym-lowcostrobot`（gym-lowcostrobot 项目）
- 可直接借鉴其 `BaseRobotEnv` 结构

**验证**：基类可被子类继承并实例化

---

### T1.4 实现 PickPlaceBlue 环境

**目标**：第一个完整任务

**步骤**：
- [ ] 创建 `sim/envs/pick_place_blue.py`
- [ ] **场景包含**：SO-ARM100 + 桌面 + 红色立方体（4cm）+ 蓝色立方体（4cm）+ 盘子（圆柱 plate，直径 12cm × 高 1cm）
- [ ] **MJCF body origin 约定**：cube 的 body origin 放在 geom 中心（避免后续 IK 计算偏移）；plate 同理
- [ ] `reset()`：
  - 工作区切分为左右两块：**plate 固定在右半区**（位置抖动 ±3cm），**两个 cube 随机分散在左半区**（最小间距 8cm，避免紧贴）
  - cube 颜色绑定固定（红/蓝），不随机化颜色（颜色是语义锚点）
  - 初始 yaw 各自随机 [-π, π]
- [ ] `step()`：动作为关节位置目标（7 维 = 6 关节 + 1 夹爪）
- [ ] `evaluate_success()`：以下全部满足 → 成功
  1. blue cube 中心到 plate 中心 xy 距离 < plate 半径（5cm）
  2. blue cube 底面 z 在 plate 表面 ±1cm 内（确认落地）
  3. red cube 位置相对初始位移 < 2cm（**未被误碰**）
  4. 夹爪在终态已松开（gripper qpos > 阈值）
- [ ] `evaluate_failure_mode()`：返回失败原因枚举（`color_confusion` / `grasp_fail` / `place_miss` / `distractor_hit` / `timeout`）
- [ ] 写 MJCF：在 `assets/scenes/pick_place_blue.xml` 加入桌子 + 双 cube + plate

**关键文件**：
- `sim/envs/pick_place_blue.py`
- `assets/scenes/pick_place_blue.xml`

**参考**：
- robosuite 的 `PickPlace` 任务（结构参考，不是直接 import）：`https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/environments/manipulation/pick_place.py`

**验证**：`env.reset()` + `env.step(zero_action)` 能跑 100 步不崩；红/蓝/plate 三者在视觉上清晰可分

---

### T1.5 添加双相机渲染

**目标**：复刻真机相机配置

**步骤**：
- [ ] 在 MJCF 里加两个 camera site：
  - `wrist_cam`：吸附在 end-effector，俯视
  - `front_cam`：固定在桌前 30cm，俯角 30°
- [ ] 在 `get_observation()` 里渲染两路 RGB（640×480@30fps 模拟）
- [ ] 可选：同时输出 depth

**关键文件**：
- `assets/scenes/pick_place_blue.xml`（更新 camera）
- `sim/envs/pick_place_blue.py`（更新 observation）

**参考**：
- MuJoCo camera 文档：`https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera`

**验证**：obs 字典里 `images.wrist` 与 `images.front` 都是 640×480×3 uint8

---

### T1.6 实现微分 IK 控制接口

**目标**：能用 ee 位姿目标驱动机械臂

**步骤**：
- [ ] 用 mink 在 `sim/controllers/ik.py` 实现：
  ```
  ik(target_pose: SE3, current_q: np.ndarray) -> np.ndarray  # 返回目标关节
  ```
- [ ] 验证：给定一个桌面上方 10cm 的目标位姿，IK 解能让 ee 到位

**关键文件**：
- `sim/controllers/ik.py`

**参考**：
- mink 官方示例：`https://github.com/kevinzakka/mink/tree/main/examples`

**验证**：调用 IK 后跑 200 step 仿真，ee 位置与目标位置距离 < 5mm

---

### T1.7 实现最小脚本策略（6 阶段）

**目标**：在仿真里用硬编码策略完成 PickPlaceBlue 全流程

**步骤**：
- [ ] 在 `sim/scripted_policies/pick_place_blue.py` 写 6 阶段状态机：
  1. **APPROACH**：移到 blue cube 上方 10cm
  2. **DESCEND**：下降到 blue cube 抓取高度
  3. **GRASP**：闭合夹爪（持续 N 步累积闭合）
  4. **LIFT**：抬起 12cm（避开桌面与红 cube）
  5. **TRANSPORT**：水平移动到 plate 上方 10cm
  6. **PLACE_RELEASE**：下降到 plate 表面上方 3cm → 松开夹爪 → 抬起 5cm 撤离
- [ ] 每阶段用 IK 求关节目标 + 线性插值
- [ ] **关键约束**：TRANSPORT 阶段的水平路径要避开红 cube（中间路点高度足够 + 必要时绕行）
- [ ] policy 直接读 `obs["blue_cube_pos"]` / `obs["plate_pos"]`，不做颜色识别（颜色识别是 VLA 的事；此脚本是 oracle 策略）
- [ ] 用 `env.evaluate_success()` 判定成功

**关键文件**：
- `sim/scripted_policies/pick_place_blue.py`

**验证**：
- 跑 100 次随机 reset 的 PickPlaceBlue，成功率 > 70%
- 失败模式分布有日志（用于改进 policy）

---

### T1.8 域随机化模块（基础）

**目标**：让仿真生成的视觉数据具备分布多样性

**步骤**：
- [ ] 创建 `sim/randomization/`：
  - `lighting.py`：随机化光源位置、强度、颜色温度
  - `textures.py`：桌面纹理库（10+ 张 procedural / 公开图）
  - `cube_pose.py`：两个 cube 在左半区的 xy + yaw 随机化（保留最小间距约束）
  - `plate_pose.py`：plate 在右半区 ±3cm 抖动 + yaw 随机化
  - `camera_pose.py`：±5cm / ±5° 抖动 front camera
- [ ] **不要随机化 cube 颜色**：红/蓝是语言锚点，必须固定（否则破坏语义对齐）
- [ ] 可选：plate 颜色/纹理可随机（不影响 instruction），增强 sim2real 鲁棒性
- [ ] 在 `env.reset()` 中按概率调用
- [ ] 留出 `--no-domain-randomization` 开关用于 debug

**关键文件**：
- `sim/randomization/*.py`

**参考**：
- robosuite domain randomization：`https://robosuite.ai/docs/modules/randomizers.html`

**验证**：连续 reset 10 次截图，视觉差异明显

---

### T1.9 LeRobot 数据格式 dummy 导出

**目标**：先打通"仿真 → LeRobot dataset"管道，为 Phase 2 做准备

**步骤**：
- [ ] 在 `data/converters/sim_to_lerobot.py` 写一个 writer
- [ ] 用 T1.7 的脚本策略跑 5 条 episode 并保存为 LeRobot 格式
- [ ] `task` 字段统一为 `"put the blue cube on the plate"`
- [ ] 用 `lerobot-dataset-viz` 验证

**关键文件**：
- `data/converters/sim_to_lerobot.py`
- `data/sim_generated/pick_place_blue_v0/`：5 条 episode

**参考**：
- LeRobot dataset format：`https://huggingface.co/docs/lerobot/lerobot_dataset`

**验证**：可视化工具能正常播放仿真采集的 5 条 episode

---

## 验收标准（全部满足后进入 Phase 2）

- [ ] SO-ARM100 在 MuJoCo 中可控、可渲染
- [ ] PickPlaceBlue 环境实现完整（reset/step/success/失败归因/双相机）
- [ ] 脚本策略在仿真里 100 次随机 pick-place 成功率 > 70%
- [ ] 域随机化模块可独立开关（cube 颜色不参与随机化）
- [ ] 5 条仿真 episode 已成功导出为 LeRobot 格式并可视化

---

## 风险与陷阱

| 风险 | 应对 |
|------|------|
| Menagerie 的 SO-ARM100 模型与真机有几何差异 | 用真机实测的关节限位 / 长度反向校准 MJCF |
| IK 解奇异时跳变 | mink 加 damping 项；或在策略里把目标位姿限制在工作空间内 |
| 渲染太慢拖累采集 | 用 `mujoco.MjvOption` 关掉阴影；离屏渲染用 EGL |
| Domain randomization 太激进导致策略不收敛 | 先在不随机化的环境验证策略，再加 DR |
| 夹爪在仿真里"抓住"了但物理不稳定 | MJCF 里给 cube 设 friction `1 0.005 0.0001`，给夹爪指尖加软材质 |

---

## 输出物

- 可用的 SO101 仿真环境（PickCube）
- 仿真 → LeRobot 数据转换工具
- 脚本策略 baseline（Phase 2 的种子）
- 域随机化模块

---

## 备选平台（已独立成档）

本 Phase 主路径用 MuJoCo + Menagerie。下面 5 个平台是不同场景下的备选 / 学习参考，**每个都有独立的详细搭建文档**：

| 备选 | 适用场景 | 详细文档 |
|------|---------|---------|
| LeRobot 自带 sim | Phase 0–1 **最快起步**（gym-lowcostrobot 含 SO-ARM 原生） | [alt-platform-lerobot-sim.md](alt-platform-lerobot-sim.md) |
| RoboCasa / MimicGen | **Phase 3 算法移植必学** | [alt-platform-robocasa-mimicgen.md](alt-platform-robocasa-mimicgen.md) |
| Genesis | GPU 极速数据生成对比 | [alt-platform-genesis.md](alt-platform-genesis.md) |
| ManiSkill 3 | 公开 demo 数据集 + GPU 并行 | [alt-platform-maniskill3.md](alt-platform-maniskill3.md) |
| Isaac Lab | NVIDIA 工业级 RTX 仿真 | [alt-platform-isaaclab.md](alt-platform-isaaclab.md) |

### 快速决策树

```
你现在的状态？

├─ Phase 0 还没出第一份 LeRobot dataset
│   → LeRobot 自带 sim（1 天闭环）
│
├─ Phase 1 主路径开干
│   → MuJoCo + Menagerie（本文档前 9 个任务）
│
├─ Phase 2 想加速大规模数据生成
│   → 跑 Genesis / ManiSkill 3 对比实验
│
├─ Phase 3 准备实现数据扩增
│   → RoboCasa / MimicGen 必学算法
│
└─ Phase 6 长期研究 / 想做 RL / 要 photorealism
    → Isaac Lab
```

完整对比矩阵见 [docs/plans/README.md](README.md) 的「备选仿真平台」表格。