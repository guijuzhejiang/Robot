# 实现状态总览

> 本文档跟踪 [docs/plans/](plans/) 中各 Phase 的**实际代码实现状态**。
> Plans 描述"要做什么"，本文档描述"已做了什么 + 怎么用"。
>
> **维护规则**：
> 1. 每次新 phase 文件落地时，在对应表格新增一行（或更新现有行）
> 2. 不在表格里写"将来要做的事"——那些留在 [docs/plans/](plans/)
> 3. 每个文件用一句话说清"实现内容"，含关键设计取舍
> 4. 表格末尾追加"启动命令"+"已知限制"两节，新发现就补
> 5. 文档末尾的"待实现"部分按 Phase 倒排，已实现的从那里移到上面

---

## 总览

| Phase | 状态 | 文件数 | Commit |
|-------|------|--------|--------|
| 0 环境与可行性验证 | 未启动（需硬件） | 0 | — |
| 1 仿真平台搭建 | ✅ 已实现 | 10 | `5f23569` |
| 2 自动轨迹生成 | ✅ 已实现 | 6 | `5f23569` |
| 3 真机 demo + sim 扩增 | ⚠️ 软件实现完成（需真机 demo 才能跑完整闭环） | 11 | (待 commit) |
| 4 VLA 微调 | 未启动 | 0 | — |
| 5 真机部署 | 未启动（需硬件） | 0 | — |
| 6 HIL recovery | 未启动（需硬件 + 踏板） | 0 | — |
| 7 扩展任务 | 未启动 | 0 | — |

---

## Phase 1：仿真平台搭建

对应计划：[phase1-simulation-platform.md](plans/phase1-simulation-platform.md)

| 文件 | 对应任务 | 实现内容 |
|------|----------|----------|
| `assets/scenes/pick_place.xml` | T1.2 / T1.4 / T1.5 | SO-ARM101 + 红/蓝 cube + plate + 双相机（front + wrist_cam），via `<include>` 引用 menagerie 的 `so101.xml`；home keyframe 把 ee 摆在工作区中央上方 14cm |
| `assets/scenes/assets` (symlink) | T1.2 | 指向 `../menagerie/robotstudio_so101/assets/`，修复 MuJoCo `<include>` 的 meshdir 解析（include 文件的 meshdir 实际相对于 PARENT 文件位置） |
| `sim/envs/base.py` | T1.3 | `BaseSoArmEnv`：Gymnasium 兼容、双相机渲染、joint/ee 双 action mode；ee 模式通过 lazy import 调用 `EeIkController` |
| `sim/envs/pick_place.py` | T1.4 | 任务 env：左半区随机化双 cube（最小间距 8cm）、右半区随机化 plate；6 条 success 判据 + 失败归因枚举（`color_confusion` / `grasp_fail` / `transport_collision_red` / `place_miss` / `lift_drop` / `joint_limit` / `timeout`） |
| `sim/controllers/ik.py` | T1.6 | mink 5-DoF position-only IK（orientation_cost=0，因为 5-DoF 无法稳定满足 6-DoF 目标）；warmstart from prev q；20 iters 可达 z=0.005 |
| `sim/scripted_policies/pick_place.py` | T1.7 | 6 阶段 oracle 策略：APPROACH → DESCEND → GRASP → LIFT → TRANSPORT → PLACE_RELEASE；策略读 `obs["blue_cube_pos"]` / `obs["plate_pos"]`（oracle，VLA 必须从图像 + 语言学到同等行为） |
| `sim/randomization/lighting.py` | T1.8 | 3 个点光源位置 + 方向 + diffuse RGB 随机化 |
| `sim/randomization/textures.py` | T1.8 | 桌布 6 色预设（procedural；PBR 纹理切换待 ambientCG 集成） |
| `sim/randomization/cube_pose.py` | T1.8 | 双 cube xy + yaw 采样（含最小间距约束） |
| `sim/randomization/plate_pose.py` | T1.8 | plate 位置 ±3cm 抖动 |
| `sim/randomization/camera_pose.py` | T1.8 | front camera xy ±3cm / z ±2cm / yaw ±5° 抖动 |
| `data/converters/sim_to_lerobot.py` | T1.9 | `make_or_resume_dataset()`：默认续接，损坏检测（缺 `meta/tasks.parquet` 时自动重建）；`DatasetWriter` 包装 `LeRobotDataset.add_frame()` 适配 0.5.x 单 dict 签名 |

**重要设计约定**：
- 红/蓝 cube **颜色固定**，DR 不参与（语言锚点）
- gripperframe 是 fingertip-between 位置；descend 目标 z ≈ 0.008（fingertip 几乎贴桌面）让夹爪从两侧抱 cube
- 自由体 cube/plate 的 body origin = geom center，避免 z 偏移

---

## Phase 2：自动轨迹生成

对应计划：[phase2-trajectory-generation.md](plans/phase2-trajectory-generation.md)

| 文件 | 对应任务 | 实现内容 |
|------|----------|----------|
| `sim/grasp/antipodal.py` | T2.1 | 候选抓取采样：两条 cube 主轴 × N 个 yaw 抖动 × xy 抖动 ±5mm；打分项 = 远离 obstacles（红 cube + plate）+ 偏好小 yaw |
| `sim/planning/waypoint.py` | T2.2 | 笛卡尔空间 N 个 waypoint 线性插值 + 逐点 IK + 关节限位检查；保存/恢复 data.qpos 不污染 env 状态 |
| `sim/scripted_policies/pick_place_pipeline.py` | T2.3 | `generate_pickplace_episode(env, seed)`：env.reset → grasp sample → 6 阶段 oracle 运行；DESCEND 阶段用 grasp sampler 的最佳 xy 覆盖 |
| `sim/collectors/parallel_runner.py` | T2.7 | `--num-workers N`：每 worker 独立 env + 独立 LeRobotDataset shard；CLI 入口；汇总 success/失败模式分布；spawn context 防止 EGL 上下文冲突 |
| `data/instructions/pick_place.txt` | T2.9 | 15 条同义指令（"put"/"place"/"move"/"grab"/"drop" + plate/dish + 是否提及红 cube） |
| `eval/audit_dataset.py` | T2.8 | 磁盘直读审计（绕过 HF Hub 401）：parse `meta/info.json` + `data/chunk-*/file-*.parquet` + `meta/tasks.parquet`；输出 num_episodes、avg duration、per-ep jerk p95、task 分布 |

**未实现的 Phase 2 任务**（plan 列出但本次不交付）：
- T2.4 失败筛除：env 已内置 `evaluate_success()` 与失败归因，但 `failures.log` 沉淀文件未做
- T2.5 完整 DR 5 维：lighting / camera / textures / cube_pose / plate_pose 都有了，但 `dynamics.py`（质量/摩擦扰动）和 HDRI 集成未做
- T2.6 LeRobot writer 正式版：基础版已能跑（writer 在 Phase 1），但 metadata 字段（`seed` / `object_color` / `sim_random_seed` / `source`）未加
- T2.10 正式批量生成：infrastructure ready，未跑 1000 条 + 推 HF Hub

---

## Phase 3：真机 demo 采集与 sim 扩增

对应计划：[phase3-real-demo-sim-augmentation.md](plans/phase3-real-demo-sim-augmentation.md)

| 文件 | 对应任务 | 实现内容 |
|------|----------|----------|
| `configs/world_frame.yaml` | T3.1 | 桌面 / robot base / camera 统一坐标 + 工作区 bound（与 Phase 1 env 同步）；含 ChArUco 标定 procedure 注释；calibrated=false 等真机标定 |
| `configs/cameras.yaml` | T3.1 | front/wrist 相机 intrinsics + distortion 模板（待真机标定后填） |
| `data/mimicgen_adapter/types.py` | 共享类型 | `ObjectPose` / `Frame` / `Segment` / `SegmentedDemo` dataclass |
| `data/mimicgen_adapter/object_tracker.py` | T3.3 | HSV color mask + connected components + 3D back-projection；`CameraModel.from_mujoco()` 用 MJCF 真值做单元测试；`CameraModel.from_yaml()` 走真机标定路径 |
| `data/mimicgen_adapter/segmenter.py` | T3.3 | gripper-event 边界检测 4 段（approach_blue / grasp / transport / place_release），每段带 anchor 标注 |
| `data/mimicgen_adapter/replayer.py` | T3.4 | 多锚点 SE(2) 变换（平移+yaw）+ transport 段 cruise-z 插值 + 红 cube avoidance 抬升；输出 ReplayResult |
| `data/mimicgen_adapter/augment.py` | T3.5 | 编排：`synthesize_source_demos()` 用脚本策略合成种子（开发期）+ `augment()` 跑 N×K 次 replay + 写 LeRobot dataset；CLI 入口 |
| `data/converters/merge_datasets.py` | T3.6 | 多 dataset 合并，per-episode source 标签；含 shape (tuple) 修正与 dtype coerce；视频特征未支持（NotImplementedError） |
| `data/converters/expand_instructions.py` | T3.7 | 把 1 episode 复制成 K 个不同 `task` 字符串的副本；视频特征未支持 |
| `eval/audit_dataset.py` (extended) | T3.8 | 增加 `_per_source_stats()`：按 `source` 列分桶统计 episode 数 / 帧数 / jerk p95 / 平均时长；markdown 输出含 source 表 |

**Phase 3 关键设计取舍**：
- **Transform 数学：** 单元测试通过 (3/3) — pure shift + pure yaw + transport+red-avoid 均符合预期（详见 commit 中的 inline tests）
- **Object tracker：** 红/蓝 cube 在 sim 渲染下平均 1.7mm 误差 ✓；plate 35mm 因透视椭圆几何不可逆已记录（真机用 ChArUco + AprilTag 解决）
- **Synthesize source demos：** 默认 `require_success=False`，因为 Phase 1 scripted policy 还不能稳定完成任务；这条 path 只用于 pipeline 调试，真实场景永远用真机遥操 demo
- **Video merge：** 显式 NotImplementedError 而非静默失败 — 三个解：降为 image features / 训练时分别加载 / 后续实现 mp4 chunk symlink 合并

**未实现的 Phase 3 任务**：
- T3.2 真机 demo 采集：CLI 命令（`lerobot-record ...`）在 plan 文档里，需真机执行
- T3.3 真机 LeRobot dataset → Frame 转换：`segmenter.episode_from_lerobot_dataset` 是 NotImplementedError stub（需 FK 帮助函数从 qpos 反推 ee pose；joint-mode 数据集才需要，ee-mode 数据集直接用 action 累积即可）
- T3.5 整合：augment 走的是合成种子 path；真机种子 path 在 `episode_from_lerobot_dataset` 落地后即可直连
- T3.8 人工抽检 50 条 mimicgen 数据：交付物是工具，人工部分留给用户

---

## 启动命令

```bash
# 激活环境
conda activate py312_cu121

# 单进程跑 5 ep（带 DR）
python -m sim.collectors.parallel_runner --num-episodes 5 --num-workers 1 \
    --repo-id local/so101_pickplace_test

# 8 worker 并行 1000 ep（生产规模）—— 产出 8 个 shard 数据集
python -m sim.collectors.parallel_runner --num-episodes 1000 --num-workers 8 \
    --repo-id local/so101_pickplace_v1

# 同上 + 自动清掉每个 shard 里的 images/ 临时 PNG（保留 data/ + videos/ + meta/）
python -m sim.collectors.parallel_runner --num-episodes 1000 --num-workers 8 \
    --repo-id local/so101_pickplace_v1 --cleanup-after-collect

# 关闭 DR（debug 用）
python -m sim.collectors.parallel_runner --num-episodes 5 --num-workers 1 \
    --repo-id local/so101_pickplace_test --no-dr

# 审计已生成的 shard（含 source-aware 统计，若 dataset 有 source 字段）
python -m eval.audit_dataset --repo-id local/so101_pickplace_v1_shard00 \
    --out-dir data/sim_generated/audit/

# 把 N 个 shard 合并成单一训练数据集（必做步骤；训练只认一个 repo_id）
python -m data.converters.merge_shards \
    --shard-glob local/so101_pickplace_v1_shard* \
    --output-repo local/so101_pickplace_v1

# 合并完验证无误后，手动清掉 shard（合并不会删源）
rm -rf ~/.cache/huggingface/lerobot/local/so101_pickplace_v1_shard*

# === Phase 3 工具 ===
# 合成种子 demo 跑 MimicGen 扩增 (pipeline 调试用；真实场景用真机种子)
python -m data.mimicgen_adapter.augment --from-sim-seeds 5 \
    --output-repo-id local/so101_sim_mimicgen_v1 --n-per-demo 20

# 合并多源 dataset（state-only；视频需要先 finalize + 重编码）
python -m data.converters.merge_datasets \
    --source local/so101_real_pickplace_v0:real \
    --source local/so101_sim_mimicgen_v1:sim_mimicgen \
    --output-repo-id local/so101_pickplace_mixed_v1

# 用指令池扩 K 倍语言多样性（不重新渲染，仅复制 task 字段）
python -m data.converters.expand_instructions \
    --source-repo-id local/so101_real_pickplace_v0 \
    --output-repo-id local/so101_real_pickplace_v0_langx3 \
    --copies 3

# 单测某个文件（示例）
python -c "
import sys; sys.path.insert(0, '.')
from sim.envs.pick_place import PickPlaceEnv
env = PickPlaceEnv(observation_mode='both', action_mode='ee')
obs, _ = env.reset(seed=0)
print('blue:', obs['blue_cube_pos'], 'plate:', obs['plate_pos'])
"
```

数据集默认写到 `~/.cache/huggingface/lerobot/<repo_id>/`（LeRobot 0.5.x 标准位置）。

---

## LeRobot 数据集目录结构和并行采集工作流

每个 LeRobot 数据集（一个 `repo_id`）的标准目录布局：

```
<repo_id>/
├── meta/
│   ├── info.json              ← schema + fps + 总帧数（必需）
│   ├── tasks.parquet          ← task_index ↔ task 字符串（必需）
│   ├── stats.json             ← 各特征的均值/方差（必需）
│   └── episodes/chunk-NNN/file-NNN.parquet  ← 每个 episode 的 [from,to] 时间戳（必需）
├── data/chunk-NNN/file-NNN.parquet          ← 每帧 state/action/timestamp（**必需**，训练的核心标签）
├── videos/
│   ├── observation.images.front/chunk-NNN/file-NNN.mp4  ← 同 chunk 内多个 episode 拼到一个 mp4（必需）
│   └── observation.images.wrist/chunk-NNN/file-NNN.mp4
└── images/                    ← LeRobot 视频编码时的临时 PNG staging dir（**可删**）
```

**训练只需要 `meta/` + `data/` + `videos/`**；`images/` 是 LeRobot 编码 mp4 后的临时残留，安全可删（`--cleanup-after-collect` 就是干这事）。

### 各目录在 VLA 训练里的角色

VLA（视觉-语言-动作）模型训练同时吃**视觉 + 本体感受 + 语言指令 → 预测动作**四类信号，分别落在不同文件里：

| 信号 | 来源 | 说明 |
|---|---|---|
| 🎥 视觉输入 | `videos/observation.images.{front,wrist}/chunk-NNN/file-NNN.mp4` | 多个 episode 拼到一个 mp4，按 `meta/episodes/*` 里的 `from_timestamp` / `to_timestamp` 切出该 episode 的视觉帧 |
| 🦾 本体感受 | `data/chunk-NNN/file-NNN.parquet` 的 `observation.state` 列 | 当前帧 6-DoF 关节角，float32×6 |
| 💬 语言指令 | `meta/tasks.parquet` 经 `task_index` 查表 | `data/` 每帧带 `task_index`（int），到这里查出字符串（当前数据集只有 `task_index=0` → `"put the red cube on the plate"`） |
| 🎯 动作标签 | `data/chunk-NNN/file-NNN.parquet` 的 `action` 列 | 预测目标，float32×6（6 维 joint position，对齐 LeRobot SO-101 teleop 惯例：`shoulder_pan.pos / shoulder_lift.pos / elbow_flex.pos / wrist_flex.pos / wrist_roll.pos / gripper.pos`） |
| 🛠️ sim 辅助 | `data/chunk-NNN/file-NNN.parquet` 的 `ee_action` 列 | sim-only，float32×4（dx, dy, dz, gripper），归一化 [-1, 1]。**VLA 训练不读这一列**，仅用作 sim 调试与 Stage-2 派生工具输入 |

`data/chunk-NNN/file-NNN.parquet` 的列：

```
observation.state  float32 [6]   ← 本体感受输入（6 个关节当前位置）
action             float32 [6]   ← 预测目标（监督标签：6 维 joint position target）
ee_action          float32 [4]   ← sim-only 辅助列（dx,dy,dz,gripper 归一化）；VLA 训练不读
task_index         int64         ← 语言指令外键（→ meta/tasks.parquet）
episode_index      int64         ← 属于哪条 episode
frame_index        int64         ← 在 episode 内的第几帧
index              int64         ← 全局帧号
timestamp          float32       ← 用于到 mp4 里取对应视觉帧
```

`meta/` 子目录详解：

| 文件 | 作用 | 训练时怎么用 |
|---|---|---|
| `meta/info.json` | 数据集 schema：feature 名、dtype、shape、fps、total_frames/episodes | LeRobotDataset 加载时按它解析 `data/*.parquet` 列和 `videos/*.mp4` 路径 |
| `meta/tasks.parquet` | `task_index` ↔ 语言字符串映射表 | 每帧从 `data/` 拿 `task_index`，到这里查出 prompt 喂 VLM 文本编码器 |
| `meta/episodes/chunk-NNN/file-NNN.parquet` | 每条 episode 的索引：长度、`dataset_from_index` / `dataset_to_index`、视频 chunk + `from_timestamp`/`to_timestamp`、per-episode stats | 把 `data/` 行号映射到 episode + 把 episode 映射到具体 mp4 时间段 |
| `meta/stats.json` | 全局 stats（state/action 的 min/max/mean/std） | 训练时 normalize 输入、unnormalize 输出 |

**`images/` 目录在训练时完全不用**（只是 LeRobot 把帧编码成 mp4 时的中转 PNG），所以 `--cleanup-after-collect` 只删它。

### 术语：action_mode 与 TCP

**`action_mode`**（`sim/envs/base.py:49`）只有两种取值：

| mode | action shape | 含义 | step 内部 |
|---|---|---|---|
| `"ee"` | (4,) `[dx, dy, dz, gripper]` | **end-effector**（末端执行器）空间增量，世界系 | `dx/dy/dz × EE_DELTA_SCALE` 加到 TCP 当前世界坐标 → DLS IK 求关节解 → 写 `ctrl[:5]`；`gripper` 线性映射到 `ctrl[5]` |
| `"joint"` | (6,) | 6 关节归一化目标（每维 ∈ [-1, 1]） | 线性映射到 `ctrl_limits` 直接写 `ctrl[:6]`，**不走 IK** |

**重要区分**：`env.action_mode` 是 **sim 内 step() 接受什么输入**，与 **LeRobot dataset 存什么 action 列**是两件事：

- `env.action_mode='ee'`：sim 采集时脚本策略输出 `[dx,dy,dz,gripper]`，env 内部走 IK 解出 `ctrl[:6]`
- **LeRobot dataset `action` 列**：记录的是 IK 解出的 `ctrl[:6]`（6 维 joint position target，对齐 LeRobot SO-101 teleop 惯例）
- **LeRobot dataset `ee_action` 列**：sim 辅助列，记录 ee delta（VLA 训练不读）

即采集流程是 **"策略输出 ee → env 解 IK → 落盘 joint（主） + ee（辅）"**。本项目 PickPlaceRed env 用 `"ee"` 是为了让脚本策略代码简洁（不必手工算 IK），但**数据集对外暴露的是 joint**——与 SmolVLA / Pi 系列等 SO-101 主流 VLA 的预训练 schema 对齐。

**TCP = Tool Center Point**（工具中心点）——机器人学通用术语，指末端工具的**参考点**。对夹爪型 ee，就是两指之间的几何中心（"夹住物体时物体所在的位置"）。

在我们的 MJCF 里：

```xml
<!-- assets/so101_pick101/so101_new_calib.xml:110 -->
<!-- Frame gripperframe (TCP - tool center point, 10cm in front, centered between fingers) -->
<site group="3" name="gripperframe" pos="0.0 0.0 -0.0981274" .../>
```

即在 wrist 坐标系沿 z 轴向前 9.81cm 处建 `gripperframe` site，正好落在两指中点。VLA 训练里 dx/dy/dz 描述的就是 **TCP 在世界系的位移**；部署时 IK 也是把 TCP 驱到目标位姿。代码里通过 `env.ee_pos()` (`= data.site("gripperframe").xpos`) 读取它的当前世界坐标。

### 为什么 ee-mode action 要 ×0.05 归一化（不直接用米）

`EE_DELTA_SCALE = 0.05 m` 同时承担**两个职责**：

**(a) 单步 TCP 位移物理上限** —— env 每帧（33 ms / 30 FPS）最大允许 TCP 移动 5 cm，即 ee 速度上限 1.5 m/s。这是 SO-101 这种小臂物理可达的极限速度，也是 DLS IK 单步迭代能稳定跟踪的范围。超过这个数 IK 雅可比饱和会让关节抖动甚至发散。

**(b) 神经网络输出归一化范围** —— VLA 最后一层通常带 tanh 或线性，自然输出 [-1, 1]。如果让网络直接输出"米"，典型位移 0.001~0.01 m 对 tanh 范围太小，梯度信号弱；而且切换机器人 / 速度配置时模型尺度都得重训。

```
神经网络输出 ∈ [-1, 1]  ←──×0.05──→  TCP 实际位移 ∈ [-0.05, 0.05] m
```

| action[0] | 实际 dx |
|---|---|
| 0.1  | TCP +x 走 5 mm |
| 1.0  | TCP +x 走 50 mm（饱和） |
| -0.5 | TCP -x 走 25 mm |

**直接用米也可行**（LeRobot 不限制 dtype），但归一化的好处是统一 schema、健康梯度、切换机器人时**只改 `EE_DELTA_SCALE` 一个常量**。

### ee-mode env.step 内部 pipeline

```
                 env.step(action)              ← action ∈ [-1, 1]^4
action[:3] ──×EE_DELTA_SCALE──→ delta_xyz (米)
                                    │
                                    ▼
       target_TCP = data.site("gripperframe").xpos + delta_xyz   (世界系绝对坐标)
                                    │
                                    ▼  DLS IK (locked_joints=[3, 4])
                                    │  solve q s.t. FK(q) ≈ target_TCP
                                    ▼
                              q_arm (5 维关节角)
                                    │
                                    ▼
                              data.ctrl[:5] = q_arm

action[3] (gripper, absolute mode)
        │
        ├──线性映射──→  ctrl[5] = (a3 + 1) / 2 × (g_hi - g_lo) + g_lo
        │              (夹爪不走 IK，是独立执行器)
        ▼
   data.ctrl[5]

                        ▼  mj_step × SIM_STEPS_PER_CTRL (5×2ms = 10ms 物理仿真)
                        ▼  期间 IK 多次 refresh ctrl (pick-101 style 500Hz IK loop)
                        ▼
                  read state → 返回 obs
```

**关键点**：
- `gripperframe` 是 MuJoCo **site**（虚拟参考点），不是运动学链节点；IK 真正解的是 wrist_link 等真实 body 的关节角，目标只是让 FK(joint_angles) 的 TCP 位姿匹配 `target_TCP`
- IK 在每个 mj_step 之间用 `gain=0.5` 指数衰减反复迭代——这就是 override 模式（固定绝对目标）下 TCP 单步能走超过 5cm 的原因（多 substep 累加）
- 夹爪 ctrl[5] 与手臂 ctrl[:5] **完全独立**，IK 不动 ctrl[5]

### LeRobot 上 VLA 训练用 ee-mode 还是 joint-mode？

**LeRobot 框架本身不挑**（`ACTION="action"` 硬编码，没有 `action_mode` 概念）；**用什么取决于 dataset 列怎么存** + **VLA 模型预训练时学的是什么语义**。本项目按 LeRobot SO-100/101 ecosystem 惯例 dataset `action` 列存 6 维 joint position（与官方 leader-follower teleop 输出一致），不同 VLA 模型走不同路径——详见下方分类。

各 VLA 预训练 action schema：

| VLA / Policy | 预训练 action 语义 | SO-101 微调路径 |
|---|---|---|
| **SmolVLA** | 6-DoF joint position（LeRobot Community Dataset，~95% SO-100/101） | **直接用** Stage-1 dataset（schema 逐位对齐） |
| **Pi-0 / Pi-0.5 / Pi-0-FAST** | 32-D zero-padded（混合 ecosystem，UR5/Franka/Trossen 双臂等） | **直接用** Stage-1 dataset（openpi loader 运行时 pad 6→32） |
| **ACT** | joint position（Aloha 14-D 双臂，无预训练） | **直接用** Stage-1 dataset（从零训，action head 适配 6-D） |
| **Diffusion Policy** | 无固定预训练 | **直接用** Stage-1 dataset |
| **OpenVLA** | 7-DoF ee delta（xyz + rpy + gripper，Open-X） | **跑 Stage-2 派生** → `ee_delta_7d_euler` |
| **X-VLA** | 10-D ee pose（xyz + Rotate6D + gripper，DROID/Robomind） | **跑 Stage-2 派生** → `ee_pose_10d_rotate6d` |
| **RT-1 / RT-2 / RT-X** | 7-DoF ee delta，离散化 | **跑 Stage-2 派生** → `ee_delta_7d_euler` |
| **gym-lowcostrobot 默认** | joint normalized | **直接用** Stage-1 dataset |

判别准则：
- 模型预训练 action 与我们 6-D joint 维度+语义**对齐或仅维度补齐** → 直接用 Stage-1（Pi 系的 zero-pad 属此类）
- 模型预训练 action **语义不同**（要末端笛卡尔位姿/旋转） → 必须跑 Stage-2 派生（`data/converters/derive_action_format.py`）

**两种语义的取舍**（仅供选 VLA 时参考，不影响 Stage-1 数据采集）：

| 维度 | ee 语义 VLA（OpenVLA/X-VLA） | joint 语义 VLA（SmolVLA/ACT） |
|---|---|---|
| **跨 DoF 机器人迁移**（如 6→7 DoF） | ✅ ee 普适，零配置 | 🟡 需换 action head + fine-tune（VLA 社区方案见下方） |
| **同形态迁移**（如 SO-100 → SO-101） | ✅ | ✅ **同样直接**，两者无差距 |
| **不同形态**（单臂 → 双臂） | ✅ 每只手独立命令拼接 | ❌ 维度爆炸，需重设计 |
| 数据可解释性 | ✅ "夹爪去哪"直观 | ❌ 6 个角度看不出意图 |
| 训练简单度 | 🟡 部署多一层 IK | ✅ 直接写 ctrl |
| 关节限位 / 奇异点 | ❌ 模型可能预测不可达 ee 位姿 | ✅ 物理上始终可执行 |
| 视觉对齐 | ✅ ee 在像素空间易 ground | 🟡 需 FK 才能算 ee |

#### 关于 joint-mode "跨机器人迁移" 的常见误解

**误解**："joint-mode 不能做跨机器人迁移，因为不同臂关节定义不同"。

**事实**：joint-mode **能**做跨机器人迁移，但靠工程手段而非 action 表示本身。VLA 社区主流四种方案：

1. **固定维度 + zero padding**（Pi-0 用）：模型内部 action 维度固定 32，不同 DoF 的臂在后部补 0，预训练时学"在 padding 位置不做有意义预测"
2. **per-robot action head**（OpenVLA / RT-X / SmolVLA 用）：共享 backbone + 共享视觉/语言编码器 + per-robot 输出头，迁移时冻 backbone 换 head
3. **embodiment token**（RT-2 / Pi-0.5 用）：输入加 "robot ID embedding"，同组权重通过 conditioning 自适应不同 DoF
4. **同形态预训练 + 同形态迁移**（SmolVLA 实际走的路）：SmolVLA 预训练集 LeRobot Community Dataset 里 ~95% 是 SO-100/SO-101 数据，DoF 一致；同形态迁移时连 head 都不用换

**因此**：用户经验"SmolVLA 在 SO-101 上 joint-mode 迁移好用"是真实的，但本质是**同形态迁移**（SO-100→SO-101），不是 joint-mode 天然跨 DoF。**真要切到 7-DoF Franka 等不同形态，joint-mode 必须换 head + 重训，ee-mode 仍可直接用**。

**ee-mode 真正的迁移优势**是"零配置跨 DoF"——任意 6/7/14 DoF 的臂只要带夹爪都用同样的 `[dx, dy, dz, gripper]` schema，pretrained 权重无需任何 head 改造直接用。joint-mode 不是不能跨，而是**跨形态时需要工程改造**，同形态时两者无差距。

#### 本项目最终选择：joint 主 + ee 辅（已落代码）

经过对 LeRobot 框架代码的实证审计（`ACTION="action"` 硬编码、SO-101 teleop 直接读舵机当前位置作 action、HF 上所有 SO-100/101 dataset 全是 6 维 joint），项目最终采用：

- **`action` 列 = 6 维 joint position**（主标签，所有 VLA 训练读这一列）
- **`ee_action` 列 = 4 维 ee delta**（sim 辅助，VLA 训练**不读**）

不同 VLA 微调的数据流：

| 模型 | 数据流 |
|---|---|
| SmolVLA / Pi-0 / Pi-0.5 / Pi-0-FAST / ACT / Diffusion Policy | **直接用 Stage-1 数据**（joint 列与 SmolVLA pretrain schema 逐位对齐；Pi 系 openpi loader 自动 pad 到 32 维，不需要数据集端改造） |
| OpenVLA / X-VLA / RT-X 系 | **必须先跑 Stage-2 派生** `data/converters/derive_action_format.py`，从 joint 列走 FK 生成 ee_delta_7d / ee_pose_10d_rotate6d 等格式的新 dataset |

判别准则：
- **维度补齐**（如 Pi 把 6 维 zero-pad 到 32 维）→ 运行时由模型 loader 完成，dataset 不变 → 直接用 Stage-1
- **语义改变**（joint 角 ↔ ee 笛卡尔位姿/旋转）→ 必须 dataset 层转换 → 跑 Stage-2

保留 `ee_action` 列的实用价值：①sim 调试时方便看世界系增量；②Stage-2 派生工具可以选择是直接复用 `ee_action` 还是用 FK 重算（FK 路径更通用，目前实现走 FK）。

### 数据集 action 列设计 —— 对齐 LeRobot SO-100/101 惯例（已落代码）

#### 重要纠错：之前的"ee/joint mode 由模型选择"理解是错的

早期分析认为不同 VLA 模型需要不同 action 表示（ee-mode 喂 Pi-0.5、joint-mode 喂 SmolVLA），所以维护"双 action 列"。**这个推理偏了**：

- **LeRobot 没有 `action_mode` 这个概念** —— 代码层 `action` 列名硬编码（`lerobot/utils/constants.py:33: ACTION = "action"`），没有 `dataset.action_key` 配置项
- **action 表示由硬件 ecosystem 决定，不是模型** —— LeRobot 官方 SO-101 leader-follower teleop（`teleoperators/so_leader/so_leader.py:148`）就是 `bus.sync_read("Present_Position")` 直接读 6 个舵机当前位置作为 action。HF 上所有 SO-100/101 dataset **全是 6 维 joint position**，没人在"选 mode"
- **SmolVLA 预训练 100% 用这种 SO-100 joint 数据**，所以 fine-tune 时数据格式天然对齐
- 其他 VLA（Pi-0/0.5、OpenVLA）即使在不同 ecosystem 上预训练，它们**自己内部用 padding / discretization 处理 shape mismatch**——dataset 端不需要为它们改格式

#### 当前最终 schema（已落代码）

| 列名 | shape | dtype | names | 含义 |
|---|---|---|---|---|
| **`action`**（主）| (6,) | float32 | `[shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos]` | **6 维 joint ctrl 目标**，弧度（5 关节）+ 夹爪 qpos。**与 LeRobot SO-101 teleop 完全对齐**，与 SmolVLA pretrain 数据集 `lerobot/svla_so100_pickplace` 同 schema |
| `ee_action`（辅）| (4,) | float32 | `[dx, dy, dz, gripper]` | sim-only ee-delta，归一化 [-1, 1]（×`EE_DELTA_SCALE`=0.05m 还原米）。仅用于：①sim 调试可视化；②Stage-2 派生工具的输入 |

**`observation.state`** 同步用 LeRobot 惯例的 `.pos` 后缀 names。

#### LeRobot 本身不做 ee↔joint 自动转换（背景）

LeRobot 框架本身没有 ee↔joint 转换逻辑——`action` 列名硬编码（`lerobot/utils/constants.py:33: ACTION = "action"`），整个框架（dataset 加载、processor、policy、trainer）只读名字叫 `"action"` 的列。无 `dataset.action_key` 配置项可切换列源。

这就是为什么我们**直接让 `action` 列存"主流要的格式"**（SO-100/101 ecosystem 是 6 维 joint），需要 ee-mode 模型时再用 Stage-2 派生工具转。

#### 为什么 `action` 记 `ctrl[:6]` 而不是 `state[t+1] - state[t]`

`ctrl[:6]` 是 IK + gripper 逻辑刚解出来发给 actuator 的命令——是 ACT/SmolVLA 训练惯例里 "joint position target" 的最干净对应，也是 LeRobot SO-101 follower 收到的 `Goal_Position` 等价物。

如果用 `state[t+1] - state[t]`（实际达到的关节角变化），会受 sts3215 严重欠阻尼（ζ≈0.26，见 `docs/lessons-learned-so101-grasp.md`）影响——smoke 实测两者每维差 0.01-0.02 弧度（夹爪因接触反弹差 0.15）。

#### 落地点

| 文件 | 改动 |
|---|---|
| `data/converters/sim_to_lerobot.py` | `action` 重定义为 6 维 joint（`.pos` 后缀 names）；新增 `ee_action` 为 sim-only 4 维辅助；`add_frame_from_obs(action, *, task, ee_action=None)` 新签名 |
| `sim/collectors/parallel_runner.py` | 变量重命名：`frame_actions` 装 joint，`frame_ee_actions` 装 ee；与列名语义一致 |
| `sim/scripted_policies/pick_place_pipeline.py` | `Episode.actions` 装 joint，`Episode.ee_actions` 装 ee |
| `data/mimicgen_adapter/replayer.py` | `ReplayResult.actions` 装 joint，`ReplayResult.ee_actions` 装 ee。replay 内部 `env.step` 还是吃 ee（因 `action_mode='ee'`），但**记录的是 joint** |
| `data/mimicgen_adapter/augment.py` | `_write_episode` 用 kwargs 接收两类 action，列 `action`=joint、`ee_action`=ee；step 喂 ee |
| `data/converters/merge_shards.py` | **无需改动**（feature-driven） |

#### Smoke 验证（新 schema）

```
parquet 列：['observation.state', 'action', 'ee_action', timestamp, ...]

action shape (6,):    wrist_flex 全程 1.5708（locked_joints ✓）
                      shoulder_pan ∈ [-0.36, +0.55] 弧度（≈±20°，合理 ✓）
                      gripper ∈ [0.02, 1.17]（全 ctrl 范围 ✓）

ee_action shape (4,): dx ∈ [-0.14, 0.15]，非零率 392/392 ✓
                      gripper(norm) ∈ [-0.8, 0.4]

mimicgen augment：371 帧产出，action 列 wrist_flex 锁定 1.5708 ✓
```

### 训练时如何使用

**训 SmolVLA / ACT / Diffusion Policy / Pi-0 / Pi-0.5 / Pi-0-FAST（所有 SO-100/101 兼容模型）**：

```bash
python -m lerobot.scripts.lerobot_train \
    --policy.type=smolvla \
    --dataset.repo_id=local/so101_pickplace_v1
```

**零配置**——LeRobot 默认读 `action` 列，我们这列就是 6 维 joint，与 SmolVLA pretrain schema 逐位对齐（同 shape、同 names、同物理单位）。Pi-0 系也接受（自动 pad 到 32 维）。

**训 OpenVLA / X-VLA / 其他 ee-pose 风格 VLA**：需要 **Stage-2 派生工具**（已实现 `data/converters/derive_action_format.py`）把 `action` 列从 6-D joint 转成 ee 格式。从 `observation.state` 走 MuJoCo FK 即可生成，**不需重采**。支持 5 种目标格式（`ee_delta_4d / ee_delta_7d_euler / ee_pose_7d_euler / ee_pose_7d_axis_angle / ee_pose_10d_rotate6d`）。详见 `docs/plans/phase4-vla-finetuning.md` Stage-2 章节。

**v1 数据集**（896 ep 已采的）：因为之前 action[:3]=0 bug 反正要重采，这次新 schema 一并解决——新代码自动产出 LeRobot-惯例 dataset。

### VLA 训练时单帧样本的精确构成

一个训练样本（DataLoader 取出来喂给 VLA 的一条数据）的输入和标签如下：

**输入（4 项）**：

| 模态 | shape | 含义 | 物理意义 |
|---|---|---|---|
| `observation.images.front` | (480, 640, 3) uint8 | 顶视相机当前帧 | RGB 像素 |
| `observation.images.wrist` | (480, 640, 3) uint8 | 腕部相机当前帧 | RGB 像素 |
| `observation.state` | (6,) float32 | 当前帧 6 个关节角（弧度） | `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper_qpos]`；本任务里 wrist_flex≈π/2、wrist_roll≈1.224 被锁定 |
| `task`（经 `task_index` 查表） | str | 语言指令 | `"put the red cube on the plate"` |

注意是 **2 张图（front + wrist）**，不是 1 张。视觉部分包含两个视角。

**标签（1 项，主）**：

| 字段 | shape | schema 含义 | 物理意义 |
|---|---|---|---|
| `action` | (6,) float32 | `[shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos]` | **6 维 joint position target**（即每帧 IK + 夹爪逻辑解出的 `ctrl[:6]`），与 LeRobot SO-101 follower 收到的 `Goal_Position` 等价；前 5 维弧度（手臂关节），末维夹爪 qpos。**这就是 VLA 监督学习的标签**，与 SmolVLA pretrain schema 逐位对齐 |

**辅助字段（不参与 VLA 训练）**：

| 字段 | shape | 含义 |
|---|---|---|
| `ee_action` | (4,) float32 | `[dx, dy, dz, gripper]` 世界系 TCP 增量（×`EE_DELTA_SCALE`=0.05m 还原米）。仅供：①sim 调试；②Stage-2 派生工具（生成 OpenVLA/X-VLA 风格 ee 数据集）作输入参考 |

action 列为何记 `ctrl[:6]` 而不是 `state[t+1] - state[t]`：`ctrl[:6]` 是 IK + gripper 逻辑刚解出来发给 actuator 的 **命令**，是 ACT / SmolVLA 训练惯例的 "joint position target" 干净对应；用实际达成的 `state` 差值会受 sts3215 严重欠阻尼（ζ≈0.26）影响，smoke 实测两者每维差 0.01–0.02 弧度。

### ee_action 列的真实化（Phase 2 fix，已落代码）

> 历史顺序：先发现并修复了 ee delta 失真问题（action[:3]=0），后又把 `action` 主列从 ee 切换成 joint。本节描述的是 **当前 `ee_action` 辅助列**（旧名 `action`）的取值正确性修复。

**历史问题**：脚本策略走 `ee_target_override` 旁路驱动手臂，`sim/scripted_policies/pick_place.py:299` 写回给 gym 的 action 是 `[0, 0, 0, gripper]`——dx/dy/dz 恒为 0。早期 `local/so101_pickplace_red_v1`（896 ep / 180454 帧）就是这样收的，ee delta[:3] 全 0，**完全不能用于 Stage-2 派生 ee 数据集**（OpenVLA/X-VLA 训练会废）。

**修复方案**（已落代码）：在采集时**根据 TCP（`gripperframe` site，夹爪两指中点向前 10cm 的标准 ee 点）实际世界系位移**重新编码 `ee_action[:3]`：

```
ee_action_recorded[:3] = clip( (ee_after - ee_before) / EE_DELTA_SCALE, -1, 1 )
ee_action_recorded[ 3] = gripper_norm  (策略输出，原样保留)
```

其中 `EE_DELTA_SCALE = 0.05 m`（`sim/envs/base.py:BaseSoArmEnv.EE_DELTA_SCALE`）是 `_apply_ee_action` 用的同一个常量——这样 `ee_action` **可以直接喂回 `env.step()`**（action_mode='ee' 模式下走标准 delta IK）实现往返一致，便于 sim 内 ee 策略回放和 Stage-2 派生工具的可靠输入。

> **注意**：当前 `action` 主列是 6 维 joint，不再受这个 fix 影响（joint 列直接记 `ctrl[:6]`，没有 override 旁路问题）。该修复对**当前 dataset 的语义贡献**是：保证 `ee_action` 辅助列存的是真实 TCP 增量而非 [0,0,0,g]，使得未来跑 Stage-2 派生 ee 数据集时有可靠参考（即便 Stage-2 默认走 FK 重算）。

**改动落地点**：

| 文件 | 改动 |
|---|---|
| `sim/envs/base.py` | 新增 `EE_DELTA_SCALE = 0.05` 类常量 + `BaseSoArmEnv.encode_ee_delta_action(ee_before, ee_after, gripper_norm)` 辅助方法；`_apply_ee_action` 改用 `self.EE_DELTA_SCALE`（行为不变） |
| `sim/collectors/parallel_runner.py` | 主采集循环 step 前后采 `ee_pos()`，落盘前用 `env.encode_ee_delta_action(...)` 重写 `ee_action`；`action` 主列另记 `ctrl[:6]` joint |
| `sim/scripted_policies/pick_place_pipeline.py` | `generate_pickplace_episode` 同样修复（MimicGen 种子要用） |

**验证**（smoke test）：

```
修复前 (v1)：dx/dy/dz min/max = [0,0,0]/[0,0,0]，非零帧 0/180454
修复后      ：ee_action dx/dy/dz min/max = [-0.23, 0.21]，非零帧 392/392
            首 3 帧 APPROACH 阶段位移逐渐放大 (-0.011 → -0.058)，符合预期
            action (joint) shoulder_pan ∈ [-0.36, +0.55] 弧度 (合理)
```

**Phase 2 数据需重采**：`so101_pickplace_red_v1` 是修复前数据，且 schema 与新的 joint+ee_action 双列设计不一致，需要丢弃并用新代码重新跑 `parallel_runner` 重采 1000 条；之后的批次（v2+）天然带正确 schema。文档命令保持不变。

**MimicGen 路径审计结果**（Phase 3）：

| 位置 | 角色 | 状态 |
|---|---|---|
| `data/mimicgen_adapter/augment.py:synthesize_source_demos` `actions.append(action.copy())` | 收集合成源 demo | ⚪ 该 list 仅用于 `episode_from_ee_actions` 的 `len()`，**值未被使用**（Frame 用的是 `ee_positions` + `gripper_states`）。已加注释防止误读 |
| `data/mimicgen_adapter/replayer.py:replay_segmented_demo` 主循环 | 新场景下重建轨迹 | ✅ 本来就正确：`action[:3] = clip((tgt - ee) / EE_DELTA_SCALE, -1, 1)`，逻辑与 `encode_ee_delta_action` 完全等价。已把魔数 `0.05` 替换为 `env.EE_DELTA_SCALE` 防漂移 |
| `data/mimicgen_adapter/augment.py:_write_episode` 写盘循环 | 把 `result.actions` 写入 LeRobot | ✅ 因为 `result.actions[:3]` 由 replayer 算出，本来就是真实归一化 delta；`env.step(action)` 此时无 override，走标准 IK，arm 被正确驱动 |

**Smoke 验证**：`--from-sim-seeds 2 --n-per-demo 2` 产出 371 帧，dx/dy/dz 非零率 371/371，dz 在 transport cruise-z 抬升阶段触发 ±1 饱和（预期行为，14cm 抬升 / 单步 5cm 上限）。

### 串起来：训练一个 batch 的数据流

```
DataLoader 想取第 i 帧
   ↓
1. 读 data/chunk-XXX/file-XXX.parquet 第 i 行
   → 拿到 observation.state、action、task_index、episode_index、timestamp
   ↓
2. 用 task_index 去 meta/tasks.parquet 查语言字符串
   → "put the red cube on the plate"
   ↓
3. 用 episode_index 去 meta/episodes/*.parquet 查
   → 这条 episode 在哪个 video chunk、起止 timestamp
   ↓
4. 用 timestamp 去 videos/observation.images.front/chunk-XXX/file-XXX.mp4 切出对应帧
   （front 和 wrist 各一帧 → 视觉输入）
   ↓
5. 用 meta/stats.json 里的 mean/std 归一化 observation.state 和 action
   ↓
6. (image_front, image_wrist, observation.state, task_str) → VLA → 预测 action
   ↓ 与标签 action 算 loss
```

所以**少任意一个目录（`meta` / `data` / `videos`）训练都跑不起来**：缺 `data/` 没标签和本体感受；缺 `videos/` 没视觉；缺 `meta/` 没法解读前两者也拿不到语言指令。

`parallel_runner --num-workers N` 写出 N 个独立数据集（`<repo_id>_shard00 .. _shardNN`），每个都是完整 LeRobot 数据集，但训练只认单一 `repo_id`，所以采集完必须合并：

```bash
# 1. 采集（每个 worker 写一个 shard；可加 --cleanup-after-collect 立即清理临时 PNG）
python -m sim.collectors.parallel_runner --num-episodes 1000 --num-workers 8 \
    --repo-id local/so101_pickplace_v1 --cleanup-after-collect

# 2. 合并（重新编码 mp4 + 重编号 episode_index；约 50–100 fr/s）
python -m data.converters.merge_shards \
    --shard-glob local/so101_pickplace_v1_shard* \
    --output-repo local/so101_pickplace_v1

# 3. 验证合并结果再删 shard（合并不会自动删源，方便回退）
python -m eval.audit_dataset --repo-id local/so101_pickplace_v1
rm -rf ~/.cache/huggingface/lerobot/local/so101_pickplace_v1_shard*
```

### 跨批次合并（v1 + v2 + ...）

如果分多次跑了 `parallel_runner`（例如改了 DR 参数补采 v2），先在各自批次内合并，再把每批的合并产物当作 "shard" 二次合并：

```bash
# 第一次采集：v1 8 worker → 8 个 shard
python -m sim.collectors.parallel_runner --num-episodes 1000 --num-workers 8 \
    --repo-id local/so101_pickplace_v1 --cleanup-after-collect
python -m data.converters.merge_shards \
    --shard-glob local/so101_pickplace_v1_shard* \
    --output-repo local/so101_pickplace_v1
rm -rf ~/.cache/huggingface/lerobot/local/so101_pickplace_v1_shard*

# 第二次采集（数日后补 500 条）：v2 4 worker → 4 个 shard
python -m sim.collectors.parallel_runner --num-episodes 500 --num-workers 4 \
    --repo-id local/so101_pickplace_v2 --cleanup-after-collect
python -m data.converters.merge_shards \
    --shard-glob local/so101_pickplace_v2_shard* \
    --output-repo local/so101_pickplace_v2
rm -rf ~/.cache/huggingface/lerobot/local/so101_pickplace_v2_shard*

# 二次合并：把 v1 v2 当成两个 "shard" 喂给 merge_shards
python -m data.converters.merge_shards \
    --shards local/so101_pickplace_v1 local/so101_pickplace_v2 \
    --output-repo local/so101_pickplace_combined_v1v2

# 验证合并结果后清掉中间产物（可选）
python -m eval.audit_dataset --repo-id local/so101_pickplace_combined_v1v2
# rm -rf ~/.cache/huggingface/lerobot/local/so101_pickplace_v{1,2}/   # 保留 v1/v2 也行，磁盘够就别删
```

**注意点**：
- `merge_shards` 只看每个目录有没有 `meta/info.json`，**不要求 repo_id 必须含 `_shardNN`**，所以已合并的 `v1` / `v2` 直接当 shard 用即可
- 各批次的 **feature schema 必须完全一致**（图像分辨率、state/action 维度、fps），第一个目录的 schema 会被采纳；后续目录帧形状对不上时 `add_frame` 会抛错。合并前先比一下：
  ```bash
  diff <(jq -S .features ~/.cache/huggingface/lerobot/local/so101_pickplace_v1/meta/info.json) \
       <(jq -S .features ~/.cache/huggingface/lerobot/local/so101_pickplace_v2/meta/info.json)
  ```
- 视频会被**再次重编码**（v1/v2 已经各编码过一次），合并耗时 ≈ v1 帧数 + v2 帧数 / 50 fr/s。如果想避免重编码，只能等以后给 `merge_shards` 加多 glob 支持，一次合所有原始 shard

### merge_shards vs merge_datasets

| | `merge_shards` | `merge_datasets` |
|---|---|---|
| 用途 | 把同 schema 的若干 LeRobot dataset 串成一个（含 v1+v2 跨批次） | 把**不同来源**（real / mimicgen / scripted）合并并打 `source` 标签 |
| 视频特征 | ✅ 支持，重编码 | ❌ 拒绝（显式 `NotImplementedError`），state-only |
| 额外字段 | 无（仅重编号 episode_index） | 给每条 episode 加 `source` 标签，训练时可按权重采样 |
| 典型场景 | Phase 2 采集收尾 / 跨批次补采 | Phase 3 多源数据集组装 |

---

## 已验证测试

每次提交前应跑通：

- ✅ MJCF 加载（27 qpos / 24 qvel / 6 actuators / 2 cameras）
- ✅ IK 在 20 iters 内可达 z=0.005（覆盖整个工作区下半）
- ✅ env.reset/step/render 不崩
- ✅ 6 阶段策略全部触发（每阶段状态打印）
- ✅ LeRobotDataset 写盘 + libsvtav1 mp4 编码
- ✅ parallel_runner 端到端：DR + instruction pool + structured stats
- ✅ audit_dataset：jerk / 任务分布 / 集数

---

## 已知限制 / 调优 backlog

| 编号 | 现象 | 影响 | 后续手段 |
|------|------|------|----------|
| L-1 | scripted-policy grasp 成功率 ~20% | Phase 2 plan 目标 ≥80% | 接入 Contact-GraspNet / cuRobo / 调宽 yaw 搜索（见 [phase2 plan §升级路径](plans/phase2-trajectory-generation.md#关键技术与工具)） |
| L-2 | gripper 闭合后 cube 偶尔滑出 | grasp_fail 主因 | MJCF cube friction tuning（增大 tangential friction）或夹爪 fingertip 软材质 |
| L-3 | TRANSPORT 阶段 ee 偶尔水平掠过红 cube 引发碰撞 | transport_collision_red ~10% | 在 pipeline 加入"中间路点避碰检查" + 重规划 |
| L-4 | DR 的 dynamics（质量/摩擦扰动）未启用 | sim2real 鲁棒性下限 | 实现 `sim/randomization/dynamics.py`，对应 Phase 2 T2.5 |
| L-5 | metadata 字段（seed / source / object_color）未写入 dataset | Phase 3 mixed dataset 分流困难 | 升级 `DatasetWriter.add_frame_from_obs` 接受 metadata kwargs |
| L-6 | audit 工具只读 episode 级聚合，没有 frame-level 渲染抽检 | 不易发现"打转 / 穿模"类异常 | 加 `--render-n` 抽 frame 存 PNG |
| L-7 | object_tracker plate 检测误差 ~35mm（透视椭圆 centroid 不等于 disc center 投影） | sim 端 plate 定位精度不够做严格 audit | 真机用 AprilTag 标记 plate 中心；sim 仅用 env GT 不用 tracker |
| L-8 | MimicGen replay 端到端 0% 成功（受 L-1 上游 grasp 失败传递） | augment pipeline 算正确但产出 0 条 | 等 L-1 修复（grasp 升级或换更小 cube）后整条 chain 自动通畅 |
| L-9 | merge_datasets / expand_instructions 不支持视频特征 | 现 Phase 2 输出含 mp4，无法直接合并 | 加 mp4 chunk symlink path；或训练时分多 dataset 加权采样 |
| L-10 | LeRobot 0.5.x `add_frame` 校验严格匹配 `tuple` shape | 从 info.json 读出的 `list` shape 不通过 | 已 fix：merge/expand 都把 shape coerce 为 tuple |

---

## 待实现 Phase（按优先级）

### Phase 4：VLA 微调
**前置**：Phase 3 mixed dataset 就绪
**关键文件**：`training/configs/{pi05,smolvla}_so101_v1.yaml`、`training/data/prepare_lerobot_for_vla.py`、`eval/sim_eval.py`、`eval/language_ood_eval.py`
config + eval 脚本可以先写，但 finetune 启动本身需要 ≥24h GPU。

### Phase 5：真机部署
**前置**：SO101 真机就绪 + USB 摄像头标定
**关键文件**：`deploy/{inference_server,robot_client,safety}.py`、`deploy/configs/deploy.yaml`、`eval/real_eval.py`

### Phase 6：HIL recovery
**前置**：Phase 5 v2 已部署 + USB 踏板
**关键文件**：`data/hil_adapter/{audit,annotate_segments,manifest}.py`、`training/configs/*_v3.yaml`

### Phase 7：扩展任务
按兴趣选 1–2 个方向（多物体 / 多容器 / real2sim / 世界模型 / 硬件升级）
