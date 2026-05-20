# 实现状态总览

> 本文档跟踪 [docs/plans/](plans/) 中各 Phase 的**实际代码实现状态**。
> Plans 描述"要做什么"，本文档描述"已做了什么 + 怎么用"。
>
> **维护规则**：新 phase 文件落地时，更新对应表格；不写"将来要做的"（那些留在 plans/）；每文件一句话说清"实现内容 + 关键设计取舍"。

---

## 总览

| Phase | 状态 | 文件数 | Commit |
|-------|------|--------|--------|
| 0 环境与可行性验证 | 未启动（需硬件） | 0 | — |
| 1 仿真平台搭建 | ✅ 已实现 | 10 | `5f23569` |
| 2 自动轨迹生成 | ✅ 已实现 | 6 | `5f23569` |
| 3 LeIsaac 遥操 + IsaacLab Mimic 扩增 | 🔄 改方案重做（旧 MuJoCo+自实现 mimicgen 代码作废） | — | — |
| 4 VLA 微调 | 未启动 | 0 | — |
| 5 真机部署 | 未启动（需硬件） | 0 | — |
| 6 HIL recovery | 未启动（需硬件 + 踏板） | 0 | — |
| 7 扩展任务 | 未启动 | 0 | — |

---

## Phase 1：仿真平台搭建

对应计划：[phase1-simulation-platform.md](plans/phase1-simulation-platform.md)

| 文件 | 对应任务 | 实现内容 |
|------|----------|----------|
| `assets/scenes/pick_place.xml` | T1.2 / T1.4 / T1.5 | SO-ARM101 + 红 cube + plate + 双相机；`<include>` 引用 pick-101 的 `so101_new_calib.xml`；home keyframe 把 ee 摆在工作区中央上方 14cm |
| `assets/so101_pick101/` | T1.2 | pick-101 整套模型（14 个 STL + 含 graspframe/finger_pad/elliptic 摩擦） |
| `sim/envs/base.py` | T1.3 | `BaseSoArmEnv`：Gymnasium 兼容、双相机渲染、joint/ee 双 action mode；ee 模式通过 `EeIkController`；`EE_DELTA_SCALE=0.05`、`encode_ee_delta_action()` |
| `sim/envs/pick_place.py` | T1.4 | red cube + plate 工作区随机化（cube yaw 量化 4 等分、wrist_roll 按 cube 位置同步）；6 条 success 判据 + 失败归因 |
| `sim/controllers/ik_dls.py` | T1.6 | 70 行 DLS Jacobian IK，原生支持 `locked_joints=[3,4]`；`locked_joints` 读 `data.ctrl[j]` 防 qpos 漂移污染 |
| `sim/scripted_policies/pick_place.py` | T1.7 | 9 阶段 oracle 策略；cube/plate anchor 首次 snapshot；用 `ee_target_override` + `gripper_action_override` 跨 substep 锁目标 |
| `sim/randomization/*.py` | T1.8 | lighting / textures / cube_pose / plate_pose / camera_pose 5 维 DR；**cube 颜色不参与**（语言锚点） |
| `data/converters/sim_to_lerobot.py` | T1.9 | `make_or_resume_dataset()`：默认续接 + 损坏检测；`DatasetWriter` 包装 `LeRobotDataset.add_frame()` 适配 0.5.x；`build_features(img_height, img_width)` 工厂动态拼 shape |

**关键设计**：
- cube 颜色固定 red（DR 不动）
- gripperframe 是 fingertip 中点向前 10cm（TCP），descend 目标 z ≈ 0.008 让夹爪贴桌面从两侧抱 cube
- 自由体 cube/plate 的 body origin = geom center，避免 z 偏移

---

## Phase 2：自动轨迹生成

对应计划：[phase2-trajectory-generation.md](plans/phase2-trajectory-generation.md)

| 文件 | 对应任务 | 实现内容 |
|------|----------|----------|
| `sim/grasp/antipodal.py` | T2.1（占位）| pick-101 风格顶抓不依赖 antipodal；接口保留待 Phase 7 扩展非 cube 物体 |
| `sim/planning/waypoint.py` | T2.2 | 笛卡尔空间 N 个 waypoint 线性插值 + 逐点 IK + 关节限位检查；保存/恢复 qpos |
| `sim/scripted_policies/pick_place_pipeline.py` | T2.3 | `generate_pickplace_episode(env, seed)`：reset → grasp sample → 9 阶段运行 |
| `sim/collectors/parallel_runner.py` | T2.7 | `--num-workers N`：每 worker 独立 env + dataset shard；spawn context 防 EGL 冲突；汇总成功/失败模式分布 |
| `data/instructions/pick_place.txt` | T2.9 | 15 条同义指令（put/place/move/grab/drop + plate/dish + 是否提及红 cube） |
| `eval/audit_dataset.py` | T2.8 | 磁盘直读审计：parse `meta/info.json` + parquet；输出 episode 数、平均时长、jerk p95、task 分布 |
| `data/converters/merge_shards.py` | T2.7.5 | N shard → 1 dataset；CLI `--shards REPO_ID ...` 或 `--shard-glob 'PATTERN'`；不删源 shard |
| `data/converters/derive_action_format.py` | Phase 4 Stage-2 | 6-D joint → 5 种 ee 格式派生（`ee_delta_4d/7d_euler/7d_axis_angle`、`ee_pose_7d_euler/10d_rotate6d`）；走 MuJoCo FK；mp4 直接复制不重编码 |

**未实现的 Phase 2 任务**：
- T2.4 `failures.log` 沉淀（env 已内置失败归因，但单独日志文件未做）
- T2.5 `dynamics.py`（质量/摩擦扰动）+ HDRI 集成
- T2.6 metadata 字段 `seed` / `object_color` / `sim_random_seed` / `source`
- T2.10 正式批量 1000 条 + 推 HF Hub

---

## Phase 3：LeIsaac 遥操 + IsaacLab Mimic 扩增（重做中）

对应计划：[phase3-real-demo-sim-augmentation.md](plans/phase3-real-demo-sim-augmentation.md)

**方案变更**：原方案（MuJoCo + 自写 `MG_EnvInterface` + 原版 MimicGen）作废。新方案走 LeIsaac × IsaacLab Mimic 全官方链路。

旧代码残留待删：`sim/collectors/sim_teleop.py`、`sim/collectors/calibrate_leader_to_sim.py`、`data/mimicgen_official/`、`data/mimicgen_adapter/`、`data/converters/lerobot_to_robomimic.py`、`data/converters/robomimic_to_lerobot.py`。

新方案不在本仓库写代码——直接用 LeIsaac 仓库的 `scripts/mimic/*.py` + `scripts/convert/isaaclab2lerobot.py`。本仓库唯一改动：在 `~/workspace/pycharm/leisaac/source/leisaac/leisaac/tasks/` 下新建 `pick_place_red/` 任务（copy lift_cube 改）。

---

## 启动命令

```bash
conda activate py312_cu121

# 单进程 5 ep（带 DR）
python -m sim.collectors.parallel_runner --num-episodes 5 --num-workers 1 \
    --repo-id local/so101_pickplace_test

# 8 worker × 1000 ep 生产规模 + 清理 images/
python -m sim.collectors.parallel_runner --num-episodes 1000 --num-workers 8 \
    --repo-id local/so101_pickplace_v1 --cleanup-after-collect

# 关 DR debug
python -m sim.collectors.parallel_runner --num-episodes 5 --num-workers 1 \
    --repo-id local/so101_pickplace_test --no-dr

# 合并 shard 成单一训练集（必做，训练只认一个 repo_id）
python -m data.converters.merge_shards \
    --shard-glob local/so101_pickplace_v1_shard* \
    --output-repo local/so101_pickplace_v1

# 验证合并后再清掉 shard（合并不会删源，方便回退）
python -m eval.audit_dataset --repo-id local/so101_pickplace_v1
rm -rf ~/.cache/huggingface/lerobot/local/so101_pickplace_v1_shard*

# Stage-2 派生 ee 格式 dataset（仅 OpenVLA/X-VLA 等需要）
python -m data.converters.derive_action_format \
    --source-repo local/so101_pickplace_v1 \
    --output-repo local/so101_pickplace_ee7d_v1 \
    --target-format ee_delta_7d_euler
```

数据集默认写到 `~/.cache/huggingface/lerobot/<repo_id>/`。

---

## LeRobot 数据集目录结构

```
<repo_id>/
├── meta/
│   ├── info.json              ← schema + fps + 总帧数
│   ├── tasks.parquet          ← task_index ↔ task 字符串
│   ├── stats.json             ← 各特征 mean/std
│   └── episodes/chunk-NNN/file-NNN.parquet  ← 每 episode [from,to] 时间戳
├── data/chunk-NNN/file-NNN.parquet          ← 每帧 state/action/timestamp（训练核心）
├── videos/
│   ├── observation.images.front/chunk-NNN/file-NNN.mp4  ← 多 episode 拼合
│   └── observation.images.wrist/chunk-NNN/file-NNN.mp4
└── images/                    ← LeRobot 编码 mp4 时的临时 PNG（**可删**）
```

`--cleanup-after-collect` 只删 `images/`。训练只需 `meta/` + `data/` + `videos/`。

### VLA 训练时的信号映射

| 信号 | 来源 | 说明 |
|---|---|---|
| 🎥 视觉 | `videos/observation.images.{front,wrist}/.../*.mp4` | 按 `meta/episodes/*` 的 `from_timestamp`/`to_timestamp` 切对应 episode |
| 🦾 本体感受 | `data/*.parquet` 的 `observation.state` 列 | 6-DoF 关节角，float32×6 |
| 💬 语言指令 | `meta/tasks.parquet` 经 `task_index` 查表 | `data/` 每帧带 `task_index`，查出字符串喂 VLM 文本编码器 |
| 🎯 动作标签 | `data/*.parquet` 的 `action` 列 | 预测目标，float32×6（6 维 joint position）|
| 🛠️ sim 辅助 | `data/*.parquet` 的 `ee_action` 列 | sim-only，float32×4（dx,dy,dz,gripper 归一化）；VLA 训练**不读** |

`data/*.parquet` 完整列：
```
observation.state  float32 [6]   ← 本体感受输入
action             float32 [6]   ← 监督标签（6 维 joint position target）
ee_action          float32 [4]   ← sim 辅助列（VLA 不读）
task_index         int64         ← 语言指令外键
episode_index      int64         ← 属于哪条 episode
frame_index/index  int64         ← 帧号
timestamp          float32       ← 切对应视频帧用
```

---

## 术语：action_mode、TCP、ee/joint 语义

### action_mode

`sim/envs/base.py` 里只有两种：

| mode | action shape | 含义 | step 内部 |
|---|---|---|---|
| `"ee"` | (4,) `[dx, dy, dz, gripper]` | end-effector 空间增量，世界系 | `dx/dy/dz × EE_DELTA_SCALE` 加到 TCP 当前世界坐标 → DLS IK → 写 `ctrl[:5]`；`gripper` 线性映射 `ctrl[5]` |
| `"joint"` | (6,) | 6 关节归一化目标 ∈ [-1, 1] | 线性映射到 `ctrl_limits` 直接写 `ctrl[:6]`，**不走 IK** |

**重要区分**：`env.action_mode` 是 **sim step() 接受什么输入**，与 **dataset 存什么 action 列**是两件事。当前 PickPlaceEnv 用 `"ee"` 是为了让脚本策略代码简洁，**dataset 对外暴露的 `action` 列是 joint**——与 LeRobot SO-101 teleop 惯例和 SmolVLA pretrain schema 对齐。即 "策略输出 ee → env 解 IK → 落盘 joint（主）+ ee（辅）"。

### TCP

**Tool Center Point**——夹爪型 ee 的两指几何中心。MJCF 里通过 site 定义：

```xml
<site name="gripperframe" pos="0.0 0.0 -0.0981274" .../>
```

VLA 训练里 dx/dy/dz 描述 TCP 在世界系的位移。代码读取：`env.ee_pos()` = `data.site("gripperframe").xpos`。

### EE_DELTA_SCALE = 0.05 m 的双职责

**(a) 单步 TCP 位移物理上限**——env 每帧 33 ms（30 FPS）最多移 5cm，即 ee 速度上限 1.5 m/s。这是 SO-101 物理可达极限，也是 DLS IK 单步迭代能稳定跟踪的范围。

**(b) 神经网络输出归一化范围**——VLA 最后一层带 tanh 自然输出 [-1, 1]。直接输出米的话，典型位移 0.001~0.01 m 对 tanh 太小，梯度信号弱。

```
神经网络输出 ∈ [-1, 1]  ←──×0.05──→  TCP 实际位移 ∈ [-0.05, 0.05] m
```

切换机器人 / 速度配置时**只改这一个常量**。

### ee-mode env.step 内部 pipeline

```
action[:3] ──×EE_DELTA_SCALE──→ delta_xyz (米)
              │
              ▼  target_TCP = ee_pos() + delta_xyz
              ▼  DLS IK (locked_joints=[3, 4])  → q_arm (5 维)
              ▼  data.ctrl[:5] = q_arm

action[3] (gripper, absolute mode)
              ▼  ctrl[5] = (a3+1)/2 × (g_hi-g_lo) + g_lo

              ▼  mj_step × SIM_STEPS_PER_CTRL (5×2ms = 10ms)
              ▼  期间 IK 多次 refresh ctrl (pick-101 风格 500Hz IK loop)
              ▼  read state → 返回 obs
```

`gripperframe` 是 MuJoCo **site**（虚拟参考点），IK 解的是真实 body 的关节角，目标只是让 FK(joint) 的 TCP 位姿匹配 `target_TCP`。夹爪 ctrl[5] 与手臂 ctrl[:5] 独立，IK 不动 ctrl[5]。

---

## VLA 训练用 ee-mode 还是 joint-mode

**LeRobot 框架本身不挑**（`ACTION="action"` 硬编码，无 `action_mode` 概念）。**用什么取决于 dataset 列 + VLA 模型预训练 schema**。

| VLA / Policy | 预训练 action 语义 | SO-101 微调路径 |
|---|---|---|
| **SmolVLA** | 6-DoF joint position（LeRobot Community ~95% SO-100/101） | **直接用** Stage-1 dataset（逐位对齐） |
| **Pi-0 / Pi-0.5 / Pi-0-FAST** | 32-D zero-padded（混合 ecosystem） | **直接用** Stage-1（openpi loader 运行时 pad 6→32）|
| **ACT** / Diffusion Policy | joint position（无预训练）/ 任意 | **直接用** Stage-1 |
| **OpenVLA** | 7-DoF ee delta（Open-X） | **跑 Stage-2 派生** → `ee_delta_7d_euler` |
| **X-VLA** | 10-D ee pose（DROID） | **跑 Stage-2 派生** → `ee_pose_10d_rotate6d` |
| **RT-1 / RT-2 / RT-X** | 7-DoF ee delta，离散化 | **跑 Stage-2 派生** → `ee_delta_7d_euler` |

**判别准则**：维度补齐（如 Pi 把 6→32 zero-pad，loader 运行时做）→ 直接用 Stage-1；语义改变（joint 角 ↔ ee 笛卡尔）→ 必须 Stage-2 派生。

**关于 joint-mode "跨机器人迁移" 的常见误解**：

> "joint-mode 不能做跨机器人迁移"——**事实**：joint-mode **能**做跨机器人迁移，靠工程手段而非 action 表示本身。VLA 社区四种方案：
> 1. 固定维度 + zero padding（Pi-0）
> 2. per-robot action head（OpenVLA / RT-X / SmolVLA）
> 3. embodiment token（RT-2 / Pi-0.5）
> 4. 同形态预训练 + 同形态迁移（SmolVLA 实际走的路；SO-100 → SO-101 连 head 都不用换）

**ee-mode 真正的迁移优势**是"零配置跨 DoF"——任意 6/7/14 DoF 的臂只要带夹爪都用同样的 `[dx,dy,dz,gripper]` schema。joint-mode 跨形态时需要工程改造，同形态时两者无差距。

### 本项目最终选择：joint 主 + ee 辅

| 列名 | shape | names | 含义 |
|---|---|---|---|
| **`action`**（主） | (6,) | `[shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos]` | **6 维 joint ctrl 目标**（即每帧 `ctrl[:6]`），与 LeRobot SO-101 teleop 完全对齐，与 SmolVLA pretrain schema 逐位对齐 |
| `ee_action`（辅） | (4,) | `[dx, dy, dz, gripper]` | sim-only ee-delta（×0.05m 还原）。仅供：①sim 调试可视化；②Stage-2 派生工具输入 |

`observation.state` 同步用 `.pos` 后缀 names。

**为什么 action 记 `ctrl[:6]` 而不是 `state[t+1] - state[t]`**：`ctrl[:6]` 是 IK + gripper 刚解出来发给 actuator 的 **命令**，是 ACT/SmolVLA 训练惯例的 "joint position target" 干净对应。用 `state` 差值会受 sts3215 严重欠阻尼（ζ≈0.26）影响——smoke 实测两者每维差 0.01–0.02 弧度。

---

## 跨批次合并

如果分多次跑了 `parallel_runner`（如改 DR 参数补 v2），先各自批次内合并，再二次合并：

```bash
# v1 8 worker → 8 shard
python -m sim.collectors.parallel_runner --num-episodes 1000 --num-workers 8 \
    --repo-id local/so101_pickplace_v1 --cleanup-after-collect
python -m data.converters.merge_shards \
    --shard-glob local/so101_pickplace_v1_shard* \
    --output-repo local/so101_pickplace_v1

# v2 4 worker → 4 shard（数日后补 500 条）
python -m sim.collectors.parallel_runner --num-episodes 500 --num-workers 4 \
    --repo-id local/so101_pickplace_v2 --cleanup-after-collect
python -m data.converters.merge_shards \
    --shard-glob local/so101_pickplace_v2_shard* \
    --output-repo local/so101_pickplace_v2

# 二次合并：v1 / v2 当成两个 shard
python -m data.converters.merge_shards \
    --shards local/so101_pickplace_v1 local/so101_pickplace_v2 \
    --output-repo local/so101_pickplace_combined_v1v2
```

**注意**：
- `merge_shards` 只看 `meta/info.json` 存在，不要求 repo_id 含 `_shardNN`
- 各批次 **feature schema 必须完全一致**（图像分辨率、state/action 维度、fps）。合并前比一下：
  ```bash
  diff <(jq -S .features ~/.cache/huggingface/lerobot/local/so101_pickplace_v1/meta/info.json) \
       <(jq -S .features ~/.cache/huggingface/lerobot/local/so101_pickplace_v2/meta/info.json)
  ```
- 视频会被**再次重编码**（v1/v2 已经各编码一次），合并耗时 ≈ 总帧数 / 50 fr/s

---

## 已验证测试

每次提交前应跑通：

- ✅ MJCF 加载（27 qpos / 24 qvel / 6 actuators / 2 cameras）
- ✅ IK 在 20 iters 内可达 z=0.005
- ✅ env.reset/step/render 不崩
- ✅ 9 阶段策略全部触发
- ✅ LeRobotDataset 写盘 + libsvtav1 mp4 编码
- ✅ parallel_runner 端到端：DR + instruction pool + structured stats
- ✅ audit_dataset：jerk / 任务分布 / 集数

---

## 已知限制 / 调优 backlog

| 编号 | 现象 | 影响 | 后续手段 |
|------|------|------|----------|
| L-1 | scripted-policy grasp 成功率 ~20%（旧 menagerie 模型）| Phase 2 目标 ≥80% | 已切到 pick-101 整套模型 + yaw 量化 + wrist_roll 同步 → 93.8% |
| L-2 | gripper 闭合后 cube 偶尔滑出 | grasp_fail 主因 | MJCF cube friction 增大 tangential / 软材质 fingertip |
| L-3 | TRANSPORT ee 偶尔掠过红 cube 碰撞（已不适用单 cube 任务） | — | 已废 |
| L-4 | DR `dynamics`（质量/摩擦扰动）未启用 | sim2real 鲁棒性下限 | 实现 `sim/randomization/dynamics.py` |
| L-5 | metadata 字段（seed/source）未写入 dataset | Phase 3 mixed dataset 分流困难 | 升级 `DatasetWriter.add_frame_from_obs` 接 metadata kwargs |
| L-6 | audit 工具只读 episode 级，没有 frame-level 渲染抽检 | 不易发现"打转/穿模" | 加 `--render-n` 抽 frame 存 PNG |
| L-7 | object_tracker plate 检测误差 ~35mm（透视椭圆 centroid ≠ disc center 投影）| sim 端 plate 定位精度不够严格 audit | 真机用 AprilTag；sim 用 env GT 不用 tracker |
| L-8 | LeRobot 0.5.x `add_frame` 严格匹配 `tuple` shape | info.json 读出的 `list` shape 不通过 | 已 fix：merge/expand 都 coerce 为 tuple |

---

## 待实现 Phase

### Phase 3（重做）：LeIsaac 遥操 + IsaacLab Mimic 扩增
**前置**：环境（已 OK：isaacsim 6.0 + isaaclab 0.47.2 + isaaclab_assets 0.2.3 + leisaac 0.4.0 + warp-lang）
**关键操作**：在 leisaac 仓库下 copy `lift_cube/` 改成 `pick_place_red/`；不在本仓库写代码

### Phase 4：VLA 微调
**前置**：Phase 3 输出的 LeRobot dataset 就绪
**关键文件**：`training/configs/{pi05,smolvla}_so101_v1.yaml`、`eval/sim_eval.py`、`eval/language_ood_eval.py`

### Phase 5：真机部署
**前置**：SO101 真机 + USB 摄像头标定
**关键文件**：`deploy/{inference_server,robot_client,safety}.py`、`eval/real_eval.py`

### Phase 6：HIL recovery
**前置**：Phase 5 v2 已部署 + USB 踏板
**关键文件**：`data/hil_adapter/{audit,annotate_segments,manifest}.py`、`training/configs/*_v3.yaml`

### Phase 7：扩展任务
按兴趣选 1–2 个方向（多物体 / 多容器 / real2sim / 世界模型 / 硬件升级）
