# Phase 2：自动轨迹生成

**周期**：2–3 周
**前置依赖**：Phase 1 完成（PickPlaceRed 仿真环境可用）
**目标**：实现一套自动化 trajectory 生成 pipeline，在仿真里产出 ≥1000 条 **PickPlaceRed** 成功 episode（带域随机化），全部存为 LeRobot 数据集格式

> **核心任务**：把红 cube 放进 plate。详细规约见 [README.md](README.md) 顶部"核心任务定义"。

---

## 代码入口（快速开始）

> 所有命令在仓库根目录 `/home/zzg/workspace/pycharm/Robot` 下、`conda activate py312_cu121` 后执行。
> Phase 2 与 Phase 1 共用 `parallel_runner` 作为采集入口——区别在于 Phase 2 启用完整 DR + 大规模并行。

| 想做的事 | 一行命令 | 产出 |
|---------------|---------|------|
| 跑单条 episode 看脚本策略是否抓得到（联调） | `python -m sim.collectors.parallel_runner --num-episodes 1 --num-workers 1 --repo-id local/so101_debug` | 一条 episode，可在 viewer 或视频里检查 |
| **正式批量采集（T2.10 Phase 2 最终交付，目标 ≥1000 条成功 episode）** | `python -m sim.collectors.parallel_runner --num-episodes 1500 --num-workers 8 --repo-id local/so101_pickplace_v1 --instructions data/instructions/pick_place.txt --cleanup-after-collect` | 8 个 shard：`~/.cache/huggingface/lerobot/local/so101_pickplace_v1_shard{00..07}/`（含 front+wrist mp4 + state/action parquet） |
| **合并 shard 成单一训练数据集**（Phase 2 收尾必做） | `python -m data.converters.merge_shards --shard-glob local/so101_pickplace_v1_shard* --output-repo local/so101_pickplace_v1` | `~/.cache/huggingface/lerobot/local/so101_pickplace_v1/`（合并后单数据集） |
| 关闭 DR 跑对照（T2.5 调试用） | 第 2 条命令追加 `--no-dr` | 同上 |
| 数据集质量审计（T2.8 验收） | `python -m eval.audit_dataset --repo-id local/so101_pickplace_v1 --n-sample 100 --out-dir runs/audit_v1` | 控制台 + `runs/audit_v1/report.md` |
| 查看任一 episode 抓取视频 | `ls ~/.cache/huggingface/lerobot/local/so101_pickplace_v1/videos/observation.images.front/chunk-000/` | `file-000.mp4`（多 episode 拼合，按 episodes.parquet 的 `[from,to]` 切片观看） |

**入口与任务清单的对应关系**：

| Task | 实现文件（库） | 由哪个 CLI 串起来 |
|------------------|--------------|-------------|
| T2.1 Antipodal grasp sampler（v21 已退化为占位） | [`sim/grasp/antipodal.py`](../../sim/grasp/antipodal.py) | pick-101 风格俯视顶抓不依赖 antipodal，保留接口便于将来扩展 |
| T2.2 Waypoint planner | [`sim/planning/waypoint.py`](../../sim/planning/waypoint.py) | 同上 |
| T2.3 9 阶段编排 | [`sim/scripted_policies/pick_place_pipeline.py`](../../sim/scripted_policies/pick_place_pipeline.py) | 被 `parallel_runner` 调用 |
| T2.4 成功判定 + 失败归因 | [`sim/envs/pick_place.py::evaluate_success / evaluate_failure_mode`](../../sim/envs/pick_place.py) | 同上 |
| T2.5 全量 DR | [`sim/randomization/`](../../sim/randomization/) | `parallel_runner` 默认开启 |
| T2.6 LeRobot writer | [`data/converters/sim_to_lerobot.py`](../../data/converters/sim_to_lerobot.py) | `parallel_runner` 内部使用 |
| T2.7 并行采集 | [`sim/collectors/parallel_runner.py`](../../sim/collectors/parallel_runner.py) | `--num-workers N` |
| **T2.7.5 shard 合并（新增）** | [`data/converters/merge_shards.py`](../../data/converters/merge_shards.py) | 独立 CLI，N 个 shard → 1 dataset |
| T2.8 数据集审计 | [`eval/audit_dataset.py`](../../eval/audit_dataset.py) | 独立 CLI |
| T2.9 多语言指令 | [`data/instructions/pick_place.txt`](../../data/instructions/pick_place.txt) | `parallel_runner --instructions ...`（内容是 red task 同义指令池） |
| T2.10 批量生成 | 上面所有合起来 | `parallel_runner --num-episodes 1500 --num-workers 8` + `merge_shards` |

> **TL;DR**：Phase 2 的核心两条命令 = `parallel_runner --num-episodes 1500 --cleanup-after-collect` + `merge_shards --shard-glob ...`。预期生成的 mp4 在合并后 dataset 的 `videos/observation.images.{front,wrist}/chunk-000/file-000.mp4`（多 episode 拼合）；用 `lerobot-dataset-viz` 按 episode 索引浏览。

---

## 关键技术与工具

| 工具 | 用途 | 备注 |
|------|------|------|
| mink | 微分 IK | Phase 1 已用 |
| 自实现 antipodal grasp sampler | 简单可行的抓取姿态采样 | 后期可升级为 GraspNet/Contact-GraspNet |
| 自实现 waypoint planner | 线性插值 + 中间无碰撞校验 | 对桌面任务足够 |
| MuJoCo MJX（可选） | GPU 并行加速采集 | 用 `pip install mujoco-mjx` |
| multiprocessing | CPU 多进程并行采集 | 默认方案 |
| LeRobot dataset API | 标准数据格式输出 | `lerobot.datasets.LeRobotDataset` |

**升级路径（如成功率不达标）**：
- 抓取采样：antipodal → **Contact-GraspNet** (`https://github.com/NVlabs/contact_graspnet`)
- 运动规划：linear waypoint → **OMPL / cuRobo** (`https://github.com/NVlabs/curobo`)
- 平台并行化：multiprocessing → **MJX (GPU 并行)**

---

## 任务清单

### T2.1 Antipodal grasp sampler（v21 已退化为占位）

**当前状态**：pick-101 风格俯视顶抓策略不依赖 antipodal 采样（夹爪垂直于桌面 + cube yaw 4 等分量化 + wrist_roll 按位置同步），所以 `sim/grasp/antipodal.py` 保留接口但不在主路径调用。

**保留它的理由**：如果将来要把任务扩展到非 cube 物体（任意形状）或解锁 wrist 5DOF 自由抓取（见 Phase 7），antipodal 采样会重新成为主路径。

**步骤（如真要重新启用）**：
- [ ] 在 `sim/grasp/antipodal.py` 实现：
  - 输入：物体 mesh（cube 简化为 OBB）+ 当前姿态
  - 算法：在物体表面对采样平行的两个接触点，对应夹爪开合方向；按抓取深度、与重力方向夹角打分
  - 输出：N 个候选 ee 位姿（按分数排序）
- [ ] 加可视化（matplotlib 3D 或 trimesh）

**参考**：
- robosuite `samplers`：思路参考
- 论文 *Antipodal grasping with one parameter sweep*（短读）

**验证**：对一个 3cm cube 任意姿态，能返回 ≥10 个候选抓取，评分合理

---

### T2.2 Waypoint motion planner

**目标**：给定起点 + 终点 ee 位姿，输出关节轨迹

**步骤**：
- [ ] 在 `sim/planning/waypoint.py` 实现：
  - 输入：start_q、target_ee_pose
  - 在笛卡尔空间生成 N 个中间 waypoint（线性插值）
  - 对每个 waypoint 用 IK 求 q，做关节限位检查
  - 输出关节轨迹（T 步）
- [ ] 加碰撞检测（用 MuJoCo `mj_collision` 一步 sim）

**关键文件**：
- `sim/planning/waypoint.py`

**参考**：
- mink examples 中的 IK 序列
- robosuite `OperationalSpaceController` 思路

**验证**：从初始姿态规划到 cube 上方 10cm，机械臂 200 步内无碰撞到达

---

### T2.3 任务级 trajectory 编排（9 阶段 pick-place，pick-101 风格）

**目标**：把整条任务拆成可复用阶段

**步骤**：
- [ ] 在 `sim/scripted_policies/pick_place_pipeline.py` 定义阶段：
  ```
  HOME → APPROACH → DESCEND → GRASP → LIFT → TRANSPORT → PLACE_DESCEND → RELEASE → RETRACT
  ```
- [ ] 每阶段返回 ee_target + gripper_action（写入 `env.ee_target_override` / `env.gripper_action_override`）
- [ ] **APPROACH/DESCEND** 用 cube_anchor（首次 snapshot 后不再变），并加 `FINGER_WIDTH_OFFSET=-0.015` 让 cube 居中
- [ ] **PLACE_DESCEND** 用闭环跟踪：`target = ee_cur + (desired_cube - cube_obs)`（对未知 wrist 旋转鲁棒）
- [ ] **TRANSPORT_Z=0.10**（不能再高，否则 elbow_flex 触限位）
- [ ] 阶段间用 `PHASE_HOLD_STEPS=15` 衔接，GRASP 用 `GRASP_RAMP_STEPS=60` 渐进闭合
- [ ] 暴露成 `generate_pickplace_episode(env, seed) -> Episode | None`

**关键文件**：
- `sim/scripted_policies/pick_place_pipeline.py`
- `sim/scripted_policies/pick_place.py`（具体的 PickPlacePolicy 实现）

**验证**：跑 100 次随机 seed，成功率 ≥ 90%（v21 实测 93.8%）

---

### T2.4 Success 判定与失败筛除

**目标**：自动判断 + 丢弃失败 episode

**步骤**：
- [ ] 在 `sim/envs/pick_place.py` 完善 `evaluate_success()`：
  1. red cube 中心 xy 到 plate 中心距离 < plate 半径
  2. red cube 底面 z 在 plate 表面 ±1cm
  3. 夹爪已松开
  4. 机械臂无超限位 / 无关节振荡
- [ ] 失败原因枚举：`grasp_fail` / `lift_drop` / `place_miss` / `plate_off` / `joint_limit` / `timeout`
- [ ] 在 pipeline 里：失败 episode 默认丢弃；`--keep-failures` 时保留并把 mode 写入 task 字段后缀 `[FAIL:<mode>]`

**关键文件**：
- `sim/envs/pick_place.py`（更新 evaluate_success + 失败归因）
- `sim/scripted_policies/pick_place_pipeline.py`（更新过滤逻辑）

**验证**：失败原因有分类统计，写入 collector 返回的 summary

---

### T2.5 全量域随机化

**目标**：让生成的数据具备 sim2real 所需的视觉多样性

**步骤**：
- [ ] 扩充 `sim/randomization/`：
  - `lighting.py`：场景里 light0/light1/light2 三个点光源随机化位置/强度
  - `textures.py`：桌面 30+ 纹理（PBR 公开素材如 ambientCG）
  - `cube_pose.py`：cube 在工作区内 xy + yaw（yaw 量化到 4 等分）+ plate 位置 ±3cm
  - `camera_pose.py`：±10cm xy / ±3cm z / ±10° rotation 抖动 front cam
  - `dynamics.py`：质量 ±20% / 摩擦 ±30%（注意不要让摩擦太低导致 cube 滑落）
- [ ] 每个 episode reset 时各项独立采样
- [ ] **不要随机化 cube 颜色**：red 是语义锚点

**关键文件**：
- `sim/randomization/*.py`（扩充）
- `assets/textures/`：本地纹理资源
- `assets/hdri/`：HDRI 资源（可选）

**参考**：
- ambientCG 免费 PBR 纹理：`https://ambientcg.com`
- Poly Haven HDRI：`https://polyhaven.com/hdris`

**验证**：连续 reset 20 次截图，所有维度都有明显差异

---

### T2.6 LeRobot dataset writer（正式版）

**目标**：把 episode 流式写入 LeRobot 数据集

**步骤**：
- [ ] 在 `data/converters/sim_to_lerobot.py` 实现：
  - 初始化 `LeRobotDataset.create(repo_id, features={...}, fps=30)`
  - features 包含：`observation.images.wrist`、`observation.images.front`、`observation.state`（6 维关节角，names 用 `*.pos` 后缀对齐 LeRobot SO-101 teleop）、**`action`（6 维 joint ctrl 目标，未归一化弧度 + 夹爪 qpos，与 LeRobot SO-100/101 ecosystem 完全对齐，SmolVLA 直接训）**、`ee_action`（4 维 ee-delta + gripper，归一化 [-1, 1]，sim 调试辅助）、`task`（语言指令字符串）
  - 写帧 → 调 `dataset.save_episode()`
- [ ] 加 metadata：`task`、`seed`、`sim_random_seed`

**关键文件**：
- `data/converters/sim_to_lerobot.py`

#### T2.6.1 action 标签真实化（v22+ 必备，否则 dx/dy/dz=0 没法训）

**问题**：脚本策略通过 `env.ee_target_override` 旁路驱动 IK，所以 `policy(env, obs)` 返回的 gym 兼容 action 永远是 `[0, 0, 0, gripper]`——dx/dy/dz 不携带任何运动信息。直接落盘的 action 列将 100% 为零，VLA 训练时模型学不到任何手臂动作（只能学夹爪开合）。

**修复**：保持 LeRobot 端 schema `action: float32 [4]`（dx, dy, dz, gripper ∈ [-1, 1]），在 step 前后采 TCP 位姿（`gripperframe` site = 夹爪两指中点向前 10cm），把真实世界系位移按 `EE_DELTA_SCALE = 0.05 m/unit` 归一化重写到 action[:3]，落盘的就是部署时 `_apply_ee_action` 能直接复用的相同 schema：

```
action_recorded[:3] = clip( (ee_after - ee_before) / EE_DELTA_SCALE, -1, 1 )
action_recorded[ 3] = policy 返回的 gripper（原样）
```

**落地点**：
- `sim/envs/base.py`：新增类常量 `EE_DELTA_SCALE = 0.05` + 方法 `encode_ee_delta_action(ee_before, ee_after, gripper_norm)`
- `sim/collectors/parallel_runner.py`：采集循环每步 step 前后调用 `env.ee_pos()` 计算位移，落盘前用 `encode_ee_delta_action` 重写 action
- `sim/scripted_policies/pick_place_pipeline.py`：`generate_pickplace_episode` 同步修复（MimicGen 种子要用）

**验证**：
```bash
python -m sim.collectors.parallel_runner --num-episodes 2 --num-workers 1 \
    --repo-id local/so101_action_check
python -c "
import pyarrow.parquet as pq, numpy as np
A = np.stack(pq.read_table('~/.cache/huggingface/lerobot/local/so101_action_check_shard00/data/chunk-000/file-000.parquet').to_pandas()['action'].values)
assert (A[:, :3] != 0).any(axis=1).mean() > 0.99, 'dx/dy/dz 仍存在大量零行——修复失效'
print('ok: nonzero rate', (A[:, :3] != 0).any(axis=1).mean())
"
```

**MimicGen 路径已审（无问题）**：`data/mimicgen_adapter/replayer.py` 的 `replay_segmented_demo` 本来就用 `action[:3] = clip((target - ee_pos) / EE_DELTA_SCALE, -1, 1)` 现算 action（逻辑与 `encode_ee_delta_action` 等价），`_write_episode` 写盘的 action 列从一开始就是真实归一化 delta。只清理了一处魔数（`/0.05` → `/env.EE_DELTA_SCALE`）和给 `synthesize_source_demos` 的 dead-code `actions` 加注释。详见 `docs/implementation-status.md` 「MimicGen 路径审计结果」。

#### T2.6.2 action 列对齐 LeRobot SO-100/101 惯例

**重要纠错**：早期方案是"双 action 列 + 训练时换 key"，后查 LeRobot 源码发现 `ACTION="action"` 是硬编码常量，**没有 `dataset.action_key` 配置项**。重新对齐 LeRobot 官方 SO-101 teleop 输出（`teleoperators/so_leader/so_leader.py:148` 用 `bus.sync_read("Present_Position")` 直接产 joint position）。

**最终设计**：

```python
features = {
    "action":     {"shape": (6,), "names": ["shoulder_pan.pos", ..., "gripper.pos"]},  # joint ctrl 目标（主）
    "ee_action":  {"shape": (4,), "names": ["dx", "dy", "dz", "gripper"]},             # ee-delta（sim 调试辅助）
}
```

**`action` 列含义**：`env.data.ctrl[:6]` 快照（IK + gripper 解出的 actuator 命令），与 LeRobot SO-101 follower `Goal_Position` 等价。SmolVLA / ACT / Pi-0 系都通过此列训练，**零配置**。

**`ee_action` 列含义**：sim-only 辅助，仅用于 ①sim 调试可视化、②Stage-2 派生 ee-pose / ee-delta-with-rotation 格式数据集（如需训 OpenVLA / X-VLA）。

**训练时**：

| 模型 | 配置 |
|---|---|
| SmolVLA / ACT / Pi-0.5 / Pi-0 / Diffusion Policy | `--dataset.repo_id=local/so101_pickplace_v1`（零配置，默认读 `action` 列）|
| OpenVLA / X-VLA / 其他 ee-pose VLA | 先 `python -m data.converters.derive_action_format --target-format <ee_*_format>` 生成派生 dataset（Stage-2 工具，Phase 4 实现）|

**落地点**：`sim_to_lerobot.py`、`parallel_runner.py`、`pick_place_pipeline.py`、`mimicgen_adapter/replayer.py`、`mimicgen_adapter/augment.py`。`merge_shards.py` 无需改。详见 `docs/implementation-status.md` 「数据集 action 列设计」。

**关键文件**：
- `data/converters/sim_to_lerobot.py`

**参考**：
- LeRobot dataset 创建：`https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/lerobot_dataset.py`

**验证**：用 `lerobot-dataset-viz --repo-id=local/so101_sim_pick_v0 --episode-index=0` 可正常播放

---

### T2.7 并行化采集

**目标**：把单进程速度从 X eps/min 提升到 ≥ 5x

**步骤**：
- [ ] 在 `sim/collectors/parallel_runner.py` 实现：
  - 主进程负责调度 N 个 worker（默认 = CPU 物理核数）
  - 每个 worker 独立 env、独立 seed、独立写本地分片
  - 主进程合并所有分片为单一 LeRobot dataset
- [ ] 增加进度条（tqdm）和 ETA 估算

**关键文件**：
- `sim/collectors/parallel_runner.py`

**参考**：
- gymnasium SyncVectorEnv / AsyncVectorEnv

**验证**：8 worker 跑 1000 episode，预计耗时 < 60 分钟（3cm cube 任务 + 480p 渲染）

---

### T2.7.5 Shard 合并（新增）

**目标**：把 parallel_runner 写出的 N 个 shard dataset 合成单一训练数据集

**步骤**：
- [ ] 在 `data/converters/merge_shards.py` 实现：
  - 输入：N 个 shard repo_id（或 glob 模式）
  - 输出：单一 merged repo_id
  - 实现：用 LeRobotDataset API 逐帧重写（视频重编码，约 50 fr/s）
- [ ] CLI：`--shards REPO_ID ...` 或 `--shard-glob 'PATTERN'`，`--output-repo`，`--reset`
- [ ] 不删源 shard（合并完用户手动 `rm -rf` 验证安全）

**关键文件**：
- `data/converters/merge_shards.py`

**验证**：合并后 dataset 的 episode 数 = 各 shard 之和，`lerobot-dataset-viz` 能正常播放

**跨批次合并**：分多次跑 `parallel_runner`（例如先 v1 1000 条，后补 v2 500 条）时，先各自合并成 `v1` / `v2`，再把它们当作两个 shard 二次合并到 `combined_v1v2`。`merge_shards` 不要求 repo_id 含 `_shardNN`，只看 `meta/info.json` 是否存在。详见 [docs/implementation-status.md](../implementation-status.md) 「跨批次合并」小节。

---

### T2.8 数据集质量审计

**目标**：避免生成出"假成功"垃圾数据

**步骤**：
- [ ] 写 `eval/audit_dataset.py`，对生成的 dataset 做：
  - 随机抽 50 条人工目视检查
  - 统计：平均时长、关节轨迹平滑度（jerk 上限）、夹爪闭合点分布
  - 检查关节是否打表（饱和限位）
- [ ] 输出 `data/sim_generated/pick_place_v0/audit.md`

**关键文件**：
- `eval/audit_dataset.py`

**验证**：人工抽检中，≥90% 的 episode 看起来合理（无打转、无穿模）

---

### T2.9 多语言指令绑定（PickPlaceRed）

**目标**：为后续 VLA 训练准备语言多样性

**步骤**：
- [ ] 创建 `data/instructions/pick_place.txt`，写 15+ 同义指令：
  ```
  put the red cube on the plate
  place the red cube on the plate
  pick up the red cube and put it on the plate
  move the red block to the plate
  grab the red cube and place it in the dish
  take the red cube and drop it on the plate
  the red cube goes on the plate
  ...
  ```
- [ ] 每条 episode 生成时随机选 1 条作为 `task` 字段
- [ ] **颜色锚定固定为 "red"**（不动态替换）
- [ ] 词汇变体覆盖：动词（put/place/move/grab/drop）、容器称呼（plate/dish）

**关键文件**：
- `data/instructions/pick_place.txt`（沿用文件名）
- `sim/collectors/parallel_runner.py`（注入逻辑）

**验证**：dataset 里 `task` 字段有 ≥10 种不同表述

---

### T2.10 正式批量生成

**目标**：生成第一份正式 sim 数据集

**步骤**：
- [ ] 全 DR 开启，生成目标 ≥1000 条成功 episode（建议 1500 条，按 93.8% 成功率算落地 ~1400 条）
- [ ] 跑 merge_shards 把 8 个 shard 合成单一数据集
- [ ] 推上 HuggingFace Hub（私有 repo）作为远端备份：`huggingface-cli upload`
- [ ] 写 dataset card（README）

**关键文件**：
- `data/sim_generated/pick_place_v1/`（说明文档）
- HuggingFace repo: `<your_id>/so101_pickplace_v1`

**验证**：
- ≥1000 条成功 episode
- audit 通过率 ≥90%
- HF Hub 可访问

---

## 验收标准（全部满足后进入 Phase 3）

- [ ] grasp + IK + planner pipeline 自动化运行（v21 配置成功率 93.8%）
- [ ] 域随机化 5 个维度全部接入并独立可控
- [ ] 单机并行采集 + shard 合并，1500 条 episode 在 2 小时内完成（含合并）
- [ ] 生成 dataset 通过质量审计（≥90% 合理）
- [ ] 至少 10 种语言指令变体
- [ ] dataset 已推送到 HF Hub

---

## 风险与陷阱

| 风险 | 应对 |
|------|------|
| 抓取在 cube yaw=±45° 时失败 | yaw 量化到 4 等分（cube 4 重对称）—— 实测从 60% 提到 92% |
| 工作区角落 cube 抓不到 | wrist_roll 按 atan2(cube.y, cube.x) 同步设定，保证总 z 旋转 = π/2 |
| IK 解收敛慢 → 500Hz 反而让成功率变差 | sts3215 严重欠阻尼，100Hz IK + 跨 substep 锁定目标 |
| TRANSPORT 阶段 elbow_flex 触下限位 | TRANSPORT_Z 从 0.15 调到 0.10；不在 TRANSPORT 做 xy 跟踪 |
| PLACE_DESCEND 因 wrist 旋转导致 FINGER_WIDTH_OFFSET 失效 | 改用闭环 `target = ee_cur + (desired_cube - cube_obs)` |
| 并行 worker 的渲染上下文冲突 | `MUJOCO_GL=egl` + spawn context，每个 worker 独立 EGL |
| 写 LeRobot 数据时图像被压缩失真 | features 用 `dtype=video`，编码 av1 高质 |
| shard 之间分布偏窄 | 每个 worker 用独立 seed_offset，cube_pose / lighting / camera_pose 独立采样 |

---

## 输出物

- 自动化 trajectory 生成 pipeline（9 阶段 pick-place，pick-101 风格）
- shard 合并工具（`merge_shards.py`）
- ≥1000 条 PickPlaceRed LeRobot 数据集（带 DR + 多语言）
- 数据集审计报告
- HuggingFace Hub 数据仓库
