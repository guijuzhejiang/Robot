# Phase 2：自动轨迹生成

**周期**：2–3 周
**前置依赖**：Phase 1 完成（PickPlaceRed 仿真环境可用，脚本策略成功率 ≥ 90%）
**目标**：在仿真里产出 ≥1000 条 **PickPlaceRed** 成功 episode（带域随机化），全部存为 LeRobot 数据集格式

---

## 代码入口（快速开始）

> 与 Phase 1 共用 `parallel_runner`，区别在于启用完整 DR + 大规模并行。

```bash
# 调试单条
python -m sim.collectors.parallel_runner --num-episodes 1 --num-workers 1 \
    --repo-id local/so101_debug

# 正式批量（≥1000 条成功 episode）
python -m sim.collectors.parallel_runner --num-episodes 1500 --num-workers 8 \
    --repo-id local/so101_pickplace_v1 \
    --instructions data/instructions/pick_place.txt \
    --cleanup-after-collect

# 合并 8 个 shard
python -m data.converters.merge_shards \
    --shard-glob 'local/so101_pickplace_v1_shard*' \
    --output-repo local/so101_pickplace_v1

# 质量审计
python -m eval.audit_dataset --repo-id local/so101_pickplace_v1 --n-sample 100 \
    --out-dir runs/audit_v1

# 看任一 episode 视频
ls ~/.cache/huggingface/lerobot/local/so101_pickplace_v1/videos/observation.images.front/chunk-000/
```

**入口与任务的对应关系**：

| Task | 实现文件 |
|------------------|--------------|
| T2.3 9 阶段编排 | [`sim/scripted_policies/pick_place_pipeline.py`](../../sim/scripted_policies/pick_place_pipeline.py) |
| T2.4 成功判定 + 失败归因 | [`sim/envs/pick_place.py`](../../sim/envs/pick_place.py) |
| T2.5 全量 DR | [`sim/randomization/`](../../sim/randomization/) |
| T2.6 LeRobot writer | [`data/converters/sim_to_lerobot.py`](../../data/converters/sim_to_lerobot.py) |
| T2.7 并行采集 | [`sim/collectors/parallel_runner.py`](../../sim/collectors/parallel_runner.py) |
| T2.7.5 shard 合并 | [`data/converters/merge_shards.py`](../../data/converters/merge_shards.py) |
| T2.8 数据集审计 | [`eval/audit_dataset.py`](../../eval/audit_dataset.py) |
| T2.9 多语言指令 | [`data/instructions/pick_place.txt`](../../data/instructions/pick_place.txt) |

> **TL;DR**：核心两步 = `parallel_runner --num-episodes 1500 --cleanup-after-collect` + `merge_shards --shard-glob ...`。

> **升级路径**（如成功率不达标）：抓取采样 antipodal → [Contact-GraspNet](https://github.com/NVlabs/contact_graspnet)；规划 linear waypoint → [cuRobo](https://github.com/NVlabs/curobo)；并行 multiprocessing → MJX GPU。

---

## 任务清单

### T2.3 任务级 trajectory 编排（9 阶段，pick-101 风格）

在 `sim/scripted_policies/pick_place_pipeline.py` 定义阶段：

```
HOME → APPROACH → DESCEND → GRASP → LIFT → TRANSPORT → PLACE_DESCEND → RELEASE → RETRACT
```

每阶段返回 ee_target + gripper_action（写入 `env.ee_target_override` / `env.gripper_action_override`）。关键参数：
- **APPROACH/DESCEND** 用 `cube_anchor`（首次 snapshot 后不变），加 `FINGER_WIDTH_OFFSET=-0.015` 让 cube 居中
- **PLACE_DESCEND** 闭环：`target = ee_cur + (desired_cube - cube_obs)`（对未知 wrist 旋转鲁棒）
- **TRANSPORT_Z=0.10**（不能再高，否则 elbow_flex 触限位）
- 阶段间 `PHASE_HOLD_STEPS=15`、GRASP `GRASP_RAMP_STEPS=60` 渐进闭合

暴露成 `generate_pickplace_episode(env, seed) -> Episode | None`。

**验证**：100 次随机 seed，成功率 ≥ 90%（v21 实测 93.8%）

---

### T2.4 Success 判定与失败归因

在 `sim/envs/pick_place.py` 完善 `evaluate_success()`：
1. red cube 中心 xy 距 plate 中心 < plate 半径
2. cube 底面 z 在 plate 表面 ±1cm
3. 夹爪已松开
4. 机械臂无超限位 / 无关节振荡

失败枚举：`grasp_fail` / `lift_drop` / `place_miss` / `plate_off` / `joint_limit` / `timeout`。

Pipeline 默认丢弃失败 episode；`--keep-failures` 时保留并把 mode 写入 task 字段 `[FAIL:<mode>]` 后缀。

---

### T2.5 全量域随机化

扩充 `sim/randomization/`：
- `lighting.py`：3 个点光源位置/强度
- `textures.py`：桌面 30+ 纹理（PBR 公开素材如 [ambientCG](https://ambientcg.com)）
- `cube_pose.py`：cube xy + yaw（量化 4 等分）+ plate 位置 ±3cm
- `camera_pose.py`：±10cm xy / ±3cm z / ±10° rotation 抖动 front cam
- `dynamics.py`：质量 ±20% / 摩擦 ±30%（注意不要让摩擦太低导致 cube 滑落）

每个 episode reset 时各项独立采样。**铁律**：cube 颜色固定 red，不随机。

**验证**：连续 reset 20 次截图，所有维度都有明显差异

---

### T2.6 LeRobot dataset writer

在 `data/converters/sim_to_lerobot.py` 实现：
- `LeRobotDataset.create(repo_id, features={...}, fps=30)`
- features 包含：`observation.images.{wrist,front}`、`observation.state`（6 维关节，names 用 `*.pos` 后缀对齐 LeRobot SO-101）、`action`（6 维 joint ctrl 目标，未归一化弧度 + gripper qpos）、`ee_action`（4 维 ee-delta + gripper，归一化 [-1,1]，sim 调试辅助）、`task`（语言指令）
- 写帧 → 调 `dataset.save_episode()`
- metadata：`task`、`seed`、`sim_random_seed`

#### T2.6.1 action 标签真实化（关键修复）

**问题**：脚本策略通过 `env.ee_target_override` 旁路驱动 IK，所以 `policy(env, obs)` 返回的 gym 兼容 action 永远是 `[0, 0, 0, gripper]`——dx/dy/dz 不携带任何运动信息。直接落盘 100% 为零，VLA 学不到任何手臂动作。

**修复**：保持 schema `action: float32 [4]`（dx, dy, dz, gripper ∈ [-1, 1]），step 前后采 TCP 位姿（`gripperframe` site = 夹爪两指中点向前 10cm），按 `EE_DELTA_SCALE = 0.05 m/unit` 归一化重写到 action[:3]：

```python
action_recorded[:3] = clip((ee_after - ee_before) / EE_DELTA_SCALE, -1, 1)
action_recorded[3]  = policy 返回的 gripper（原样）
```

**落地点**：
- `sim/envs/base.py`：`EE_DELTA_SCALE = 0.05` + `encode_ee_delta_action(ee_before, ee_after, gripper_norm)`
- `sim/collectors/parallel_runner.py`：采集循环每步 step 前后调 `env.ee_pos()` 算位移，落盘前用 `encode_ee_delta_action` 重写
- `sim/scripted_policies/pick_place_pipeline.py`：`generate_pickplace_episode` 同步修复

**验证**：
```bash
python -m sim.collectors.parallel_runner --num-episodes 2 --num-workers 1 --repo-id local/so101_action_check
python -c "
import pyarrow.parquet as pq, numpy as np
A = np.stack(pq.read_table('~/.cache/huggingface/lerobot/local/so101_action_check_shard00/data/chunk-000/file-000.parquet').to_pandas()['action'].values)
assert (A[:, :3] != 0).any(axis=1).mean() > 0.99, 'dx/dy/dz 仍是零'
"
```

#### T2.6.2 action 列对齐 LeRobot SO-100/101 惯例

LeRobot `ACTION="action"` 是硬编码常量，**无 `dataset.action_key` 配置项**。最终设计：

| 列 | 形状 | 含义 |
|---|---|---|
| `action` | (6,) | `env.data.ctrl[:6]` 快照（IK + gripper 解出的 actuator 命令），与 LeRobot SO-101 follower `Goal_Position` 等价。**SmolVLA / ACT / Pi-0 默认读这列，零配置** |
| `ee_action` | (4,) | dx,dy,dz,gripper，sim-only 辅助，用于调试可视化或派生 ee-pose 格式（如 OpenVLA / X-VLA） |

训练时 SmolVLA 等默认走 `--dataset.repo_id=local/so101_pickplace_v1`；OpenVLA/X-VLA 先跑 `python -m data.converters.derive_action_format --target-format <ee_*_format>` 派生。

**验证**：`lerobot-dataset-viz --repo-id=local/so101_pickplace_v1 --episode-index=0` 正常播放

---

### T2.7 并行化采集

在 `sim/collectors/parallel_runner.py`：主进程调度 N worker（默认 = CPU 物理核数），每个独立 env / seed / 写本地分片。增加 tqdm 进度条 + ETA。

**验证**：8 worker 跑 1000 episode，预计 < 60 分钟（3cm cube + 480p 渲染）

---

### T2.7.5 Shard 合并

`data/converters/merge_shards.py`：输入 N 个 shard repo_id（或 glob），输出单一 merged repo_id。用 LeRobotDataset API 逐帧重写（视频重编码 ~50 fr/s）。

CLI：`--shards REPO_ID ...` 或 `--shard-glob 'PATTERN'`、`--output-repo`、`--reset`。不删源 shard，用户手动 `rm -rf` 验证安全。

**跨批次合并**：分多次跑 `parallel_runner`（例如先 v1 1000 条，后补 v2 500 条）时，先各自合并成 v1/v2，再当两个 shard 二次合并到 combined。`merge_shards` 不要求 repo_id 含 `_shardNN`，只看 `meta/info.json` 存在。

---

### T2.8 数据集质量审计

`eval/audit_dataset.py`：
- 随机抽 50 条人工目视
- 统计：平均时长、关节轨迹平滑度（jerk 上限）、夹爪闭合点分布
- 检查关节是否打表（饱和限位）
- 输出 `runs/audit_v1/report.md`

**验证**：人工抽检 ≥90% episode 合理

---

### T2.9 多语言指令池

`data/instructions/pick_place.txt` 写 15+ 同义指令：

```
put the red cube on the plate
place the red cube on the plate
pick up the red cube and put it on the plate
move the red block to the plate
grab the red cube and place it in the dish
take the red cube and drop it on the plate
...
```

每条 episode 随机抽 1 条作 `task` 字段。**颜色锚定固定为 "red"**。词汇覆盖动词（put/place/move/grab/drop）+ 容器称呼（plate/dish）。

**验证**：dataset 里 `task` 字段有 ≥10 种不同表述

---

### T2.10 正式批量生成

- [ ] 全 DR 开启，目标 ≥1000 条成功 episode（建议 1500 条，93.8% 成功率落地 ~1400）
- [ ] 合并 8 个 shard
- [ ] 推 HuggingFace Hub（私有 repo）：`huggingface-cli upload`
- [ ] 写 dataset card

---

## 验收标准

- [ ] grasp + IK + planner pipeline 自动化运行（v21 配置 93.8%）
- [ ] DR 5 个维度独立可控
- [ ] 8 worker 并行 + shard 合并，1500 条 < 2 小时
- [ ] audit 通过率 ≥90%
- [ ] ≥10 种语言指令变体
- [ ] dataset 推到 HF Hub

---

## 风险与陷阱

| 风险 | 应对 |
|------|------|
| cube yaw=±45° 时抓取失败 | yaw 量化到 4 等分（cube 4 重对称）—— 实测 60% → 92% |
| 工作区角落 cube 抓不到 | wrist_roll 按 `atan2(cube.y, cube.x)` 同步设定，保证总 z 旋转 = π/2 |
| 500Hz IK 反而让成功率变差 | sts3215 严重欠阻尼，100Hz IK + 跨 substep 锁目标 |
| TRANSPORT 时 elbow_flex 触下限位 | TRANSPORT_Z 0.15 → 0.10；不在 TRANSPORT 做 xy 跟踪 |
| PLACE_DESCEND 因 wrist 旋转致 FINGER_WIDTH_OFFSET 失效 | 改闭环 `target = ee_cur + (desired_cube - cube_obs)` |
| 并行 worker 渲染上下文冲突 | `MUJOCO_GL=egl` + spawn context，每个 worker 独立 EGL |
| LeRobot 图像被压缩失真 | features 用 `dtype=video`，编码 av1 高质 |
| shard 之间分布偏窄 | 每 worker 独立 seed_offset，cube/光照/相机独立采样 |
