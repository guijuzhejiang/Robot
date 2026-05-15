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
| 3 真机 demo + sim 扩增 | 未启动（需真机 demo） | 0 | — |
| 4 VLA 微调 | 未启动 | 0 | — |
| 5 真机部署 | 未启动（需硬件） | 0 | — |
| 6 HIL recovery | 未启动（需硬件 + 踏板） | 0 | — |
| 7 扩展任务 | 未启动 | 0 | — |

---

## Phase 1：仿真平台搭建

对应计划：[phase1-simulation-platform.md](plans/phase1-simulation-platform.md)

| 文件 | 对应任务 | 实现内容 |
|------|----------|----------|
| `assets/scenes/pick_place_blue.xml` | T1.2 / T1.4 / T1.5 | SO-ARM101 + 红/蓝 cube + plate + 双相机（front + wrist_cam），via `<include>` 引用 menagerie 的 `so101.xml`；home keyframe 把 ee 摆在工作区中央上方 14cm |
| `assets/scenes/assets` (symlink) | T1.2 | 指向 `../menagerie/robotstudio_so101/assets/`，修复 MuJoCo `<include>` 的 meshdir 解析（include 文件的 meshdir 实际相对于 PARENT 文件位置） |
| `sim/envs/base.py` | T1.3 | `BaseSoArmEnv`：Gymnasium 兼容、双相机渲染、joint/ee 双 action mode；ee 模式通过 lazy import 调用 `EeIkController` |
| `sim/envs/pick_place_blue.py` | T1.4 | 任务 env：左半区随机化双 cube（最小间距 8cm）、右半区随机化 plate；6 条 success 判据 + 失败归因枚举（`color_confusion` / `grasp_fail` / `transport_collision_red` / `place_miss` / `lift_drop` / `joint_limit` / `timeout`） |
| `sim/controllers/ik.py` | T1.6 | mink 5-DoF position-only IK（orientation_cost=0，因为 5-DoF 无法稳定满足 6-DoF 目标）；warmstart from prev q；20 iters 可达 z=0.005 |
| `sim/scripted_policies/pick_place_blue.py` | T1.7 | 6 阶段 oracle 策略：APPROACH → DESCEND → GRASP → LIFT → TRANSPORT → PLACE_RELEASE；策略读 `obs["blue_cube_pos"]` / `obs["plate_pos"]`（oracle，VLA 必须从图像 + 语言学到同等行为） |
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
| `data/instructions/pick_place_blue.txt` | T2.9 | 15 条同义指令（"put"/"place"/"move"/"grab"/"drop" + plate/dish + 是否提及红 cube） |
| `eval/audit_dataset.py` | T2.8 | 磁盘直读审计（绕过 HF Hub 401）：parse `meta/info.json` + `data/chunk-*/file-*.parquet` + `meta/tasks.parquet`；输出 num_episodes、avg duration、per-ep jerk p95、task 分布 |

**未实现的 Phase 2 任务**（plan 列出但本次不交付）：
- T2.4 失败筛除：env 已内置 `evaluate_success()` 与失败归因，但 `failures.log` 沉淀文件未做
- T2.5 完整 DR 5 维：lighting / camera / textures / cube_pose / plate_pose 都有了，但 `dynamics.py`（质量/摩擦扰动）和 HDRI 集成未做
- T2.6 LeRobot writer 正式版：基础版已能跑（writer 在 Phase 1），但 metadata 字段（`seed` / `object_color` / `sim_random_seed` / `source`）未加
- T2.10 正式批量生成：infrastructure ready，未跑 1000 条 + 推 HF Hub

---

## 启动命令

```bash
# 激活环境
conda activate py312_cu121

# 单进程跑 5 ep（带 DR）
python -m sim.collectors.parallel_runner --num-episodes 5 --num-workers 1 \
    --repo-id local/so101_pickplace_blue_test

# 8 worker 并行 1000 ep（生产规模）
python -m sim.collectors.parallel_runner --num-episodes 1000 --num-workers 8 \
    --repo-id local/so101_pickplace_blue_v1

# 关闭 DR（debug 用）
python -m sim.collectors.parallel_runner --num-episodes 5 --num-workers 1 \
    --repo-id local/so101_pickplace_blue_test --no-dr

# 审计已生成的 shard
python -m eval.audit_dataset --repo-id local/so101_pickplace_blue_v1_shard00 \
    --out-dir data/sim_generated/audit/

# 单测某个文件（示例）
python -c "
import sys; sys.path.insert(0, '.')
from sim.envs.pick_place_blue import PickPlaceBlueEnv
env = PickPlaceBlueEnv(observation_mode='both', action_mode='ee')
obs, _ = env.reset(seed=0)
print('blue:', obs['blue_cube_pos'], 'plate:', obs['plate_pos'])
"
```

数据集默认写到 `~/.cache/huggingface/lerobot/<repo_id>/`（LeRobot 0.5.x 标准位置）。

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

---

## 待实现 Phase（按优先级）

### Phase 3：真机 demo + sim 扩增
**前置**：先采 ~100 条真机 demo（Phase 0 T0.5 之后）
**关键文件**：`data/mimicgen_adapter/{segmenter,replayer,augment,object_tracker}.py`、`data/converters/merge_datasets.py`
建议在采到 demo 之后单独开新会话调试 segmenter 边界检测和 replayer 多锚点变换。

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
