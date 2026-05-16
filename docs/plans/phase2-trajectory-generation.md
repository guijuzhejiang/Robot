# Phase 2：自动轨迹生成

**周期**：2–3 周
**前置依赖**：Phase 1 完成（PickPlaceBlue 仿真环境可用）
**目标**：实现一套自动化 trajectory 生成 pipeline，在仿真里产出 ≥1000 条 **PickPlaceBlue** 成功 episode（带域随机化），全部存为 LeRobot 数据集格式

> **核心任务**：抓蓝 cube 放进 plate（红 cube 是干扰物）。详细规约见 [README.md](README.md) 顶部"核心任务定义"。

---

## 代码入口（快速开始）

> 所有命令在仓库根目录 `/home/zzg/workspace/pycharm/Robot` 下、`conda activate py312_cu121` 后执行。
> Phase 2 与 Phase 1 共用 `parallel_runner` 作为采集入口——区别在于 Phase 2 启用完整 DR + 大规模并行。

| 想做的事 | 一行命令 | 产出 |
|---------|---------|------|
| 跑单条 episode 看脚本策略是否抓得到（T2.1 / T2.2 / T2.3 联调） | `python -m sim.collectors.parallel_runner --num-episodes 1 --num-workers 1 --repo-id local/so101_debug` | 一条 episode，可在 viewer 或视频里检查 |
| **正式批量采集（T2.10 Phase 2 最终交付，目标 ≥1000 条成功 episode）** | `python -m sim.collectors.parallel_runner --num-episodes 1500 --num-workers 8 --repo-id local/so101_pickplace_blue_v1 --instructions data/instructions/pick_place_blue.txt` | `~/.cache/huggingface/lerobot/local/so101_pickplace_blue_v1/`（含 front+wrist mp4 + state/action parquet） |
| 关闭 DR 跑对照（T2.5 调试用） | 上一条命令追加 `--no-dr` | 同上 |
| 数据集质量审计（T2.8 验收） | `python -m eval.audit_dataset --repo-id local/so101_pickplace_blue_v1 --n-sample 100 --out-dir runs/audit_v1` | 控制台 + `runs/audit_v1/report.md` |
| 查看任一 episode 抓取视频 | `ls ~/.cache/huggingface/lerobot/local/so101_pickplace_blue_v1/videos/chunk-000/observation.images.front/` | `episode_*.mp4` |

**入口与任务清单的对应关系**：

| Task | 实现文件（库） | 由哪个 CLI 串起来 |
|------|--------------|------------------|
| T2.1 Antipodal grasp sampler | [`sim/grasp/antipodal.py`](../../sim/grasp/antipodal.py) | 被 `pick_place_pipeline` 调用 |
| T2.2 Waypoint planner | [`sim/planning/waypoint.py`](../../sim/planning/waypoint.py) | 同上 |
| T2.3 6 阶段编排 | [`sim/scripted_policies/pick_place_pipeline.py`](../../sim/scripted_policies/pick_place_pipeline.py) | 被 `parallel_runner` 调用 |
| T2.4 成功判定 + 失败归因 | [`sim/envs/pick_place_blue.py::evaluate_success / evaluate_failure_mode`](../../sim/envs/pick_place_blue.py) | 同上 |
| T2.5 全量 DR | [`sim/randomization/`](../../sim/randomization/) | `parallel_runner` 默认开启 |
| T2.6 LeRobot writer | [`data/converters/sim_to_lerobot.py`](../../data/converters/sim_to_lerobot.py) | `parallel_runner` 内部使用 |
| T2.7 并行采集 | [`sim/collectors/parallel_runner.py`](../../sim/collectors/parallel_runner.py) | `--num-workers N` |
| T2.8 数据集审计 | [`eval/audit_dataset.py`](../../eval/audit_dataset.py) | 独立 CLI |
| T2.9 多语言指令 | [`data/instructions/pick_place_blue.txt`](../../data/instructions/pick_place_blue.txt) | `parallel_runner --instructions ...` |
| T2.10 批量生成 | 上面所有合起来 | `parallel_runner --num-episodes 1500 --num-workers 8` |

> **TL;DR**：Phase 2 的"一条命令"就是表中第 2 行 `parallel_runner --num-episodes 1500`。预期生成的 mp4 视频在 dataset 的 `videos/` 目录下，每个 episode 各一个文件，可直接用任意播放器打开。

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

### T2.1 Antipodal grasp sampler（针对 blue cube）

**目标**：给定 blue cube 姿态，返回若干候选抓取位姿（红 cube 不参与采样）

**步骤**：
- [ ] 在 `sim/grasp/antipodal.py` 实现：
  - 输入：物体 mesh（cube 简化为 OBB）+ 当前姿态
  - 算法：在物体表面对采样平行的两个接触点，对应夹爪开合方向；按抓取深度、与重力方向夹角打分
  - **避碰评分项**：候选抓取位姿对红 cube 与 plate 的接近度作为惩罚项（避免抓蓝时撞红或撞 plate）
  - 输出：N 个候选 ee 位姿（按分数排序）
- [ ] 加可视化（matplotlib 3D 或 trimesh），可视化时把红 cube + plate 也画上

**关键文件**：
- `sim/grasp/antipodal.py`
- `sim/grasp/__init__.py`

**参考**：
- robosuite `samplers`：思路参考
- 论文 *Antipodal grasping with one parameter sweep*（短读）

**验证**：对一个 4cm blue cube 任意姿态（红 cube + plate 在场），能返回 ≥10 个候选抓取，避碰评分合理

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

### T2.3 任务级 trajectory 编排（6 阶段 pick-place）

**目标**：把整条任务拆成可复用阶段

**步骤**：
- [ ] 在 `sim/scripted_policies/pick_place_pipeline.py` 定义阶段：
  ```
  approach_blue → pre_grasp → grasp → lift → transport_above_plate → place_release → retract
  ```
- [ ] 每阶段是一个返回关节轨迹的函数
- [ ] **transport_above_plate** 需检查中间路点是否与红 cube 发生水平投影冲突；冲突时绕行（提高 lift 高度或加中转点）
- [ ] **place_release** 阶段：下降到 plate 表面上方 3cm → 松开 → 抬起 5cm 撤离
- [ ] 阶段间用 `wait_until_stable` 衔接
- [ ] 暴露成 `generate_pickplace_episode(env, seed) -> Episode | None`

**关键文件**：
- `sim/scripted_policies/pick_place_pipeline.py`

**验证**：跑 100 次随机 seed，成功率 ≥ 80%（无 DR）；其中红 cube 被误碰比例 < 5%

---

### T2.4 Success 判定与失败筛除

**目标**：自动判断 + 丢弃失败 episode

**步骤**：
- [ ] 在 `sim/envs/pick_place_blue.py` 完善 `evaluate_success()`（与 Phase 1 T1.4 一致）：
  1. blue cube 中心到 plate 中心 xy 距离 < plate 半径
  2. blue cube 底面 z 在 plate 表面 ±1cm
  3. red cube 位移 < 2cm
  4. 夹爪已松开
  5. 机械臂无超限位 / 无关节振荡
- [ ] 失败原因枚举：`color_confusion` / `grasp_fail` / `lift_drop` / `transport_collision_red` / `place_miss` / `joint_limit` / `timeout`
- [ ] 在 pipeline 里：失败 episode 直接丢弃，记录失败原因到 `data/sim_generated/failures.log`

**关键文件**：
- `sim/envs/pick_place_blue.py`（更新 evaluate_success + 失败归因）
- `sim/scripted_policies/pick_place_pipeline.py`（更新过滤逻辑）

**验证**：失败原因有分类统计；其中 `transport_collision_red` 占比作为 transport 阶段质量的核心指标

---

### T2.5 全量域随机化

**目标**：让生成的数据具备 sim2real 所需的视觉多样性

**步骤**：
- [ ] 扩充 `sim/randomization/`：
  - `lighting.py`：3 个随机点光源 + 1 个 HDRI 环境光（公开 HDRI 包：Poly Haven 子集）
  - `textures.py`：桌面 30+ 纹理（PBR 公开素材如 ambientCG）
  - `object.py`：**保留** cube 尺寸（3–5cm）、初始姿态、初始角度的随机化；**禁止**随机化红/蓝两个 cube 的颜色（颜色是语义锚点）
  - `plate.py`：plate 直径（10–15cm）、颜色/纹理（可随机，不影响 instruction）、位置 ±3cm
  - `camera_pose.py`：±10cm xy / ±3cm z / ±10° rotation 抖动 front cam
  - `dynamics.py`：质量 ±20% / 摩擦 ±30%
- [ ] 每个 episode reset 时各项独立采样
- [ ] **位置约束**：两个 cube 在工作区左半区，最小间距 8cm；plate 在右半区

**关键文件**：
- `sim/randomization/*.py`（扩充）
- `assets/textures/`：本地纹理资源
- `assets/hdri/`：HDRI 资源

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
  - features 包含：`observation.images.wrist`、`observation.images.front`、`observation.state`（关节）、`action`（关节目标）、`task`（语言指令字符串）
  - 写帧 → 调 `dataset.save_episode()`
- [ ] 加 metadata：`task`、`seed`、`object_color`、`sim_random_seed`

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

**验证**：8 worker 跑 1000 episode，预计耗时 < 30 分钟（4cm cube 任务）

---

### T2.8 数据集质量审计

**目标**：避免生成出"假成功"垃圾数据

**步骤**：
- [ ] 写 `eval/audit_dataset.py`，对生成的 dataset 做：
  - 随机抽 50 条人工目视检查
  - 统计：平均时长、关节轨迹平滑度（jerk 上限）、夹爪闭合点分布
  - 检查关节是否打表（饱和限位）
- [ ] 输出 `data/sim_generated/pick_cube_v0/audit.md`

**关键文件**：
- `eval/audit_dataset.py`

**验证**：人工抽检中，≥90% 的 episode 看起来合理（无打转、无穿模）

---

### T2.9 多语言指令绑定（PickPlaceBlue）

**目标**：为后续 VLA 训练准备语言多样性

**步骤**：
- [ ] 创建 `data/instructions/pick_place_blue.txt`，写 15+ 同义指令：
  ```
  put the blue cube on the plate
  place the blue cube on the plate
  pick up the blue cube and put it on the plate
  move the blue block to the plate
  grab the blue cube and place it in the dish
  take the blue cube and drop it on the plate
  the blue cube goes on the plate
  put the blue block on the plate, leave the red one
  ...
  ```
- [ ] 每条 episode 生成时随机选 1 条作为 `task` 字段
- [ ] **颜色锚定固定为 "blue"**（不动态替换，红 cube 始终是干扰物）
- [ ] 词汇变体覆盖：动词（put/place/move/grab/drop）、容器称呼（plate/dish）、是否提到干扰物（"leave the red one"）

**关键文件**：
- `data/instructions/pick_place_blue.txt`
- `sim/collectors/parallel_runner.py`（注入逻辑）

**验证**：dataset 里 `task` 字段有 ≥10 种不同表述

---

### T2.10 正式批量生成

**目标**：生成第一份正式 sim 数据集

**步骤**：
- [ ] 全 DR 开启，生成目标 ≥1000 条成功 episode
- [ ] 推上 HuggingFace Hub（私有 repo）作为远端备份：`huggingface-cli upload`
- [ ] 写 dataset card（README）

**关键文件**：
- `data/sim_generated/pick_place_blue_v1/`
- HuggingFace repo: `<your_id>/so101_pick_place_blue_v1`

**验证**：
- ≥1000 条成功 episode
- audit 通过率 ≥90%
- HF Hub 可访问

---

## 验收标准（全部满足后进入 Phase 3）

- [ ] grasp sampling + IK + planner pipeline 自动化运行
- [ ] 域随机化 5 个维度全部接入并独立可控
- [ ] 单机并行采集，1000 条 episode 在 1 小时内完成
- [ ] 生成 dataset 通过质量审计（≥90% 合理）
- [ ] 至少 10 种语言指令变体
- [ ] dataset 已推送到 HF Hub

---

## 风险与陷阱

| 风险 | 应对 |
|------|------|
| antipodal 抓取在物体倾斜时失败 | 加 "抓取前 normalize 物体到水平" 的 pre-grasp 阶段 |
| IK 解收敛慢拖累采集速度 | mink 加 warmstart；缓存上一帧 q |
| 域随机化的随机纹理太花引起策略不稳定 | 给红/蓝 cube 颜色独立染色，桌面纹理与 cube 颜色对比明显 |
| transport 阶段撞到红 cube | 提高 lift 高度阈值；加水平绕行；红/蓝最小间距约束放大到 10cm |
| place 时 blue cube 弹出 plate | 降低释放高度（plate 表面上方 2cm 而非 3cm）；松开夹爪后等 5 帧让物理稳定 |
| 并行 worker 的渲染上下文冲突 | EGL 上下文每个 worker 独立创建；或用 osmesa fallback |
| 写 LeRobot 数据时图像被压缩失真 | 用 `image_writer_processes=4`，写 png 而非 jpg |
| 数据量到了但分布偏窄（如蓝 cube 永远在左红 cube 永远在右） | reset 时强制让红/蓝的相对位置覆盖 4 个象限 |

---

## 输出物

- 自动化 trajectory 生成 pipeline（6 阶段 pick-place）
- ≥1000 条 PickPlaceBlue LeRobot 数据集（带 DR + 多语言）
- 数据集审计报告
- HuggingFace Hub 数据仓库
