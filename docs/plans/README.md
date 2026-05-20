# SO101 + Pi0.5 VLA 开发计划总索引

> 项目主文档：[../../README.md](../../README.md)

本目录是项目的执行手册。每个 Phase 对应一个独立文件，按顺序执行。

---

## 核心任务定义（贯穿全 Phase）

> **任务代号**：`PickPlaceRed`
> **语言指令模板**：`"put the red cube on the plate"` （及 10+ 同义变体）
> **场景**：桌面上随机摆放 1 个红 cube（3cm）+ 1 个 plate（6cm 半径）
> **机器人**：SO-ARM101（6-DoF + 夹爪），俯视顶抓（wrist_flex/wrist_roll 锁定 π/2）
> **执行流**：approach → descend → grasp → lift → transport above plate → place → release → retract
> **成功判据**（全部满足）：
>   1. red cube 中心 xy 距 plate 中心 < plate 半径
>   2. red cube 底面 z 在 plate 表面 ±1cm 内
>   3. 夹爪在终态已松开
>   4. 机械臂无关节超限/振荡

**命名约定**：
- 文档统一用 `PickPlaceRed` 描述任务概念
- 代码使用颜色无关命名：`sim/envs/pick_place.py`、`PickPlaceEnv`、`local/so101_pickplace_v1` 等，便于未来扩展到其他颜色/多物体变体

---

## 全 Phase 代码入口速查（TL;DR）

> 仓库根目录 `/home/zzg/workspace/pycharm/Robot`，先 `conda activate py312_cu121`。

```bash
# Phase 1：跑通仿真 + 渲染抓取视频
python -m mujoco.viewer --mjcf=assets/scenes/pick_place.xml               # 视觉自检
python -m sim.collectors.parallel_runner --num-episodes 20 --num-workers 4 \
    --repo-id local/so101_debug --keep-failures --cleanup-after-collect

# Phase 2：批量生成 ≥1000 条仿真 episode
python -m sim.collectors.parallel_runner --num-episodes 1500 --num-workers 8 \
    --repo-id local/so101_pickplace_v1 \
    --instructions data/instructions/pick_place.txt \
    --cleanup-after-collect
python -m data.converters.merge_shards \
    --shard-glob 'local/so101_pickplace_v1_shard*' \
    --output-repo local/so101_pickplace_v1
python -m eval.audit_dataset --repo-id local/so101_pickplace_v1 --n-sample 100

# Phase 3：LeIsaac 录制 + IsaacLab Mimic 扩增（在 leisaac 仓库内执行）
cd /home/zzg/workspace/pycharm/leisaac
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-SO101-PickPlaceRed-v0 --teleop_device=so101leader \
    --port=/dev/ttyACM0 --num_envs=1 --device=cuda --enable_cameras \
    --record --dataset_file=./datasets/pickplace_seeds.hdf5
python scripts/mimic/{eef_action_process,annotate_demos,generate_dataset}.py ...
python scripts/convert/isaaclab2lerobot.py --input ./datasets/pickplace_mimic.hdf5 \
    --output-repo-id local/so101_pickplace_mimic_v0

# Phase 4–7：待实现，各文档"代码入口"为占位计划
```

各 Phase 文档顶部 **「代码入口（快速开始）」** 小节给出对应命令的详细含义。

---

## Phase 一览

| Phase | 主题 | 周期 | 文件 |
|-------|------|------|------|
| 0 | 环境与可行性验证 | 1 周 | [phase0-environment-setup.md](phase0-environment-setup.md) |
| 1 | 仿真平台搭建 | 1–2 周 | [phase1-simulation-platform.md](phase1-simulation-platform.md) |
| 2 | 自动轨迹生成 | 2–3 周 | [phase2-trajectory-generation.md](phase2-trajectory-generation.md) |
| 3 | LeIsaac 遥操采集 + IsaacLab Mimic 扩增 | 1–2 周 | [phase3-real-demo-sim-augmentation.md](phase3-real-demo-sim-augmentation.md) |
| 4 | VLA 微调 | 2–3 周 | [phase4-vla-finetuning.md](phase4-vla-finetuning.md) |
| 5 | 真机部署与 co-finetune | 2–3 周 | [phase5-real-deployment.md](phase5-real-deployment.md) |
| 6 | HIL recovery 数据采集与回炉微调 | 2–3 周 | [phase6-hil-recovery.md](phase6-hil-recovery.md) |
| 7 | 扩展任务与高级指令 | 持续 | [phase7-extensions.md](phase7-extensions.md) |

**总预估周期：12–18 周（约 3–4.5 个月）打通主链路，Phase 7 持续迭代**

**三阶段训练范式**：

| 阶段 | Phase | 数据来源 | 目的 |
|------|-------|----------|------|
| 1️⃣ 仿真预训练 | Phase 1–4 | 大规模 sim（IsaacLab Mimic 扩增）+ 真机 demo | 学会任务基本能力 |
| 2️⃣ 少量真实微调 | Phase 5 | 真机 demo + Phase 5 targeted demo | 真机分布对齐 |
| 3️⃣ HIL recovery/correction | Phase 6 | 模型自跑 → 人介入纠正的恢复数据集 | 修复 distribution shift |

---

## 备选平台与备用方案

| 文档 | 角色 |
|---|---|
| [alt-platform-groot-n15.md](alt-platform-groot-n15.md) | **VLA 备用方案**：GR00T N1.5 + SO-101 微调（HF × NVIDIA 联合官方） |
| [alt-platform-isaaclab.md](alt-platform-isaaclab.md) | IsaacLab 基础学习（Phase 3 LeIsaac 是其子集） |
| [alt-platform-genesis.md](alt-platform-genesis.md) | Phase 6 GPU 速度对比实验候选 |
| [alt-platform-maniskill3.md](alt-platform-maniskill3.md) | Phase 4 VLA 预训练数据源（公开 demo 数据集） |
| [alt-platform-robocasa-mimicgen.md](alt-platform-robocasa-mimicgen.md) | MimicGen 算法原理参考（实操路径已改走 IsaacLab Mimic） |

---

## 全局技术栈

| 类别 | 工具 | 用途 |
|------|------|------|
| 系统 / Python | Ubuntu 22.04 + CPython 3.10–3.12 | 开发环境 |
| 加速 | CUDA 12.4+ + PyTorch 2.4+ | GPU 训练/推理 |
| 仿真 | MuJoCo 3.2+ + mujoco_menagerie + mink（IK） | Phase 1/2 主线 |
| 仿真（扩增） | Isaac Sim 6.0 + IsaacLab 0.47 + LeIsaac 0.4 | Phase 3 LeIsaac 路线 |
| 机器人栈 | lerobot | 数据采集/训练/部署 |
| VLA 主路径 | smolvla / openpi (Pi0.5) | Phase 4 微调 |
| VLA 备用 | NVIDIA Isaac GR00T N1.5 | 备用方案 |
| 评估 / 日志 | SimplerEnv + wandb | 训练监控 |

---

## 使用方式

1. 按顺序打开每个 Phase 的 md 文件
2. 每个任务有 `- [ ]` checkbox，完成后勾选
3. 每个 Phase 末尾的"验收标准"全部满足后再进入下一 Phase
4. 卡住时回看本 Phase 的"风险与陷阱"

## 任务命名

任务编号：`T<phase>.<seq>`，例如 `T0.1` 表示 Phase 0 的第 1 个任务。每个任务包含：目标 / 步骤 / 关键文件 / 参考 / 验证。
