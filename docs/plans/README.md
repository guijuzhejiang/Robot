# SO101 + Pi0.5 VLA 开发计划总索引

> 项目主文档：[../../README.md](../../README.md)
> 计划撰写日期：2026-05-13

本目录是项目的执行手册。每个 Phase 对应一个独立文件，按顺序执行。

---

## 核心任务定义（贯穿全 Phase）

> **任务代号**：`PickPlaceBlue`
> **语言指令模板**：`"put the blue cube on the plate"` （及 10+ 同义变体）
> **场景**：桌面上随机摆放 1 个红 cube + 1 个蓝 cube + 1 个盘子（plate）
> **机器人**：SO-ARM101（6-DoF + 夹爪）
> **执行流**：approach blue → grasp blue → lift → transport above plate → place → release → retract
> **成功判据**（全部满足）：
>   1. blue cube 中心位于 plate 半径内
>   2. blue cube 与 plate 接触（z 落在 plate 表面 ±1cm）
>   3. red cube 位移 < 2cm（**未被误碰/误抓**）
>   4. 夹爪在终态已松开

**与旧版"PickCube"的差异**（旧文档里看到 PickCube/单红 cube 的地方按以下规则替换）：

| 维度 | 旧 PickCube | 新 PickPlaceBlue |
|------|-------------|-----------------|
| 场景物体 | 1 × 红 cube | 1 × 红 + 1 × 蓝 + 1 × plate |
| 任务动作 | pick + lift | pick + lift + transport + place |
| 语言锚定 | "pick the red cube" | "put the **blue** cube on the plate"（必须区分颜色） |
| 关键挑战 | 抓取几何 | 抓取几何 + **干扰物语义识别** + **放置精度** |
| 成功判据 | cube 抬起 > 5cm | blue 在 plate 上 + red 未动 |

**任务命名贯穿全文档**：`PickPlaceBlue` / `pick_place_blue` / `assets/scenes/pick_place_blue.xml`。

---

## 全 Phase 代码入口速查（TL;DR）

> 仓库根目录 `/home/zzg/workspace/pycharm/Robot`，先 `conda activate py312_cu121`。
> 每行后面括号是 Phase 文档内对应小节"代码入口（快速开始）"的更详细表格。

```bash
# Phase 0：环境验证（用 LeRobot 上游 CLI，本仓库无脚本）

# Phase 1：跑通仿真 + 渲染抓取视频
python -m mujoco.viewer --mjcf=assets/scenes/pick_place_blue.xml          # 视觉自检
# 当前 L-1：脚本策略成功率 0-20%。下面这条加 --keep-failures --cleanup-after-collect，
# 失败 episode 也切成 mp4 集中输出，shard 缓存自动删除。
python -m sim.collectors.parallel_runner --num-episodes 20 --num-workers 4 \
    --repo-id local/so101_debug --keep-failures --cleanup-after-collect
# 输出：~/.cache/huggingface/lerobot/local/so101_debug_videos/front/shardXX_epNNNN_FAIL_grasp_fail.mp4

# Phase 2：批量生成 ≥1000 条仿真 episode（Phase 1 同入口，规模放大）
python -m sim.collectors.parallel_runner --num-episodes 1500 --num-workers 8 \
    --repo-id local/so101_pickplace_blue_v1 \
    --instructions data/instructions/pick_place_blue.txt
python -m eval.audit_dataset --repo-id local/so101_pickplace_blue_v1 --n-sample 100

# Phase 3：MimicGen 扩增 + 数据集合并
python -m data.mimicgen_adapter.augment --from-sim-seeds 5 \
    --output-repo-id local/so101_sim_mimicgen_smoke --n-per-demo 10       # smoke
python -m data.converters.merge_datasets \
    --source local/so101_real_pickplace_blue_v0:real \
    --source local/so101_sim_mimicgen_v1:sim_mimicgen \
    --source local/so101_pickplace_blue_v1:sim_scripted \
    --output-repo-id local/so101_pickplace_blue_mixed_v1
python -m data.converters.expand_instructions --source-repo-id local/... --copies 3

# Phase 4–7：尚未实现，各文档内"代码入口"为占位计划
```

详细的每条命令含义、产出路径、与任务清单的对应关系，见各 Phase 文档顶部 **「代码入口（快速开始）」** 小节。

---

## Phase 一览

| Phase | 主题 | 周期 | 文件 |
|-------|------|------|------|
| 0 | 环境与可行性验证 | 1 周 | [phase0-environment-setup.md](phase0-environment-setup.md) |
| 1 | 仿真平台搭建 | 1–2 周 | [phase1-simulation-platform.md](phase1-simulation-platform.md) |
| 2 | 自动轨迹生成 | 2–3 周 | [phase2-trajectory-generation.md](phase2-trajectory-generation.md) |
| 3 | 真机 demo 采集与 sim 扩增 | 2 周 | [phase3-real-demo-sim-augmentation.md](phase3-real-demo-sim-augmentation.md) |
| 4 | VLA 微调 | 2–3 周 | [phase4-vla-finetuning.md](phase4-vla-finetuning.md) |
| 5 | 真机部署与 co-finetune | 2–3 周 | [phase5-real-deployment.md](phase5-real-deployment.md) |
| **6** | **HIL recovery 数据采集与回炉微调** | **2–3 周** | **[phase6-hil-recovery.md](phase6-hil-recovery.md)** |
| 7 | 扩展任务与高级指令 | 持续 | [phase7-extensions.md](phase7-extensions.md) |

**总预估周期：14–20 周（约 3.5–5 个月）打通主链路，Phase 7 持续迭代**

**三阶段训练范式**（从模型训练视角看主链路）：

| 阶段 | Phase | 数据来源 | 目的 |
|------|-------|----------|------|
| 1️⃣ 仿真预训练 | Phase 1–4 | 大规模 sim（MimicGen 扩增）+ 真机 demo | 学会任务的基本能力 |
| 2️⃣ 少量真实微调 | Phase 5 | Phase 3 真机 demo + Phase 5 补采的 targeted demo | 真机分布对齐 |
| 3️⃣ HIL recovery/correction | **Phase 6** | 模型自跑 → 人介入纠正的"恢复数据集" | 修复 distribution shift |

第 3 阶段（HIL）是 Pi0.6 / RECAP 论文证明的核心增益来源，对 sim2real 至关重要。

## 备选仿真平台（学习参考，非主路径）

主路径用 **MuJoCo + Menagerie**（见 Phase 1）。下列平台作为快速起步 / 算法学习 / 对比实验 / Phase 6 研究方向：

| 平台 | 用途 | 文件 |
|------|------|------|
| LeRobot 自带 sim | **Phase 0–1 最快起步**（gym-lowcostrobot 含 SO-ARM 原生支持） | [alt-platform-lerobot-sim.md](alt-platform-lerobot-sim.md) |
| RoboCasa / MimicGen | **Phase 3 算法移植必学**（数据扩增核心） | [alt-platform-robocasa-mimicgen.md](alt-platform-robocasa-mimicgen.md) |
| Genesis | GPU 极速数据生成对比，研究前沿 | [alt-platform-genesis.md](alt-platform-genesis.md) |
| ManiSkill 3 | 公开 demo 数据集借用 + GPU 并行 | [alt-platform-maniskill3.md](alt-platform-maniskill3.md) |
| Isaac Lab | NVIDIA 工业级 RTX 仿真栈 | [alt-platform-isaaclab.md](alt-platform-isaaclab.md) |

**5 个备选平台的角色定位**：

```
Phase 0 → Phase 1 起步
   └─ LeRobot 自带 sim（1 天闭环）

Phase 1 → Phase 2 主路径
   └─ MuJoCo + Menagerie（项目主线）

Phase 3 数据扩增
   └─ RoboCasa / MimicGen（学算法，移植到主路径）

Phase 2 / Phase 6 GPU 加速对比实验
   ├─ Genesis（速度最快）
   ├─ ManiSkill 3（任务库 + 公开 demo）
   └─ Isaac Lab（NVIDIA 工业级，硬件门槛高）
```

## 全局技术栈

| 类别 | 工具 | 版本（推荐） | 用途 |
|------|------|--------------|------|
| 系统 | Ubuntu | 22.04 LTS | 开发环境 |
| Python | CPython | 3.10 | LeRobot/openpi 最低要求 |
| 加速 | CUDA + cuDNN | 12.4 + 9.x | GPU 训练/推理 |
| 深度学习 | PyTorch | 2.4+ | 主框架 |
| 仿真 | MuJoCo | 3.2+ | 物理引擎 |
| 机器人资产 | mujoco_menagerie | latest | SO-ARM100 官方 MJCF |
| IK | mink | latest | 微分 IK |
| 机器人栈 | lerobot | latest | 数据采集/训练/部署 |
| VLA | openpi | latest | Pi0.5 微调主框架 |
| VLA 备选 | smolvla | latest | 轻量微调备选 |
| 数据扩增 | mimicgen | 参考实现 | 仿真数据扩增 |
| 评估 | SimplerEnv | latest | 仿真内真机风格评估 |
| 日志 | wandb | latest | 训练监控 |
| 数据 | huggingface datasets | latest | 数据集托管 |

## 使用方式

1. 按顺序打开每个 Phase 的 md 文件
2. 每个任务有 `- [ ]` checkbox，完成后勾选
3. 每个 Phase 末尾的"验收标准"全部满足后再进入下一 Phase
4. 卡住时回看本 Phase 的"风险与陷阱"

## 命名约定

- 任务编号：`T<phase>.<seq>`，例如 `T0.1` 表示 Phase 0 的第 1 个任务
- 每个任务包含：目标 / 步骤 / 关键文件 / 参考 / 验证
