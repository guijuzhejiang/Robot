# Phase 7：扩展任务与高级指令（持续迭代）

**周期**：持续（按兴趣 / 课题选做）
**前置依赖**：Phase 6 完成（v3+ 模型经 HIL recovery/correction 微调后真机鲁棒可用）
**目标**：把 PickPlaceBlue（双 cube + 单 plate + 颜色锚定）能力进一步扩展到 N 物体场景、多步骤复合指令、多容器选择、更复杂场景，并探索 real2sim 与世界模型的前沿方向

> **主路径已覆盖**：单干扰物（红）+ 单容器（plate）+ 单步 pick-place + 颜色锚定 + HIL 鲁棒化。Phase 7 在此基础上加复杂度。
> **Phase 6 的 HIL 方法可在每个新任务上重复使用**——每扩展一个能力，跑一轮 HIL 吃掉新失败模式。

---

## 代码入口（快速开始）

> **状态**：Phase 7 是开放性扩展，无单一入口。每个子方向都复用 Phase 1–6 的 CLI，只是改任务名 / 物体配置 / 指令池。下面给出典型套用模式：

| 子方向 | 复用入口 | 需要新增的资产 |
|-------|---------|--------------|
| 新颜色 / 多干扰物（如绿 cube） | `parallel_runner` + 新 MJCF | `assets/scenes/pick_place_<task>.xml`、`data/instructions/<task>.txt`、`sim/envs/<task>.py` |
| 多容器选择（多个 plate） | 同上 | 新 MJCF + 新 evaluate_success |
| 多步骤复合指令 | 新脚本策略 + 新 segmenter | `sim/scripted_policies/<task>.py`、`data/mimicgen_adapter/segmenter.py` 扩展段数 |
| Real2Sim（数字孪生场景） | Phase 3 标定 + 新 MJCF 自动生成 | `real2sim/` 工具链（未实现） |
| 世界模型 / RL 探索 | 独立训练栈，复用 PickPlaceBlueEnv | `rl/` 训练目录（未实现） |

> 实现新任务时建议：(1) 复制 `sim/envs/pick_place_blue.py` 为新任务文件，(2) 复制 `assets/scenes/pick_place_blue.xml` 为新场景，(3) 复制 `data/instructions/pick_place_blue.txt` 为新指令池，(4) 在 `parallel_runner` 加 `--env` 选项（当前硬编码为 PickPlaceBlue）即可走完整 Phase 1–6 流水线。

---

## 关键技术与工具

| 工具 | 用途 | 备注 |
|------|------|------|
| RialTo | 真实场景三维重建到仿真 | `https://github.com/real-to-sim-to-real/rialto` |
| SplatSim / PhysGaussian | 3D Gaussian Splatting + 物理仿真 | `https://github.com/qureshinomaan/SplatSim` |
| URDFormer | 单图重建可铰接物体 URDF | `https://github.com/WEIRDLabUW/urdformer` |
| GR00T-Dreams | 用世界模型生成想象 trajectory | NVIDIA, 2025 |
| Dreamer V3 / 1X World Model | 通用世界模型 | 学习参考 |
| robocasa-gen | 程序化场景生成 | RoboCasa 子模块 |

---

## 子方向一览

Phase 6 不是单一线性流程，而是多个可独立推进的子方向。建议根据兴趣 / 论文需求选 1–2 个深入。

### 方向 A：多物体多颜色乱堆场景

**目标**：从 1 红 + 1 蓝 + 1 plate → N 个不同颜色/形状物体 + 多容器，按指令抓指定物

**与主路径的差异**：主路径只学了"区分蓝 vs 红"；本方向要让模型理解"任意颜色 token"。

**任务清单**：

- [ ] **T6A.1** 扩展 PickPlaceBlue env 为 PickPlaceClutter：3–5 个不同颜色/形状物体 + 1 plate 随机摆放
- [ ] **T6A.2** 抓取采样升级：contact-aware antipodal（避免抓时碰到邻近物体）
- [ ] **T6A.3** 指令池扩展：`"put the {color} {shape} on the plate"` 模板，color ∈ {red, blue, green, yellow, orange}，shape ∈ {cube, block, sphere}
- [ ] **T6A.4** 真机采 50 条 cluttered demo + MimicGen 扩增 5K
- [ ] **T6A.5** Pi0.5 微调 + 真机评估
- [ ] **T6A.6** 评估"指令到物体"对应正确率（识别 + 抓对），与主路径"颜色锚定"指标对比

**关键参考**：
- *DexMimicGen* (NVIDIA, 2024)
- robosuite NutAssembly task

---

### 方向 B：多步骤复合指令 + 多容器选择

**目标**：在主路径单 plate 之外，处理多容器、多步骤复合任务

**与主路径的差异**：主路径只有 1 个 plate（隐式目标）；本方向引入容器歧义（红盘/蓝盘/碗）+ 多步顺序（先 A 后 B）。

**任务清单**：

- [ ] **T6B.1** 仿真加多容器：plate + bowl + tray，颜色各异
- [ ] **T6B.2** 单步带容器选择：`"put the blue cube in the {red plate | blue plate | bowl}"`
- [ ] **T6B.3** 两步顺序指令：`"first put the blue cube on the plate, then put the red cube in the bowl"`
- [ ] **T6B.4** 脚本策略支持多步：状态机扩展为 task-level FSM
- [ ] **T6B.5** 真机采 30 条多步 demo + 扩增
- [ ] **T6B.6** 微调与评估
- [ ] **T6B.7** 探索 chain-of-action：是否需要给 VLA 显式 sub-task token？两步任务在 LIBERO benchmark 上对比

**关键参考**：
- LIBERO benchmark：`https://github.com/Lifelong-Robot-Learning/LIBERO`（多步任务标准评估）
- Pi0.5 论文中关于 chain-of-thought action 的讨论

---

### 方向 C：Real2Sim 视觉对齐

**目标**：把真实工作台 1:1 重建到仿真，缩小 sim2real 视觉差

**任务清单**：

- [ ] **T6C.1** 用手机扫真实工作台（Polycam / 自己装 COLMAP）
- [ ] **T6C.2** 选 RialTo 或 SplatSim 重建
- [ ] **T6C.3** 把重建场景接入 MuJoCo（或 SplatSim 原生物理）
- [ ] **T6C.4** 用 Phase 2/3 的 pipeline 在重建场景里再生成 5K 数据
- [ ] **T6C.5** 微调评估：与"通用 DR sim"对比真机成功率提升幅度

**关键参考**：
- RialTo 论文：*Real-to-Sim-to-Real Transfer of Manipulation Policies through Real-World Object Modeling*
- SplatSim：`https://github.com/qureshinomaan/SplatSim`

**预期收益**：sim2real gap 缩小 20–40%

---

### 方向 D：世界模型增广（GR00T-Dreams 风格）

**目标**：用世界模型生成"想象 trajectory"，把数据规模扩大一个数量级

**任务清单**：

- [ ] **T6D.1** 调研：GR00T-Dreams / Dreamer V3 / 1X World Model 哪个最容易接入 SO101 数据
- [ ] **T6D.2** 在混合 dataset 上训练一个轻量视觉动作 world model
- [ ] **T6D.3** 用 world model rollout 生成"想象 episode"（注意只取物理上合理的）
- [ ] **T6D.4** 想象数据加入训练集，对比微调收益
- [ ] **T6D.5** 失败案例：world model 生成的"假数据"是否反而损害策略？

**关键参考**：
- NVIDIA GR00T-Dreams (2025)
- Dreamer V3 论文与代码：`https://github.com/danijar/dreamerv3`
- 1X World Model

**风险**：这是研究前沿，可能"想象数据"质量不够；建议先把方向 A/B 跑通再做

---

### 方向 E：硬件升级（WidowX / PiPER）

**目标**：当 SO101 精度成为瓶颈，把整套流水线迁移到更强的臂

**任务清单**：

- [ ] **T6E.1** 选定目标硬件（推荐 WidowX 250s 或 AgileX PiPER）
- [ ] **T6E.2** 在 mujoco_menagerie 找对应 MJCF（WidowX 已有，PiPER 可能要自己改 URDF→MJCF）
- [ ] **T6E.3** Phase 1–3 流水线 portage（环境改写、数据格式不变）
- [ ] **T6E.4** Phase 4 重新微调：Pi0.5 在 WidowX 上有更多公开 baseline 可比较
- [ ] **T6E.5** 部署新硬件评估

**关键参考**：
- WidowX 在 LeRobot 中的支持
- BridgeData V2 数据集（WidowX 大量真实数据）

---

### 方向 F：开源贡献

**目标**：把工程成果回馈社区

**任务清单**：

- [ ] **T6F.1** 整理 SO101 仿真环境为独立 pip 包
- [ ] **T6F.2** 写 LeRobot 教程：从零到部署 Pi0.5 on SO101
- [ ] **T6F.3** dataset 上 HuggingFace Hub（公开版）
- [ ] **T6F.4** 模型 checkpoint 上 HuggingFace Hub
- [ ] **T6F.5** 在 LeRobot Discord / GitHub 分享

---

## Phase 6 总体节奏建议

```
Month 1 (Phase 5 完成后):
  方向 A 或 B 选一个深入（让 VLA 真正"能用"）

Month 2-3:
  方向 C（real2sim）— 解决持续遇到的 sim2real 痛点
  + 方向 F 部分（开源准备）

Month 4+:
  方向 D（世界模型）或 方向 E（升级硬件）
  — 选研究热度高的那个
```

---

## 持续度量与里程碑

设定可量化里程碑，避免开环漫游：

- **M1**：真机 PickPlaceClutter（5 物体 + 颜色 OOD）成功率 > 50%
- **M2**：真机两步顺序指令 ("first X then Y") 成功率 > 40%
- **M3**：Real2Sim 重建后真机成功率比 v2 提升 ≥ 10 个百分点
- **M4**：dataset / 模型在 HF Hub 公开，月活下载 > 0

---

## 风险与陷阱

| 风险 | 应对 |
|------|------|
| 想做的太多，每个方向都浅尝辄止 | 强制每月只主推一个方向 |
| 世界模型方向研究不出结果 | 设 4 周时间盒，没拿到结果就回主线 |
| Real2Sim 工具链装环境麻烦 | 先用 SplatSim 公开 demo 跑通再用自己的扫描 |
| 升级硬件后所有数据都要重采 | 升级前确保有强复用基础设施，不一定要重新 from scratch |

---

## 输出物（依方向不同）

- 多任务 VLA 模型 v3 / v4 ...
- 真机评估视频集
- 论文 / blog（如做研究）
- 开源 repo + HF Hub 资产
