# Phase 7：扩展任务与高级指令（持续迭代）

**周期**：持续（按兴趣 / 课题选做）
**前置依赖**：Phase 6 完成（v3+ 模型经 HIL recovery/correction 后真机鲁棒可用）
**目标**：把单 cube + 单 plate 能力扩展到多物体、多颜色、多步骤、多容器、复杂场景，并探索 real2sim 与世界模型方向

> Phase 6 的 HIL 方法可在每个新任务上重复使用——每扩展一个能力，跑一轮 HIL 吃掉新失败模式。

---

## 复用模式

Phase 7 是开放性扩展，每个子方向都复用 Phase 1–6 的 CLI，只是改任务名 / 物体配置 / 指令池：

| 子方向 | 复用入口 | 需要新增的资产 |
|-------|---------|--------------|
| 新颜色 / 多干扰物 | `parallel_runner` + 新 MJCF | `assets/scenes/<task>.xml`、`data/instructions/<task>.txt`、`sim/envs/<task>.py` |
| 多容器选择 | 同上 + 新 evaluate_success | 同上 |
| 多步骤复合指令 | 新脚本策略 + 新 segmenter | `sim/scripted_policies/<task>.py`、扩展段数 |
| Real2Sim 数字孪生 | Phase 3 标定 + 自动生成 MJCF | `real2sim/` 工具链（未实现） |
| 世界模型 / RL | 独立训练栈，复用 `PickPlaceEnv` | `rl/` 训练目录（未实现） |

实现新任务时：(1) 复制 `sim/envs/pick_place.py`；(2) 复制场景与指令池；(3) 给 `parallel_runner` 加 `--env` 选项即可走完整 Phase 1–6 流水线。

---

## 子方向一览

按热度 + 工程化难度排序，建议选 1–2 个深入：

### 方向 A：多物体多颜色乱堆

3–5 个不同颜色/形状物体 + 多容器，按指令抓指定物。指令模板 `"put the {color} {shape} on the {target}"`，重点测**指令到物体的对应正确率**（颜色 anchor + 形状 anchor 两个维度）。参考 *DexMimicGen*、robosuite NutAssembly。

### 方向 B：多步骤复合指令 + 多容器选择

多容器（plate / bowl / tray）+ 两步顺序指令（`"first put X on plate, then put Y in bowl"`）。状态机扩展为 task-level FSM。**探索**：是否需要给 VLA 显式 sub-task token？参考 [LIBERO benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO)。

### 方向 C：Real2Sim 视觉对齐

手机扫真实工作台 → RialTo / SplatSim 重建 → 接入 MuJoCo 重跑 Phase 2/3。**预期收益**：sim2real gap 缩 20–40%。工具：[RialTo](https://github.com/real-to-sim-to-real/rialto)、[SplatSim](https://github.com/qureshinomaan/SplatSim)。

### 方向 D：世界模型增广（前沿）

调研 GR00T-Dreams / Dreamer V3 / 1X World Model，用 world model rollout 生成"想象 episode"扩增数据。**风险**：想象数据可能反损害策略；先把 A/B 跑通再做。**时间盒 4 周**，没结果就回主线。

### 方向 E：硬件升级（WidowX / PiPER）

SO101 精度成瓶颈时，迁移到更强的臂。Phase 1–3 流水线 portage，数据格式不变。Pi0.5 在 WidowX 上有更多公开 baseline 可比较，BridgeData V2 提供大量 WidowX 真实数据。

### 方向 F：开源贡献

整理 SO101 仿真环境为独立 pip 包；写 LeRobot 教程；dataset / checkpoint 上 HuggingFace Hub。

---

## 持续度量与里程碑

避免开环漫游，设可量化里程碑：

- **M1**：真机多物体场景（5 物体 + 颜色 OOD）成功率 > 50%
- **M2**：真机两步顺序指令成功率 > 40%
- **M3**：Real2Sim 重建后真机成功率比基线提升 ≥ 10 个百分点
- **M4**：dataset / 模型在 HF Hub 公开，月活下载 > 0

---

## 风险与陷阱

| 风险 | 应对 |
|------|------|
| 想做的太多，每个方向都浅尝辄止 | 强制每月只主推一个方向 |
| 世界模型方向研究不出结果 | 设 4 周时间盒，没拿到结果就回主线 |
| Real2Sim 工具链装环境麻烦 | 先用 SplatSim 公开 demo 跑通再用自己的扫描 |
| 升级硬件后所有数据都要重采 | 升级前确保有强复用基础设施 |
