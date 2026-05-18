# Phase 4：VLA 微调

**周期**：2–3 周
**前置依赖**：Phase 3 完成（混合数据集 v1 已就绪）
**目标**：在 **SmolVLA（首选）** 上微调，sim 内 **PickPlaceRed** 成功率 > 90%，并能响应多种语言指令变体；Pi-0.5 / Pi-0 作为对比实验

> **核心任务**：把红 cube 放进 plate。详细规约见 [README.md](README.md)。
> **命名约定**：训练 config 统一使用颜色无关命名，例如 `configs/smolvla_pickplace.yaml`；任务概念仍是 PickPlaceRed。

---

## VLA 选型（基于预训练数据 + action schema 匹配度）

详见 `docs/implementation-status.md` 「数据集 action 列设计」。一句话总结：**我们 `action` 列 = 6 维 joint position（LeRobot SO-100/101 惯例），与 SmolVLA pretrain 数据集 `lerobot/svla_so100_pickplace` 逐位对齐**。

| 模型 | 预训练形态 | action 表示 | SO-101 适配度 | 微调策略 |
|---|---|---|---|---|
| **SmolVLA** | 100% SO-100 community | 6-D joint | ⭐⭐⭐⭐⭐ 完全对齐 | **首选**，直接训 |
| Pi-0 / Pi-0.5 / Pi-0-FAST | PI 内部混合（UR5、Franka、双臂 Trossen 等）| 32-D zero-padded（运行时 pad，非数据列）| ⭐⭐ | 6-D joint 直接喂 openpi loader，自动 pad 到 32；action head 部分迁移 |
| OpenVLA | Open-X（WidowX、Franka、Google Robot 等）| 7-D discretized ee-delta（含旋转）| ⭐⭐ | 需 Stage-2 派生 7-D ee dataset |
| X-VLA | DROID / Robomind / Agibot | 10-D ee-pose（xyz + Rotate6D + gripper）| ⭐⭐ | 需 Stage-2 派生 10-D ee-pose dataset |
| ACT | Aloha（无预训练）| 关节角，14-D 双臂 | ⭐ 无预训练红利 | 从零训，速度快 |
| Diffusion Policy | 无 | 任意 | ⭐ 无预训练红利 | 从零训 |

**预训练数据来源关键事实**：所有非-SO-1xx VLA 都在不同 robot ecosystem 上预训练。SmolVLA 是唯一同形态预训练，其他都属"借用 VLM backbone + 重学 action head"——红利大幅打折。

---

## 数据流水线分支（关键概念）

我们 Phase-2 生成的 LeRobot dataset `action` 列 = **6-D joint position**（`shoulder_pan.pos ... gripper.pos`）。下游 VLA 模型按是否能直接消费这一格式分两类：

### ✅ 直接复用 Stage-1 数据（无需转换）

| 模型 | 为什么能直接用 |
|------|---------------|
| **SmolVLA** | 预训练就是 SO-100 6-D joint，逐位对齐 |
| **Pi-0 / Pi-0.5 / Pi-0-FAST** | openpi loader 运行时把 6-D zero-pad 到 32-D，**dataset 不变** |
| **ACT** | action 维度可配置，从零训不挑格式 |
| **Diffusion Policy** | action 维度可配置，从零训不挑格式 |

→ 微调命令直接指 `--repo-id local/so101_pickplace_v1`，无中间步骤。

### 🔧 需先跑 Stage-2 派生（数据格式不兼容）

| 模型 | 需要的 action 格式 | 派生命令 |
|------|---------------------|----------|
| **OpenVLA** | 7-D ee-delta（xyz + rpy + gripper） | `--target-format ee_delta_7d_euler` |
| **X-VLA** | 10-D ee-pose（xyz + Rotate6D + gripper） | `--target-format ee_pose_10d_rotate6d` |
| **RT-X 系**（如 RT-2-X） | 7-D ee-delta，离散化 | `--target-format ee_delta_7d_euler` |

→ 微调前先跑 `python -m data.converters.derive_action_format ...` 生成派生 dataset，再用派生 repo-id 训练。

**判断准则**：模型预训练 action 与 6-D joint **维度+语义都对齐** → 直接用；**语义不同**（要末端笛卡尔位姿/旋转）→ 必须派生。Pi 的 padding 是维度补齐而非语义改变，所以仍属直接用。

---

## Stage-2 派生工具（已实现）

**适用场景**：上表中"需先跑 Stage-2 派生"那一组模型。SmolVLA / Pi 系列 / ACT / Diffusion Policy **不需要**这一步——直接用 source dataset。

**实现**：[`data/converters/derive_action_format.py`](../../data/converters/derive_action_format.py)

**工作原理**：
1. 加载 source dataset（必须是 Phase-2 joint schema，`action` shape (6,)）
2. 对每帧的 `observation.state`（6 维关节角）走 MuJoCo FK → 拿 `gripperframe` site 的 `xpos` (3,) + `xmat` (3,3)
3. 按 `--target-format` 转换：ee_pose 类直接用 `xpos+xmat`；ee_delta 类算 `pose[t+1] - pose[t]`（episode 边界处零 delta）
4. 调 LeRobot 的 `_copy_data_with_feature_changes` 重写 parquet（callable 路径支持多维特征），`_copy_videos` **复制 mp4 不重编码**（shutil.copy），`recompute_stats` 自动重算新 action 列的 mean/std
5. 输出新 dataset，`action` 列已替换、`ee_action` 列已删除，其余原样保留

**CLI**：

```bash
python -m data.converters.derive_action_format \
    --source-repo local/so101_pickplace_v1 \
    --output-repo local/so101_pickplace_<format>_v1 \
    --target-format <name> \
    [--scene-path assets/scenes/pick_place.xml] \
    [--ee-site gripperframe] \
    [--reset]
```

**`--target-format` 候选（已全部实现）**：

| 格式 | shape | action names | 推荐适用 |
|---|---|---|---|
| `ee_delta_4d` | (4,) | `[dx, dy, dz, gripper]` | 位置-only delta，等价于把现有 `ee_action` 列取出来（但 gripper 用 joint qpos 而非归一化）|
| `ee_delta_7d_euler` | (7,) | `[dx, dy, dz, droll, dpitch, dyaw, gripper]` | **OpenVLA** / RT-X 风格 |
| `ee_pose_7d_euler` | (7,) | `[x, y, z, roll, pitch, yaw, gripper]` | LIBERO 风格 |
| `ee_pose_7d_axis_angle` | (7,) | `[x, y, z, ax, ay, az, gripper]` | `robocerebra_unified` 风格 |
| `ee_pose_10d_rotate6d` | (10,) | `[x, y, z, r11, r21, r31, r12, r22, r32, gripper]` | **X-VLA** 风格 |

**Smoke 验证**（已通过）：
- 2 ep / 392 帧 source 跑全 5 格式 → 全部产出 ✓
- ee_pose TCP xyz ∈ tabletop 工作空间 ✓
- ee_pose_10d_rotate6d 的 r31 ≈ -1（gripper z 轴指世界 -z，对应顶视顶抓）✓
- video mp4 复制不重编码（耗时 < 1s）✓
- stats.json 自动重算新 action 分布 ✓

**已知细节 / 后续可调**：
- 所有格式的 `gripper` 维度沿用 source `action[5]`（joint qpos，范围约 0.02-1.17），**未归一化**。OpenVLA / X-VLA 训练惯例是二值（开/合）或归一化 [-1, 1]——如果训练时报 stats 异常，加个 `--gripper-mode {qpos|binary|normalized}` 选项即可。
- 多 chunk 数据集（>1 GB / 数千 episode）未经充分压测，但 LeRobot 的 `_copy_data_with_feature_changes` 是 per-parquet-file 流式处理，理论上不受规模限制。Phase 2 v1 dataset（896 ep / ~180K 帧）应能直接跑通。

---

## 代码入口（快速开始）

> **状态**：Phase 4 代码尚未实现。下表是规划好的入口，待开发；当前留作占位 / 设计目标。所有命令将在 `conda activate py312_cu121` 后执行。

| 想做的事 | 计划命令（占位） | 计划产出 |
|---------|-----------------|----------|
| 准备训练 dataset（从 Phase 3 mixed v1 派生 train/val split） | `python -m training.prepare_data --repo-id local/so101_pickplace_mixed_v1 --out-dir runs/phase4_data` | train/val parquet 索引 |
| 微调 SmolVLA（首选）| `python -m lerobot.scripts.lerobot_train --policy.type=smolvla --dataset.repo_id=local/so101_pickplace_mixed_v1` | `runs/phase4/smolvla/checkpoints/` |
| 微调 Pi0.5（对比）| `python -m training.finetune_pi0 --config configs/pi0_pickplace.yaml` | `runs/phase4/pi0/checkpoints/` |
| 派生 ee dataset（如需对比 OpenVLA / X-VLA）| `python -m data.converters.derive_action_format --source-repo local/so101_pickplace_v1 --target-format ee_delta_7d_euler --output-repo local/so101_pickplace_ee7d_v1` | 新 dataset，`action` 列 = 7-D ee-delta |
| 仿真内评估 | `python -m eval.sim_eval --checkpoint runs/phase4/smolvla/checkpoints/best --n-trials 100` | 成功率报告 + 失败模式分布 |

**计划实现文件**：`training/`、`configs/smolvla_pickplace.yaml`、`configs/pi0_pickplace.yaml`、`eval/sim_eval.py`、`data/converters/derive_action_format.py`（Stage-2）。

> 实现时请回写本表，把"计划命令"换成实际命令，并在 [docs/implementation-status.md](../implementation-status.md) 中登记。

---

## 关键技术与工具

| 工具 | 用途 | 备注 |
|------|------|------|
| openpi | Pi0/Pi0.5 官方代码 + 权重 | `https://github.com/Physical-Intelligence/openpi` |
| LeRobot | SmolVLA 训练 + 数据加载 | 主仓库自带 SmolVLA finetune scripts |
| HuggingFace Accelerate | 多卡训练协调 | `pip install accelerate` |
| HuggingFace transformers | 模型加载 | `pip install transformers` |
| PEFT / LoRA | 参数高效微调（备选） | `pip install peft` |
| Weights & Biases | 训练日志 / 实验对比 | Phase 0 已配置 |
| SimplerEnv | sim 内"近真机"评估 | `https://github.com/simpler-env/SimplerEnv` |

**SmolVLA vs Pi-0.5 选型**（已切换首选为 SmolVLA）：

| 维度 | SmolVLA（首选）| Pi-0.5（对比）|
|------|-----------------|-----------------|
| 参数量 | ~450M | ~3B |
| 显存（fp16 微调） | 单卡 12GB 够 | 单卡 24GB 紧张，48GB 舒服 |
| 微调速度 | 快（迭代成本低）| 慢 |
| **预训练 action schema 匹配** | ⭐⭐⭐⭐⭐ 100% SO-100 同形态 joint，与我们 `action` 列逐位对齐 | ⭐⭐ 32-D zero-padded 混合形态（运行时 pad，前 6 维有效）；action head 部分迁移 |
| **预训练数据来源**（参考）| LeRobot Community Dataset 481 datasets / 22.9K episodes 全 SO-100 | PI 内部 ~10000h + Open-X + DROID，0% SO-1xx |
| 真机泛化 | 已为 SO-ARM 优化，sim2real 数据天然对齐 | 大模型容量更高，但 SO-1xx 迁移有 gap |

**结论**：先用 SmolVLA 跑通主路径（零配置训练，schema 自然对齐），稳定后再上 Pi-0.5 做对比实验看是否更大容量带来收益。这与我们 dataset action 列设计直接吻合。

---

## 任务清单

### T4.1 微调环境搭建

**目标**：openpi + LeRobot 训练栈可用

**步骤**：
- [ ] `git clone https://github.com/Physical-Intelligence/openpi training/openpi`
- [ ] 跟随 openpi README 安装依赖（注意 JAX/PyTorch 选择）
- [ ] 拉取 Pi0.5 base checkpoint：`huggingface-cli download lerobot/pi05_base`（或官方指定路径）
- [ ] 拉取 SmolVLA：`huggingface-cli download lerobot/smolvla_base`
- [ ] 验证两者都能加载并做一次 dummy forward

**关键文件**：
- `training/openpi/`：openpi 仓库
- `training/configs/pi05_so101.yaml`：将创建的微调 config

**参考**：
- openpi 微调指南：`https://github.com/Physical-Intelligence/openpi#fine-tuning`
- SmolVLA finetune 例子：`https://huggingface.co/docs/lerobot/smolvla`

**验证**：两个模型在 GPU 上能各跑一次 forward 不 OOM

---

### T4.2 数据集预处理与切分

**目标**：把 Phase 3 dataset 转换为 VLA 训练可用格式

**步骤**：
- [ ] 写 `training/data/prepare_lerobot_for_vla.py`：
  - 加载 `so101_pickplace_mixed_v1`
  - 检查 features 完整：`observation.images.{wrist,front}`、`observation.state`、`action`、`task`
  - 切分：train 90% / val 10%（按 episode 切，不按 frame）
  - 真机 episode 训练集中独立标识，方便后续加权
- [ ] 把 sim/real 比例写入 dataset card

**关键文件**：
- `training/data/prepare_lerobot_for_vla.py`
- `data/lerobot/so101_pickplace_mixed_v1_train_val/`

**验证**：用 LeRobot dataloader 能拉一个 batch（含图像 + 文本 + 动作）

---

### T4.3 SmolVLA 微调（baseline）

**目标**：先跑通完整 finetune 流程，拿到第一个可评估模型

**步骤**：
- [ ] 配置 `training/configs/smolvla_so101.yaml`：
  - base = `lerobot/smolvla_base`
  - dataset = `so101_pickplace_mixed_v1`
  - lr = 1e-5（VLA 微调常见值）
  - batch_size = 8（单卡 24GB）
  - epochs = 5–10
  - 真机 episode sampling weight = 5×
- [ ] 启动：`lerobot-train --config training/configs/smolvla_so101.yaml`
- [ ] wandb 监控 loss、grad norm

**关键文件**：
- `training/configs/smolvla_so101.yaml`
- `training/checkpoints/smolvla_so101_v1/`

**参考**：
- LeRobot training script：`https://github.com/huggingface/lerobot/tree/main/src/lerobot/scripts/lerobot_train.py`

**验证**：
- 训练 loss 平稳下降
- val loss 不发散
- 训练完成（5 epoch）：单卡 24GB 约 12–24 小时

---

### T4.4 Sim-only 评估

**目标**：在仿真里量化 SmolVLA finetuned 的能力

**步骤**：
- [ ] 写 `eval/sim_eval.py`：加载 checkpoint → 在 Phase 1 的 PickPlaceRed env 跑 100 次随机评估
- [ ] **评估维度**：
  - 总成功率
  - 按 cube 位置 4 象限（左/右/前/后）拆分（每象限 25 条）
  - 按 cube yaw 4 等分拆分（每种 25 条）
  - 按指令变体拆分成功率（10 种）
  - 失败模式归类：`grasp_fail` / `lift_drop` / `place_miss` / `plate_off` / `joint_limit` / `timeout`
- [ ] 输出 `eval/results/smolvla_so101_v1.md`

**关键文件**：
- `eval/sim_eval.py`
- `eval/results/smolvla_so101_v1.md`

**验证**：
- sim 内 PickPlaceRed 总成功率 > 80%
- 各 yaw / 象限均衡（无明显偏置）

---

### T4.5 SimplerEnv 集成（可选但推荐）

**目标**：用更接近真机分布的 sim 评估，sim2real 预测更准

**步骤**：
- [ ] `git clone https://github.com/simpler-env/SimplerEnv eval/SimplerEnv`
- [ ] SimplerEnv 默认是 WidowX/Google Robot，需要写 SO101 adapter：
  - 把我们的 PickCube env 包装成 SimplerEnv 风格 protocol
  - 加视觉风格匹配（colorjitter 模拟真机相机）
- [ ] 跑 100 次评估

**关键文件**：
- `eval/SimplerEnv/`（外部）
- `eval/simpler_so101_adapter.py`

**参考**：
- SimplerEnv 论文 *Evaluating Real-World Robot Manipulation Policies in Simulation*

**验证**：SimplerEnv 评估结果与 sim_eval 结果差距在 ±15% 内（如果差距很大说明哪边有 bug）

---

### T4.6 Pi0.5 微调（主目标）

**目标**：在最强 base 上微调

**步骤**：
- [ ] 配置 `training/configs/pi05_so101.yaml`：
  - base = `lerobot/pi05_base` (或 openpi 官方权重路径)
  - 显存不够时启用 LoRA（rank=64）或 gradient checkpointing
  - lr = 5e-6（Pi0.5 微调更小 lr）
  - batch_size = 4，gradient_accumulation = 8（等效 batch 32）
  - epochs = 3–5
- [ ] 启动：参考 openpi 训练 entry point
- [ ] wandb 对比 SmolVLA baseline

**关键文件**：
- `training/configs/pi05_so101.yaml`
- `training/checkpoints/pi05_so101_v1/`

**参考**：
- openpi finetune 文档
- Pi0.5 论文（Physical Intelligence 2025）

**验证**：
- 训练完成（约 24–48 小时单卡 4090；多卡更快）
- val loss 收敛
- sim_eval 成功率 > SmolVLA baseline

---

### T4.7 Co-training 实验（关键）

**目标**：明确"真机 + sim 混合"vs"纯 sim"的差异

**步骤**：
- [ ] 跑 3 个实验：
  - **A**：纯 sim（5K mimicgen + 5K scripted）
  - **B**：sim + 100 real（real 权重 1×）
  - **C**：sim + 100 real（real 权重 5×）— 主推荐
- [ ] 每个实验同样 epoch、同样种子
- [ ] 全部用 sim_eval + SimplerEnv 评估
- [ ] 形成 `eval/results/cotraining_ablation.md` 报告

**关键文件**：
- `training/configs/pi05_so101_{A,B,C}.yaml`
- `eval/results/cotraining_ablation.md`

**验证**：得到清晰的 co-training 收益数字，决定 Phase 5 用哪个权重

---

### T4.8 语言泛化测试

**目标**：验证 VLA 真的理解了语言（不只是记住了"看见 cube 就抓"）

**步骤**：
- [ ] **指令 OOD**：准备训练时没见过的同义表述，5 条（如 `"deposit the red block into the dish"`），在 sim 里测成功率
- [ ] **颜色 OOD（可选）**：临时把 cube 颜色改成绿/黄/橙（修改 MJCF material rgba），看模型是否仍抓得到。
  - 训练数据全是红 cube，所以本测试是 OOD 视觉评估，验证模型是否过拟合到颜色
- [ ] **容器 OOD**：把 plate 换成不同形状/颜色的盘子

**关键文件**：
- `eval/language_ood_eval.py`
- `eval/results/language_generalization.md`

**验证**：
- 指令 OOD 成功率 ≥ in-distribution 的 70%
- 颜色 OOD 不显著降低抓取成功率（衰减 < 30%，宽容度可以放高，因为视觉差异较大）

---

### T4.9 选定 Phase 5 部署模型

**目标**：从所有实验里选一个进入真机

**步骤**：
- [ ] 综合 sim_eval、SimplerEnv、language OOD 三个维度排序
- [ ] 选一个 winner，记录决策依据到 `eval/results/phase4_summary.md`
- [ ] 把 winner 推 HF Hub：`<your_id>/so101_vla_v1`

**关键文件**：
- `eval/results/phase4_summary.md`
- HF Hub: `<your_id>/so101_vla_v1`

**验证**：决策有量化依据，winner 在 HF Hub 可访问

---

## 验收标准（全部满足后进入 Phase 5）

- [ ] SmolVLA baseline 已训练并评估
- [ ] Pi0.5 微调完成并评估
- [ ] Co-training 消融实验完整（A/B/C）
- [ ] sim_eval 成功率 > 90%（winner）
- [ ] 各 cube yaw / 工作区象限均衡成功
- [ ] 语言 OOD 成功率 ≥ 70%
- [ ] 推送 winner 到 HF Hub

---

## 风险与陷阱

| 风险 | 应对 |
|------|------|
| 单卡显存放不下 Pi0.5 微调 | 启用 LoRA + gradient checkpointing；或退到 SmolVLA |
| 训练 loss 平稳但 sim_eval 不涨 | 检查图像归一化、动作 normalize 是否与 base 模型一致；ee_pose 顺序（xyz_quat vs quat_xyz） |
| 真机数据过拟合 | 加 sim 占比；early stopping 看 val loss |
| sim_eval 95% 但 SimplerEnv 30% | 视觉 DR 还不够；augment 图像（光照、色温抖动） |
| Pi0.5 finetune 训不动（loss 平坦） | 检查 base checkpoint 是否正确加载；学习率太小；动作头是否被 freeze |
| 评估慢拖累迭代 | sim_eval 用 16-worker 并行 |

---

## 输出物

- SmolVLA + Pi0.5 微调 checkpoint
- 完整的 co-training 消融报告
- 语言泛化评估
- 选定的 v1 部署模型（推 HF Hub）
