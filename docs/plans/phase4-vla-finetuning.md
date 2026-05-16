# Phase 4：VLA 微调

**周期**：2–3 周
**前置依赖**：Phase 3 完成（混合数据集 v1 已就绪）
**目标**：在 Pi0.5（首选）或 SmolVLA（备选）上微调，sim 内 **PickPlaceBlue** 成功率 > 90%，并能响应多种语言指令变体；尤其要测出 **颜色锚定能力**（红/蓝位置互换时仍正确抓蓝）

> **核心任务**：抓蓝 cube 放进 plate（红 cube 是干扰物）。详细规约见 [README.md](README.md)。

---

## 代码入口（快速开始）

> **状态**：Phase 4 代码尚未实现。下表是规划好的入口，待开发；当前留作占位 / 设计目标。所有命令将在 `conda activate py312_cu121` 后执行。

| 想做的事 | 计划命令（占位） | 计划产出 |
|---------|-----------------|----------|
| 准备训练 dataset（从 Phase 3 mixed v1 派生 train/val split） | `python -m training.prepare_data --repo-id local/so101_pickplace_blue_mixed_v1 --out-dir runs/phase4_data` | train/val parquet 索引 |
| 微调 Pi0.5（首选） | `python -m training.finetune_pi0 --config configs/pi0_pickplace_blue.yaml` | `runs/phase4/pi0/checkpoints/` |
| 微调 SmolVLA（备选） | `python -m training.finetune_smolvla --config configs/smolvla_pickplace_blue.yaml` | `runs/phase4/smolvla/checkpoints/` |
| 仿真内评估（含颜色锚定测试） | `python -m eval.sim_eval --checkpoint runs/phase4/pi0/checkpoints/best --n-trials 100 --color-swap` | 成功率报告 + 失败模式分布 |

**计划实现文件**：`training/`、`configs/pi0_pickplace_blue.yaml`、`configs/smolvla_pickplace_blue.yaml`、`eval/sim_eval.py`。

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

**Pi0.5 vs SmolVLA 选型**：

| 维度 | Pi0.5 | SmolVLA |
|------|-------|---------|
| 参数量 | ~3B | ~450M |
| 显存（fp16 微调） | 单卡 24GB+ 紧张，48GB 舒服 | 单卡 12GB 够 |
| 真机表现 | 更强，泛化更好 | 已为 SO-ARM 优化 |
| 微调速度 | 慢 | 快（迭代成本低） |

**建议**：先用 SmolVLA 快速跑通整条流水线 → 切到 Pi0.5 做最终模型

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
  - 加载 `so101_pickplace_blue_mixed_v1`
  - 检查 features 完整：`observation.images.{wrist,front}`、`observation.state`、`action`、`task`
  - 切分：train 90% / val 10%（按 episode 切，不按 frame）
  - 真机 episode 训练集中独立标识，方便后续加权
- [ ] 把 sim/real 比例写入 dataset card

**关键文件**：
- `training/data/prepare_lerobot_for_vla.py`
- `data/lerobot/so101_pickplace_blue_mixed_v1_train_val/`

**验证**：用 LeRobot dataloader 能拉一个 batch（含图像 + 文本 + 动作）

---

### T4.3 SmolVLA 微调（baseline）

**目标**：先跑通完整 finetune 流程，拿到第一个可评估模型

**步骤**：
- [ ] 配置 `training/configs/smolvla_so101.yaml`：
  - base = `lerobot/smolvla_base`
  - dataset = `so101_pickplace_blue_mixed_v1`
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
- [ ] 写 `eval/sim_eval.py`：加载 checkpoint → 在 Phase 1 的 PickPlaceBlue env 跑 100 次随机评估
- [ ] **评估维度**：
  - 总成功率
  - 按红/蓝相对位置 4 象限拆分（每象限 25 条）
  - 按指令变体拆分成功率（10 种）
  - **颜色锚定专项**：固定 plate 位置，红/蓝位置互换，验证模型不会抓错颜色
  - 失败模式归类：`color_confusion` / `grasp_fail` / `transport_collision_red` / `place_miss` / `timeout`
- [ ] 输出 `eval/results/smolvla_so101_v1.md`

**关键文件**：
- `eval/sim_eval.py`
- `eval/results/smolvla_so101_v1.md`

**验证**：
- sim 内 PickPlaceBlue 总成功率 > 80%
- 颜色锚定准确率 > 95%（即"抓对颜色"的概率，独立于"放进 plate"成功率）

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

**目标**：验证 VLA 真的理解了语言（不只是记住了"看见蓝色就抓"）

**步骤**：
- [ ] **指令 OOD**：准备训练时没见过的同义表述，5 条（如 `"deposit the blue block into the dish"`），在 sim 里测成功率
- [ ] **反向指令测试**（关键）：把指令改成 `"put the red cube on the plate"`，观察模型行为
  - 若模型仍抓蓝 → 说明只学了视觉显著性，没真正理解语言
  - 若模型改抓红 → 说明真的理解了颜色锚定
  - 训练数据里没有"抓红"的样本，因此本测试是 OOD 行为评估
- [ ] **干扰物 OOD**：把红 cube 换成绿/黄/橙 cube，看蓝抓取是否仍稳定
- [ ] **容器 OOD**：把 plate 换成不同形状/颜色的盘子

**关键文件**：
- `eval/language_ood_eval.py`
- `eval/results/language_generalization.md`

**验证**：
- 指令 OOD 成功率 ≥ in-distribution 的 70%
- 反向指令测试时模型改变行为（即使成功率低也算成功，证明真的依赖语言）
- 干扰物颜色 OOD 不显著降低蓝抓取成功率（衰减 < 15%）

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
- [ ] 颜色锚定准确率 > 95%
- [ ] 语言 OOD 成功率 ≥ 70%
- [ ] 反向指令测试通过（模型行为随语言改变）
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
