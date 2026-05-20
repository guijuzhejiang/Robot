# Phase 4：VLA 微调

**周期**：2–3 周
**前置依赖**：Phase 3 完成（混合数据集 v1 已就绪）
**目标**：在 **SmolVLA（首选）** 上微调，sim 内 PickPlaceRed 成功率 > 90%，能响应多种语言指令变体；Pi-0.5 作为对比实验

> **备用方案**：[alt-platform-groot-n15.md](alt-platform-groot-n15.md)（GR00T N1.5 + SO-101）

---

## VLA 选型（基于预训练数据 + action schema 匹配度）

我们 `action` 列 = **6 维 joint position**（LeRobot SO-100/101 惯例），与 SmolVLA pretrain 数据集 `lerobot/svla_so100_pickplace` 逐位对齐。

| 模型 | 预训练形态 | action 表示 | SO-101 适配 | 微调策略 |
|---|---|---|---|---|
| **SmolVLA** | 100% SO-100 community | 6-D joint | ⭐⭐⭐⭐⭐ 完全对齐 | **首选**，直接训 |
| Pi-0 / Pi-0.5 | PI 内部混合（UR5、Franka、Trossen 等） | 32-D zero-padded（运行时 pad，非数据列）| ⭐⭐ | 6-D joint 直接喂 openpi loader，自动 pad；action head 部分迁移 |
| OpenVLA | Open-X | 7-D discretized ee-delta | ⭐⭐ | 需 Stage-2 派生 7-D ee dataset |
| X-VLA | DROID / Robomind | 10-D ee-pose（xyz + Rotate6D + gripper） | ⭐⭐ | 需 Stage-2 派生 10-D dataset |
| ACT / Diffusion Policy | 无预训练 | 任意 | ⭐ | 从零训 |

**关键事实**：所有非-SO-1xx VLA 都在不同 robot ecosystem 上预训练。SmolVLA 是唯一同形态预训练，其他都属"借用 VLM backbone + 重学 action head"——红利打折。

---

## 数据流水线分支

### ✅ 直接复用 Stage-1 数据（无需转换）

SmolVLA、Pi-0/Pi-0.5、ACT、Diffusion Policy → 直接 `--repo-id local/so101_pickplace_v1`。

### 🔧 需先跑 Stage-2 派生

OpenVLA / X-VLA / RT-X 系：模型预训练 action 语义 = 末端笛卡尔位姿/旋转，必须派生。

```bash
python -m data.converters.derive_action_format \
    --source-repo local/so101_pickplace_v1 \
    --output-repo local/so101_pickplace_<format>_v1 \
    --target-format <name>
```

`--target-format` 候选：

| 格式 | shape | 推荐适用 |
|---|---|---|
| `ee_delta_4d` | (4,) | 位置 delta，等价 ee_action 列 |
| `ee_delta_7d_euler` | (7,) | **OpenVLA** / RT-X |
| `ee_pose_7d_euler` | (7,) | LIBERO 风格 |
| `ee_pose_7d_axis_angle` | (7,) | `robocerebra_unified` 风格 |
| `ee_pose_10d_rotate6d` | (10,) | **X-VLA** |

工作原理：加载 source → 对每帧 6 维关节角走 MuJoCo FK → 拿 `gripperframe` site 的 xpos+xmat → 转换 → 调 LeRobot `_copy_data_with_feature_changes` 重写 parquet + `_copy_videos` 不重编码 + `recompute_stats`。

**已知**：gripper 维度沿用 source `action[5]`（joint qpos，未归一化）。若 OpenVLA/X-VLA 训练 stats 异常，加 `--gripper-mode {qpos|binary|normalized}` 选项即可。

---

## 代码入口（快速开始）

> **状态**：Phase 4 代码尚未实现，下表为规划入口。

| 想做的事 | 计划命令 |
|---------|---------|
| 准备 train/val split | `python -m training.prepare_data --repo-id local/so101_pickplace_mixed_v1 --out-dir runs/phase4_data` |
| 微调 SmolVLA | `python -m lerobot.scripts.lerobot_train --policy.type=smolvla --dataset.repo_id=local/so101_pickplace_mixed_v1` |
| 微调 Pi-0.5（对比） | `python -m training.finetune_pi0 --config configs/pi0_pickplace.yaml` |
| 派生 ee dataset | `python -m data.converters.derive_action_format --target-format ee_delta_7d_euler ...` |
| 仿真评估 | `python -m eval.sim_eval --checkpoint runs/phase4/smolvla/checkpoints/best --n-trials 100` |

实现时回写本表并登记到 [docs/implementation-status.md](../implementation-status.md)。

---

## 任务清单

### T4.1 微调环境搭建

- [ ] `git clone https://github.com/Physical-Intelligence/openpi training/openpi`
- [ ] `huggingface-cli download lerobot/smolvla_base` + `lerobot/pi05_base`
- [ ] 两个模型各跑一次 dummy forward

### T4.2 数据集预处理与切分

写 `training/data/prepare_lerobot_for_vla.py`：加载 `so101_pickplace_mixed_v1` → 按 episode 切 train 90%/val 10% → 真机 episode 独立标识便于加权。

### T4.3 SmolVLA 微调（baseline）

Config `training/configs/smolvla_so101.yaml`：base = `lerobot/smolvla_base`、lr=1e-5、batch=8（单卡 24GB）、epochs=5–10、真机 episode 采样权重 5×。`lerobot-train --config ...`。

**验证**：train/val loss 稳降；单卡 24GB 约 12–24 小时

### T4.4 Sim-only 评估

`eval/sim_eval.py`：加载 checkpoint → Phase 1 env 跑 100 次随机评估。维度：总成功率、cube 4 象限拆分、cube yaw 4 等分拆分、10 种指令变体、6 类失败模式。

**验证**：sim 总成功率 > 80%，各 yaw/象限均衡

### T4.5 SimplerEnv 集成（可选但推荐）

`git clone simpler-env/SimplerEnv` → 写 SO101 adapter（包装成 SimplerEnv 风格 + 视觉风格匹配 colorjitter）→ 100 次评估。

**验证**：与 sim_eval 差距在 ±15% 内

### T4.6 Pi-0.5 微调（对比）

Config base = `lerobot/pi05_base`，显存不够时 LoRA rank=64 + gradient checkpointing；lr=5e-6；batch=4/grad_accum=8；epochs=3–5。

**验证**：sim_eval > SmolVLA baseline

### T4.7 Co-training 实验（关键）

3 个对比实验：
- **A**：纯 sim（5K mimicgen + 5K scripted）
- **B**：sim + 100 real（real 权重 1×）
- **C**：sim + 100 real（real 权重 5×）← 主推荐

同样 epoch + seed，全部用 sim_eval + SimplerEnv 评估。形成 `eval/results/cotraining_ablation.md` 决定 Phase 5 用哪个权重。

### T4.8 语言泛化测试

- **指令 OOD**：训练时没见过的同义表述 5 条（如 `"deposit the red block into the dish"`）
- **颜色 OOD**：临时改 cube 颜色为绿/黄（修改 MJCF rgba），看是否仍抓得到（验证是否过拟合颜色）
- **容器 OOD**：换不同形状/颜色的盘子

**验证**：指令 OOD ≥ in-distribution 的 70%；颜色 OOD 衰减 < 30%

### T4.9 选定 Phase 5 部署模型

综合 sim_eval + SimplerEnv + language OOD 三个维度排序 → 选 winner → 推 HF Hub `<your_id>/so101_vla_v1`。

---

## 验收标准

- [ ] SmolVLA + Pi-0.5 微调完成并评估
- [ ] Co-training 消融实验完整（A/B/C）
- [ ] sim_eval winner 成功率 > 90%
- [ ] 各 yaw/象限均衡
- [ ] 语言 OOD ≥ 70%
- [ ] winner 推送到 HF Hub

---

## 风险与陷阱

| 风险 | 应对 |
|------|------|
| 单卡显存放不下 Pi0.5 | 启用 LoRA + gradient checkpointing；或退到 SmolVLA |
| train loss 平稳但 sim_eval 不涨 | 检查图像归一化、动作 normalize 是否与 base 一致；ee_pose 顺序（xyz_quat vs quat_xyz） |
| 真机数据过拟合 | 加 sim 占比；early stopping 看 val loss |
| sim_eval 95% 但 SimplerEnv 30% | 视觉 DR 还不够；augment 光照、色温抖动 |
| Pi-0.5 finetune 训不动（loss 平坦） | 检查 base 是否正确加载；lr 太小；action head 是否被 freeze |
| 评估慢拖累迭代 | sim_eval 用 16-worker 并行 |
