# Phase 6：HIL recovery 数据采集与回炉微调

**周期**：2–3 周
**前置依赖**：Phase 5 完成（v2 模型已部署真机，失败模式已归类）
**目标**：用 LeRobot 官方 HIL（Human-In-the-Loop）pipeline，针对 Phase 5 暴露的失败模式专门采集 **recovery + correction** 数据，与原 dataset 合并做 co-finetune，把真机 PickPlaceRed 成功率从 60% 推到 80%+

> **核心任务**：把红 cube 放进 plate。详细规约见 [README.md](README.md)。

---

## 代码入口（快速开始）

> **状态**：Phase 6 代码尚未实现，依赖 LeRobot 上游 HIL pipeline + 真机 + 脚踏开关。下表为规划入口。

| 想做的事 | 计划命令（占位） | 计划产出 |
|---------|-----------------|----------|
| 启动 HIL 数据采集（VLA 主导 + 人工 takeover via foot pedal） | `python -m hil.collect_recovery --policy runs/phase5/pi0_v2/checkpoints/best --foot-pedal /dev/input/event3 --output local/so101_hil_recovery_v1` | 真机 dataset，含 `is_takeover` flag |
| 标注 recovery 段（自动 / 半自动） | `python -m hil.label_takeovers --repo-id local/so101_hil_recovery_v1` | 带 `is_recovery` 标签的 dataset |
| Co-finetune v3 模型 | `python -m training.cofinetune_pi0 --base runs/phase5/pi0_v2/checkpoints/best --add local/so101_hil_recovery_v1 --weight-recovery 2.0` | `runs/phase6/pi0_v3/checkpoints/` |
| 真机验收（≥80% 成功率） | `python -m deploy.real_eval --checkpoint runs/phase6/pi0_v3/checkpoints/best --n-trials 100 --robot so101 --report runs/phase6/final_report.md` | 最终验收报告 |

**计划实现文件**：`hil/collect_recovery.py`、`hil/label_takeovers.py`、`hil/foot_pedal.py`。

---

## 为什么必须做 HIL？

**标准行为克隆只学"成功轨迹"**：模型只见过 demo 里"一切正常"的状态分布。真机部署时一旦出现小偏差，状态会漂到训练时从未见过的分布外区域（**compounding error / distribution shift**），错误不断放大。

**HIL 直接解决 distribution shift**：
1. 让 v2 模型在真机上跑
2. 人**只在快要失败时介入**（不是每步都标）
3. 人接管 → **recovery**（把机器人摇回 in-distribution 状态）→ **correction**（演示正确行为）
4. 同一条 episode 里 policy ↔ human 可多次交接，全程录制
5. 用合并 dataset 微调 → v3 → 再跑 HIL → v4 → ...

**与 Phase 3 真机 demo 采集的根本区别**：
- Phase 3：人**全程**遥操，采集"理想成功轨迹"
- Phase 6：模型先跑，人**只在失败临界点介入**——recovery 段的起点正好覆盖了 v2 模型实际会漂到的状态分布

这是 **DAgger / HG-DAgger / RaC** 思想在 SO101 上的落地，也是 Pi0.6（RECAP）大幅超越 Pi0.5 的核心方法。

---

## 关键技术与工具

| 工具 | 用途 | 备注 |
|------|------|------|
| LeRobot HIL script | `examples/hil/hil_data_collection.py` | LeRobot 仓库自带 |
| `so_leader` / `so_follower` 配置 | SO-ARM101 主从臂 HIL 兼容 | LeRobot 已注册 |
| USB foot pedal（强烈推荐） | 释放双手做 pause / takeover / return 切换 | PCsensor FootSwitch ~¥80 |
| RTC（Real-Time Chunking） | Pi0.5 / SmolVLA 推理时降低延迟卡顿 | `--rtc.enabled=true` |
| `interpolation_multiplier` | 介入接管时关节插值平滑 | 默认 2–3 即可 |
| LeRobot dataset merge | 多份 HIL dataset 合并 | LeRobot 原生支持 |
| Weights & Biases | 多轮迭代版本对比 | Phase 0 已配置 |

**官方文档**：[`https://huggingface.co/docs/lerobot/hil_data_collection`](https://huggingface.co/docs/lerobot/hil_data_collection)

---

## HIL 交互协议（每条 episode 内的人机协作流程）

```
START
  │
  ▼
[Policy 自主跑] ──────── 看着，不动 ─────────┐
  │                                          │
  │ 觉察失败临界（颜色看错 / 撞红 / 偏离 plate）│
  ▼                                          │
[踩 PAUSE 踏板]                              │
  │  - Policy 停止                            │
  │  - Leader 臂 torque on，移动到 Follower    │
  │    当前位姿（自动镜像）                     │
  │  - 此时不录任何 frame                      │
  ▼                                          │
[踩 TAKEOVER 踏板]                           │
  │  - Leader 臂 torque off，自由可动          │
  │  - 你用 Leader 遥操：                       │
  │    1) Recovery：摇回 in-distribution 状态  │
  │    2) Correction：演示正确动作              │
  │  - 全程录制为 intervention 段              │
  ▼                                          │
[踩 RETURN-TO-POLICY 踏板]                   │
  │  - Policy 从当前状态继续接管               │
  │  └──────────── 同一 episode 内循环 ────────┘
  │
  ▼
[任务完成或超时] → 保存 episode → 下一条
```

**关键点**：
- 一条 episode 里**可多次交接**，不需要重置
- Recovery 段的起点正好是 v2 模型"会漂到的失败前状态"——这是普通 demo 永远拿不到的数据
- Correction 不要过度——只演示完成当前 subtask，别加多余动作

---

## 任务清单

### T6.1 HIL 环境与硬件搭建

**目标**：本地 HIL pipeline 可启动并接收踏板信号

**步骤**：
- [ ] 升级 LeRobot 到含 `examples/hil/` 的版本（≥ 当前 main）
- [ ] 验证 LeRobot 已注册 `so_follower` + `so_leader` HIL 配置：
  ```bash
  python -c "from lerobot.teleoperators import so_leader; print('so_leader OK')"
  ```
- [ ] 接入 USB 脚踏开关（推荐 PCsensor 3 键款）：
  ```bash
  sudo setfacl -m u:$USER:rw /dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd
  # 验证：evtest 看是否能读到按键事件
  ```
- [ ] 用键盘 fallback 跑一遍 dry-run（不接 SO101），确认 pause/takeover/return 三个命令位映射正确
- [ ] 录制脚本启动时打印的实际按键/踏板绑定到 `configs/hil_controls.md`（每台机器可能不同，以脚本输出为准）

**关键文件**：
- `configs/hil_controls.md`：本机的按键/踏板映射记录
- `deploy/scripts/hil_smoke_test.sh`：冒烟测试脚本

**验证**：dry-run 跑 30s，3 个状态切换均能记录到 stdout

---

### T6.2 失败模式聚焦与采集计划

**目标**：把 Phase 5 T5.3 的失败模式翻译成"HIL 触发场景清单"

**步骤**：
- [ ] 读 `eval/results/real_v1_baseline.md` 和 T5.4 失败分析，按频次排序失败模式
- [ ] 对 PickPlaceRed 任务，按经验**目标采集量**：
  | 失败模式 | 推荐 HIL episode 数 | 触发场景设计 |
  |---------|---------------------|--------------|
  | `grasp_fail` | 35 | cube yaw 接近 ±π/4 边界 / 工作区边缘 |
  | `place_miss` | 25 | plate 在工作区边缘 / plate 颜色与桌布接近 |
  | `lift_drop` | 15 | cube 表面打磨光滑 / 加涂层 |
  | `plate_off` | 15 | cube 释放高度太高反弹弹出 |
  | `joint_limit` | 10 | 极端 cube/plate 位置触发关节限位 |
  | **总计** | **~100** | |
- [ ] **不要均匀采集**：v2 模型最差的失败模式权重最高（80/20 法则）
- [ ] 写采集计划 `eval/hil/collection_plan_v3.md`：每个场景几条、谁来采、何时采

**关键文件**：
- `eval/hil/collection_plan_v3.md`

**验证**：计划评审，每种失败模式都有明确的 trigger setup + 目标条数

---

### T6.3 第一轮 HIL 采集（基于 v2 模型）

**目标**：跑出 ~100 条 HIL episode

**步骤**：
- [ ] 启动 HIL 采集（Pi0.5/SmolVLA winner 用 RTC，ACT/Diffusion 不需要）：

  ```bash
  # 主推：Pi0.5 winner + RTC
  python examples/hil/hil_data_collection.py \
      --rtc.enabled=true \
      --rtc.execution_horizon=20 \
      --rtc.max_guidance_weight=5.0 \
      --rtc.prefix_attention_schedule=LINEAR \
      --robot.type=so_follower \
      --robot.port=/dev/ttyUSB0 \
      --robot.cameras='{wrist: {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30}}' \
      --teleop.type=so_leader \
      --teleop.port=/dev/ttyUSB1 \
      --policy.path=training/checkpoints/pi05_so101_v2/last/pretrained_model \
      --dataset.repo_id=local/so101_pickplace_hil_v3 \
      --dataset.single_task="put the red cube on the plate" \
      --dataset.fps=30 \
      --dataset.episode_time_s=60 \
      --dataset.num_episodes=100 \
      --interpolation_multiplier=2
  ```

- [ ] **采集纪律**：
  - 每条 episode 开始前按 T6.2 计划摆好 cube / plate
  - **只在临界点介入**：不要因为"看起来不够流畅"就接管（否则 dataset 全是人类轨迹，HIL 失去意义）
  - Recovery 要把机器人摇回**像 demo 一样的姿态**才放手（in-distribution）
  - Correction 要果断，3–5 秒完成 subtask 即可
  - 同一条 episode 多次交接是好事，正好教会 policy 多种失败的恢复
- [ ] **录像**：另开一个手机相机录全程，事后可视化 review

**关键文件**：
- `data/real_demos/so101_pickplace_hil_v3/`
- `eval/recordings/hil_v3/`

**验证**：
- 100 条 episode 完成，每条至少 1 次人机交接
- LeRobot dataset viewer 能回放并可视化 intervention 段
- 失败模式覆盖与 T6.2 计划比例偏差 < 20%

---

### T6.4 HIL 数据后处理与质量控制

**目标**：标注 + 筛选高质量 HIL 数据

**步骤**：
- [ ] 写 `data/hil_adapter/audit.py`：
  - 每条 episode 的 intervention 段统计（数量 / 时长 / 时序位置）
  - 标记 "全程介入" 的 episode（人接管 > 70% 时长）→ 这种本质是普通 demo，单独打 source 标签
  - 标记 "无介入" 的 episode（policy 全程跑成）→ 也保留，作为成功示例
- [ ] 给每帧加 `control_source` 元数据（`policy` / `human_recovery` / `human_correction` / `policy_resume`）
- [ ] 人工抽检 20 条，看 recovery 段是否真的从"失败临界状态"出发
- [ ] 失败案例（人都救不回来）单独存到 `data/real_demos/.../discarded/`

**关键文件**：
- `data/hil_adapter/audit.py`
- `data/hil_adapter/annotate_segments.py`
- `data/real_demos/so101_pickplace_hil_v3/audit.md`

**验证**：
- 抽检 20 条中 ≥ 18 条 intervention 段从 "失败临界" 状态起步
- `control_source` 元数据完整无 NaN

---

### T6.5 Recovery dataset 混合训练（v3 模型）

**目标**：在 v2 基础上做 HIL co-finetune 得到 v3

**步骤**：
- [ ] 写 `training/configs/{pi05|smolvla}_so101_v3.yaml`：
  - `pretrained_path` = v2 checkpoint（warm start）
  - `dataset.repo_id` = 合并 dataset：原 `so101_pickplace_mixed_v1` + `so101_real_pickplace_targeted_v1`（Phase 5）+ **`so101_pickplace_hil_v3`（新加）**
  - HIL data 采样权重 **5–8×**（这是 v3 的核心信号，不要被淹没）
  - lr 进一步降到 v2 的 1/2（已经接近收敛）
  - epochs = 2（避免过拟合到 HIL）
- [ ] 启动训练，wandb 同时显示 v1 / v2 / v3 三条 loss 曲线对比
- [ ] 训完后必跑 `eval/sim_eval.py`：sim 内不能比 v2 退化超过 5%
- [ ] 训完后必跑反向指令测试（Phase 4 T4.8）：确认 HIL 没破坏语言锚定

**关键文件**：
- `training/configs/{pi05|smolvla}_so101_v3.yaml`
- `training/checkpoints/{pi05|smolvla}_so101_v3/`

**参考**：
- LeRobot fine-tune 命令：
  ```bash
  python src/lerobot/scripts/lerobot_train.py \
      --dataset.repo_id=local/so101_pickplace_combined_v3 \
      --policy.type=pi0 \
      --policy.pretrained_path=training/checkpoints/pi05_so101_v2/last/pretrained_model \
      --output_dir=training/checkpoints/pi05_so101_v3 \
      --steps=20000
  ```

**验证**：sim_eval 不掉超过 5%；val loss 单调；wandb 看到 HIL 段 loss 显著更低（说明模型学到了 recovery）

---

### T6.6 真机评估 v3 与失败模式对比

**目标**：量化 HIL 微调真的修复了 Phase 5 的失败模式

**步骤**：
- [ ] 完全复用 Phase 5 T5.3 的评估 setup（30 次试验 + 4 象限 + yaw 多样性）
- [ ] 跑 v3 评估，按相同失败模式枚举统计
- [ ] 写 `eval/results/real_v3_eval.md`，表格对比 v2 vs v3：
  | 失败模式 | v2 (Phase 5) | v3 (Phase 6 round 1) | Δ |
  |---------|--------------|----------------------|---|
  | 总成功率 | X/30 | Y/30 | |
  | grasp_fail | a | b | |
  | place_miss | c | d | |
  | lift_drop | e | f | |
  | plate_off | g | h | |
  | ... | | | |
- [ ] 若某类失败**没下降**：说明该类 HIL 数据采少了 / 质量不够，回到 T6.2 调整

**关键文件**：
- `eval/results/real_v3_eval.md`

**验证**：
- v3 总成功率 > v2 至少 10 个百分点
- 没有出现新的退化失败模式

---

### T6.7 多轮迭代（v3 → v4 → v5）

**目标**：每轮把上一轮最差的失败模式吃掉

**步骤**：
- [ ] 把 T6.2 ↔ T6.3 ↔ T6.5 ↔ T6.6 串成循环：
  ```
  v3 评估 → 选 top-2 残留失败模式 → 针对性采 30–50 条 HIL → 训 v4 → 评估
  v4 评估 → 选 top-2 残留失败模式 → 针对性采 30–50 条 HIL → 训 v5 → 评估
  ```
- [ ] **何时停止**：
  - 真机成功率连续 2 轮提升 < 3% → 停（边际收益用尽）
  - 所有失败模式占比都 < 5% → 停（任务已解决）
  - HIL 数据总量到 300 条 → 停（过多 HIL 数据反而压制 demo 分布）
- [ ] 每轮版本独立 wandb run + checkpoint，可随时回退

**关键文件**：
- `training/checkpoints/<model>_so101_v4/`、`v5/`
- `eval/results/real_v{4,5}_eval.md`
- `data/real_demos/so101_pickplace_hil_v{4,5}/`

**验证**：v5（或停止点版本）真机成功率 > 80%

---

### T6.8 Recovery vs Correction 解耦消融（关键科研项）

**目标**：验证 RaC 论文的核心 claim——recovery 段单独有价值

**步骤**：
- [ ] 拿 T6.3 的 100 条 HIL 数据，按 `control_source` 拆三个子集：
  - **A**：只用 `human_correction` 段（标准 BC）
  - **B**：只用 `human_recovery` 段（recovery-only，理论上单独不完整）
  - **C**：A + B 全用（标准 HIL，T6.5 已做）
- [ ] 各训一个 v3 版本，sim_eval + 真机 mini-eval（10 次）对比
- [ ] 结论应该是 C > A > B，**且 C - A > A - B**（说明 recovery 段的边际价值高）

**关键文件**：
- `training/configs/*_v3_ablation_{A,B,C}.yaml`
- `eval/results/rac_ablation.md`

**参考**：
- RaC 论文：*Robot Learning for Long-Horizon Tasks by Scaling Recovery and Correction* (Hu et al., 2025) arXiv:2509.07953

**验证**：消融报告完整；如果 B 单独不退化甚至有用 → 写下来当 finding（可发 workshop）

---

### T6.9 RTC 调优（仅 Pi0.5 / SmolVLA winner）

**目标**：保证大模型推理不卡 HIL pause/takeover 节奏

**步骤**：
- [ ] 测当前推理延迟（Pi0.5 fp16 在你 GPU 上）：典型 80–200ms / step
- [ ] 若延迟 > 100ms，启用 RTC 关键参数：
  - `--rtc.execution_horizon`：每次推理跨多少步（默认 20，可降到 15）
  - `--rtc.max_guidance_weight`：guidance 上限（默认 5.0）
  - `--rtc.prefix_attention_schedule=LINEAR`
  - `--interpolation_multiplier=3`：关节插值更细，掩盖推理跳变
- [ ] 调参后实测真机动作平滑度（用 IMU 或视觉 jitter 指标）
- [ ] HIL 采集时记录 RTC 配置到 dataset metadata

**关键文件**：
- `deploy/configs/rtc_tuning.md`：调参日志

**验证**：HIL session 60s 无明显卡顿；joint velocity rms 比 RTC off 时低 ≥ 30%

---

### T6.10 数据治理与版本化

**目标**：多轮迭代产生很多 dataset / checkpoint，要可追溯

**步骤**：
- [ ] 在 `data/hil_adapter/manifest.py` 维护 dataset 谱系：
  ```yaml
  v3_combined:
    base: so101_pickplace_mixed_v1
    + so101_real_pickplace_targeted_v1
    + so101_pickplace_hil_v3 (weight=5x)
  v4_combined:
    base: v3_combined
    + so101_pickplace_hil_v4 (weight=5x)
  ...
  ```
- [ ] 每个 checkpoint 在 `training/checkpoints/<v>/manifest.yaml` 记录：
  - 训练 dataset 版本
  - 训练超参
  - sim_eval + 真机 eval 结果
  - 失败模式分布
- [ ] 推 winner 到 HF Hub：`<your_id>/so101_pickplace_v3` / `_v4` / `_v5`

**关键文件**：
- `data/hil_adapter/manifest.py`
- `training/checkpoints/<v>/manifest.yaml`

**验证**：从最终 winner 版本可一路追溯到原始 sim/真机数据

---

## 验收标准（全部满足后进入 Phase 7）

- [ ] HIL pipeline 跑通，至少完成 1 轮（T6.3 + T6.5 + T6.6）
- [ ] v3 真机成功率比 v2 提升 ≥ 10 个百分点
- [ ] 至少完成 2 轮迭代（v3 + v4），或迭代收敛（边际收益 < 3%）
- [ ] 终版模型真机 PickPlaceRed 成功率 > 80%
- [ ] RaC 消融实验报告完成（recovery vs correction 拆解）
- [ ] dataset 谱系与 checkpoint manifest 齐全

---

## 风险与陷阱

| 风险 | 应对 |
|------|------|
| 介入太频繁，dataset 退化为普通 demo | 强制纪律：只在"快要失败"时介入；事后审计 intervention 时长占比应 30%–60% |
| 介入太少，HIL 段太短没信号 | 反过来：把 v2 模型放到更难的 setup（边缘位置 / 紧邻物体），自然多失败多介入 |
| Recovery 段质量差（人也手抖） | 多人采集 / 同一操作员先练 20 条再正式采；不熟练的 recovery 比没有更糟 |
| HIL 数据权重过高，原 demo 分布被压制 | 5–8× 上限；同时保留全部原 dataset；用 sim_eval 监控 demo 任务有无退化 |
| HIL 数据让某些 yaw 角性能变差 | 检查 HIL 数据 yaw 分布是否过度集中（偏向某一类失败模式），保持各 yaw 角覆盖均衡 |
| 多轮迭代后过拟合到评估 setup | 评估 setup 每 2 轮换一次（同任务，但物体位置 / 光照变） |
| 推理延迟造成介入瞬间机器人冲过头 | 启用 RTC + `interpolation_multiplier=3`；pause 触发时 follower 立即 freeze（torque hold） |
| 踏板抽风（短按被识别成长按） | 用 evtest 校准 debounce；或换有源踏板 |

---

## 输出物

- HIL 采集 pipeline（含踏板配置、采集脚本、审计工具）
- 多版本 HIL dataset（v3 / v4 / v5 ...），共 200–300 条
- 多版本 checkpoint（v3 / v4 / v5 ...），各自带评估报告
- 失败模式收敛曲线（每轮 vs 残留失败比例）
- RaC 消融报告（recovery vs correction）
- 最终 winner 推 HF Hub

---

## 与其他 Phase 的关系

```
Phase 3：纯成功 demo 数据（in-distribution）
Phase 5：v2 + 失败模式归类（暴露 distribution shift）
Phase 6（本文档）：HIL recovery/correction（针对性修复 distribution shift）  ← 你在这里
Phase 7：扩展任务（多物体 / 多容器 / 多步指令）
```

**Phase 6 是真机性能从"勉强能用"到"鲁棒可用"的核心一步**。跳过 Phase 6 直接做 Phase 7（更难任务），等同于在脆弱基础上加复杂度，注定崩。

---

## 参考

- LeRobot HIL 官方文档：`https://huggingface.co/docs/lerobot/hil_data_collection`
- DAgger：Ross et al. 2011，*A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*
- HG-DAgger：Kelly et al. 2019，*HG-DAgger: Interactive Imitation Learning with Human Experts*（HIL 的现代雏形）
- RaC：Hu et al. 2025，arXiv:2509.07953（recovery/correction 解耦）
- Pi0.6 / RECAP：Physical Intelligence 2025（VLA 上的工业级 HIL 实践）
