# Phase 6：HIL recovery 数据采集与回炉微调

**周期**：2–3 周
**前置依赖**：Phase 5 完成（v2 模型已部署真机，失败模式已归类）
**目标**：用 LeRobot 官方 HIL（Human-In-the-Loop）pipeline，针对 Phase 5 暴露的失败模式专门采集 **recovery + correction** 数据，与原 dataset 合并做 co-finetune，把真机 PickPlaceRed 成功率从 60% 推到 80%+

---

## 代码入口（快速开始）

> **状态**：Phase 6 代码尚未实现，依赖 LeRobot 上游 HIL pipeline + 真机 + 脚踏开关。

| 想做的事 | 计划命令 |
|---------|---------|
| 启动 HIL 采集（VLA 主导 + foot pedal takeover） | `python -m hil.collect_recovery --policy runs/phase5/pi0_v2/checkpoints/best --foot-pedal /dev/input/event3 --output local/so101_hil_recovery_v1` |
| 标注 recovery 段 | `python -m hil.label_takeovers --repo-id local/so101_hil_recovery_v1` |
| Co-finetune v3 | `python -m training.cofinetune_pi0 --base runs/phase5/pi0_v2/checkpoints/best --add local/so101_hil_recovery_v1 --weight-recovery 2.0` |
| 真机验收（≥80%） | `python -m deploy.real_eval --checkpoint runs/phase6/pi0_v3/checkpoints/best --n-trials 100 --report runs/phase6/final_report.md` |

---

## 为什么必须做 HIL

**标准行为克隆只学"成功轨迹"**：模型只见过 demo 里"一切正常"的状态分布。真机一旦出现小偏差就漂到训练时未见过的分布外区域（**compounding error / distribution shift**），错误放大。

**HIL 直接解决**：
1. v2 模型在真机跑 → 2. 人**只在快要失败时介入**（不是每步都标）→ 3. 接管做 recovery（摇回 in-distribution 状态）+ correction（演示正确行为）→ 4. 同一 episode 可多次交接 → 5. 合并 dataset 微调 → v3 → 再跑 HIL → v4...

**与 Phase 3 demo 采集的根本区别**：Phase 3 人**全程**遥操采"理想成功轨迹"；Phase 6 模型先跑，人**只在失败临界点介入**——recovery 段起点正好覆盖 v2 模型实际会漂到的状态分布。

这是 DAgger / HG-DAgger / RaC 在 SO101 上的落地，也是 Pi0.6（RECAP）大幅超越 Pi0.5 的核心方法。

**官方文档**：[LeRobot HIL data collection](https://huggingface.co/docs/lerobot/hil_data_collection)

---

## HIL 交互协议

```
[Policy 自主跑] ──觉察临界（看错色 / 撞红 / 偏离 plate）──┐
                                                          │
[踩 PAUSE 踏板]：policy 停，leader 镜像 follower 当前位姿  │
                                                          │
[踩 TAKEOVER 踏板]：leader 自由，遥操做 recovery+correction │
                                                          │
[踩 RETURN 踏板]：policy 从当前状态继续 ─── 同 episode 循环 ┘
```

**关键点**：
- 一条 episode 可多次交接，不重置
- Recovery 段起点是 v2 模型"会漂到的失败前状态"——普通 demo 永远拿不到
- Correction 不过度——只演示完成当前 subtask，别加多余动作

---

## 任务清单

### T6.1 HIL 环境与硬件搭建

- [ ] 升级 LeRobot 到含 `examples/hil/` 的版本
- [ ] 验证 `so_follower` + `so_leader` HIL 配置已注册
- [ ] 接入 USB 脚踏开关（推荐 PCsensor 3 键款 ~¥80）：
  ```bash
  sudo setfacl -m u:$USER:rw /dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd
  evtest  # 确认能读到按键事件
  ```
- [ ] 键盘 fallback 跑 dry-run 验证 pause/takeover/return 映射
- [ ] 把按键/踏板绑定写到 `configs/hil_controls.md`

**验证**：dry-run 30s，3 个状态切换均能记录到 stdout

### T6.2 失败模式聚焦与采集计划

读 `eval/results/real_v1_baseline.md` + T5.4 失败分析，按频次排序失败模式。**目标采集量**（不要均匀采，按失败模式权重分配）：

| 失败模式 | 推荐 HIL ep 数 | 触发场景设计 |
|---------|---------------|--------------|
| `grasp_fail` | 35 | cube yaw 接近 ±π/4 边界 / 工作区边缘 |
| `place_miss` | 25 | plate 在工作区边缘 / 颜色与桌布接近 |
| `lift_drop` | 15 | cube 表面打磨光滑 / 加涂层 |
| `plate_off` | 15 | cube 释放高度太高反弹 |
| `joint_limit` | 10 | 极端 cube/plate 位置触发限位 |
| **总** | **~100** | |

写采集计划 `eval/hil/collection_plan_v3.md`。

### T6.3 第一轮 HIL 采集（基于 v2 模型）

```bash
python examples/hil/hil_data_collection.py \
    --rtc.enabled=true \
    --rtc.execution_horizon=20 \
    --rtc.max_guidance_weight=5.0 \
    --rtc.prefix_attention_schedule=LINEAR \
    --robot.type=so_follower --robot.port=/dev/ttyUSB0 \
    --robot.cameras='{wrist: {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30}}' \
    --teleop.type=so_leader --teleop.port=/dev/ttyUSB1 \
    --policy.path=training/checkpoints/pi05_so101_v2/last/pretrained_model \
    --dataset.repo_id=local/so101_pickplace_hil_v3 \
    --dataset.single_task="put the red cube on the plate" \
    --dataset.fps=30 --dataset.episode_time_s=60 \
    --dataset.num_episodes=100 \
    --interpolation_multiplier=2
```

**采集纪律**：
- 每条 episode 开始前按 T6.2 计划摆好 cube/plate
- **只在临界点介入**：不要因为"不够流畅"就接管（否则 dataset 全是人类轨迹，HIL 失去意义）
- Recovery 要把机器人摇回**像 demo 一样的姿态**才放手（in-distribution）
- Correction 果断，3–5 秒完成 subtask
- 同 episode 多次交接是好事
- 另开手机相机录全程

**验证**：100 条完成；每条至少 1 次交接；失败模式覆盖与计划偏差 < 20%

### T6.4 HIL 数据后处理与质控

`data/hil_adapter/audit.py`：
- 每条 episode 的 intervention 段统计（数量/时长/时序位置）
- 标 "全程介入" 的 episode（接管 > 70% 时长）→ 本质是普通 demo
- 标 "无介入" 的 episode → 也保留作成功示例
- 每帧加 `control_source` 元数据（`policy` / `human_recovery` / `human_correction` / `policy_resume`）
- 人工抽检 20 条，看 recovery 段是否真从失败临界状态出发
- 救不回来的失败案例存到 `discarded/`

**验证**：抽检 20 条中 ≥18 条 intervention 段从失败临界起步；`control_source` 完整

### T6.5 Recovery dataset 混合训练（v3）

Config `training/configs/<model>_so101_v3.yaml`：
- `pretrained_path` = v2（warm start）
- `dataset.repo_id` = v1 mixed + Phase 5 targeted + **`so101_pickplace_hil_v3`**
- HIL data 采样权重 **5–8×**（核心信号，不要被淹没）
- lr 降到 v2 的 1/2；epochs = 2

启动：
```bash
python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=local/so101_pickplace_combined_v3 \
    --policy.type=pi0 \
    --policy.pretrained_path=training/checkpoints/pi05_so101_v2/last/pretrained_model \
    --output_dir=training/checkpoints/pi05_so101_v3 \
    --steps=20000
```

训完跑 sim_eval（不能比 v2 退化 > 5%）+ 反向指令测试（确认 HIL 没破坏语言锚定）。

**验证**：val loss 单调；wandb 看到 HIL 段 loss 显著更低

### T6.6 真机评估 v3 vs v2

复用 Phase 5 T5.3 setup（30 次 + 4 象限 + yaw 多样性），按相同失败模式枚举统计：

| 失败模式 | v2 | v3 | Δ |
|---------|----|----|---|
| 总成功率 | X/30 | Y/30 | |
| grasp_fail | a | b | |
| place_miss | c | d | |
| ... | | | |

若某类失败**没下降** → 该类 HIL 数据采少了 / 质量不够，回 T6.2 调整。

**验证**：v3 总成功率 > v2 至少 10 个百分点；无新退化失败模式

### T6.7 多轮迭代（v3 → v4 → v5）

把 T6.2 ↔ T6.3 ↔ T6.5 ↔ T6.6 串成循环：每轮选 top-2 残留失败 → 针对性采 30–50 条 HIL → 训下一版 → 评估。

**停止条件**：
- 真机成功率连续 2 轮提升 < 3% → 停（边际收益用尽）
- 所有失败模式占比 < 5% → 停（任务已解决）
- HIL 数据总量到 300 条 → 停（过多 HIL 反而压制 demo 分布）

每轮版本独立 wandb run + checkpoint，可随时回退。

**验证**：终版（或停止点）真机成功率 > 80%

### T6.8 Recovery vs Correction 解耦消融（科研项）

验证 [RaC 论文](https://arxiv.org/abs/2509.07953)（Hu et al. 2025）核心 claim——recovery 段单独有价值。

拿 T6.3 的 100 条 HIL，按 `control_source` 拆三个子集训 v3：
- **A**：只用 `human_correction` 段（标准 BC）
- **B**：只用 `human_recovery` 段（recovery-only）
- **C**：A + B 全用（标准 HIL，T6.5 已做）

各跑 sim_eval + 真机 mini-eval（10 次）。预期 C > A > B，**且 C - A > A - B**（说明 recovery 段的边际价值高）。如果 B 单独有用 → 可写 workshop paper。

### T6.9 RTC 调优（仅 Pi0.5/SmolVLA winner）

测当前推理延迟（Pi0.5 fp16 典型 80–200ms/step）。若 > 100ms，启用 RTC：
- `--rtc.execution_horizon=15`（默认 20）
- `--rtc.max_guidance_weight=5.0`
- `--rtc.prefix_attention_schedule=LINEAR`
- `--interpolation_multiplier=3`

**验证**：HIL 60s 无明显卡顿；joint velocity rms 比 RTC off 时低 ≥ 30%

### T6.10 数据治理与版本化

`data/hil_adapter/manifest.py` 维护 dataset 谱系：

```yaml
v3_combined:
  base: so101_pickplace_mixed_v1
  + so101_real_pickplace_targeted_v1
  + so101_pickplace_hil_v3 (weight=5x)
v4_combined:
  base: v3_combined
  + so101_pickplace_hil_v4 (weight=5x)
```

每个 checkpoint 在 `training/checkpoints/<v>/manifest.yaml` 记录训练 dataset 版本 + 超参 + sim/真机 eval 结果 + 失败模式分布。推 winner 到 HF Hub。

**验证**：从最终 winner 可一路追溯到原始 sim/真机数据

---

## 验收标准

- [ ] HIL pipeline 跑通，至少完成 1 轮（T6.3 + T6.5 + T6.6）
- [ ] v3 真机成功率比 v2 提升 ≥ 10 个百分点
- [ ] 至少完成 2 轮迭代，或迭代收敛（边际 < 3%）
- [ ] 终版真机 PickPlaceRed 成功率 > 80%
- [ ] RaC 消融实验报告完成
- [ ] dataset 谱系与 checkpoint manifest 齐全

---

## 风险与陷阱

| 风险 | 应对 |
|------|------|
| 介入太频繁，dataset 退化为普通 demo | 强制纪律：只在"快要失败"时介入；事后审计 intervention 时长占比应 30–60% |
| 介入太少，HIL 段太短没信号 | 反过来：把 v2 放到更难 setup（边缘位置/紧邻物体）自然多失败 |
| Recovery 段质量差（人也手抖） | 多人采集 / 同一操作员先练 20 条再正式；不熟练的 recovery 比没有更糟 |
| HIL 数据权重过高，原 demo 分布被压制 | 5–8× 上限；保留全部原 dataset；用 sim_eval 监控有无退化 |
| HIL 数据让某些 yaw 角性能变差 | 检查 HIL 数据 yaw 分布是否过度集中；保持均衡 |
| 多轮迭代后过拟合到评估 setup | 评估 setup 每 2 轮换一次（同任务，物体位置/光照变） |
| 推理延迟造成介入瞬间机器人冲过头 | 启用 RTC + `interpolation_multiplier=3`；pause 触发时 follower 立即 freeze（torque hold） |
| 踏板抽风（短按被识别成长按） | 用 evtest 校准 debounce；或换有源踏板 |

---

## 与其他 Phase 的关系

```
Phase 3：纯成功 demo（in-distribution）
Phase 5：v2 + 失败模式归类（暴露 distribution shift）
Phase 6（本文档）：HIL recovery/correction（针对性修复 distribution shift）  ← 你在这里
Phase 7：扩展任务（多物体 / 多容器 / 多步指令）
```

Phase 6 是真机性能从"勉强能用"到"鲁棒可用"的核心一步。跳过 Phase 6 直接做 Phase 7（更难任务），等同于在脆弱基础上加复杂度，注定崩。

---

## 参考

- [LeRobot HIL data collection 官方文档](https://huggingface.co/docs/lerobot/hil_data_collection)
- DAgger（Ross et al. 2011）：*A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*
- HG-DAgger（Kelly et al. 2019）：HIL 的现代雏形
- [RaC（Hu et al. 2025）](https://arxiv.org/abs/2509.07953)：recovery/correction 解耦
- Pi0.6 / RECAP（Physical Intelligence 2025）：VLA 上的工业级 HIL 实践
