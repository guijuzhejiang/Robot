# Phase 5：真机部署与 co-finetune

**周期**：2–3 周
**前置依赖**：Phase 4 完成（v1 模型已选定）
**目标**：把 Phase 4 winner 部署到真机 SO101，量化真机性能，针对失败模式补采少量真机数据做 co-finetune，最终真机 PickPlaceRed 成功率 > 60%

---

## 代码入口（快速开始）

> **状态**：Phase 5 代码尚未实现，需要真机 + Phase 4 checkpoint。

| 想做的事 | 计划命令 |
|---------|---------|
| 真机评估（v1） | `python -m deploy.real_eval --checkpoint runs/phase4/pi0/checkpoints/best --n-trials 50 --robot so101` |
| 失败模式补采 | `python -m lerobot.scripts.control_robot record ...`（上游 CLI） |
| Co-finetune | `python -m training.cofinetune_pi0 --base ...best --add local/so101_real_recovery_v1` |
| 部署 v2 验收 | `python -m deploy.real_eval --checkpoint runs/phase5/pi0_v2/checkpoints/best --n-trials 100 --report runs/phase5/final_report.md` |

**部署架构**：推荐推理与控制解耦——GPU 机跑推理服务（zmq/websocket），本地控制 PC 跑 client。一体机也可。

---

## 任务清单

### T5.1 真机推理 pipeline

`deploy/inference_server.py`：加载 winner checkpoint，提供 `{images, joint_state, task} → action` 接口（openpi 用其 wrapper；SmolVLA 用 LeRobot 原生 inference）。

`deploy/robot_client.py`：读真机相机 + joint state → 调推理服务 → 写 action 回真机（30Hz 闭环）。

**验证**：闭环可启动无报错；推理延迟 < 100ms（不含相机抓帧）

### T5.2 安全护栏

`deploy/safety.py` + 接入 client：
- 关节速度限位（每步变化 < 0.3 rad）
- 关节角度限位（MJCF 硬限位的 90%）
- 紧急停止键（ESC）
- 推理输出 NaN / out-of-range 时挂起
- 终端 `rich.live` 显示当前 action / state

**验证**：超限 action 不执行有警告；ESC 触发回 home pose

### T5.3 真机 zero-shot 评估（v1）

Setup：1 红 cube + 1 plate，**30 次试验**，cube 位置覆盖 4 象限（每象限 ≥6 次），plate 位置 ±3cm 抖动，每种 cube yaw（0/π/2/π/3π/2）≥6 次。

每次：给定语言指令（随机选 1 种变体）→ 启动 client，最多 30 秒 → 记录成功/失败 + 失败模式（grasp_fail / lift_drop / place_miss / plate_off / joint_limit / timeout）。同步全程录像。

**验证**：得到量化基线（X/30）+ 失败归类

### T5.4 失败模式分析

把 T5.3 失败分至少 4 类：
- 几何：approach 角度偏 / 高度错
- 抓取：闭合时机错 / cube yaw 难
- 搬运：transport 脱手
- 放置：放在 plate 外或弹出

对每类列出"假设的补救数据"。

### T5.5 补采针对性 demo

按 T5.4 列表专项补采：
- 几何失败 → 工作区边缘
- 抓取失败 → 不同 cube yaw（特别是 ±π/4 边界附近）
- 搬运失败 → cube 在 plate 同侧近距离
- 放置失败 → 不同 plate 位置 / 颜色

总量 20–50 条，存 `data/real_demos/so101_real_pickplace_targeted_v1/`。

### T5.6 Co-finetune

Config `training/configs/<model>_so101_v2.yaml`：v1 checkpoint warm start；dataset = v1 mixed + targeted（targeted 权重 10×）；lr 降到 v1 的 1/3；epochs 2–3（避免过拟合 targeted）。

**验证**：sim_eval 不掉超过 5%；val loss 不发散

### T5.7 真机再评估（v2）

复用 T5.3 setup（同位置、同 yaw 分布），跑评估，写 `eval/results/real_v2_eval.md` 对比 v1。

**验证**：v2 > v1，且 > 60%

### T5.8 鲁棒性扩展评估

每项 10 次干扰：
- 桌面铺新颜色桌布
- 加 1 个干扰物（绿/黄 cube；语言锚定仍是 red）
- 调环境光（开关 1 盏灯）
- 摄像头位置 ±3cm 抖动
- plate 换不同形状/颜色（白瓷盘、黑色塑料盘）

**验证**：每种干扰下衰减 < 30%

### T5.9 部署文档

`deploy/README.md`：启动命令 + 安全清单 + 常见问题。录 3 分钟"完整任务演示"视频。

**验证**：另开终端按 README 能从零启动跑通

---

## 验收标准

- [ ] 真机推理 + 安全护栏可用
- [ ] v1 基线已建立
- [ ] 失败模式归类清晰
- [ ] v2 真机成功率 > 60%
- [ ] 鲁棒性评估通过（5 项干扰平均衰减 < 30%）
- [ ] 部署文档 + 演示视频齐全

---

## 风险与陷阱

| 风险 | 应对 |
|------|------|
| 推理延迟过高致真机抖动 | fp16；调小 image resolution；增大 action chunk |
| 真机相机与仿真白平衡差异 | inference server 加 colorjitter / 白平衡校正；Phase 3 训练时抖色相 ±15° |
| 真机红 cube 在某些光照偏暗/偏粉 | 色卡校正白平衡；选饱和度高的红（如 RAL 3020 交通红） |
| 真机 zero-shot < 20% 让你怀疑路线 | sim2real 典型情况，先 co-finetune 再下结论；最差只用真机数据小幅微调 |
| Co-finetune 过拟合 targeted | targeted 权重不要太高（≤10×）；保留 v1 broad sim 数据 |
| 安全护栏触发频繁 | 用更严格的 action smoothing（action chunk + EMA） |
| 推理 server 与 client 跨机器延迟 | 同一局域网；先 local 同机验证再分机 |

---

## 输出物

- 真机部署 pipeline（含安全护栏）
- v1/v2 真机评估报告
- 鲁棒性评估
- 部署文档 + 演示视频
- **Phase 6 HIL 起点**：v2 模型 + 失败模式分类报告（正是 [phase6](phase6-hil-recovery.md) T6.2 要专门修复的目标）
