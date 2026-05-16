# Phase 5：真机部署与 co-finetune

**周期**：2–3 周
**前置依赖**：Phase 4 完成（v1 模型已选定）
**目标**：把 Phase 4 的 winner 模型部署到真机 SO101，量化真机性能，针对失败模式补采少量真机数据并做 co-finetune，最终真机 **PickPlaceBlue** 成功率 > 60%（含颜色锚定 + place 精度）

> **核心任务**：抓蓝 cube 放进 plate（红 cube 是干扰物）。详细规约见 [README.md](README.md)。

---

## 代码入口（快速开始）

> **状态**：Phase 5 代码尚未实现，需要真机硬件 + Phase 4 训练好的 checkpoint。下表为规划入口。

| 想做的事 | 计划命令（占位） | 计划产出 |
|---------|-----------------|----------|
| 真机评估（同分布） | `python -m deploy.real_eval --checkpoint runs/phase4/pi0/checkpoints/best --n-trials 50 --robot so101` | 真机成功率 + 失败模式分布 |
| 失败模式补采 demo | `python -m lerobot.scripts.control_robot record ...`（上游 CLI） | 50 条 recovery demo |
| Co-finetune（mixed_v1 + recovery v1） | `python -m training.cofinetune_pi0 --base runs/phase4/pi0/checkpoints/best --add local/so101_real_recovery_v1` | `runs/phase5/pi0_v2/checkpoints/` |
| 部署 v2 模型做 final 验收 | `python -m deploy.real_eval --checkpoint runs/phase5/pi0_v2/checkpoints/best --n-trials 100 --robot so101 --report runs/phase5/final_report.md` | 真机 success ≥60% 报告 |

**计划实现文件**：`deploy/real_eval.py`、`deploy/robot_runner.py`、`training/cofinetune_pi0.py`。

---

## 关键技术与工具

| 工具 | 用途 | 备注 |
|------|------|------|
| LeRobot inference / control | 真机推理 SDK | 主仓库自带 |
| openpi inference server | Pi0.5 推理后端（可选） | 推理速度更优 |
| websocket / zmq | 模型推理与机器人控制解耦 | 推理放 GPU 机，控制放本地 |
| OpenCV | 真机相机预处理 | 已用 |
| LeRobot record | 补采 demo | Phase 0 已用 |
| Weights & Biases | 真机评估日志 | 已配置 |

**部署架构（推荐）**：
```
[USB 相机 + SO101 控制] --joint state--> [推理服务 GPU 机]
       ^                                        |
       |---------- action ----------------------+
```
推理与控制解耦能让 GPU 资源复用、推理频率独立调优。本地一体机方案也可。

---

## 任务清单

### T5.1 真机推理 pipeline 搭建

**目标**：模型 → 真机控制的最小闭环

**步骤**：
- [ ] 在 `deploy/inference_server.py` 实现：
  - 加载 Phase 4 winner checkpoint
  - 提供推理接口：输入 `{images, joint_state, task}` → 输出 `action`
  - openpi 模型用 openpi 提供的 inference wrapper；SmolVLA 用 LeRobot 原生 inference
- [ ] 在 `deploy/robot_client.py` 实现：
  - 读真机相机 + joint state
  - 调用推理服务（local function call 或 zmq）
  - 写 action 回真机（30Hz 闭环）

**关键文件**：
- `deploy/inference_server.py`
- `deploy/robot_client.py`
- `deploy/configs/deploy.yaml`：模型路径、推理频率、安全限位

**参考**：
- LeRobot 真机推理示例：`lerobot-eval` CLI 中的 inference loop
- openpi inference server：`https://github.com/Physical-Intelligence/openpi/tree/main/scripts/serve_policy.py`

**验证**：
- 推理 + 控制闭环可启动，无报错
- 实测推理延迟 < 100ms（不含相机抓帧）

---

### T5.2 安全护栏

**目标**：避免模型输出异常导致机械臂损坏

**步骤**：
- [ ] 在 `deploy/robot_client.py` 加：
  - 关节速度限位（每步关节变化 < 0.3 rad）
  - 关节角度限位（不超过 MJCF 里读到的硬限位的 90%）
  - 紧急停止键（键盘 ESC）
  - 推理输出 NaN / out-of-range 时挂起
- [ ] 加上 GUI 显示当前 action / state（终端用 `rich.live`）

**关键文件**：
- `deploy/safety.py`
- `deploy/robot_client.py`（接入）

**验证**：
- 故意给一个超限 action，机器人不会执行，给出警告
- ESC 触发后机械臂回到 home pose

---

### T5.3 真机 zero-shot 评估（v1 模型）

**目标**：拿到 Phase 4 winner 在真机的基线数字

**步骤**：
- [ ] 准备评估 setup：
  - 1 红 cube + 1 蓝 cube + 1 plate
  - **30 次试验**，红/蓝相对位置覆盖 4 象限（每象限 ≥ 6 次），plate 位置 ±3cm 抖动
  - 其中 **6 次"颜色锚定难度测试"**：红 cube 离机械臂更近（容易被默认抓到）
- [ ] 每次试验：
  - 给定语言指令（随机选 1 种变体）
  - 启动 client，最多 45 秒（pick-place 比 pick 长）
  - 记录：成功 / 失败、失败模式
- [ ] **失败模式细分**：
  - `color_confusion`：抓了红 cube（**最严重**）
  - `grasp_fail`：抓蓝失败
  - `transport_hit_red`：搬运时把红 cube 撞飞
  - `place_miss`：放在 plate 外
  - `lift_drop`：抓到了但途中掉
  - `timeout`：45s 内未完成
- [ ] 同步全程录像（前置相机外加一个手机相机）

**关键文件**：
- `eval/real_eval.py`：评估主脚本
- `eval/results/real_v1_baseline.md`：报告
- `eval/recordings/real_v1/`：视频

**验证**：得到量化基线（如 X/30），失败模式有归类；尤其 `color_confusion` 比例是 Phase 5 核心要降的指标

---

### T5.4 失败模式分析

**目标**：知道下一步该补什么数据

**步骤**：
- [ ] 把 T5.3 的失败分类成至少 5 类：
  - **颜色识别**：抓了红 cube（color_confusion）
  - 几何：approach 角度偏 / 高度错
  - 抓取：闭合时机错 / 闭合不到位
  - 搬运：transport 中撞红 cube 或脱手
  - 放置：blue cube 放在 plate 外或弹出
- [ ] 对每类列出"假设的补救数据"
- [ ] 颜色识别失败重点排查：训练数据红蓝对比度 / 真机光照下蓝色饱和度 / camera 白平衡

**关键文件**：
- `eval/results/real_v1_failure_analysis.md`

**验证**：每类失败有 ≥1 条补救数据假设

---

### T5.5 补采针对性 demo

**目标**：用 20–50 条针对性 demo 矫正主要失败模式

**步骤**：
- [ ] 按 T5.4 列表，专门针对每类失败补采：
  - 颜色识别失败 → 红 cube 距机械臂更近的极端位置多采（强迫学习颜色而非空间近邻）
  - 几何失败 → 在工作区边缘补采
  - 抓取失败 → 在不同 blue cube 朝向下补采
  - 搬运失败 → 在红/蓝距离最近的几种摆放下重点采
  - 放置失败 → 在不同 plate 位置 / 不同 plate 颜色下补采
- [ ] 总量目标 20–50 条
- [ ] 保存为 `data/real_demos/so101_real_pickplace_blue_targeted_v1/`

**关键文件**：
- `data/real_demos/so101_real_pickplace_blue_targeted_v1/`

**验证**：补采数据已 ingest，且能被 LeRobot 加载

---

### T5.6 Co-finetune 训练

**目标**：在 v1 基础上做轻量微调

**步骤**：
- [ ] 配置 `training/configs/{pi05|smolvla}_so101_v2.yaml`：
  - 从 v1 checkpoint warm start
  - dataset = v1 mixed + targeted（targeted 权重 10×）
  - lr 降到 v1 的 1/3
  - epochs = 2–3（避免过拟合到 targeted）
- [ ] 启动训练
- [ ] 训练完成后跑 sim_eval 确认没回退

**关键文件**：
- `training/configs/<model>_so101_v2.yaml`
- `training/checkpoints/<model>_so101_v2/`

**验证**：
- sim_eval 不掉超过 5%
- val loss 不发散

---

### T5.7 真机再评估（v2 模型）

**目标**：验证 co-finetune 真的有提升

**步骤**：
- [ ] 复用 T5.3 的评估 setup（同样 30 次试验，同样位置）
- [ ] 跑评估，记录
- [ ] 写 `eval/results/real_v2_eval.md`，对比 v1

**关键文件**：
- `eval/results/real_v2_eval.md`

**验证**：v2 成功率 > v1，且 > 60%

---

### T5.8 鲁棒性扩展评估

**目标**：除了"标准 setup"还有哪些场景能用

**步骤**：
- [ ] 干扰评估（每项 10 次）：
  - 桌面铺一张新颜色桌布
  - 加 1 个**额外**干扰物（如绿/黄 cube；总共 3 个 cube + plate）
  - 调整环境光（开关一盏灯）
  - 摄像头位置抖动 ±3cm
  - **plate 替换**：换不同形状/颜色的盘子（白瓷盘、黑色塑料盘）
- [ ] 记录成功率衰减幅度
- [ ] 单独记录"颜色锚定准确率"在每种干扰下的衰减

**关键文件**：
- `eval/real_robustness.py`
- `eval/results/real_v2_robustness.md`

**验证**：每种干扰下成功率衰减 < 30%（说明 sim DR 起了作用）

---

### T5.9 部署文档

**目标**：未来自己 / 别人能复现部署

**步骤**：
- [ ] 写 `deploy/README.md`：
  - 启动命令
  - 安全检查清单
  - 常见问题（推理太慢 / 关节抖动 / 抓不到等）
- [ ] 录一个 3 分钟"完整任务演示"短视频

**关键文件**：
- `deploy/README.md`
- `eval/recordings/deployment_demo.mp4`

**验证**：另开终端按 README 能从零启动跑通

---

## 验收标准（Phase 5 完成）

- [ ] 真机推理 pipeline + 安全护栏可用
- [ ] v1 模型真机基线已建立
- [ ] 失败模式归类清晰
- [ ] v2 模型在真机成功率 > 60%
- [ ] 颜色锚定准确率 > 85%（即使整体成功率 60%，"抓对颜色"必须更高）
- [ ] 鲁棒性评估通过（5 项干扰平均衰减 < 30%）
- [ ] 部署文档 + 演示视频齐全

---

## 风险与陷阱

| 风险 | 应对 |
|------|------|
| 推理延迟过高导致真机抖动 | 推理服务用 fp16；调小 image resolution；增大 action chunk |
| 真机相机与仿真相机白平衡差异大导致蓝/红识别偏差 | 在 inference server 加 colorjitter / 白平衡校正；Phase 3 训练时强制抖色相 ±15° |
| 真机蓝 cube 在某些光照下偏紫/偏青 | 用色卡校正白平衡；选饱和度高的蓝（如 RAL 5005 信号蓝） |
| 真机 zero-shot 成功率 < 20% 让你怀疑路线 | 这是 sim2real 典型情况，先做 co-finetune 再下结论；最差就只用真机数据小幅微调 |
| Co-finetune 过拟合 targeted，泛化变差 | targeted 权重不要太高（≤ 10×）；保留 v1 的 broad sim 数据 |
| 安全护栏触发频繁 | 用更严格的 action smoothing（action chunk + EMA） |
| 推理 server 与 client 跨机器延迟 | 同一局域网；先 local 同机部署确认逻辑正确再分机 |

---

## 输出物

- 真机部署 pipeline（含安全护栏）
- v1 / v2 真机评估报告
- 鲁棒性评估
- 部署文档 + 演示视频
- **Phase 6 HIL 的起点**：v2 模型 + 失败模式分类报告（这些失败模式正是 Phase 6 HIL 采集要专门修复的目标，见 [phase6-hil-recovery.md](phase6-hil-recovery.md) T6.2）
