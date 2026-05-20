# Phase 0：环境与可行性验证

**周期**：1 周
**前置依赖**：SO-ARM101 套件已到货（leader + follower 双臂）、Ubuntu 22.04 主机、NVIDIA GPU（≥12GB 显存推荐）
**目标**：跑通从真机控制到 LeRobot 数据格式的最小闭环；在真机上评估 SmolVLA 预训练权重对 **PickPlaceRed** 任务的零样本表现，确定起点

> **核心任务**：详细规约见 [README.md](README.md) 顶部"核心任务定义"。

---

## 代码入口（快速开始）

> Phase 0 主要是**环境搭建 + 上游 LeRobot CLI 验证**，本仓库内没有专门脚本——所有命令来自 LeRobot 官方。

| 想做的事 | 命令 |
|---------|------|
| 安装环境 | `conda create -n py312_cu121 python=3.12 -y` + `pip install lerobot mujoco mink dm_control gymnasium "imageio[ffmpeg]"` |
| 真机连通性自检 | `lerobot-find-port` + `lerobot-calibrate ...` |
| 第一条遥操 demo | `lerobot-teleoperate ...` |
| SmolVLA 零样本评估 | LeRobot 官方 `eval` CLI |

> 本仓库代码从 Phase 1 起开始有自定义实现。Phase 0 完成的标志是上面命令都跑通，可以进入 [phase1-simulation-platform.md](phase1-simulation-platform.md)。

详细技术栈见 [README.md](README.md) §全局技术栈。

---

## 任务清单

### T0.1 硬件装配与电气检查

**目标**：双臂物理装配完成并能上电

**步骤**：
- [ ] 按官方手册装配 leader 与 follower 臂（每条 6 个 STS3215 舵机）
- [ ] 连接 USB 转 TTL 适配器（双臂各一根），确认 `/dev/ttyUSB0` / `/dev/ttyUSB1` 出现
- [ ] 给 `dialout` 组加用户：`sudo usermod -aG dialout $USER` 后重新登录
- [ ] 上电后，用万用表确认每个舵机供电稳定在 7.4V

**参考**：[SO-ARM101 装配手册](https://github.com/TheRobotStudio/SO-ARM100)

**验证**：`ls /dev/ttyUSB*` 输出两个设备节点

---

### T0.2 软件环境搭建

**目标**：Python + CUDA + LeRobot 可用

**步骤**：
- [ ] 安装 CUDA 12.4 + cuDNN 9.x，`nvidia-smi` 正常
- [ ] 创建 conda 环境：`conda create -n py312_cu121 python=3.12 -y`
- [ ] 安装 PyTorch：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`
- [ ] 安装 LeRobot（含 SmolVLA + feetech）：`pip install -e ".[smolvla,feetech]"`

**验证**：`python -c "import torch; print(torch.cuda.is_available())"` → `True`；`python -c "import lerobot"` 正常

---

### T0.3 摄像头接入

**目标**：双路 USB 摄像头可用

**步骤**：
- [ ] 接 wrist 相机 + front 相机
- [ ] `v4l2-ctl --list-devices` 确认设备节点
- [ ] `lerobot-find-cameras opencv` 测试帧率
- [ ] 固定 front 相机位置

**关键文件**：`configs/cameras.yaml` 记录 device_index、分辨率、帧率

**验证**：脚本里能同时读取两路 640×480@30fps 画面

---

### T0.4 SO101 校准与遥操测试

**目标**：leader-follower 主从控制可用

**步骤**：
- [ ] `lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyUSB0`
- [ ] 对 leader 重复（`--teleop.type=so101_leader --teleop.port=/dev/ttyUSB1`）
- [ ] `lerobot-teleoperate --robot.type=so101_follower --teleop.type=so101_leader ...`
- [ ] 手动操作 leader，确认 follower 跟随

**参考**：[LeRobot SO-101 教程](https://huggingface.co/docs/lerobot/so101)

**验证**：6 个关节均跟随，延迟 < 100ms

---

### T0.5 采集 5 条 LeRobot 格式数据

**目标**：拿到第一份 LeRobot 格式数据集

**步骤**：
- [ ] 准备 1 红 cube（3cm）+ 1 plate（直径 12cm）
- [ ] 场景摆放：cube 在 x≈0.18、plate 在 x≈0.28（与仿真 home pose 一致）
- [ ] `lerobot-record --robot.type=so101_follower --teleop.type=so101_leader --dataset.repo_id=local/so101_pickplace_test --dataset.num_episodes=5 --dataset.single_task="put the red cube on the plate"`
- [ ] 每条 demo：俯视顶抓 → 抬起 → 移到 plate 上方 → 释放
- [ ] `lerobot-dataset-viz` 回放验证

**验证**：5 条 episode 均可回放，画面与关节同步，末态 cube 在 plate 上

---

### T0.6 SmolVLA 真机 zero-shot 评估

**目标**：建立 baseline（注：`smolvla_base` 是 SO-100/101 通用 BC 权重，对 PickPlaceRed 多半零样本失败，这步是建数字基线）

**步骤**：
- [ ] `huggingface-cli download lerobot/smolvla_base`
- [ ] 加载到推理脚本，指令 `"put the red cube on the plate"`
- [ ] 跑 20 次试验（不同初始位置），记录：
  - 成功次数
  - 失败模式归类：grasp_fail / lift_drop / place_miss / joint_limit / timeout
- [ ] 录前 5 次屏作基线对比素材

**关键文件**：`eval/phase0_smolvla_baseline.py`、`eval/results/phase0_baseline.md`

**验证**：有数字化的 baseline（X/20）+ 失败模式归类

---

## 验收标准（进入 Phase 1）

- [ ] SO101 leader-follower 遥操稳定运行 ≥ 5 分钟无掉链
- [ ] LeRobot 数据集 ≥ 5 条 episode 可回放
- [ ] SmolVLA 真机 zero-shot 有 20 次试验的量化数据
- [ ] 项目目录结构与 git 仓库就绪

---

## 风险与陷阱

| 风险 | 应对 |
|------|------|
| STS3215 通信不稳定，episode 中断 | 屏蔽线 + 远离 USB 3.0 接口；降 baud rate；leader/follower 分两根 USB |
| 摄像头延迟 > 100ms | 关 OpenCV 自动曝光；降到 480p |
| LeRobot 安装与 CUDA 版本冲突 | 优先固定 PyTorch 版本，再装 LeRobot |
| SmolVLA 真机零样本成功率为 0 | 正常，说明需要微调；记失败模式后进入 Phase 1 |
| 校准时夹爪显示错位 | 重做校准，务必让所有关节到限位 |
