# Phase 0：环境与可行性验证

**周期**：1 周
**前置依赖**：SO-ARM101 套件已到货（leader + follower 双臂）、Ubuntu 22.04 主机、NVIDIA GPU（≥12GB 显存推荐）
**目标**：跑通从真机控制到 LeRobot 数据格式的最小闭环；在真机上评估 SmolVLA 预训练权重对 **PickPlaceRed** 任务的零样本表现，确定起点

> **核心任务（贯穿全 Phase）**：桌面上随机摆放 1 个红 cube（3cm）+ 1 个 plate（6cm 半径），机器人需把红 cube 放进 plate 中。详细规约见 [README.md](README.md) 顶部"核心任务定义"一节。

---

## 代码入口（快速开始）

> Phase 0 主要是**环境搭建 + 上游 LeRobot CLI 验证**，本仓库内没有专门脚本——所有命令来自 LeRobot 官方。

| 想做的事 | 命令 | 产出 |
|---------|------|------|
| 安装环境 | `conda create -n py312_cu121 python=3.12 -y && conda activate py312_cu121` + `pip install lerobot mujoco mink dm_control gymnasium "imageio[ffmpeg]"` | conda env |
| 真机连通性自检（识别串口 / 校准 / 抓取） | LeRobot 官方 `python -m lerobot.scripts.control_robot ...`（参数见 LeRobot docs） | 真机响应 + 第一条 demo |
| SmolVLA 零样本评估 | LeRobot 官方 `python -m lerobot.scripts.eval --policy.path lerobot/smolvla --task ...` | success rate 报告 |

> **本仓库代码从 Phase 1 起开始有自定义实现**。Phase 0 完成的标志是上面三条命令都跑通，可以进入 [phase1-simulation-platform.md](phase1-simulation-platform.md)。

---

## 关键技术与工具

| 工具 | 用途 | 获取方式 |
|------|------|---------|
| LeRobot | 真机控制 + 数据采集 + 部署 SDK | `pip install lerobot` 或 `git clone https://github.com/huggingface/lerobot` |
| feetech-servo SDK | STS3215 舵机底层通信 | LeRobot 依赖自动安装 |
| SmolVLA | HuggingFace 轻量 VLA，已为 SO-ARM 系列优化 | `huggingface.co/lerobot/smolvla_base` |
| OpenCV / v4l2 | USB 摄像头 | `apt install v4l-utils` + `pip install opencv-python` |
| Weights & Biases | 训练 / 评估日志（提前注册） | `pip install wandb && wandb login` |

---

## 任务清单

### T0.1 硬件装配与电气检查

**目标**：双臂物理装配完成并能上电

**步骤**：
- [ ] 按官方手册装配 leader 与 follower 臂（每条 6 个 STS3215 舵机）
- [ ] 连接 USB 转 TTL 适配器（双臂各一根），确认 `/dev/ttyUSB0` / `/dev/ttyUSB1` 出现
- [ ] 给 `dialout` 组加用户：`sudo usermod -aG dialout $USER` 后重新登录
- [ ] 上电后，用万用表确认每个舵机供电稳定在 7.4V

**关键文件**：无（物理任务）

**参考**：
- SO-ARM101 装配手册：`https://github.com/TheRobotStudio/SO-ARM100`
- LeRobot 硬件文档：`https://huggingface.co/docs/lerobot/index`

**验证**：`ls /dev/ttyUSB*` 输出两个设备节点

---

### T0.2 软件环境搭建

**目标**：Python + CUDA + LeRobot 可用

**步骤**：
- [ ] 安装 CUDA 12.4 + cuDNN 9.x，`nvidia-smi` 正常显示 GPU
- [ ] 创建 conda 环境：`conda create -n robot python=3.10 -y && conda activate robot`
- [ ] 安装 PyTorch：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`
- [ ] 安装 LeRobot（含 SmolVLA 依赖）：`pip install -e ".[smolvla,feetech]"` （从 `git clone` 后的目录）
- [ ] 验证：`python -c "import torch; print(torch.cuda.is_available())"` 输出 `True`

**关键文件**：
- 新建 `requirements.txt`（项目根目录）记录确切版本

**参考**：
- LeRobot 安装：`https://github.com/huggingface/lerobot#installation`

**验证**：`python -c "import lerobot; print(lerobot.__version__)"` 正常输出

---

### T0.3 摄像头接入

**目标**：双路 USB 摄像头可用并校准

**步骤**：
- [ ] 接入 wrist 相机 + front 相机
- [ ] 用 `v4l2-ctl --list-devices` 确认设备节点
- [ ] 用 LeRobot 自带工具试拍：`lerobot-find-cameras opencv`
- [ ] 固定 front 相机位置（用支架），记录视角
- [ ] 简单内参标定（可选，用 ChArUco 板）

**关键文件**：
- `configs/cameras.yaml`：记录 device_index、分辨率、帧率

**参考**：
- LeRobot cameras 文档：`https://huggingface.co/docs/lerobot/cameras`

**验证**：在脚本里能同时读取两路 640x480@30fps 的画面

---

### T0.4 SO101 机械臂校准与控制测试

**目标**：leader-follower 主从控制可用

**步骤**：
- [ ] 运行 LeRobot 校准脚本：`lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyUSB0`
- [ ] 对 leader 重复校准
- [ ] 运行遥操：`lerobot-teleoperate --robot.type=so101_follower --teleop.type=so101_leader ...`
- [ ] 手动操作 leader，确认 follower 跟随

**关键文件**：
- `configs/so101.yaml`：保存校准结果（LeRobot 默认会写到 `~/.cache/huggingface/lerobot/calibration/`）

**参考**：
- LeRobot SO-101 教程：`https://huggingface.co/docs/lerobot/so101`

**验证**：leader 与 follower 6 个关节均跟随，无明显延迟（<100ms）

---

### T0.5 采集 5 条标准 LeRobot 数据（PickPlaceRed 任务）

**目标**：拿到第一份 LeRobot 格式数据集

**步骤**：
- [ ] 准备桌面 + 1 红 cube（边长 3cm）+ 1 plate（直径 12cm 浅口盘）
- [ ] **场景摆放**：cube 与 plate 分别放在工作区两侧（默认 cube 在 x≈0.18 附近、plate 在 x≈0.28 附近，参考仿真 home pose）
- [ ] 执行 `lerobot-record --robot.type=so101_follower --teleop.type=so101_leader --dataset.repo_id=local/so101_pickplace_test --dataset.num_episodes=5 --dataset.single_task="put the red cube on the plate"`
- [ ] 每条 demo 操作要点：俯视顶抓 cube → 抬起 → 移到 plate 上方 → 释放
- [ ] 检查输出目录结构（episode、camera frames、joint trajectories）
- [ ] 用 `lerobot-dataset-viz` 回放验证

**关键文件**：
- `data/real_demos/so101_pickplace_test/`：原始数据

**参考**：
- LeRobot record 文档

**验证**：
- 5 条 episode 均可回放，画面与关节同步
- 每条 episode 末态：红 cube 在 plate 上

---

### T0.6 SmolVLA 真机 zero-shot 评估

**目标**：知道现有 VLA 在你的 SO101 + PickPlaceRed 任务上的起点能力（注：smolvla_base 是 SO-100/101 上 LeRobot 团队预训练的通用 BC 权重，对 red cube pick-place 多半不能 zero-shot 成功；这一步主要是建立 baseline 数字）

**步骤**：
- [ ] 拉取 SmolVLA 权重：`huggingface-cli download lerobot/smolvla_base`
- [ ] 加载到推理脚本，输入指令 `"put the red cube on the plate"`
- [ ] 场景：红 cube + plate 在工作区（同 T0.5 setup）
- [ ] 跑 20 次试验（不同初始位置），记录：
  - 成功次数（红 cube 进 plate）
  - 失败模式归类：
    - **抓取失败**：抓空 / 推开 / 抓住但抬不起 / 抓取角度错（grasp_fail）
    - **抬起掉落**：抓到但运输中掉了（lift_drop）
    - **放置失败**：放在 plate 外（place_miss / plate_off）
    - **超限**：关节限位 / 超时（joint_limit / timeout）
- [ ] 录屏前 5 次作为基线对比素材

**关键文件**：
- `eval/phase0_smolvla_baseline.py`：评估脚本
- `eval/results/phase0_baseline.md`：结果记录

**参考**：
- SmolVLA 文档：`https://huggingface.co/lerobot/smolvla_base`

**验证**：
- 有数字化的 baseline（成功率 X/20）
- 失败模式有归类记录

---

### T0.7 项目骨架初始化

**目标**：把后续 Phase 需要的目录结构先建出来

**步骤**：
- [ ] 创建目录树（与主 README §8 一致）：
  ```
  assets/  sim/  data/  training/  eval/  deploy/  configs/
  ```
- [ ] 初始化 `pyproject.toml` 或 `setup.py`（用 LeRobot 风格）
- [ ] 加 `.gitignore`（排除 `data/`、`*.pth`、`wandb/`、`__pycache__`、`.venv`）
- [ ] `git init` 并初始 commit

**关键文件**：
- `.gitignore`、`pyproject.toml`、空目录的 `.gitkeep`

**验证**：`git status` 干净，目录结构齐全

---

## 验收标准（全部满足后进入 Phase 1）

- [ ] SO101 leader-follower 遥操可稳定运行 ≥ 5 分钟无掉链
- [ ] LeRobot 格式数据集已采集 ≥ 5 条 episode 并能回放
- [ ] SmolVLA 真机 zero-shot 评估有 20 次试验的量化数据
- [ ] 项目目录结构与 git 仓库就绪

---

## 风险与陷阱

| 风险 | 应对 |
|------|------|
| STS3215 通信不稳定，episode 中断 | 用屏蔽线 + 远离 USB 3.0 接口；降低 baud rate；分两根 USB 线分别接 leader/follower |
| 摄像头延迟过高（>100ms） | 关闭 OpenCV 自动曝光；降到 480p |
| LeRobot 安装时与 CUDA 版本冲突 | 优先固定 PyTorch 版本，再装 LeRobot |
| SmolVLA 真机零样本成功率为 0 | 正常，说明需要微调；记录失败模式即可，不要停在这里调参 |
| 校准时夹爪闭合到位但显示错位 | 重做校准并务必让所有关节到限位 |

---

## 输出物

- 真机可控的 SO101 双臂
- 第一份 LeRobot 数据集（5 条 episode）
- SmolVLA 真机 baseline 报告
- 项目仓库骨架
