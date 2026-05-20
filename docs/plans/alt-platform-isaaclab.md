# 备选平台：Isaac Lab

> **本文档定位**：Isaac Lab 平台的概念介绍 + GO/STOP 决策框架。**学习价值高**（工业 / 学界 SOTA 仿真栈），**安装与上手成本最高**。
>
> Phase 3 已经走 LeIsaac（IsaacLab 子集，已配置好 SO-101），所以**本项目实际上已经在用 IsaacLab**，只是通过 LeIsaac 这层包装。本文档作为 Isaac Lab 通用学习的参考——如果你以后要做 RL / 工业项目超出 LeIsaac 范畴时再深入。

---

## Isaac Lab 是什么

- **维护**：NVIDIA
- **GitHub**：[isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab)
- **官方文档**：[isaac-sim.github.io/IsaacLab](https://isaac-sim.github.io/IsaacLab/)

```
Isaac Sim（NVIDIA 通用机器人仿真器，GUI / USD / RTX 渲染）
    │
    ├─ Omniverse Kit（底层引擎）
    └─ Isaac Lab（专为 RL / IL 研究的轻量封装）
            ├─ 任务库（Manipulation / Locomotion / Navigation）
            ├─ RL 训练接口
            └─ GPU 并行环境
            └─ LeIsaac（LightwheelAI 在此基础上做的 SO-101 适配）← 本项目走这条
```

**关键区别**：Isaac Gym（旧）已弃用，被 Isaac Lab 取代。

### 核心卖点

| 特性 | 说明 |
|------|------|
| **NVIDIA RTX 实时渲染** | photorealism 业界顶级 |
| **GPU 并行** | PhysX GPU，单卡可跑数千并行环境 |
| **USD 资产** | 工业级 3D 资产格式，可与 Omniverse 生态打通 |
| **任务库** | Manipulation / Locomotion / Navigation 都有 |
| **学界采用率高** | RL / sim2real 论文常用 |

### 对本项目的适用性

| 维度 | 现状 |
|------|------|
| 通过 LeIsaac 的便捷性 | ⭐⭐⭐⭐⭐ Phase 3 已经在用 |
| 直接用裸 Isaac Lab | ⚠️⚠️⚠️ 学习曲线陡，本项目不需要 |
| 安装复杂度 | 通过 LeIsaac 走源码安装最简单（[官方教程](https://lightwheelai.github.io/leisaac/docs/getting_started/installation/)）|
| 硬件要求 | RTX 30+ / ≥16GB 显存 / ≥50GB 磁盘 |

**结论**：
- **如果只做本项目**：用 LeIsaac 即可，不需要专门学裸 Isaac Lab
- **如果要超出 SO-101 范围**（其他机器人 / RL 训练 / 大规模并行）：用本文档下面的"GO/STOP 决策框架"评估

---

## GO/STOP 决策框架（评估是否值得投入 2 周）

> 早失败、早决断。每一步都有判断点，卡住超预算时间就回主路径。

### Step 1：硬件 + 驱动检查（30 分钟，决定能否继续）

```bash
nvidia-smi   # 看 GPU 型号 + 驱动版本
df -h .      # 看磁盘空间
```

| 必要条件 | 阈值 |
|---|---|
| GPU 型号 | **必须 RTX 系列**（RTX 20/30/40/50；Quadro RTX；A 系列）；GTX/T4 无 RT Core 的卡直接放弃 |
| 显存 | ≥ 8GB 跑基础 demo，≥ 16GB 推荐 |
| 驱动 | NVIDIA Driver ≥ 535 |
| 磁盘 | ≥ 50GB |

**GO/STOP**：任一不满足 → **直接放弃** Isaac Lab。

### Step 2：装通 Isaac Sim（≤ 1 天）

**推荐方式**：通过 LeIsaac 源码安装（自动拉 IsaacLab + isaaclab_assets 配套版本）：

```bash
git clone https://github.com/LightwheelAI/leisaac.git ~/workspace/pycharm/leisaac
cd ~/workspace/pycharm/leisaac && pip install -e .
pip install warp-lang  # 必装
```

**常见坑**：
- `libGL.so 找不到` → `sudo apt install libgl1 libglib2.0-0`
- shader 编译卡住 → 等 5–10 分钟，第一次正常
- `Vulkan 不支持` → 升级驱动；或装 `vulkan-tools` 验证
- Headless 服务器 → 必须配 virtual display：`Xvfb :99 &; export DISPLAY=:99`

**GO/STOP**：超过 1 天搞不定 → **放弃**。

### Step 3：跑通 LeIsaac LiftCube（≤ 半天，最小端到端验证）

```bash
cd ~/workspace/pycharm/leisaac
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-SO101-LiftCube-v0 \
    --teleop_device=so101leader --port=/dev/ttyACM0 \
    --num_envs=1 --device=cuda --enable_cameras
```

**GO**：Isaac Sim 窗口弹出，leader 同步 follower → 你的环境已经满足 Phase 3 所需，**到此即可**。

**继续深入裸 Isaac Lab 仅在以下情况有价值**：
- 要做 RL（PPO/SAC）训练而非 IL
- 要在 SO-101 之外的机器人上做 manipulation
- 要打通 Isaac Sim + Omniverse 生态做大规模合成数据

否则直接走 LeIsaac 路径（[phase3](phase3-real-demo-sim-augmentation.md)）。

---

## 与本项目主路径的衔接

| 衔接方式 | 状态 |
|---|---|
| 通过 LeIsaac 用 Isaac Lab 跑 SO-101 数据扩增 | ✅ Phase 3 主路径 |
| 直接用 Isaac Lab 自写任务 + 训 RL | ❌ 超出本项目范围，可作为 Phase 6+ 研究扩展 |
| 用 Isaac Sim RTX 渲染做 sim2real 视觉对齐 | ❌ Phase 3 LeIsaac 已经在做这事 |

---

## 关键参考链接

| 资源 | URL |
|------|-----|
| Isaac Lab GitHub | https://github.com/isaac-sim/IsaacLab |
| Isaac Lab 文档 | https://isaac-sim.github.io/IsaacLab/ |
| Isaac Sim 文档 | https://docs.isaacsim.omniverse.nvidia.com/ |
| Orbit / Isaac Lab 论文（ICRA 2024）| *A Unified and Modular Framework for Robot Learning* |
| LeIsaac（本项目实际用的封装）| https://lightwheelai.github.io/leisaac/ |

---

## 风险与陷阱

| 风险 | 应对 |
|------|------|
| GPU 不是 RTX 系无法跑 | 提前 `nvidia-smi` 确认；旧 GPU 直接放弃 |
| URDF → USD 转换中关节失效 | 用 Isaac Sim GUI 手动修正 articulation；或直接用 LeIsaac 提供好的 SO-101 USD |
| 安装失败（驱动 / Vulkan 不兼容） | 升级 NVIDIA 驱动到 535+；用 Docker 版 Isaac Sim |
| 学习曲线陡 → 长时间不出成果 | 设硬时间盒 2 周；裸 IsaacLab 入门比 LeIsaac 慢 5 倍 |
| 文档与代码版本经常不一致 | 锁定 Isaac Sim / Lab 版本（用 LeIsaac/dependencies 里的版本）|
| 真实感渲染慢拖累数据生成 | 训练时关 RTX 路径追踪，用 rasterization |
