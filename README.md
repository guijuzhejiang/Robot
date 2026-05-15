# SO101 + Pi0.5 VLA 微调项目

> **项目定位**：以 SO-ARM101 为入门平台，深入掌握 `仿真数据生成 → VLA 微调 → 真机部署` 全链路；最终把微调后的 VLA 模型部署到机械臂上执行多种语言指令任务。

---

## 1. 项目目标

| 维度 | 现实目标 | 不切实际的目标（避免） |
|------|----------|------------------------|
| 模型 | **微调** Pi0.5 / SmolVLA / OpenVLA | 从零训练 VLA |
| 数据 | 真机少量 demo + 仿真大规模扩增 | 纯仿真零样本到真机 |
| 任务 | 桌面级抓取/放置/分类，语言指令驱动 | 复杂双臂/灵巧手任务 |
| 部署 | 真机 SO101 跑通基础语言指令任务 | 工业级稳定性 |

---

## 2. 关键概念

### 2.1 ACT / Diffusion Policy / VLA 不是一类东西

| 模型 | 类型 | 视觉 | 语言 | 数据需求（典型） |
|------|------|------|------|------------------|
| ACT | 行为克隆 (Transformer) | ✅ | ❌ | 50–200 episodes/任务 |
| Diffusion Policy | 行为克隆 (扩散) | ✅ | ❌ | 几十到几百 episodes |
| **Pi0 / Pi0.5** | VLA | ✅ | ✅ | 微调 50–500 episodes/任务 |
| **SmolVLA** | 轻量 VLA | ✅ | ✅ | 专为 SO-ARM100/101 设计 |
| **OpenVLA / GR00T N1** | VLA | ✅ | ✅ | 微调 200–1000+ episodes |

**结论**：ACT 和 DP 是 baseline，不是 VLA 的"前置阶段"。本项目直接走 VLA 微调路线。

### 2.2 不再从零训 VLA

**唯一现实路径：在已有 VLA 权重上微调。**

---

## 3. 推荐技术路线

```
SO101（真机最小可用）
   ↓
SmolVLA / Pi0.5 真机零样本评估（看现有 VLA 能做什么）
   ↓
真机采集 50–200 条 demo（LeRobot 格式）
   ↓
仿真侧 MimicGen 风格扩增到 5K–50K 条
   ↓
Pi0.5 / SmolVLA 微调（真机 demo + 仿真扩增联合训练）
   ↓
真机部署 + 小规模 co-finetune
```

1. **真机 demo 是源头**，不是仿真先行
2. **仿真用于扩增而非起点**
3. **联合训练（sim + real）是默认设定**

---

## 4. 仿真数据生成平台对比

> 我们最终选 **MuJoCo + Menagerie + LeRobot 格式** 作为主路径，**MimicGen 风格扩增** 作为关键技术。原因见下文。

### 4.1 RoboCasa / MimicGen — 最适合 VLA 数据生成

**核心思想**：拿少量人类演示（10–50 条），按 object-centric 子任务自动分段，在随机化场景中重放扩增到 1000×。

- **MimicGen**（NVIDIA, 2023）：github.com/NVlabs/mimicgen
- **RoboCasa**（NVIDIA, 2024）：github.com/robocasa/robocasa — 大规模厨房场景
- **DexMimicGen**：灵巧手扩展

**生成流程示例**：
```python
# 1. 人类遥操采集少量 source demos（10 条）
collect_demos(env, num=10)

# 2. MimicGen 自动分段（按物体交互边界）
segments = parse_object_centric_subtasks(demos)

# 3. 随机化新场景，把每段重新接续
for new_scene in randomize_object_poses(N=1000):
    new_traj = []
    for seg in segments:
        new_traj += transform_and_replay(seg, new_scene)
    if success(new_traj):
        save_episode(new_traj)
```

**优势**：成功率高、数据多样性强、专为 imitation learning 设计
**劣势**：依赖少量真机/仿真种子 demo；目前主要基于 RoboSuite

### 4.2 LeRobot 自带 sim — 闭环最短

**核心思想**：直接用 HuggingFace LeRobot 的仿真环境采集，输出原生 LeRobot 数据集格式。

- 代码：github.com/huggingface/lerobot
- 相关环境：`gym-aloha`, `gym-pusht`, `gym-xarm`, `gym-lowcostrobot`（含 SO-ARM 模型）

**生成流程示例**：
```python
import gymnasium as gym
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

env = gym.make("gym_lowcostrobot/PickPlaceSO100-v0")
dataset = LeRobotDataset.create("user/so101_pick", fps=30, ...)

for ep in range(1000):
    obs, _ = env.reset()
    while not done:
        action = scripted_policy(obs)  # 或 IK / motion planner
        obs, _, term, trunc, _ = env.step(action)
        dataset.add_frame({"observation.image": obs["pixels"], ...})
    dataset.save_episode()
```

**优势**：与 HuggingFace 训练栈零摩擦衔接，社区有 SO-ARM 资产
**劣势**：仿真环境数量少；扩增能力不如 MimicGen

### 4.3 MuJoCo + Menagerie — 最灵活的底层

**核心思想**：用 MuJoCo 物理引擎 + DeepMind Menagerie 的 SO-ARM100 官方模型，自己写脚本生成。

- Menagerie：github.com/google-deepmind/mujoco_menagerie （含 `trs_so_arm100/`）
- MuJoCo MJX：GPU 并行
- 控制：mink（IK）/ robosuite-style scripted policies

**生成流程示例**：
```python
import mujoco, mujoco.mjx as mjx
import mink  # 微分 IK

model = mujoco.MjModel.from_xml_path("trs_so_arm100/so_arm100.xml")
data = mujoco.MjData(model)

for ep in range(N):
    randomize_scene(model, data)             # domain randomization
    grasp_pose = sample_grasp(target_obj)    # antipodal grasp sampling
    waypoints = plan_motion(grasp_pose)      # IK + RRT/线性插值
    traj = execute_with_recording(waypoints, data)
    if check_success(data):
        export_to_lerobot(traj)
```

**优势**：物理稳定、调试简单、社区资产成熟、sim2real 经验最丰富
**劣势**：需要自己实现采集与扩增逻辑

### 4.4 选型建议

| 阶段 | 推荐 | 理由 |
|------|------|------|
| 起步原型 | **LeRobot + gym-lowcostrobot** | 1 天内打通 dataset → 训练闭环 |
| 主路径 | **MuJoCo + Menagerie + 自写采集器** | SO101 资产官方、灵活 |
| 数据扩增 | **MimicGen 思路移植** | 把少量种子 demo 扩到 10× ~ 100× |

不推荐：
- ManiSkill 3 — GPU 并行强但 SO101 资产需自接，社区案例少
- Isaac Sim 原生 — 学习曲线对本项目过陡
- Genesis — 太新，生态不成熟

---

## 5. Sim2Real 与 Real2Sim 技术

### 5.1 Sim2Real：仿真训练的策略部署到真机

**核心难题**：视觉、动力学、感知噪声三大 gap。

| 技术 | 思路 | 推荐实现 |
|------|------|----------|
| **Domain Randomization (DR)** | 随机化纹理/光照/物理参数，迫使策略对差异鲁棒 | 任何仿真器自带 |
| **Domain Adaptation** | 用对抗或对比学习把 sim/real 特征对齐 | RCAN, RetinaGAN |
| **System Identification** | 测量真机动力学参数后回填仿真 | mujoco MPC + 优化 |
| **Co-training** | 仿真 + 少量真机数据混合训练（VLA 的主流做法） | LeRobot + HF Accelerate |
| **Real2Sim2Real** | 真实场景三维重建到仿真中训练 | RialTo / SplatSim |

**对 VLA 的实践建议**：DR + Co-training 是最划算的组合。视觉 DR 比动力学 DR 更重要（因为 VLA 大头是视觉策略）。

### 5.2 Real2Sim：把真实世界搬进仿真

2024–2026 年的研究热点，核心是 3D Gaussian Splatting / NeRF 重建后接入物理引擎。

| 项目 | 简介 | 代码 |
|------|------|------|
| **RialTo**（MIT, 2024） | 手机扫描真实场景 → 自动建仿真 → 训练 → 部署 | github.com/real-to-sim-to-real/rialto |
| **SplatSim** | 3DGS + 物理碰撞 | github.com/qureshinomaan/SplatSim |
| **PhysGaussian** | 3DGS 加物理 | github.com/XPandora/PhysGaussian |
| **URDFormer** | 单图重建可铰接物体 URDF | github.com/WEIRDLabUW/urdformer |
| **Robocasa-Gen** | 场景生成 + 物体增广 | RoboCasa 子模块 |

**对本项目的实用价值**：第一阶段不必投入；当真机微调遇到瓶颈、需要 sim 数据贴近真实场景时，再用 RialTo 或 SplatSim 把工作台重建到仿真里。

---

## 6. VLA vs 世界模型（2026 现状）

| 维度 | VLA | 世界模型 |
|------|-----|----------|
| 输出 | 直接动作 | 未来观测/状态预测 |
| 部署 | 直接用 | 通常配 planner / actor |
| 成熟度 | 已商用化 | 多数仍是研究阶段 |
| 代表作 | Pi0/Pi0.5, OpenVLA, GR00T N1, SmolVLA | Dreamer V3, GR00T-Dreams, 1X World Model, Genie 2, UniSim |
| 数据效率 | 中 | 高（自监督） |
| 长程任务 | 一般 | 强 |

**2026 年的真实情况**：

1. **生产落地仍是 VLA 主导**。Pi0.5、GR00T N1、SmolVLA 都已在真机上稳定跑任务。
2. **世界模型主要作为辅助**：
   - 作为 VLA 的**数据增广器**（GR00T-Dreams：用世界模型生成想象 trajectory 喂给 VLA）
   - 作为**动作先验**或**长程规划器**（Dreamer 系列）
   - 作为**评估器**（SimplerEnv 思路）
3. **趋势**：VLA + World Model 融合，例如 Pi0.5 内部就有 latent world prediction 的成分。

**对你的建议**：**主线走 VLA 微调，世界模型作为后期优化方向**。Phase 6+ 可以试 GR00T-Dreams 风格的想象数据增广。

---

## 7. 机械臂选型建议

### 7.1 SO-ARM101 的真实定位

- **优点**：~$300–500、HuggingFace 全栈支持、SmolVLA/LeRobot 原生兼容、社区活跃
- **缺点**：STS3215 舵机背隙明显、负载小（<500g）、重复精度仅毫米级、gripper 是简单平行夹爪
- **结论**：**学习和原型验证非常好，但精度任务和长期研究瓶颈很快会出现**

### 7.2 阶梯式推荐

| 档位 | 机械臂 | 价格 | 适用 |
|------|--------|------|------|
| 入门 | **SO-ARM101 / Koch v1.1** | $300–600 | 入门、原型、本项目 Phase 1–3 |
| 进阶 | **AgileX PiPER** | ~$2.5K | 6 DoF、ROS 支持、性价比高 |
| 研究主力 | **WidowX 250s (Trossen)** | ~$5K | BridgeData / ALOHA 同款，论文复现首选 |
| 研究主力 | **UFactory xArm 6/7** | ~$10–15K | 工业级精度、ROS、国内可购 |
| 顶配 | **Franka FR3 / Galaxea R1** | $30K+ | SOTA 论文标准平台 |

### 7.3 给你的具体建议

继续用 SO101 完成本项目（Phase 1–5），打通全链路后，**Phase 6 升级到 WidowX 250s 或 AgileX PiPER**。理由：
- SO101 的精度极限会在做精细放置 / 多步任务时暴露
- WidowX 是 BridgeData、RT-X 大量训练数据的来源，VLA 微调和评估生态最好

---

## 8. 项目结构

```text
Robot/
├── assets/                    # SO101 URDF / MJCF（从 Menagerie 拉取）
│   └── so_arm100/
├── sim/                       # 仿真核心
│   ├── envs/                  # MuJoCo 任务环境
│   ├── scripted_policies/     # IK + grasp sampling
│   ├── randomization/         # 视觉/动力学 DR
│   └── mimicgen_adapter/      # 种子 demo 扩增
├── data/                      # 数据
│   ├── real_demos/            # 真机采集
│   ├── sim_generated/         # 仿真生成
│   └── lerobot/               # 最终训练格式
├── training/                  # 训练
│   ├── pi0_finetune/          # 主路径
│   └── smolvla_finetune/      # baseline
├── eval/                      # 评估
│   ├── sim_eval/              # SimplerEnv 风格
│   └── real_eval/             # 真机评估
├── deploy/                    # 真机部署
└── configs/
```

---

## 9. 开发步骤（按顺序执行）

### Phase 0：环境与可行性验证（1 周）

- [ ] 真机 SO101 装配、运动校验
- [ ] LeRobot SDK 跑通：键盘/leader-follower 遥操采集 5 条 demo
- [ ] 在真机上跑 **SmolVLA 预训练权重**，记录 zero-shot 行为
- **验收**：能用脚本指令让 SO101 按 LeRobot 控制接口动起来，并保存 1 条标准格式 episode

### Phase 1：仿真平台搭建（1–2 周）

- [ ] 从 mujoco_menagerie 导入 SO-ARM100 模型
- [ ] 验证关节控制、夹爪开合、相机渲染（wrist + front）
- [ ] 实现 1 个最小任务 `PickCube`，包含 reset / step / success / DR
- **验收**：在仿真中用脚本策略完成 100 次 PickCube，成功率 > 70%

### Phase 2：自动轨迹生成（2–3 周）

- [ ] 实现 grasp sampling（antipodal 起步即可）
- [ ] 实现 IK + 直线/RRT motion planning
- [ ] 实现 success 自动判定与数据筛选
- [ ] 加入域随机化（光照、纹理、物体姿态、相机视角）
- **验收**：自动生成 1000 条 PickCube 成功 episode，保存为 LeRobot 数据集

### Phase 3：真机 demo 采集与 sim 扩增（2 周）

- [ ] 真机遥操采集 50–100 条 PickCube demo
- [ ] 把真机 demo 作为 MimicGen 种子，在仿真中扩增到 5K–10K 条
- [ ] 加入语言指令标注（"pick the red cube", "grab the cube" 等同义变体 10–20 种）
- **验收**：5K+ 条带语言标签、通过物理校验的 episode

### Phase 4：VLA 微调（2–3 周）

- [ ] 选定模型：**Pi0.5**（首选）或 SmolVLA（更轻）
- [ ] 在 sim 数据上单独 finetune，看 sim 内任务成功率
- [ ] Co-training：sim 大批量 + 真机少量混合
- [ ] 评估：sim 内 success rate + SimplerEnv 风格的 real-evaluable sim eval
- **验收**：sim 内 PickCube 成功率 > 90%，多语言指令变体均能响应

### Phase 5：真机部署与 co-finetune（2–3 周）

- [ ] 接入 LeRobot 真机推理 pipeline
- [ ] 真机 zero-shot 评估，记录失败模式
- [ ] 针对失败模式补采 20–50 条真机 demo
- [ ] 小批量 real co-finetune
- **验收**：真机 PickCube 在 5 种新物体姿态下成功率 > 60%

### Phase 6：扩展任务与高级指令（持续）

- [ ] 多物体场景（pick from clutter）
- [ ] 多步指令（"pick the red cube and place it in the box"）
- [ ] 引入 RialTo / 3DGS 真实场景重建增强数据
- [ ] 引入世界模型（GR00T-Dreams 思路）做想象数据增广

---

## 10. 关键技术决策清单

| 决策 | 选择 | 备选 |
|------|------|------|
| 仿真器 | **MuJoCo + Menagerie** | LeRobot gym, ManiSkill |
| 机器人模型 | trs_so_arm100 | — |
| 数据扩增 | **MimicGen 思路** | 纯 procedural |
| 数据格式 | **LeRobot dataset (parquet)** | HDF5 |
| 训练框架 | **HuggingFace LeRobot** | — |
| 微调目标 | **Pi0.5** | SmolVLA, OpenVLA |
| Sim2Real | **DR + co-training** | adversarial DA |
| 部署 | **LeRobot 真机 SDK** | ROS2 |

---

## 11. 避免的陷阱

1. **不要从零训 VLA** — 算力不够，必须微调
2. **不要纯仿真训完就期望部署** — 必须有真机 co-training
3. **不要在 Phase 1 就追求视觉 photorealism** — 优先确保 trajectory 数量与多样性
4. **不要把 ACT/DP 当 VLA 的前置** — 它们是独立的 baseline
5. **不要忽略 SO101 的精度上限** — 任务设计要符合 ±5mm 精度的现实
6. **不要忽视语言指令多样性** — VLA 的语言泛化能力需要在数据侧充分覆盖

---

## 12. 核心参考资源

**模型与训练**：
- Pi0 / Pi0.5：github.com/Physical-Intelligence/openpi
- SmolVLA：huggingface.co/lerobot/smolvla_base
- LeRobot：github.com/huggingface/lerobot
- OpenVLA：github.com/openvla/openvla
- GR00T N1：github.com/NVIDIA/Isaac-GR00T

**仿真与数据**：
- MuJoCo Menagerie：github.com/google-deepmind/mujoco_menagerie
- MimicGen：github.com/NVlabs/mimicgen
- RoboCasa：github.com/robocasa/robocasa
- gym-lowcostrobot：github.com/perezjln/gym-lowcostrobot

**Real2Sim**：
- SplatSim：github.com/qureshinomaan/SplatSim

**评估**：
- SimplerEnv：github.com/simpler-env/SimplerEnv

---

## 13. 项目核心理念

> **数据驱动 + 模型复用 + sim-real 协同**：
> 不重复造 VLA 轮子；用 SO101 跑通完整链路；让模型最终真正落到机械臂上执行任意语言指令。
