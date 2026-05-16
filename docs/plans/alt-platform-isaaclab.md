# 备选平台：Isaac Lab

> **本文档定位**：Isaac Lab 平台的搭建与使用指引。**学习价值高**（工业 / 学界 SOTA 仿真栈），但**安装与上手成本最高**，不建议作为本项目主路径。

---

## 1. Isaac Lab 是什么

- **维护**：NVIDIA
- **GitHub**：`https://github.com/isaac-sim/IsaacLab`
- **官方文档**：`https://isaac-sim.github.io/IsaacLab/`
- **底层**：Isaac Sim（基于 USD 资产 + PhysX 物理）
- **关系图**：

```
Isaac Sim（NVIDIA 通用机器人仿真器，全套 GUI / USD / RTX 渲染）
    │
    ├─ Omniverse Kit（底层引擎）
    │
    └─ Isaac Lab（专为 RL / IL 研究的轻量封装）
            ├─ 任务库
            ├─ RL 训练接口
            └─ GPU 并行环境
```

**关键区别**：Isaac Gym（旧）已弃用，被 Isaac Lab 取代。本文聚焦 Isaac Lab。

### 1.1 核心卖点

| 特性 | 说明 |
|------|------|
| **NVIDIA RTX 实时渲染** | photorealism 业界顶级，对视觉策略训练有优势 |
| **GPU 并行** | PhysX GPU，单卡可跑数千并行环境 |
| **USD 资产** | 工业级 3D 资产格式，可与 Omniverse 生态打通 |
| **任务库** | Manipulation、Locomotion、Navigation 都有 |
| **学界采用率高** | RL / sim2real 论文常用 |

### 1.2 诚实评估：对本项目的适用性

| 维度 | 现状（2026） |
|------|--------------|
| 安装复杂度 | ⚠️⚠️⚠️ 三平台中最高（需 Isaac Sim 4.x + 显卡 RTX 系列） |
| 硬件要求 | 至少 RTX 3060+，推荐 RTX 4080+ / A6000 |
| SO-ARM100/101 原生支持 | ❌ 需 URDF Importer 转 USD |
| GPU 并行 | ⭐⭐⭐⭐⭐ |
| 视觉真实度 | ⭐⭐⭐⭐⭐（RTX 路径追踪） |
| 学习曲线 | ⚠️⚠️⚠️ 比 MuJoCo / ManiSkill 都陡 |
| sim2real 案例 | ⭐⭐⭐⭐⭐ 学界主流 |
| LeRobot 衔接 | ⚠️ 需自写转换器 |
| 与本项目时间预算契合度 | ⚠️ 投入 1–2 周才入门，性价比不高 |

**结论**：**值得学习但不要作为本项目主线**。建议作为 Phase 6 的研究扩展，或在你后续做 RL / 工业项目时再深入。

---

## 2. 安装

### 2.1 环境要求

- **OS**：Ubuntu 22.04（推荐）或 Windows 11
- **GPU**：NVIDIA RTX 30 系及以上（必须 RTX，不支持纯 CUDA 卡）
- **显存**：≥ 8GB，推荐 ≥ 16GB
- **磁盘**：≥ 50GB（Isaac Sim 本身就 20GB+）
- **驱动**：NVIDIA 535+
- **Python**：3.10（Isaac Sim 4.x 内置）

### 2.2 安装步骤（pip 安装，2025 后推荐）

```bash
# 独立 conda 环境
conda create -n isaaclab python=3.10 -y
conda activate isaaclab

# 1. 装 PyTorch（CUDA 12.x）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. 装 Isaac Sim（pip 版，比 Omniverse Launcher 简单）
pip install isaacsim==4.5.0.* --extra-index-url https://pypi.nvidia.com

# 3. 装 Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```

> **遇坑提示**：
> - 首次启动会下载几 GB 模型 cache
> - 若卡在 shader 编译，等 5–10 分钟
> - 必须在 X11 桌面环境运行（headless 模式需要额外配置）

### 2.3 验证

```bash
./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py
```

弹出 Isaac Sim 窗口看到空场景 → 成功。

---

## 3. 最小可运行示例

### 3.1 跑官方 Cartpole 任务

```bash
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
    --task Isaac-Cartpole-Direct-v0 \
    --num_envs 64
```

可以看到 64 个 cartpole 并行学习平衡。

### 3.2 跑 Franka Cabinet（manipulation 基准任务）

```bash
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
    --task Isaac-Franka-Cabinet-Direct-v0 \
    --num_envs 256
```

### 3.3 自定义 PickCube 环境（关键代码骨架）

Isaac Lab 任务定义比 MuJoCo / Genesis 复杂——分两层：

```python
# tasks/pick_cube/pick_cube_env_cfg.py
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import ArticulationCfg, RigidObjectCfg
from omni.isaac.lab.utils import configclass

@configclass
class PickCubeEnvCfg(DirectRLEnvCfg):
    episode_length_s = 10.0
    decimation = 2
    action_space = 7
    observation_space = 20

    # 机器人
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=...,
        actuators={...},
    )
    # 物体
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        spawn=...,
    )
```

```python
# tasks/pick_cube/pick_cube_env.py
from omni.isaac.lab.envs import DirectRLEnv

class PickCubeEnv(DirectRLEnv):
    cfg: PickCubeEnvCfg

    def _reset_idx(self, env_ids):
        # 随机化 cube 位置
        ...

    def _get_observations(self):
        return {"policy": obs_tensor}

    def _get_rewards(self):
        return reward_tensor

    def _apply_action(self):
        self.robot.set_joint_position_target(self.actions)
```

完整骨架见 Isaac Lab `source/extensions/omni.isaac.lab_tasks/` 目录里的任意 task 仿写。

### 3.4 导入 SO-ARM100

```bash
# 1. 用 Isaac Sim URDF Importer 把 SO-ARM100.urdf 转 USD
./isaaclab.sh -p source/standalone/tools/convert_urdf.py \
    --input assets/so_arm100.urdf \
    --output assets/so_arm100.usd

# 2. 在 ArticulationCfg 里引用 USD
```

> **URDF → USD 转换是 Isaac 系列的固定坑**：actuator / mimic joint / collision mesh 经常需要手工修正。

---

## 4. GPU 并行数据生成示例

Isaac Lab 内置数据生成示例（不一定原生支持 LeRobot 格式，需要 hook）：

```bash
# 用 RSL RL 训练 + 同时记录 obs/action
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
    --task Isaac-Lift-Cube-Franka-v0 \
    --num_envs 4096 \
    --headless \
    --enable_cameras

# 训完后用 play.py 跑出 trajectory：
./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    --task Isaac-Lift-Cube-Franka-v0 \
    --checkpoint logs/.../model_xxx.pt
```

要保存为 LeRobot 格式，需要在 `play.py` 里加 hook 把每帧写出。

---

## 5. 与本项目主路径的衔接

### 5.1 不衔接（推荐）

Phase 1–5 完全跳过 Isaac Lab。

### 5.2 弱衔接（Phase 6 研究扩展）

把 Isaac Lab 作为：
- **真实感渲染对比**：同一 SO101 任务，Isaac 渲染 vs MuJoCo 渲染，看 sim2real 差异
- **RL 探索**：若以后想做 RL 而非纯 IL，Isaac Lab 是公认的 RL 平台

### 5.3 数据格式转换思路

```python
# data/converters/isaaclab_to_lerobot.py 骨架
# 在 Isaac Lab play.py 的每个 env step 之后 hook：
def on_step_hook(env, obs, action):
    for env_id in range(env.num_envs):
        buffers[env_id].append({
            "rgb": env.cameras["front"][env_id].rgb,
            "qpos": env.robot.data.joint_pos[env_id].cpu().numpy(),
            "action": action[env_id].cpu().numpy(),
        })
        if env.terminated[env_id] or env.truncated[env_id]:
            flush_to_lerobot(buffers[env_id])
            buffers[env_id] = []
```

---

## 6. 学习资源

| 资源 | URL |
|------|-----|
| Isaac Lab GitHub | `https://github.com/isaac-sim/IsaacLab` |
| Isaac Lab 文档 | `https://isaac-sim.github.io/IsaacLab/` |
| Isaac Sim 文档 | `https://docs.isaacsim.omniverse.nvidia.com/` |
| 教程视频 | NVIDIA Developer Channel（YouTube） |
| 社区论坛 | `https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/` |
| 论文 | `Orbit / Isaac Lab: A Unified and Modular Framework`（ICRA 2024）|

---

## 7. 推荐学习路径

```
2 天：装 Isaac Sim + Isaac Lab，跑通官方 Cartpole / Franka 示例
2 天：研究一个 manipulation 任务的代码（Franka-Cabinet）
3 天：把 SO-ARM100 URDF 转 USD 并加载
4 天：写自定义 PickCube 任务 + 数据导出
3 天：跑 4096 并行 + 与 MuJoCo 数据对比

总计 ≈ 2 周入门
```

---

## 8. 风险与陷阱

| 风险 | 应对 |
|------|------|
| GPU 不是 RTX 系无法跑 | 提前确认；旧 GPU 直接跳过 Isaac Lab |
| URDF → USD 转换中关节失效 | 用 Isaac Sim GUI 手动修正 articulation |
| 安装失败（驱动 / Vulkan 不兼容） | 升级 NVIDIA 驱动到 535+；用 Docker 版 Isaac Sim |
| 学习曲线陡 → 长时间不出成果 | 设硬时间盒：2 周拿不出 demo 就放弃 |
| 文档与代码版本经常不一致 | 锁定 Isaac Sim / Lab 版本，别盲跟 main |
| 真实感渲染慢拖累数据生成 | 训练时关 RTX 路径追踪，用 rasterization |

---

## 9. 决策建议

| 你的情况 | 是否用 Isaac Lab |
|----------|-----------------|
| Phase 1–5 主线 | ❌ 用 MuJoCo + Menagerie |
| GPU 不是 RTX 30+ | ❌ 跑不了 |
| 想学最先进工业仿真栈 | ✅ 但只在 Phase 6 之后 |
| 做 RL 研究而非 VLA / IL | ✅ 首选 |
| 需要 photorealism（视觉 sim2real 突破） | ✅ Isaac 是最佳选项 |
| 只想最快真机部署 | ❌ 完全跳过 |

---

## 10. 三个备选平台对比

下表汇总 Genesis / ManiSkill 3 / Isaac Lab 的取舍，方便你做最终决策：

| 维度 | Genesis | ManiSkill 3 | Isaac Lab |
|------|---------|-------------|-----------|
| GitHub stars（2026 中） | ~30K | ~5K | ~3K |
| 安装难度 | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| GPU 必需 | 推荐 | 推荐 | **必需且必须 RTX** |
| 仿真速度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 视觉真实度 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| SO-ARM 原生 | ❌ | ❌ | ❌ |
| 任务库丰富度 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 公开 demo 数据 | ❌ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 论文采用率 | 上升中 | 高 | 高 |
| 适合本项目 | 实验对比 | 借数据 | 长期研究 |

**最终建议**：本项目主路径不动（MuJoCo + Menagerie）。学有余力时按 **Genesis（快） → ManiSkill 3（数据） → Isaac Lab（深度研究）** 的顺序逐个学。

---

## 11. 实战手册：Isaac Lab 可行性评估 + 最小工作流

> **目标**：用 ~2 周时间评估 Isaac Lab 对本项目的真实价值。**重点不是把所有东西迁过来，而是用最小投入判断是否值得继续投入**。
>
> Isaac Lab 是 5 个备选平台中**安装最重、上手最难、硬件门槛最高**的。这一节遵循 "**早失败、早决断**" 原则：每一步都有"继续 / 放弃"判断点。

### 11.1 总流程

```
T-E.1 硬件 + 驱动检查（决定能否继续）
T-E.2 Isaac Sim 安装（最大坑点）
T-E.3 Isaac Lab 安装 + 跑通官方 Cartpole（最小验证）
T-E.4 跑 Franka manipulation 任务（中等验证）
T-E.5 URDF → USD 导入 SO-ARM100
T-E.6 写最小自定义 PickCube 任务骨架
T-E.7 数据导出适配 LeRobot
T-E.8 最终决策：是否在 Phase 6 继续投入
```

每一步都设了 **GO/STOP 判断点**——卡住超过预算时间就停，回主路径。

---

### T-E.1 硬件 + 驱动检查（决定能否继续）

**目标**：先排除"装不上"的情况

**步骤**：
- [ ] 确认 GPU 型号：`nvidia-smi` 输出里看 GPU 名称
  - 必须是 RTX 系列（RTX 20/30/40/50；Quadro RTX；A 系列）
  - GTX 系列 / T4 等无 RT Core 的卡**直接放弃** Isaac Lab
- [ ] 确认显存：≥ 8GB 才能跑通基础 demo，≥ 16GB 推荐
- [ ] 确认驱动：`nvidia-smi` 输出顶部"Driver Version"≥ 535
  - 版本低就升级：`sudo apt install nvidia-driver-555` 然后重启
- [ ] 确认磁盘：≥ 50GB 可用（Isaac Sim 本体 + cache）

**GO/STOP 判断点**：
- ✅ GPU = RTX + 显存 ≥ 8GB + 驱动 ≥ 535 + 磁盘 ≥ 50GB → 继续 T-E.2
- ❌ 任一不满足 → **直接放弃 Isaac Lab**，本节剩余跳过

**关键文件**：`docs/plans/isaac_lab_check.txt`（记录你的硬件配置）

---

### T-E.2 Isaac Sim 安装

**目标**：装通 Isaac Sim，能启动空场景

**推荐方式**：pip 安装（比 Omniverse Launcher 简单）

```bash
# 独立 conda 环境
conda create -n isaaclab python=3.10 -y
conda activate isaaclab

# 装 PyTorch（CUDA 12.x）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 装 Isaac Sim（pip 版）
pip install isaacsim==4.5.0.* --extra-index-url https://pypi.nvidia.com

# 首次启动（会下载 cache 文件，5–10 分钟）
isaacsim
```

**常见坑**：
- "libGL.so 找不到" → `sudo apt install libgl1 libglib2.0-0`
- shader 编译卡住 → 等 5–10 分钟，第一次正常
- "Vulkan 不支持" → 升级驱动；或装 `vulkan-tools` 验证
- Headless 服务器 → 必须配 virtual display：`Xvfb :99 &; export DISPLAY=:99`

**预算时间**：4 小时

**GO/STOP 判断点**：
- ✅ Isaac Sim GUI 启动看到空场景 → 继续
- ❌ 超过 1 天搞不定 → **放弃**

---

### T-E.3 Isaac Lab 安装 + 跑通官方 Cartpole

**目标**：装通 Isaac Lab 并跑通最小任务

```bash
# 在同一 conda 环境
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install

# 验证安装
./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py
```

**跑 Cartpole**：
```bash
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
    --task Isaac-Cartpole-Direct-v0 \
    --num_envs 64 \
    --headless
```

**期望输出**：1–2 分钟后看到 reward 上升日志

**GO/STOP 判断点**：
- ✅ Cartpole 训练能跑 → 继续
- ❌ 报错"找不到 task" / "USD load failed" → 检查版本（IsaacLab 与 Isaac Sim 必须配套）

---

### T-E.4 跑 Franka manipulation 任务

**目标**：跑通一个 manipulation 任务，确认 Isaac Lab 对你的项目"工程上可达"

```bash
# Franka Lift-Cube（与 PickCube 概念一致）
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
    --task Isaac-Lift-Cube-Franka-v0 \
    --num_envs 256 \
    --headless

# 训完后用 play 看动作
./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    --task Isaac-Lift-Cube-Franka-v0 \
    --num_envs 4 \
    --checkpoint logs/rsl_rl/franka_lift/.../model_xxx.pt
```

**步骤**：
- [ ] 训练 30 分钟看 reward 是否上升
- [ ] play.py 看 Franka 真的抓起 cube
- [ ] 记录吞吐：256 envs 的 FPS（可在 stdout 看）

**GO/STOP 判断点**：
- ✅ Franka 能完成 Lift-Cube → 继续
- ❌ 训练 reward 不动 → 检查 reward 配置；或换 demo 任务

**关键文件**：`eval/results/isaaclab_franka_baseline.md`（记录 FPS / 训练曲线）

---

### T-E.5 URDF → USD 导入 SO-ARM100

**目标**：把 SO-ARM100 装进 Isaac Lab——**最大瓶颈点**

**准备 URDF**：
- 从 `https://github.com/TheRobotStudio/SO-ARM100` 拉原始 URDF
- 或者从 `mujoco_menagerie/trs_so_arm100/` 用工具把 MJCF 转 URDF（mjcf2urdf 风险高，建议拉原始 URDF）

**用 Isaac Sim GUI 转 USD**（最稳）：

1. 启动 Isaac Sim：`isaacsim`
2. 菜单 `Isaac Utils → URDF Importer`
3. 选择 `so_arm100.urdf`
4. 配置：
   - Create Articulation Root: ✅
   - Fix Base Link: ✅
   - Self-Collision: ❌
   - Default Joint Drive Type: position
5. 输出到 `assets/so_arm100/so_arm100.usd`

**或者用 CLI**：
```bash
./isaaclab.sh -p source/standalone/tools/convert_urdf.py \
    --input assets/so_arm100.urdf \
    --output assets/so_arm100/so_arm100.usd \
    --merge-joints \
    --fix-base
```

**验证导入**：
```python
# isaaclab_scripts/inspect_so101.py
from omni.isaac.lab.app import AppLauncher
app = AppLauncher(headless=False).app

import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core import World

world = World()
stage_utils.add_reference_to_stage(
    usd_path="assets/so_arm100/so_arm100.usd",
    prim_path="/World/SO101"
)
world.reset()

# 打印 DoF
view = ArticulationView(prim_paths_expr="/World/SO101", name="so101")
view.initialize(world.physics_sim_view)
print(f"DoF names: {view.dof_names}")
print(f"DoF count: {view.num_dof}")
```

**常见坑**：
- mimic joint（夹爪指尖联动）→ Isaac 不识别，转 USD 后手动改成独立 actuated joint
- collision mesh 缺失 → URDF 里要有 `<collision>` 标签；可用 `simplify_collision` 工具
- DoF count 不对 → URDF 里 fixed joint 也被算了，转换时加 `--merge-joints`

**预算时间**：1–2 天

**GO/STOP 判断点**：
- ✅ SO-ARM100 在 Isaac Sim 中正确显示，DoF = 6（+ 1 夹爪）→ 继续
- ❌ 关节缺失 / 几何错位 → 时间盒 2 天，超时**放弃 Isaac Lab**

---

### T-E.6 写最小自定义 PickCube 任务骨架

**目标**：把 SO-ARM100 USD 接入 Isaac Lab 的任务框架，跑 1 个 episode

```python
# isaaclab_scripts/so101_pickcube/env_cfg.py
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import ArticulationCfg, RigidObjectCfg
from omni.isaac.lab.sim import UsdFileCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.utils import configclass

@configclass
class So101PickCubeEnvCfg(DirectRLEnvCfg):
    episode_length_s = 8.0
    decimation = 2
    action_space = 7
    observation_space = 24

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UsdFileCfg(usd_path="assets/so_arm100/so_arm100.usd"),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["J[1-6]"],
                stiffness=200.0,
                damping=20.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper"],
                stiffness=200.0,
                damping=20.0,
            ),
        },
    )

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        spawn=...,    # 详见 Isaac Lab examples/cube
    )
```

```python
# isaaclab_scripts/so101_pickcube/env.py
from omni.isaac.lab.envs import DirectRLEnv
import torch

class So101PickCubeEnv(DirectRLEnv):
    cfg: So101PickCubeEnvCfg

    def _setup_scene(self):
        # 在 Direct workflow 里手动加 scene 元素
        ...

    def _reset_idx(self, env_ids):
        # 随机化 cube 位置
        cube_pos = torch.zeros((len(env_ids), 3), device=self.device)
        cube_pos[:, 0] = torch.rand(len(env_ids), device=self.device) * 0.1 + 0.15
        cube_pos[:, 1] = (torch.rand(len(env_ids), device=self.device) - 0.5) * 0.2
        cube_pos[:, 2] = 0.02
        self.cube.write_root_pose_to_sim(cube_pos, env_ids=env_ids)

    def _get_observations(self):
        return {"policy": torch.cat([self.robot.data.joint_pos,
                                      self.cube.data.root_pos_w], dim=-1)}

    def _get_rewards(self):
        cube_z = self.cube.data.root_pos_w[:, 2]
        return cube_z - 0.02   # 越高越好

    def _get_dones(self):
        cube_z = self.cube.data.root_pos_w[:, 2]
        success = cube_z > 0.08
        truncated = self.episode_length_buf >= self.max_episode_length
        return success, truncated

    def _apply_action(self):
        self.robot.set_joint_position_target(self.actions)
```

**注册任务**：
```python
# isaaclab_scripts/so101_pickcube/__init__.py
import gymnasium as gym

gym.register(
    id="Isaac-SO101-PickCube-v0",
    entry_point="so101_pickcube.env:So101PickCubeEnv",
    kwargs={"env_cfg_entry_point": "so101_pickcube.env_cfg:So101PickCubeEnvCfg"},
)
```

**步骤**：
- [ ] 把上面三个文件写出来（参考 Isaac Lab `source/extensions/.../tasks/manipulation/lift/` 仿写）
- [ ] 启动：`./isaaclab.sh -p isaaclab_scripts/run.py --task Isaac-SO101-PickCube-v0 --num_envs 8`
- [ ] 看 viewer 里 SO-ARM100 在动 + cube 在不同位置 reset

**关键文件**：
- `isaaclab_scripts/so101_pickcube/env_cfg.py`
- `isaaclab_scripts/so101_pickcube/env.py`
- `isaaclab_scripts/so101_pickcube/__init__.py`

**预算时间**：2–3 天

**GO/STOP 判断点**：
- ✅ 8 envs 同步跑 PickCube reset/step 无报错 → 继续
- ❌ 超 3 天 → 评估 Isaac Lab 的 ROI（投入产出比），考虑放弃

---

### T-E.7 数据导出适配 LeRobot

**目标**：把 Isaac Lab 跑出来的 trajectory 写到 LeRobot

```python
# isaaclab_scripts/data_export.py
"""
在 play.py 的每步加 hook，把 obs/action 写到 LeRobot dataset
"""
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset.create(
    repo_id="local/isaaclab_so101_pickcube_v0",
    fps=30,
    features={
        "observation.images.front": {"dtype": "video", "shape": (240, 320, 3),
                                      "names": ["height","width","channels"]},
        "observation.state":        {"dtype": "float32", "shape": (6,),
                                      "names": [f"q{i}" for i in range(6)]},
        "action":                   {"dtype": "float32", "shape": (7,),
                                      "names": [f"a{i}" for i in range(7)]},
    },
)

class IsaacLabRecorder:
    def __init__(self, env, dataset):
        self.env = env
        self.dataset = dataset
        self.episode_buffers = [[] for _ in range(env.num_envs)]

    def record_step(self, obs, action, done, success):
        for i in range(self.env.num_envs):
            self.episode_buffers[i].append({
                "rgb":   obs["sensor_data"]["front_cam"]["rgb"][i].cpu().numpy(),
                "qpos":  self.env.robot.data.joint_pos[i].cpu().numpy(),
                "action": action[i].cpu().numpy(),
            })
            if done[i]:
                if success[i]:
                    for frame in self.episode_buffers[i]:
                        self.dataset.add_frame({
                            "observation.images.front": frame["rgb"],
                            "observation.state":        frame["qpos"][:6].astype("float32"),
                            "action":                   frame["action"].astype("float32"),
                            "task": "pick up the cube",
                        })
                    self.dataset.save_episode()
                self.episode_buffers[i] = []
```

**关键文件**：`isaaclab_scripts/data_export.py`

**验证**：跑 100 个 episode，生成 LeRobot dataset，可被 `lerobot-dataset-viz` 加载

---

### T-E.8 最终决策：是否在 Phase 6 继续投入

跑完 T-E.1 ~ T-E.7（**或在中间因为 GO/STOP 判断点退出**），写一份决策报告：

```markdown
# Isaac Lab 可行性评估报告

## 硬件 / 驱动
- GPU: RTX 4090 / 24GB
- 驱动: 555.xx
- ⇒ 通过

## 安装 + 基础 demo
- Isaac Sim: 装通，启动正常
- Isaac Lab: 装通
- Cartpole: 通过
- Franka Lift-Cube: 256 envs 跑通，FPS = ___
- ⇒ 通过

## SO-ARM100 移植
- URDF→USD: 成功 / 失败（具体问题）
- 自定义任务: 成功 / 失败
- 投入时间: ___ 小时
- ⇒ 可行 / 不可行

## 数据导出
- LeRobot 兼容性: 通过 / 失败
- 每 1000 条数据耗时: ___ 分钟
- 视觉真实度（vs MuJoCo + Menagerie）: 显著更好 / 略好 / 差不多

## 最终建议
□ Phase 6 全面切到 Isaac Lab（视觉 sim2real 收益明显）
□ Phase 6 部分使用（只用其渲染做数据增广）
□ 不再投入（投入产出比不达预期）

## 备注
（任何额外观察）
```

**关键文件**：`eval/results/isaac_lab_feasibility_report.md`

---

### 11.2 决策树（合并 GO/STOP 点）

```
T-E.1 硬件检查
  ├─ ❌ 不达标 → 整段放弃（节省两周）
  └─ ✅ 继续

T-E.2 Isaac Sim 安装
  ├─ ❌ 装不通 → 放弃
  └─ ✅ 继续

T-E.3/T-E.4 官方任务跑通
  ├─ ❌ 失败 → 放弃
  └─ ✅ 继续

T-E.5 SO-ARM100 USD 移植
  ├─ ❌ 2 天搞不定 → 放弃（这是最高风险点）
  └─ ✅ 继续

T-E.6/T-E.7 自定义任务 + 数据导出
  ├─ ❌ 3 天搞不定 → 评估 ROI，倾向放弃
  └─ ✅ 写决策报告

T-E.8 决策报告
  → 三选一：全面切换 / 部分使用 / 不再投入
```

---

### 11.3 常见问题排查

| 症状 | 原因 | 修复 |
|------|------|------|
| Isaac Sim 启动黑屏 | Vulkan / OpenGL 驱动问题 | 升级 NVIDIA 驱动到 555+，重启 |
| `./isaaclab.sh --install` 卡在编译 | 第一次构建慢 | 等 10–20 分钟，正常 |
| 跑任务时报 OOM | num_envs 太多 | 减半 |
| URDF Importer 弹错 "mimic joint" | mimic 不支持 | URDF 里去掉 mimic，写两个独立 joint |
| 自定义任务运行后 viewer 黑屏 | camera 路径错 | 用 Isaac Sim GUI 调一遍 prim path |
| 数据导出图像质量差 | RTX 路径追踪关了 | 训练用 rasterization，导出最终数据时开 RTX |
| 训练吞吐远低于 Franka 任务 | SO-ARM 物理碰撞配置不当 | 简化 collision mesh；调 solver iters |

---

### 11.4 完成本节后的状态（无论决策结果）

跑完这一节，无论最终是否采用 Isaac Lab，你拥有：
- **量化的可行性评估报告**（不是凭感觉判断）
- **Isaac Lab 工程经验**：可在简历 / 论文里列
- **决策依据**：清楚 Isaac Lab 对本项目的真实价值

**如果决策是"全面切换"**：
- Phase 6 重新写 sim 数据 pipeline
- 利用 RTX 渲染做视觉 sim2real 突破

**如果决策是"部分使用"**：
- 仅用 Isaac Sim 的渲染器（USD scene + RTX）给主路径 MuJoCo 数据做"视觉增强"
- 不用 Isaac Lab 的训练栈

**如果决策是"不再投入"（最可能）**：
- 这一周时间不白费：你已经了解了这个工具栈的能力边界
- 主路径继续走 MuJoCo + Menagerie，无后顾之忧
