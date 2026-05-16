# 备选平台：Genesis

> **本文档定位**：Genesis 平台的搭建与使用指引。不替换主路径（MuJoCo + Menagerie），但可作为 **GPU 加速数据生成对比实验**、未来可能的主路径候选。

---

## 1. Genesis 是什么

- **发布**：2024 年 12 月，由 CMU / 上海 AI Lab / UMD / MIT / 北大 / Stanford 等多机构联合发起
- **GitHub**：`https://github.com/Genesis-Embodied-AI/Genesis`
- **官网**：`https://genesis-embodied-ai.github.io/`
- **定位**：通用机器人物理仿真平台 + 生成式机器人数据引擎

### 1.1 它为什么 star 这么高

| 特性 | 说明 |
|------|------|
| **极高速度** | 官方宣称单卡 RTX4090 可达 4300 万 FPS（远超 MuJoCo MJX / Isaac Sim） |
| **多物理引擎** | 刚体 / 软体 / 流体 / 颗粒物 统一框架 |
| **可微物理** | 原生支持梯度反传，对策略学习友好 |
| **Pythonic API** | 比 MuJoCo XML / Isaac Sim USD 上手简单 |
| **跨平台 GPU** | 支持 NVIDIA / Apple Silicon / CPU |

### 1.2 诚实评估：对本项目的适用性

| 维度 | 现状（2026 中） |
|------|----------------|
| SO-ARM100/101 原生支持 | ❌ 需要自己导入 URDF |
| 生态成熟度 | ⚠️ 比 MuJoCo / SAPIEN 弱一截，文档还在补 |
| sim2real 公开经验 | ⚠️ 公开 SO-ARM sim2real 案例很少 |
| 数据生成速度 | ⭐⭐⭐⭐⭐ 真比 MuJoCo 快很多 |
| 与 LeRobot 衔接 | ⚠️ 需要自己写 dataset writer |
| 学习价值 | ⭐⭐⭐⭐ 代表未来方向，值得跟进 |

**结论**：作为 **Phase 2 / Phase 6 的对比实验平台**很有价值（验证"GPU 大规模数据生成 → VLA 微调"是否能压过 MuJoCo），但不建议替代 MuJoCo + Menagerie 作为主路径。

---

## 2. 安装

### 2.1 环境要求

- Linux / macOS / Windows（推荐 Linux）
- Python 3.10–3.12
- CUDA 11.8+ 或 12.x（如用 GPU）
- 16GB+ 内存

### 2.2 安装步骤

```bash
# 推荐：新建独立 conda 环境（避免与 LeRobot 主环境冲突）
conda create -n genesis python=3.11 -y
conda activate genesis

# 安装 Genesis（pip 直装）
pip install genesis-world

# 或从源码安装（拿最新功能）
git clone https://github.com/Genesis-Embodied-AI/Genesis
cd Genesis
pip install -e .
```

### 2.3 验证安装

```bash
python -c "import genesis as gs; gs.init(); print('Genesis OK')"
```

第一次运行会下载 cache。

---

## 3. 最小可运行示例

### 3.1 Hello World：让一个机械臂动起来

```python
import genesis as gs

# 1. 初始化（gpu / cpu）
gs.init(backend=gs.gpu)

# 2. 创建场景
scene = gs.Scene(
    show_viewer=True,
    sim_options=gs.options.SimOptions(dt=1/60),
)

# 3. 添加平面 + 机械臂（用 Franka 内置资产先跑通）
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
)

# 4. 编译并运行
scene.build()
for _ in range(1000):
    scene.step()
```

### 3.2 SO-ARM100 接入

Genesis 没有官方 SO-ARM 资产，**需要复用 mujoco_menagerie 的 MJCF**：

```python
import genesis as gs

gs.init(backend=gs.gpu)
scene = gs.Scene(show_viewer=True)

scene.add_entity(gs.morphs.Plane())

# 直接加载 mujoco_menagerie 里的 SO-ARM100 MJCF
so101 = scene.add_entity(
    gs.morphs.MJCF(
        file="assets/menagerie/trs_so_arm100/so_arm100.xml"
    ),
)

scene.build()

# 控制：和 MuJoCo 类似的 joint 接口
import numpy as np
target_qpos = np.zeros(7)   # 6 关节 + 1 夹爪
for step in range(500):
    so101.set_dofs_position(target_qpos)
    scene.step()
```

> **注意**：Genesis 对 MJCF 的支持还在迭代，部分 actuator / equality constraint 可能不被识别。第一次接入要测试通过。

### 3.3 PickCube 简化示例

```python
import genesis as gs
import numpy as np

gs.init(backend=gs.gpu)

scene = gs.Scene(
    show_viewer=False,         # headless 加速
    rigid_options=gs.options.RigidOptions(dt=1/120),
)

# 平面 + 桌子 + 机械臂 + cube
scene.add_entity(gs.morphs.Plane())
so101 = scene.add_entity(gs.morphs.MJCF(file="assets/menagerie/trs_so_arm100/so_arm100.xml"))
cube = scene.add_entity(
    gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.2, 0.0, 0.02)),
    surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0)),
)

# 加相机
cam = scene.add_camera(
    pos=(0.5, 0.0, 0.3),
    lookat=(0.2, 0.0, 0.0),
    fov=45, GUI=False,
)

scene.build()

# 简单脚本策略（伪代码 -- 实际要调好 IK）
for ep in range(100):
    cube.set_pos(np.array([np.random.uniform(0.15, 0.25),
                            np.random.uniform(-0.1, 0.1), 0.02]))
    # ... approach / grasp / lift ...
    rgb = cam.render()[0]
    # 记录到 dataset
```

---

## 4. GPU 并行数据生成（Genesis 的杀手锏）

Genesis 最大卖点是**单卡并行数千个环境**。完整示例：

```python
import genesis as gs

gs.init(backend=gs.gpu)

# 一次开 1024 个并行 env
n_envs = 1024
scene = gs.Scene(show_viewer=False)
scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(gs.morphs.MJCF(file="..."))
cube  = scene.add_entity(gs.morphs.Box(size=(0.04,)*3))

scene.build(n_envs=n_envs, env_spacing=(1.0, 1.0))

# 并行 step：所有 env 同步前进
import torch
target = torch.zeros((n_envs, 7), device="cuda")
for _ in range(1000):
    robot.set_dofs_position(target)
    scene.step()
    # rgb_all = cam.render()  # batched
```

**何时值得用**：
- Phase 2 想生成 100K+ 条数据（MuJoCo 跑要几天，Genesis 可能几小时）
- 对比实验：同任务、MuJoCo 与 Genesis 各生成 10K 条，看哪个训出的 VLA 真机表现更好

---

## 5. 与本项目主路径的衔接

### 5.1 数据格式转换

Genesis 没有原生 LeRobot writer。需要自己写桥接：

```python
# data/converters/genesis_to_lerobot.py
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def genesis_episode_to_lerobot(episode_data, dataset):
    for frame in episode_data:
        dataset.add_frame({
            "observation.images.front": frame["rgb_front"],
            "observation.images.wrist": frame["rgb_wrist"],
            "observation.state":        frame["qpos"].astype("float32"),
            "action":                   frame["action"].astype("float32"),
            "task": frame["instruction"],
        })
    dataset.save_episode()
```

### 5.2 与 Phase 2 集成

把 Genesis 作为 Phase 2 的"对比组"：

```
Phase 2 主流程：MuJoCo + Menagerie → 1000 条 → LeRobot
Phase 2 对比组：Genesis        → 10000 条 → LeRobot
Phase 4：分别微调 VLA → 真机评估 → 看哪个数据更有效
```

---

## 6. 学习资源

| 资源 | URL |
|------|-----|
| GitHub | `https://github.com/Genesis-Embodied-AI/Genesis` |
| 官方文档 | `https://genesis-world.readthedocs.io/` |
| 示例集 | `Genesis/examples/` 目录 |
| 社区论坛 | GitHub Discussions |
| 中文社区 | 知乎、B站搜 "Genesis 仿真" |

---

## 7. 推荐学习路径

```
1 天：跑通官方 Hello World + Franka 示例
2 天：把 mujoco_menagerie 的 SO-ARM100 MJCF 导入 Genesis
2 天：实现 PickCube 简化版（无 grasp 优化）
3 天：开 1024 并行 env 跑数据生成基准
1 天：写 Genesis → LeRobot 数据转换器
```

总计约 1–2 周可达"能用作对比实验"的水平。

---

## 8. 风险与陷阱

| 风险 | 应对 |
|------|------|
| MJCF 兼容性不完整（actuator / sensor 解析失败） | 简化 MJCF；或在 Genesis 里重新定义 actuator |
| 渲染与 MuJoCo 视觉风格差异大 → 训出的策略迁移性差 | 把视觉 DR 调到与主路径一致 |
| 1024 并行很美但内存爆 | 先从 64 并行起步逐步扩 |
| API 在版本间变化（新平台） | 锁定 pip 版本；记录 `requirements.txt` |
| 公开 sim2real 案例少 | 视为实验，结果不及预期不要硬走 |

---

## 9. 决策建议

| 你的情况 | 是否用 Genesis |
|----------|---------------|
| Phase 1–5 主线 | ❌ 用 MuJoCo + Menagerie |
| Phase 2 想生成 50K+ 数据加速实验 | ✅ 值得 1 周尝试 |
| Phase 6 想做"数据生成平台对比"研究 | ✅ 主要对比对象 |
| 你有论文 / 公开课题需求 | ✅ 是个热点，写出来好讲故事 |
| 你只想最快部署到真机 | ❌ 不要分心 |

---

## 10. 实战手册：Genesis GPU 并行生成 1000 条数据 + 与 MuJoCo 速度对比

> **目标**：把 SO-ARM100 PickCube 任务移植到 Genesis，单卡跑 256 并行环境，1 小时内生成 1000 条 LeRobot 数据；同时与主路径 MuJoCo + Menagerie 做速度对比，验证 Genesis 在本项目中的实际价值。

### 10.1 总流程

```
T-C.1 GPU 后端验证 + 性能基线
T-C.2 SO-ARM100 MJCF 接入 Genesis
T-C.3 PickCube 场景搭建（单 env 先跑通）
T-C.4 Genesis 风格的脚本策略
T-C.5 单 env 完整 episode 验证
T-C.6 升级到 256 并行
T-C.7 数据写 LeRobot 格式
T-C.8 与 MuJoCo 速度对比
```

预计时间：1 周（含 MJCF 兼容性排查）

---

### T-C.1 GPU 后端验证 + 性能基线

**目标**：确认 Genesis GPU 后端在你的机器上能跑，并测出基线 FPS

```python
# sim/genesis/00_benchmark.py
import genesis as gs
import time

gs.init(backend=gs.gpu, logging_level="warning")

scene = gs.Scene(
    show_viewer=False,
    rigid_options=gs.options.RigidOptions(dt=1/120),
)
scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
scene.build(n_envs=256)

t0 = time.time()
for _ in range(1000):
    scene.step()
elapsed = time.time() - t0
fps = 256 * 1000 / elapsed
print(f"Genesis @ 256 envs: {fps:,.0f} FPS")
```

**期望输出**：单卡 RTX 4090 应该看到 50K–200K FPS（物理 step，不含渲染）

**步骤**：
- [ ] 跑出上面的数字，与 MuJoCo MJX 同任务对比（MuJoCo 通常 5K–20K FPS）
- [ ] 若 FPS < 10K，检查是否真的在用 GPU：`nvidia-smi` 看 GPU 占用

**验证**：FPS 数字落在合理区间，nvidia-smi 显示 GPU 在工作

---

### T-C.2 SO-ARM100 MJCF 接入 Genesis

**目标**：把 mujoco_menagerie 的 SO-ARM100 MJCF 加载进 Genesis，不报错

```python
# sim/genesis/01_load_so101.py
import genesis as gs

gs.init(backend=gs.gpu)
scene = gs.Scene(show_viewer=True, sim_options=gs.options.SimOptions(dt=1/60))
scene.add_entity(gs.morphs.Plane())

so101 = scene.add_entity(
    gs.morphs.MJCF(file="assets/menagerie/trs_so_arm100/so_arm100.xml"),
)
scene.build()

# 列出关节名（用于后续控制）
print("Joints:", [j.name for j in so101.joints])
print("DOFs:", so101.n_dofs)

import numpy as np
target_q = np.zeros(so101.n_dofs)
for _ in range(500):
    so101.set_dofs_position(target_q)
    scene.step()
```

**常见 MJCF 兼容性问题**：
- `<actuator>` 标签个别属性不支持 → 简化为 position actuator
- `<equality>` 约束不识别 → 删除或自己实现
- `<sensor>` 不被 Genesis 解析 → 注释掉，物理本身不受影响

**修复模板**（建一个 Genesis 友好的副本）：
```bash
cp -r assets/menagerie/trs_so_arm100 assets/menagerie/trs_so_arm100_genesis
# 编辑 assets/menagerie/trs_so_arm100_genesis/so_arm100.xml
# 删除 <sensor>...</sensor> 整个块
# actuator 改为：<position joint="J1" kp="200"/> 风格
```

**验证**：viewer 显示完整 SO-ARM100，关节滑块能动

---

### T-C.3 PickCube 场景搭建（单 env 先跑通）

**目标**：在 Genesis 里搭一个最小 PickCube 场景

```python
# sim/genesis/02_pickcube_scene.py
import genesis as gs
import numpy as np

gs.init(backend=gs.gpu)
scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=1/60))

# 平面
scene.add_entity(gs.morphs.Plane())

# 机械臂
so101 = scene.add_entity(
    gs.morphs.MJCF(file="assets/menagerie/trs_so_arm100_genesis/so_arm100.xml"),
)

# Cube
cube = scene.add_entity(
    gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.2, 0.0, 0.02)),
    surface=gs.surfaces.Default(color=(0.9, 0.1, 0.1)),
)

# 两路相机
cam_front = scene.add_camera(
    pos=(0.5, 0.0, 0.3), lookat=(0.2, 0.0, 0.0),
    res=(320, 240), fov=45, GUI=False,
)
cam_wrist = scene.add_camera(
    pos=(0.0, 0.0, 0.5), lookat=(0.0, 0.0, 0.0),    # 后面要绑到 ee
    res=(320, 240), fov=60, GUI=False,
)

scene.build()
```

**关键文件**：
- `sim/genesis/02_pickcube_scene.py`
- `assets/menagerie/trs_so_arm100_genesis/`（Genesis 友好版 MJCF）

**验证**：scene.build() 不报错，渲染一帧能看到 SO-ARM + cube

---

### T-C.4 Genesis 风格的脚本策略

**目标**：在 Genesis 里实现 4 阶段 PickCube 脚本

Genesis 没有原生 IK 接口（截至 2026 中），通用做法是**自己写简化版差分 IK** 或**借用 mink**。下面给最小可工作版（雅可比伪逆 IK）：

```python
# sim/genesis/policy.py
import torch
import numpy as np

class GenesisPickCubePolicy:
    """雅可比伪逆 IK + 4 阶段脚本。"""

    def __init__(self, robot, cube, ee_link_idx: int):
        self.robot = robot
        self.cube = cube
        self.ee_link_idx = ee_link_idx
        self.phase = "APPROACH"
        self.phase_steps = 0
        self.gripper_pos = 0.04   # open

    def reset(self):
        self.phase = "APPROACH"
        self.phase_steps = 0
        self.gripper_pos = 0.04

    def _ee_pos(self):
        # Genesis 提供 robot.get_links_pos() 接口
        return self.robot.get_links_pos()[self.ee_link_idx].cpu().numpy()

    def _cube_pos(self):
        return self.cube.get_pos().cpu().numpy()

    def _ik_step(self, target_xyz, max_step=0.01):
        """返回单步关节增量"""
        cur_ee = self._ee_pos()
        delta_xyz = target_xyz - cur_ee
        if np.linalg.norm(delta_xyz) > max_step:
            delta_xyz = delta_xyz / np.linalg.norm(delta_xyz) * max_step
        J = self.robot.get_jacobian(link_idx=self.ee_link_idx).cpu().numpy()[:3]
        dq = np.linalg.pinv(J) @ delta_xyz
        return dq

    def __call__(self):
        cube = self._cube_pos()

        if self.phase == "APPROACH":
            goal = cube + np.array([0, 0, 0.05])
            dq = self._ik_step(goal)
            if np.linalg.norm(self._ee_pos() - goal) < 0.01:
                self.phase = "DESCEND"

        elif self.phase == "DESCEND":
            goal = cube + np.array([0, 0, 0.005])
            dq = self._ik_step(goal, max_step=0.005)
            if np.linalg.norm(self._ee_pos() - goal) < 0.005:
                self.phase = "GRASP"

        elif self.phase == "GRASP":
            dq = np.zeros(self.robot.n_dofs - 1)
            self.gripper_pos = 0.0   # close
            self.phase_steps += 1
            if self.phase_steps > 15:
                self.phase = "LIFT"

        elif self.phase == "LIFT":
            goal = cube + np.array([0, 0, 0.10])
            dq = self._ik_step(goal)
            if self._ee_pos()[2] > cube[2] + 0.08:
                self.phase = "DONE"

        else:
            dq = np.zeros(self.robot.n_dofs - 1)

        # 组装完整 action（arm + gripper）
        cur_q = self.robot.get_dofs_position().cpu().numpy()
        new_q = cur_q.copy()
        new_q[:-1] += dq
        new_q[-1] = self.gripper_pos
        return new_q
```

**关键文件**：
- `sim/genesis/policy.py`

**Tip**：实际 Genesis API 在版本间略有变化，把 `get_links_pos / get_jacobian / get_dofs_position / set_dofs_position` 当占位接口，按你装的版本对齐。

**验证**：能创建 policy 对象，调用 `__call__()` 返回合理形状

---

### T-C.5 单 env 完整 episode 验证

**目标**：单环境下跑通 1 条 PickCube，看脚本能否成功

```python
# sim/genesis/03_run_one_episode.py
import genesis as gs
import imageio
import numpy as np
from policy import GenesisPickCubePolicy

gs.init(backend=gs.gpu)
scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=1/60))
scene.add_entity(gs.morphs.Plane())
so101 = scene.add_entity(gs.morphs.MJCF(file="assets/menagerie/trs_so_arm100_genesis/so_arm100.xml"))
cube  = scene.add_entity(gs.morphs.Box(size=(0.04,)*3, pos=(0.2, 0.0, 0.02)),
                          surface=gs.surfaces.Default(color=(0.9,0.1,0.1)))
cam   = scene.add_camera(pos=(0.5, 0.0, 0.3), lookat=(0.2, 0.0, 0.0), res=(320,240), GUI=False)
scene.build()

# 找 ee link index（一次性查表）
ee_idx = next(i for i, l in enumerate(so101.links) if l.name == "Moving_Jaw")
policy = GenesisPickCubePolicy(so101, cube, ee_link_idx=ee_idx)

frames = []
for t in range(400):
    action = policy()
    so101.set_dofs_position(action)
    scene.step()
    frames.append(cam.render()[0])
    if policy.phase == "DONE":
        break

print(f"phase: {policy.phase}, cube_z: {cube.get_pos()[2].item():.3f}")
imageio.mimsave("genesis_episode.mp4", frames, fps=30)
```

**验证**：
- `cube_z > 0.08`（cube 被抬起来）
- `genesis_episode.mp4` 看起来合理

**调试 tips**：
- 抓不住 → 增大 dt（1/120 → 1/240）让物理更稳
- IK 跳变 → max_step 减半
- cube 被推飞 → 增大 cube friction，或减慢 DESCEND 速度

---

### T-C.6 升级到 256 并行

**目标**：把单 env 改成 256 并行 env，所有环境同步跑脚本策略

```python
# sim/genesis/04_run_parallel.py
import genesis as gs
import torch
import numpy as np

N_ENVS = 256

gs.init(backend=gs.gpu)
scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=1/60))
scene.add_entity(gs.morphs.Plane())
so101 = scene.add_entity(gs.morphs.MJCF(file="assets/menagerie/trs_so_arm100_genesis/so_arm100.xml"))
cube  = scene.add_entity(gs.morphs.Box(size=(0.04,)*3),
                          surface=gs.surfaces.Default(color=(0.9,0.1,0.1)))
cam   = scene.add_camera(pos=(0.5, 0.0, 0.3), lookat=(0.2, 0.0, 0.0), res=(320,240), GUI=False)

scene.build(n_envs=N_ENVS, env_spacing=(1.5, 1.5))

# 每个 env 独立 reset cube 位置
cube_x = torch.rand(N_ENVS, device="cuda") * 0.1 + 0.15
cube_y = torch.rand(N_ENVS, device="cuda") * 0.2 - 0.1
cube_z = torch.full((N_ENVS,), 0.02, device="cuda")
cube.set_pos(torch.stack([cube_x, cube_y, cube_z], dim=1))

# 注意：脚本策略在 batched 模式下需要重写为 torch 张量操作
# 这里给最小可工作版骨架
from policy_batched import BatchedPickCubePolicy
policy = BatchedPickCubePolicy(so101, cube, n_envs=N_ENVS, ee_link_idx=...)

records = [[] for _ in range(N_ENVS)]
for t in range(400):
    action = policy()                 # (N_ENVS, n_dofs)
    so101.set_dofs_position(action)
    scene.step()
    rgbs = cam.render()               # (N_ENVS, H, W, 3)
    for i in range(N_ENVS):
        records[i].append({"rgb": rgbs[i].cpu().numpy(),
                           "qpos": action[i].cpu().numpy()})

# 检查每个 env 是否成功
cube_zs = cube.get_pos()[:, 2].cpu().numpy()
success_mask = cube_zs > 0.08
print(f"success: {success_mask.sum()} / {N_ENVS}")
```

**关键改动**：
- 策略要 batched 化（所有 `np` 操作改 `torch`，所有 IK 操作 batched）
- `set_pos` / `set_dofs_position` 输入维度变 (N_ENVS, ...)
- 状态读取（`get_pos`, `get_dofs_position`）都返回 batched 张量

**关键文件**：
- `sim/genesis/policy_batched.py`：批量化版策略
- `sim/genesis/04_run_parallel.py`

**验证**：256 并行成功率 > 60%；GPU 占用接近满

**Tip**：先 8 envs → 64 → 256 逐步扩，便于定位 OOM 边界

---

### T-C.7 数据写 LeRobot 格式

**目标**：把 Genesis 并行采集的成功 episode 写到 LeRobot

```python
# sim/genesis/05_to_lerobot.py
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

dataset = LeRobotDataset.create(
    repo_id="local/genesis_pickcube_v0",
    fps=30,
    features={
        "observation.images.front": {"dtype": "video", "shape": (240, 320, 3),
                                      "names": ["height","width","channels"]},
        "observation.state":        {"dtype": "float32", "shape": (6,),
                                      "names": [f"q{i}" for i in range(6)]},
        "action":                   {"dtype": "float32", "shape": (6,),
                                      "names": [f"a{i}" for i in range(6)]},
    },
)

# 从 T-C.6 的 records + success_mask 写入
for env_id in range(N_ENVS):
    if not success_mask[env_id]:
        continue
    for frame in records[env_id]:
        dataset.add_frame({
            "observation.images.front": frame["rgb"],
            "observation.state":        frame["qpos"][:6].astype("float32"),
            "action":                   frame["qpos"][:6].astype("float32"),
            "task": "pick up the red cube",
        })
    dataset.save_episode()
```

**验证**：`lerobot-dataset-viz --repo-id local/genesis_pickcube_v0 --episode-index 0` 正常回放

---

### T-C.8 与 MuJoCo 速度对比（实验报告）

**目标**：量化 Genesis 是否真的比主路径快

**对比维度**：

| 指标 | MuJoCo + Menagerie | Genesis | 差异 |
|------|-------------------|---------|------|
| 1000 条 episode 总耗时 | ___ 秒 | ___ 秒 | ___ |
| 平均 episode 长度 | ___ 帧 | ___ 帧 | — |
| 成功率 | ___ % | ___ % | ___ |
| GPU 占用 | ___ % | ___ % | — |
| 数据空间占用 | ___ MB | ___ MB | — |

**步骤**：
- [ ] 用主路径（Phase 2 已实现的 MuJoCo + Menagerie）跑 1000 条 PickCube
- [ ] 用本节 Genesis pipeline 跑 1000 条 PickCube
- [ ] 写 `eval/results/genesis_vs_mujoco.md` 填上表
- [ ] **VLA 微调对比**（终极判别）：两份数据各微调一个 SmolVLA，真机评估

**预期结论**：
- 物理 step 速度 Genesis 应该 5–10× 快
- **但总耗时未必快 5× 倍**——渲染 / 数据写盘 / Python overhead 可能成为瓶颈
- VLA 训练结果是最终判别器：如果两边训出的真机成功率差不多，**Genesis 没有显著优势**；如果 Genesis 数据多 10× 但训出来一样差，说明数据"量大但质量低"

**关键文件**：
- `eval/results/genesis_vs_mujoco.md`

---

### 10.2 常见问题排查

| 症状 | 原因 | 修复 |
|------|------|------|
| MJCF 加载报错"unsupported element" | Genesis 不识别某些 actuator/sensor | 简化 MJCF，删除不必要的标签 |
| 256 并行 OOM | n_envs 太多 / 渲染分辨率太高 | 降到 64 envs；res 改 160x120 |
| GPU 占用 < 30% 但 FPS 不高 | Python 调用 overhead 大 | 用 `torch.no_grad()` + 减少 cpu↔gpu 拷贝 |
| 256 并行成功率比单 env 低很多 | 批量初始化时位置分布偏 | 用 `torch.manual_seed` 固定测试 seed，逐个排查 |
| Genesis API 在不同版本不一致 | 新平台迭代快 | pip 锁版本；用 `gs.__version__` 记录 |
| 雅可比伪逆 IK 在奇异位姿跳变 | 没加 damping | 改用 damped least squares：`(J@J.T + λI)^-1 @ J @ delta` |

---

### 10.3 完成本节后的状态

跑完这一节，你拥有：
- **`sim/genesis/`** 一整套代码：场景搭建 / 单 env / batched 策略 / 数据导出
- **256 并行 PickCube** 可工作，1000 条成功 episode
- **量化的 Genesis vs MuJoCo 对比报告**
- **VLA 训练结果对比**（如已跑完 Phase 4）

**判断后续投入的依据**（基于本节结果）：
- 如果 Genesis 速度 > 5× MuJoCo **且** VLA 成功率持平或更好 → 值得在 Phase 6 全面切换
- 如果 Genesis 速度快但 VLA 成功率明显更差 → 继续主路径，Genesis 仅作研究兴趣
- 如果 Genesis 速度提升不明显（< 2×）→ 不值得继续投入，回到主路径
