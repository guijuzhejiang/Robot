# 备选平台：RoboCasa / MimicGen

> **本文档定位**：RoboCasa + MimicGen 体系的搭建与使用指引。**核心价值是学习 MimicGen 算法**——本项目 Phase 3 会把它的算法思想移植到 MuJoCo + Menagerie 主路径上，而不是直接拿它训练 SO101。

---

## 1. RoboCasa / MimicGen 是什么

这是一组**互相关联但分属不同项目**的工具，理清关系再说怎么用：

```
        robosuite（UCSD ARISE Lab）
       MuJoCo + Franka/UR5/Sawyer + 任务库
              │
        ┌─────┴─────┐
        │           │
   robomimic    MimicGen（NVIDIA）
   IL 训练 +    数据扩增框架
   demo 格式    （依赖 robosuite）
                    │
              DexMimicGen
              （灵巧手扩展）

   RoboCasa（NVIDIA）
   100+ 厨房场景任务（基于 robosuite + MimicGen）
```

| 组件 | 维护方 | 角色 |
|------|--------|------|
| **robosuite** | ARISE Initiative | 底层物理 + 任务环境 |
| **robomimic** | ARISE Initiative | IL 训练框架 + hdf5 数据格式 |
| **MimicGen** | NVIDIA | **数据扩增核心算法** |
| **DexMimicGen** | NVIDIA | MimicGen 的灵巧手版本 |
| **RoboCasa** | NVIDIA | 大规模厨房任务库（学习参考）|

### 1.1 MimicGen 核心思想（必须理解）

这是本文档**最重要的部分**——本项目 Phase 3 就是把这套思路移植到 SO101 上。

```
输入：10 条人类遥操 source demos
  ↓
按 "物体交互边界" 切成子任务段：
  ├─ approach 段（接近物体，不依赖物体）
  ├─ contact / manipulate 段（与物体强耦合）
  └─ retract 段（离开物体）
  ↓
对新场景（新物体姿态）：
  ├─ approach 段：重新规划（不依赖物体）
  ├─ contact 段：T_new = T_new_object × T_old_object⁻¹ × T_segment
  │              （对物体姿态变化做齐次变换重放）
  └─ retract 段：重新规划
  ↓
物理仿真验证 → 成功的保留，失败的丢弃
  ↓
输出：1000+ 条扩增 demos
```

**为什么这套方法对 VLA 数据生成最有效**：
- 只需 10 条种子，能扩到 1000+
- 物理可行（不是合成图像）
- 自动覆盖物体姿态分布
- 与原 demo 的"风格"接近，VLA 学到的策略迁移性好

### 1.2 诚实评估：对本项目的适用性

| 维度 | 现状（2026） |
|------|--------------|
| 安装难度 | ⭐⭐⭐ 中（robosuite + robomimic + mimicgen 多包配合）|
| SO-ARM 原生支持 | ❌ 默认 Franka / UR5 / Sawyer，需写 robot adapter |
| 数据格式 | robomimic hdf5（**非 LeRobot**），需要转换 |
| MimicGen 算法成熟度 | ⭐⭐⭐⭐⭐ ICRA 2023 起多次工业级验证 |
| RoboCasa 任务库 | ⭐⭐⭐⭐⭐ 100+ 厨房任务，对 SO101 桌面任务**用处不大** |
| 直接训练 SO101 | ⚠️ 不推荐（机器人 + 数据格式双不匹配）|
| **算法借鉴价值** | ⭐⭐⭐⭐⭐ **本项目 Phase 3 必读** |
| sim2real 案例 | ⭐⭐⭐⭐ 多篇顶会 |

**结论**：
- **不作为本项目训练数据生成主平台**（适配 SO-ARM 性价比低）
- **作为 Phase 3 算法移植参考**（必须跑通 demo + 读源码）
- **作为多任务先验数据源**（其公开 demos 可转 LeRobot 用作 Phase 4 辅助）

---

## 2. 安装

### 2.1 环境要求

- Linux / macOS / Windows
- Python 3.9–3.11
- MuJoCo（robosuite 依赖，自动装）
- 推荐独立 conda 环境（避免与 LeRobot 主环境冲突）

### 2.2 安装步骤

```bash
# 独立环境
conda create -n robocasa python=3.10 -y
conda activate robocasa

# 1. robosuite（基础物理 + 任务）
pip install robosuite

# 2. robomimic（数据格式 + IL 训练）
pip install robomimic

# 3. MimicGen（核心 —— 装源码版便于读代码）
git clone https://github.com/NVlabs/mimicgen
cd mimicgen
pip install -e .
cd ..

# 4. RoboCasa（可选，提供厨房任务）
git clone https://github.com/robocasa/robocasa
cd robocasa
pip install -e .
python robocasa/scripts/download_kitchen_assets.py    # 下载场景资产，约 5GB
cd ..

# 5. DexMimicGen（可选，灵巧手扩展）
git clone https://github.com/NVlabs/dexmimicgen
cd dexmimicgen
pip install -e .
```

### 2.3 验证

```bash
# robosuite
python -c "import robosuite as suite; \
  env = suite.make('Lift', robots='Panda', has_renderer=True); \
  env.reset(); print('robosuite OK')"

# MimicGen
python -c "import mimicgen; print('MimicGen', mimicgen.__version__)"
```

---

## 3. 最小可运行示例

### 3.1 robosuite 基础（不涉及扩增）

```python
import robosuite as suite

env = suite.make(
    env_name="Lift",            # 任务名
    robots="Panda",             # 机器人
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
)
env.reset()
for _ in range(200):
    action = env.action_spec[1] * 0   # 零动作
    obs, reward, done, info = env.step(action)
    env.render()
env.close()
```

### 3.2 跑 MimicGen 官方 demo（核心学习路径）

**这是必跑步骤**——理解 MimicGen 流水线最快的方式。

**Step 1：下载官方 source datasets**
```bash
# MimicGen 官方提供的 source demos（人类遥操采集，每任务 10 条）
python mimicgen/scripts/download_datasets.py --tasks stack square threading
# 默认下载到 datasets/source/
```

**Step 2：看任务配置**（理解 subtask 边界怎么定义）
```bash
cat mimicgen/exps/templates/robosuite/stack.json
```
关键字段：
```json
{
  "name": "stack_d0",
  "experiment": {
    "task": {"name": "Stack_D0"},
    "source_dataset_path": "datasets/source/stack.hdf5",
    ...
  },
  "task": {
    "task_spec": {
      "subtask_1": {
        "object_ref": "cubeA",           // 第一段关注的物体
        "subtask_term_signal": "grasp",  // 段终止条件
        "selection_strategy": "random"
      },
      "subtask_2": {
        "object_ref": "cubeB",
        "subtask_term_signal": null
      }
    }
  },
  "obs": { ... },
  "experiment": {"generation": {"num_trials": 1000, ...}}
}
```

**Step 3：生成扩增数据集**
```bash
python mimicgen/scripts/generate_dataset.py \
    --config mimicgen/exps/templates/robosuite/stack.json \
    --num-trajs 1000 \
    --auto-remove-exp
```
约 30 分钟内，从 10 条 source demo 扩到 1000 条新 demo。

**Step 4：可视化结果**
```bash
python mimicgen/scripts/playback_dataset.py \
    --dataset datasets/generated/stack_d0/demo.hdf5 \
    --n 5 --render
```

---

## 4. MimicGen 算法核心代码（Phase 3 移植必读）

### 4.1 关键文件

| 文件 | 角色 |
|------|------|
| `mimicgen/datagen/data_generator.py` | 主入口：`DataGenerator.generate()` |
| `mimicgen/datagen/datagen_utils.py` | object-centric 变换数学 |
| `mimicgen/datagen/selection_strategy.py` | source demo 选择策略 |
| `mimicgen/datagen/waypoint.py` | waypoint 插值与重放 |
| `mimicgen/configs/task_spec.py` | 任务规范数据结构 |
| `mimicgen/env_interfaces/robosuite.py` | 与 robosuite 的桥接（移植参考）|

### 4.2 算法核心流程（伪代码）

```python
# 来自 mimicgen/datagen/data_generator.py 的简化版
class DataGenerator:
    def generate_one_trajectory(self, new_initial_state):
        # 1. 从 source demos 中选一条种子
        src_traj = self.select_source_traj(new_initial_state)

        # 2. 按 subtask spec 切段
        segments = self.parse_subtask_segments(src_traj)

        # 3. 对每段做 object-centric 变换 + 重放
        full_traj = []
        for seg in segments:
            obj_pose_old = src_traj.get_object_pose(seg.object_ref, t=seg.start)
            obj_pose_new = self.env.get_object_pose(seg.object_ref)
            transform = obj_pose_new @ np.linalg.inv(obj_pose_old)

            new_ee_traj = [transform @ ee_pose for ee_pose in seg.ee_traj]
            new_joint_traj = [self.ik(ee) for ee in new_ee_traj]

            # 物理仿真重放
            for q in new_joint_traj:
                obs, _, _, _ = self.env.step(self.controller(q))
                full_traj.append(obs)

        # 4. 检查成功
        if self.env.is_success():
            return full_traj
        else:
            return None
```

### 4.3 移植到本项目 Phase 3 的对应表

| MimicGen 元素 | 本项目对应实现 |
|---------------|----------------|
| `RoboSuite` env | MuJoCo + Menagerie 的 `sim/envs/pick_cube.py` |
| `task_spec.json` | Python 字典写在 `data/mimicgen_adapter/configs.py` |
| `subtask_term_signal: grasp` | 用夹爪闭合点检测 |
| `object_ref` | 用 cube body 名 |
| 数据格式 robomimic hdf5 | LeRobot dataset（直接写） |
| robot Panda | SO-ARM100 |

---

## 5. RoboCasa 厨房任务（次要参考）

RoboCasa 自己也是一个数据扩增 + 任务生成系统，但任务都是厨房场景：

```python
import robocasa
import robosuite as suite

env = suite.make(
    env_name="PnPCounterToCab",     # Pick-and-place from counter to cabinet
    robots="PandaMobile",
    has_renderer=True,
)
env.reset()
# ... 同 robosuite 风格
```

**对本项目的价值**：
- 看它怎么用 MimicGen + 程序化场景生成造任务
- **不直接用其任务**（厨房 + 移动底盘对 SO101 桌面任务无关）

---

## 6. 数据格式转换（robomimic hdf5 → LeRobot）

如果你想把 MimicGen 公开数据集塞进 Phase 4 训练，写一个转换器：

```python
# data/converters/mimicgen_to_lerobot.py
import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def convert(h5_path: str, repo_id: str, task_desc: str):
    f = h5py.File(h5_path, "r")
    demos = f["data"]   # 每个 demo 是 demo_0, demo_1, ...

    # 探测 obs 结构（一次性）
    first_demo = demos[list(demos.keys())[0]]
    img_shape = first_demo["obs/agentview_image"][0].shape
    state_dim = first_demo["obs/robot0_joint_pos"].shape[1]
    action_dim = first_demo["actions"].shape[1]

    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=20,
        features={
            "observation.images.agentview": {
                "dtype": "video", "shape": img_shape,
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32", "shape": (state_dim,),
                "names": [f"q{i}" for i in range(state_dim)],
            },
            "action": {
                "dtype": "float32", "shape": (action_dim,),
                "names": [f"a{i}" for i in range(action_dim)],
            },
        },
    )

    for demo_key in demos.keys():
        demo = demos[demo_key]
        T = demo["actions"].shape[0]
        for t in range(T):
            dataset.add_frame({
                "observation.images.agentview": demo["obs/agentview_image"][t],
                "observation.state": demo["obs/robot0_joint_pos"][t].astype("float32"),
                "action": demo["actions"][t].astype("float32"),
                "task": task_desc,
            })
        dataset.save_episode()

    f.close()

# 用法
convert(
    "datasets/generated/stack_d0/demo.hdf5",
    "local/mimicgen_stack_v0",
    "stack the red cube on the green cube",
)
```

---

## 7. 与本项目主路径的衔接

### 7.1 主线方式（推荐）：算法移植

Phase 3 时**不安装 RoboCasa / MimicGen 跑训练数据**，而是：
1. 跑一次 MimicGen 官方 stack demo（≈ 1 小时）感受流程
2. 通读上面列出的 4 个核心文件（≈ 1 天）
3. 在 `data/mimicgen_adapter/` 重新实现核心算法，输出到 LeRobot 格式（≈ 1 周）

### 7.2 辅助方式：数据转换

下载 MimicGen 几个公开任务的扩增数据集 → 转 LeRobot → 在 Phase 4 微调时作为**辅助多任务数据**（注意机器人是 Franka，仅作先验提取）。

### 7.3 不推荐

- **直接用 RoboCasa 训 SO101** —— 机器人 / 任务 / 渲染都不匹配
- **把 SO-ARM100 全套移植进 robosuite** —— 工作量大，性价比远低于 MuJoCo + Menagerie 主路径

---

## 8. 学习资源

| 资源 | URL |
|------|-----|
| MimicGen GitHub | `https://github.com/NVlabs/mimicgen` |
| MimicGen 论文 | `MimicGen: A Data Generation System for Scalable Robot Learning`（CoRL 2023）|
| MimicGen 文档 | `https://mimicgen.github.io/` |
| DexMimicGen GitHub | `https://github.com/NVlabs/dexmimicgen` |
| RoboCasa GitHub | `https://github.com/robocasa/robocasa` |
| RoboCasa 论文 | `RoboCasa: Large-Scale Simulation of Everyday Tasks`（RSS 2024）|
| robosuite 文档 | `https://robosuite.ai/docs/overview.html` |
| robomimic 文档 | `https://robomimic.github.io/docs/introduction/overview.html` |
| 教程视频 | NVIDIA Robotics Channel（YouTube）搜 "MimicGen" |

---

## 9. 推荐学习路径

```
1 天：装 robosuite + robomimic + mimicgen，跑通官方 stack 任务
1 天：用 playback_dataset.py 看扩增数据，理解 subtask 切分
2 天：精读 data_generator.py 与 datagen_utils.py
1 天：写一个 mimicgen → LeRobot 转换器，转一个公开 dataset
1–2 周（Phase 3）：把算法移植到本项目，输出 LeRobot 格式
```

---

## 10. 风险与陷阱

| 风险 | 应对 |
|------|------|
| robosuite / robomimic / mimicgen 版本错配 | 全部源码安装且 git checkout 到一致的 release tag |
| MimicGen 扩增数据中含失败 episode（is_success=False） | 转换器里只取 `mask/successful_demos` 子集 |
| h5py 兼容性问题（hdf5 1.10 vs 1.12） | 用 conda install -c conda-forge h5py |
| 误以为 RoboCasa 的厨房任务能直接给 SO101 用 | 不能，机器人 + 场景都不匹配 |
| 算法移植时忽略 subtask 边界检测 → 扩增成功率低 | 务必先跑通官方 demo 摸清 subtask_term_signal 含义 |
| MuJoCo 版本差异导致 robosuite 物理不稳 | robosuite 4.x 配 MuJoCo 3.2.x；锁定版本 |

---

## 11. 决策建议

| 你的情况 | 是否用 RoboCasa / MimicGen |
|----------|---------------------------|
| Phase 1–2 仿真主路径 | ❌ 用 MuJoCo + Menagerie |
| Phase 3 数据扩增 | ✅ **必学算法**（跑 demo + 读源码） |
| Phase 4 想加多任务先验 | ✅ 转其公开 dataset 到 LeRobot |
| 想发 imitation learning 论文 | ✅ MimicGen 是必引用 baseline |
| 只想最快真机部署 | ⚠️ 至少跑一次官方 demo，对算法有感觉就行 |
| 想做厨房 / 长程任务 | ✅ RoboCasa 学习参考 |

---

## 12. 实战手册：MimicGen 算法移植到本项目（Phase 3 实现指南）

> **目标**：把 MimicGen 的核心算法移植到 **MuJoCo + Menagerie + LeRobot** 栈，对应 Phase 3 的 T3.3 / T3.4 / T3.5 任务。
> **不依赖** robosuite / robomimic 整个栈，**只借鉴算法**。

### 12.1 移植总流程

```
T-B.1 准备：种子 demo 来源决策（sim-source vs real-source）
T-B.2 物体姿态追踪器（sim 用 MuJoCo state，real 用视觉估计）
T-B.3 子任务分段器（按 grasp 边界切段）
T-B.4 EE 轨迹提取 + object-centric 变换
T-B.5 IK 重放器（mink 求解 + MuJoCo 仿真）
T-B.6 成功判定与失败筛除
T-B.7 整合 augmentation pipeline
T-B.8 跑通最小规模扩增（5 → 50）
T-B.9 大规模扩增（100 → 10K）+ 质量审计
```

预计实现时间：核心算法 3–5 天，调通到大规模 1 周左右。

---

### T-B.1 准备：种子 demo 来源决策

**目标**：先想清楚扩增的"种子"从哪来——这决定后续物体追踪的实现复杂度

**两种种子来源**：

| 来源 | 物体姿态怎么拿 | 难度 | 用途 |
|------|---------------|------|------|
| **Sim-source**（Phase 1/2 脚本策略产出） | 直接读 MuJoCo `data.body("cube").xpos` | ⭐ | 调通算法的最快路径 |
| **Real-source**（Phase 3 真机遥操产出） | 需要视觉追踪（颜色掩膜 → 桌面平面投影） | ⭐⭐⭐⭐ | 最终生产数据 |

**推荐顺序**：先用 sim-source 跑通整套 pipeline（T-B.2–T-B.7），再切换到 real-source（T-B.2 那一步换实现）。

**关键文件**：
- `data/mimicgen_adapter/__init__.py`
- `data/mimicgen_adapter/configs.py`：任务规范字典

**配置示例**（对应 MimicGen 的 task_spec.json）：

```python
# data/mimicgen_adapter/configs.py
PICK_CUBE_SPEC = {
    "task_name": "pick_cube",
    "object_ref": "cube",                # 仿真中 body 名
    "subtasks": [
        {"name": "approach",   "term_signal": "near_object",  "object_centric": False},
        {"name": "grasp",      "term_signal": "gripper_closed", "object_centric": True},
        {"name": "lift",       "term_signal": "object_lifted", "object_centric": True},
    ],
    "success_condition": "object_lifted_height>0.08",
    "generation": {"num_trials": 100, "max_attempts_per_trial": 3},
}
```

**验证**：能 import 配置，无语法错误

---

### T-B.2 物体姿态追踪器

**目标**：给定一帧 episode 数据，返回该帧目标物体在世界系下的 6D 姿态 `T_obj ∈ SE(3)`

#### 12.2.A Sim-source 版（先实现这个）

```python
# data/mimicgen_adapter/object_tracker.py
import numpy as np
import mujoco

class SimObjectTracker:
    """从 MuJoCo state 直接读物体姿态。仅用于 sim-source demos。"""

    def __init__(self, model: mujoco.MjModel, object_body_name: str):
        self.model = model
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body_name)
        if self.body_id < 0:
            raise ValueError(f"body '{object_body_name}' not found")

    def get_pose(self, data: mujoco.MjData) -> np.ndarray:
        """返回 4x4 齐次变换矩阵"""
        T = np.eye(4)
        T[:3, 3] = data.xpos[self.body_id]
        T[:3, :3] = data.xmat[self.body_id].reshape(3, 3)
        return T

    def get_pose_from_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """从存档的 qpos 重建——用于离线分析"""
        data = mujoco.MjData(self.model)
        data.qpos[:] = qpos
        mujoco.mj_forward(self.model, data)
        return self.get_pose(data)
```

#### 12.2.B Real-source 版（先跳过，T-B.7 之后再做）

```python
# data/mimicgen_adapter/object_tracker.py（续）
import cv2

class VisualObjectTracker:
    """
    从前置相机图像估计 cube 的 xy 位置（z 假设在桌面上）。
    用 HSV 颜色掩膜定位 + 已知 camera extrinsic 反投影。
    """

    def __init__(self, color_range_hsv, camera_extrinsic, camera_intrinsic, table_z=0.0):
        self.color_lo, self.color_hi = color_range_hsv
        self.K = camera_intrinsic        # 3x3
        self.T_cam_world = camera_extrinsic   # 4x4: camera→world
        self.table_z = table_z

    def get_pose(self, rgb_image: np.ndarray) -> np.ndarray | None:
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.color_lo, self.color_hi)
        moments = cv2.moments(mask)
        if moments["m00"] < 100:    # 像素太少 → 没检到
            return None
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        # 反投影：从像素射线与桌面 z=table_z 平面求交点
        K_inv = np.linalg.inv(self.K)
        ray_cam = K_inv @ np.array([cx, cy, 1.0])
        ray_world = self.T_cam_world[:3, :3] @ ray_cam
        cam_origin = self.T_cam_world[:3, 3]
        t = (self.table_z - cam_origin[2]) / ray_world[2]
        xy_world = cam_origin + t * ray_world

        T = np.eye(4)
        T[:3, 3] = [xy_world[0], xy_world[1], self.table_z + 0.02]   # cube 半高
        return T
```

**关键文件**：
- `data/mimicgen_adapter/object_tracker.py`

**验证**：
- Sim：用 PickCube env 跑一步，调用 `tracker.get_pose(env.unwrapped.data)` 得到的 xy 与 `cube_pos` 一致
- Real：拿一张含 cube 的真机图像，输出 xy 与手测值误差 < 1cm

---

### T-B.3 子任务分段器

**目标**：把一条 episode 切成 [approach, grasp, lift] 三段

**核心信号**：
- `gripper_closed`：夹爪开度从张开变为闭合的那一帧
- `near_object`：ee 与物体距离首次小于阈值的那一帧
- `object_lifted`：物体高度首次大于阈值的那一帧

```python
# data/mimicgen_adapter/segmenter.py
import numpy as np
from dataclasses import dataclass

@dataclass
class Subtask:
    name: str
    start: int        # frame index inclusive
    end: int          # frame index inclusive
    object_centric: bool
    obj_pose_at_start: np.ndarray | None  # 4x4，object_centric=True 时填

class Segmenter:
    def __init__(self, spec: dict,
                 gripper_close_threshold: float = 0.02,
                 near_object_threshold: float = 0.03,
                 lift_height_threshold: float = 0.05):
        self.spec = spec
        self.gripper_close_thr = gripper_close_threshold
        self.near_obj_thr = near_object_threshold
        self.lift_thr = lift_height_threshold

    def segment(self, episode: dict) -> list[Subtask]:
        """
        episode 字典需要至少包含：
          - ee_pose: (T, 4, 4)
          - gripper_pos: (T,)  夹爪开度，越小越闭
          - object_pose: (T, 4, 4)
        """
        T = len(episode["gripper_pos"])
        ee_pos = episode["ee_pose"][:, :3, 3]
        obj_pos = episode["object_pose"][:, :3, 3]
        gripper = episode["gripper_pos"]

        # 1. 找 near_object：ee 与 obj 距离首次小于阈值
        dist = np.linalg.norm(ee_pos - obj_pos, axis=1)
        near_idx = self._first_below(dist, self.near_obj_thr)

        # 2. 找 grasp：在 near 之后，夹爪首次闭合
        grasp_idx = self._first_below(gripper[near_idx:], self.gripper_close_thr)
        grasp_idx = (near_idx + grasp_idx) if grasp_idx is not None else None

        # 3. 找 lift：grasp 之后，cube 高度首次超过阈值
        if grasp_idx is not None:
            cube_z0 = obj_pos[grasp_idx, 2]
            lift_offsets = obj_pos[grasp_idx:, 2] - cube_z0
            lift_idx = self._first_above(lift_offsets, self.lift_thr)
            lift_idx = (grasp_idx + lift_idx) if lift_idx is not None else None
        else:
            lift_idx = None

        # 边界判错处理
        if near_idx is None or grasp_idx is None or lift_idx is None:
            return []   # 该 demo 无法分段，丢弃

        subtasks = [
            Subtask("approach", 0, grasp_idx - 1, object_centric=False,
                    obj_pose_at_start=episode["object_pose"][0]),
            Subtask("grasp", grasp_idx, lift_idx - 1, object_centric=True,
                    obj_pose_at_start=episode["object_pose"][grasp_idx]),
            Subtask("lift", lift_idx, T - 1, object_centric=True,
                    obj_pose_at_start=episode["object_pose"][lift_idx]),
        ]
        return subtasks

    @staticmethod
    def _first_below(arr: np.ndarray, thr: float) -> int | None:
        idx = np.where(arr < thr)[0]
        return int(idx[0]) if len(idx) > 0 else None

    @staticmethod
    def _first_above(arr: np.ndarray, thr: float) -> int | None:
        idx = np.where(arr > thr)[0]
        return int(idx[0]) if len(idx) > 0 else None
```

**验证步骤**：
- [ ] 用 Phase 1 脚本策略生成 5 条 source episode
- [ ] 调 `segment()` 看 3 个边界 index
- [ ] 可视化：把 episode 视频按 3 段切色，肉眼看分段合理

**Tip**：阈值参数对实际数据敏感，先在 5 条上调通再扩

---

### T-B.4 EE 轨迹提取 + object-centric 变换

**目标**：实现 MimicGen 算法的数学核心 —— 给定新场景物体姿态，把每段 ee 轨迹做齐次变换

```python
# data/mimicgen_adapter/transform.py
import numpy as np
from segmenter import Subtask

def transform_subtask_ee_trajectory(
    ee_traj_src: np.ndarray,        # (N, 4, 4)  source 段的 ee 轨迹
    subtask: Subtask,
    obj_pose_new: np.ndarray,       # (4, 4)    新场景中物体姿态
) -> np.ndarray:
    """
    返回新场景下应该执行的 ee 轨迹 (N, 4, 4)
    """
    if not subtask.object_centric:
        # 非物体相关段（如 approach 前段的"home → 接近"段）→ 不变换
        # 但 approach 段的最后一帧实际是 grasp 起点，所以这里 approach 也用 object-centric
        # 取决于你的设计，这里给最简版：approach 也变换，但用第一帧物体姿态
        T_rel = obj_pose_new @ np.linalg.inv(subtask.obj_pose_at_start)
        return np.array([T_rel @ T for T in ee_traj_src])

    # object-centric 段：把 ee 轨迹整体跟着物体姿态变化做齐次变换
    T_rel = obj_pose_new @ np.linalg.inv(subtask.obj_pose_at_start)
    return np.array([T_rel @ T for T in ee_traj_src])


def extract_ee_pose_from_episode(
    episode: dict,
    model: mujoco.MjModel,
    ee_body_name: str,
) -> np.ndarray:
    """
    从 episode 的 qpos 序列重建 ee 轨迹（4x4 序列）
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
    data = mujoco.MjData(model)
    T_list = []
    for qpos in episode["qpos"]:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        T = np.eye(4)
        T[:3, 3] = data.xpos[body_id]
        T[:3, :3] = data.xmat[body_id].reshape(3, 3)
        T_list.append(T)
    return np.array(T_list)
```

**变换数学解释**（直观版）：
- `T_rel = obj_pose_new @ inv(obj_pose_at_start)` 表示从老物体姿态到新物体姿态的相对变换
- 把它 left-apply 到 ee 轨迹上 = "**让 ee 跟着物体一起搬动**"
- approach 段：ee 接近 → 跟着物体新位置走 ✓
- grasp/lift 段：ee 与物体相对位姿不变 → 直接跟随 ✓

**验证**：
- [ ] 单元测试：`obj_pose_new == obj_pose_at_start` 时，`T_new == T_src`
- [ ] 平移一致：`obj_pose_new` 比 src 平移 +10cm，输出 ee 轨迹也平移 +10cm

---

### T-B.5 IK 重放器（mink + MuJoCo）

**目标**：把变换后的 ee 轨迹 `(N, 4, 4)` 转回关节序列，并在 MuJoCo 中物理执行

```python
# data/mimicgen_adapter/replayer.py
import numpy as np
import mujoco
import mink
from mink import SE3

class Replayer:
    def __init__(self, model: mujoco.MjModel, ee_frame_name: str,
                 step_dt: float = 1/30, max_iters: int = 100):
        self.model = model
        self.ee_frame = ee_frame_name
        self.step_dt = step_dt
        self.max_iters = max_iters

    def replay(
        self,
        new_ee_traj: np.ndarray,      # (N, 4, 4)
        gripper_traj: np.ndarray,     # (N,)  来自 source 段
        init_qpos: np.ndarray,
        env_step_callback,             # 接收 (qpos_target, gripper) → 推进物理仿真
    ) -> dict | None:
        """
        返回 replay 出来的完整 episode dict；失败返回 None
        """
        data = mujoco.MjData(self.model)
        data.qpos[:] = init_qpos
        mujoco.mj_forward(self.model, data)

        config = mink.Configuration(self.model)
        ee_task = mink.FrameTask(
            frame_name=self.ee_frame,
            frame_type="body",
            position_cost=1.0,
            orientation_cost=0.5,
        )
        config.update(data.qpos)

        replayed = {"qpos": [], "ee_pose": [], "gripper": [], "obs": []}
        for t in range(len(new_ee_traj)):
            target_T = new_ee_traj[t]
            ee_task.set_target(SE3.from_matrix(target_T))

            # 迭代 IK
            for _ in range(self.max_iters):
                vel = mink.solve_ik(config, [ee_task], dt=self.step_dt, solver="quadprog")
                config.integrate_inplace(vel, self.step_dt)
                err = ee_task.compute_error(config)
                if np.linalg.norm(err) < 5e-3:
                    break
            else:
                # IK 不收敛 → 重放失败
                return None

            # 写入物理 sim
            q_target = config.q.copy()
            obs = env_step_callback(q_target, gripper_traj[t])
            replayed["qpos"].append(data.qpos.copy())
            replayed["ee_pose"].append(target_T)
            replayed["gripper"].append(gripper_traj[t])
            replayed["obs"].append(obs)

        for k in replayed:
            replayed[k] = np.array(replayed[k])
        return replayed
```

**关键点**：
- `mink.FrameTask` 表示"让 ee_frame 到达目标 SE3"
- `solve_ik` 返回关节速度，`integrate_inplace` 积分得到下一时刻关节位置
- IK 不收敛时**直接放弃这条扩增**，不勉强重放

**验证**：
- [ ] 给 `new_ee_traj == ee_traj_src`，replay 出来的 qpos 与 source 一致（重放原 demo）
- [ ] 给小幅偏移的 new_ee_traj，replay 成功率 > 60%

---

### T-B.6 成功判定与失败筛除

**目标**：自动判断扩增的 episode 是不是真的完成任务

```python
# data/mimicgen_adapter/success.py
import numpy as np

def evaluate_pick_success(replayed_episode: dict, object_tracker, model, init_obj_pose) -> bool:
    """
    PickCube 成功条件：
      - 末态物体高度 > 初始 + 5cm
      - 物体仍被夹爪持有（ee 与物体距离 < 5cm）
      - 整段没有出现关节超限位
    """
    final_qpos = replayed_episode["qpos"][-1]
    final_obj_T = object_tracker.get_pose_from_qpos(final_qpos)
    final_obj_z = final_obj_T[2, 3]
    init_obj_z = init_obj_pose[2, 3]
    if final_obj_z - init_obj_z < 0.05:
        return False

    final_ee_T = replayed_episode["ee_pose"][-1]
    ee_obj_dist = np.linalg.norm(final_ee_T[:3, 3] - final_obj_T[:3, 3])
    if ee_obj_dist > 0.05:
        return False

    # 关节限位检查
    qpos_all = replayed_episode["qpos"]
    jnt_lo = model.jnt_range[:, 0]
    jnt_hi = model.jnt_range[:, 1]
    margin = 0.02
    if np.any(qpos_all < jnt_lo + margin) or np.any(qpos_all > jnt_hi - margin):
        return False

    return True
```

**记录失败原因**（便于调试）：

```python
def evaluate_with_reason(...) -> tuple[bool, str]:
    if final_obj_z - init_obj_z < 0.05:
        return False, "object_not_lifted"
    ...
```

---

### T-B.7 整合 augmentation pipeline

**目标**：把 T-B.2 ~ T-B.6 串成端到端

```python
# data/mimicgen_adapter/augment.py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from segmenter import Segmenter
from transform import transform_subtask_ee_trajectory, extract_ee_pose_from_episode
from replayer import Replayer
from object_tracker import SimObjectTracker
from success import evaluate_with_reason

class MimicGenAugmenter:
    def __init__(self, model, spec):
        self.model = model
        self.spec = spec
        self.tracker = SimObjectTracker(model, spec["object_ref"])
        self.segmenter = Segmenter(spec)
        self.replayer = Replayer(model, ee_frame_name="Moving_Jaw")

    def augment_one(self, source_episode: dict, new_obj_pose: np.ndarray, env) -> dict | None:
        """
        source_episode 字典需要：qpos (T, n_q), gripper_pos (T,), object_pose (T, 4, 4)
        """
        ee_traj_src = extract_ee_pose_from_episode(source_episode, self.model, "Moving_Jaw")
        source_episode["ee_pose"] = ee_traj_src

        subtasks = self.segmenter.segment(source_episode)
        if not subtasks:
            return None

        # reset env 到新场景
        env.reset()
        env.set_object_pose(new_obj_pose)

        replayed_full = {"qpos": [], "ee_pose": [], "gripper": [], "obs": []}
        for subtask in subtasks:
            ee_seg_src = ee_traj_src[subtask.start:subtask.end + 1]
            gripper_seg = source_episode["gripper_pos"][subtask.start:subtask.end + 1]

            ee_seg_new = transform_subtask_ee_trajectory(ee_seg_src, subtask, new_obj_pose)

            result = self.replayer.replay(
                ee_seg_new, gripper_seg,
                init_qpos=env.get_qpos(),
                env_step_callback=env.step_with_target,
            )
            if result is None:
                return None
            for k in replayed_full:
                replayed_full[k].append(result[k])

        for k in replayed_full:
            replayed_full[k] = np.concatenate(replayed_full[k])

        ok, reason = evaluate_with_reason(replayed_full, self.tracker, self.model, new_obj_pose)
        if not ok:
            return None
        replayed_full["instruction"] = source_episode.get("instruction", "pick up the cube")
        return replayed_full


def run_augmentation(
    source_dataset_path: str,
    out_repo_id: str,
    spec: dict,
    n_per_demo: int = 50,
):
    # 1. 加载 source demos
    source_episodes = load_source_episodes(source_dataset_path)

    # 2. 创建 LeRobot writer
    out_dataset = LeRobotDataset.create(repo_id=out_repo_id, fps=30, features={...})

    # 3. 加载仿真 env + augmenter
    env = make_pickcube_env()
    augmenter = MimicGenAugmenter(env.unwrapped.model, spec)

    n_success, n_fail = 0, 0
    failure_log = {}
    for src in source_episodes:
        for trial in range(n_per_demo):
            new_pose = sample_random_object_pose()
            result = augmenter.augment_one(src, new_pose, env)
            if result is None:
                n_fail += 1
                continue
            write_episode_to_lerobot(result, out_dataset)
            n_success += 1
    print(f"Augmentation done: {n_success} success / {n_success+n_fail} total")
```

**关键文件**：
- `data/mimicgen_adapter/augment.py`

---

### T-B.8 跑通最小规模扩增（5 → 50）

**目标**：用 5 条 sim-source demo 扩增到 50 条，端到端验证

**步骤**：
- [ ] 用 Phase 1 的 PickCube 脚本策略生成 5 条 source（保存 qpos / gripper / object_pose）
- [ ] 跑：`python data/mimicgen_adapter/augment.py --source 5_demos.pkl --n-per-demo 10`
- [ ] 检查 `out_dataset`：应该有 30–50 条成功 episode（扩增成功率 60–100%）
- [ ] 可视化检查 5 条扩增结果是否合理

**期望成功率**：60% 以上。低于此说明：
- 阈值（near_object / gripper_close）需要调
- IK 不收敛 → 检查 `Moving_Jaw` body 名 / mink 的 task cost
- 物体随机化范围太大，超出机械臂工作空间

---

### T-B.9 大规模扩增 + 质量审计

**目标**：把 5 条扩增到 500（100×）；100 条扩到 5000；最终配合 Phase 3 真机 demo 扩到目标量

**步骤**：
- [ ] 接入域随机化（Phase 2 的 `sim/randomization/`），每条扩增 episode 独立随机化
- [ ] 接入多语言指令（Phase 2 的 `data/instructions/`）
- [ ] 跑大规模：`python ... --n-per-demo 100`
- [ ] 跑 `eval/audit_dataset.py`（Phase 2 已写）做质量审计

**预期产出**：
- 5K+ 扩增 episode
- audit 通过率 > 95%
- 失败原因分布有日志（`object_not_lifted` / `IK_failed` / `joint_limit` 各占多少）

---

### 12.2 常见问题排查

| 症状 | 原因 | 修复 |
|------|------|------|
| segmenter 返回空列表（无法分段） | 阈值不对 / source demo 本身不规范 | 在 5 条上逐个 print 边界 index，目视调阈值 |
| IK 不收敛率 > 50% | mink 任务权重不对 / 工作空间外采样 | 减小 `orientation_cost`；缩小物体随机化范围 |
| 扩增成功率 < 30% | 物体姿态变换后 ee 撞桌 | approach 段加 z-clamp（强制保持 ≥ 桌面 +2cm） |
| Replay 出来 cube 被推飞 | grasp 段 IK 误差累积 | 减小 IK 收敛阈值；grasp 段 contact 检测后再触发 lift |
| Real-source demo 物体追踪误差大 | HSV 阈值过窄 / camera 标定不准 | 重做 ChArUco 标定；HSV 阈值在 5 张样本上人工调 |
| 不同 source demo 扩增成功率差异大 | source 本身质量参差 | 用"扩增成功率 ≥ 50%"作为 source 质量过滤 |
| 写 LeRobot 时关节维度对不上 | source 与扩增的 qpos 维度不同 | 检查 `model.nq`，确认有没有 free joint 物体污染 qpos |

---

### 12.3 完成本节后的状态

跑完这一节，你拥有：
- **`data/mimicgen_adapter/`** 一整套核心代码（5 个文件）：
  - `object_tracker.py` — sim + visual 双版物体追踪
  - `segmenter.py` — 子任务分段
  - `transform.py` — object-centric 变换数学
  - `replayer.py` — mink IK + MuJoCo 重放
  - `augment.py` — pipeline 整合
- **可输出 LeRobot 格式**的扩增数据集（5 条种子 → 数千条扩增）
- **失败原因日志**，便于持续优化
- **Phase 3 T3.3 / T3.4 / T3.5 全部完成**

**关键差异点（对比 MimicGen 官方实现）**：
| 维度 | MimicGen 官方 | 本项目实现 |
|------|---------------|------------|
| 物理底层 | robosuite | MuJoCo + Menagerie |
| 机器人 | Panda / UR5 | SO-ARM100 |
| IK | robosuite controllers | mink |
| 数据格式 | robomimic hdf5 | LeRobot dataset |
| 物体追踪 | 仿真直读 | sim 直读 + 视觉估计双版 |
| Subtask 配置 | JSON 模板 | Python dict |

**下一步**：把这一节产出的数据与 Phase 3 真机 demo 合并，进入 Phase 4 VLA 微调。
