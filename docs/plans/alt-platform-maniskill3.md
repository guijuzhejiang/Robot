# 备选平台：ManiSkill 3

> **本文档定位**：ManiSkill 3 平台的搭建与使用指引。主要作为**学习参考 + GPU 数据生成对比**，不替换主路径（MuJoCo + Menagerie）。

---

## 1. ManiSkill 3 是什么

- **维护**：UC San Diego HaoSu Lab
- **GitHub**：`https://github.com/haosulab/ManiSkill`
- **官方文档**：`https://maniskill.readthedocs.io/`
- **底层物理**：SAPIEN（自研，GPU 加速）

### 1.1 核心卖点

| 特性 | 说明 |
|------|------|
| **GPU 并行** | SAPIEN 原生 GPU 并行，30000+ FPS 数据生成 |
| **任务库丰富** | 30+ benchmark 任务（PickCube、Stack、Peg-Insert、Mobile Manipulation 等）|
| **多机器人** | Franka、xArm、WidowX、Fetch 等内置；SO-ARM 需导入 |
| **多模态观测** | RGB、深度、点云、segmentation 全支持 |
| **RL + IL 双友好** | gym API + 内置 demo 数据 |
| **Demo 数据集** | 多任务有公开示教数据，方便快速上手 |

### 1.2 诚实评估：对本项目的适用性

| 维度 | 现状（2026） |
|------|--------------|
| SO-ARM100/101 原生支持 | ❌ 需要自己导入 URDF |
| GPU 数据生成 | ⭐⭐⭐⭐⭐ 比 MuJoCo MJX 略快、比 Genesis 慢但稳 |
| 生态成熟度 | ⭐⭐⭐⭐ 比 Genesis 成熟，论文广泛使用 |
| sim2real 案例 | ⭐⭐⭐⭐ 学界有大量 ManiSkill→真机 案例 |
| LeRobot 衔接 | ⚠️ 需自写 dataset writer |
| Demo 数据集可借鉴 | ⭐⭐⭐⭐⭐ 数百 K 条公开 demo 可直接用作 VLA 预训练 |
| 学习价值 | ⭐⭐⭐⭐ 学界主流，benchmark 必备 |

**结论**：作为 **VLA 预训练数据来源**（用其公开 demo 数据集）很有价值；作为本项目的 **SO-ARM 主仿真平台**性价比不高（要花时间适配机器人）。

---

## 2. 安装

### 2.1 环境要求

- Linux（推荐）；Windows / macOS 部分支持
- Python 3.9–3.11
- CUDA 11.8+ 或 12.x
- NVIDIA GPU（GPU 模式必需，CPU 模式可跑但慢）

### 2.2 安装步骤

```bash
# 独立环境
conda create -n maniskill python=3.10 -y
conda activate maniskill

# 安装核心
pip install --upgrade mani-skill

# 下载资产（机器人模型 / 场景）
python -m mani_skill.utils.download_asset all  # 全量
# 或选择性下载：
# python -m mani_skill.utils.download_asset ycb partnet_mobility
```

### 2.3 验证

```bash
python -m mani_skill.examples.demo_random_action -e "PickCube-v1"
```

弹出 viewer 看到 Franka 在动 → 成功。

---

## 3. 最小可运行示例

### 3.1 跑官方 PickCube（基于 Franka）

```python
import gymnasium as gym
import mani_skill.envs  # 注册环境

env = gym.make(
    "PickCube-v1",
    obs_mode="rgbd",          # state / rgbd / pointcloud
    control_mode="pd_joint_delta_pos",
    render_mode="human",
    num_envs=1,
)
obs, info = env.reset(seed=0)
for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated.any() or truncated.any():
        obs, info = env.reset()
env.close()
```

### 3.2 GPU 并行采集（ManiSkill 的杀手锏）

```python
import gymnasium as gym
import mani_skill.envs
import torch

env = gym.make(
    "PickCube-v1",
    num_envs=512,                 # 512 个并行环境
    obs_mode="rgb",
    control_mode="pd_joint_delta_pos",
    sim_backend="gpu",
)
obs, _ = env.reset(seed=0)
for step in range(1000):
    action = torch.randn(env.action_space.shape, device="cuda")
    obs, reward, term, trunc, info = env.step(action)
env.close()
```

单卡 RTX 4090 上约 25K FPS。

### 3.3 导入 SO-ARM100

ManiSkill 的 robot 系统比 MuJoCo 复杂——需要写一个 robot adapter 类：

```python
# robots/so_arm100.py
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.registration import register_agent

@register_agent()
class SoArm100(BaseAgent):
    uid = "so_arm100"
    urdf_path = "assets/menagerie/trs_so_arm100/so_arm100.urdf"

    @property
    def _controller_configs(self):
        from mani_skill.agents.controllers import PDJointPosControllerConfig
        return dict(
            pd_joint_pos=dict(
                arm=PDJointPosControllerConfig(
                    joint_names=["joint_1", "joint_2", "joint_3",
                                 "joint_4", "joint_5", "joint_6"],
                    lower=-3.14, upper=3.14, stiffness=200, damping=20,
                    normalize_action=False,
                ),
                gripper=PDJointPosControllerConfig(
                    joint_names=["gripper"],
                    lower=0.0, upper=0.04, stiffness=200, damping=20,
                ),
            ),
        )
```

然后在任务里用：
```python
env = gym.make("PickCube-v1", robot_uids="so_arm100", ...)
```

> **注意**：mujoco_menagerie 提供 MJCF 不是 URDF。需要从原始 SO-ARM100 仓库拉 URDF，或用 `mjcf_to_urdf` 工具转。

---

## 4. ManiSkill 公开 Demo 数据集（最有用的部分）

ManiSkill 维护一批高质量 demo，对 VLA 预训练有用：

```bash
# 下载某任务的 demo
python -m mani_skill.utils.download_demo "PickCube-v1"
# 默认存到 ~/.maniskill/demos/PickCube-v1/

# 转成 LeRobot 格式（要自己写转换器）
```

**任务库**：`https://maniskill.readthedocs.io/en/latest/tasks/index.html`

**怎么用到本项目**：
- 把多个 ManiSkill 公开 demo 转成 LeRobot 格式
- 在 Phase 4 微调 Pi0.5 时作为**多任务预训练 / 辅助任务数据**
- 不直接用于 SO-ARM 单任务训练（机器人不同）

---

## 5. 与本项目主路径的衔接

### 5.1 数据格式转换

```python
# data/converters/maniskill_to_lerobot.py
import h5py
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def convert_maniskill_demo(h5_path, dataset):
    with h5py.File(h5_path, "r") as f:
        for ep_key in f.keys():
            ep = f[ep_key]
            T = len(ep["actions"])
            for t in range(T):
                dataset.add_frame({
                    "observation.images.front": ep["obs/rgb"][t],
                    "observation.state": ep["obs/qpos"][t].astype("float32"),
                    "action": ep["actions"][t].astype("float32"),
                    "task": ep.attrs.get("task", "unknown"),
                })
            dataset.save_episode()
```

### 5.2 集成到工作流

```
Phase 4 VLA 微调：
  主数据：Phase 3 的 SO101 混合数据（10K 条）
  + 辅助：ManiSkill PickCube + Stack demo 转 LeRobot（数千条 Franka 数据）
  → 期望：多任务先验提升 SO101 任务泛化
```

---

## 6. 学习资源

| 资源 | URL |
|------|-----|
| GitHub | `https://github.com/haosulab/ManiSkill` |
| 文档 | `https://maniskill.readthedocs.io/` |
| 任务列表 | `https://maniskill.readthedocs.io/en/latest/tasks/index.html` |
| Demo dataset | `https://maniskill.readthedocs.io/en/latest/user_guide/demos/index.html` |
| 论文 | `ManiSkill3: GPU Parallelized Robotics Simulation`（ICRA 2025）|

---

## 7. 推荐学习路径

```
1 天：装 ManiSkill + 跑 PickCube 官方示例
1 天：跑 GPU 并行 demo（512 envs）
2 天：下载 + 阅读 3 个任务的 demo 数据集结构
2 天：写 ManiSkill demo → LeRobot 数据格式转换器
3 天（可选）：写 SO-ARM100 robot adapter，跑通 SO-ARM PickCube
```

如不需要把 SO-ARM 移植进去，最后 3 天可以省。

---

## 8. 风险与陷阱

| 风险 | 应对 |
|------|------|
| URDF → SAPIEN 转换中关节失效 | 先用官方 Franka 验证 pipeline，再换 SO-ARM |
| SAPIEN 渲染与 MuJoCo 视觉风格不同 | 不混用训练，要混用就做 colorjitter DR |
| ManiSkill demo 是 Franka 数据，对 SO-ARM 直接迁移会差 | 仅作多任务辅助，主信号靠 SO-ARM 数据 |
| 控制器配置不当导致仿真不稳 | 复用任务模板里的 PD 参数，逐步调 |
| 版本变更（ManiSkill 2 → 3 API 有差异） | 锁定 `mani-skill==X.Y.Z` |

---

## 9. 决策建议

| 你的情况 | 是否用 ManiSkill 3 |
|----------|-------------------|
| Phase 1–5 主线 | ❌ 用 MuJoCo + Menagerie |
| Phase 4 想要更多预训练数据 | ✅ 转其 demo 数据集到 LeRobot |
| 你做 benchmark 论文 / 想报 ManiSkill 任务成绩 | ✅ 必学 |
| 想体验 GPU 并行仿真但 Genesis 太新不放心 | ✅ ManiSkill 更稳 |
| 只想最快真机部署 | ❌ 跳过 |

---

## 10. 实战手册：把 ManiSkill 3 公开 demo 转 LeRobot，用作 VLA 多任务先验

> **目标**：下载 3 个 ManiSkill 3 公开 demo 数据集（PickCube / StackCube / PickClutterYCB），转换成 LeRobot 格式，作为 Phase 4 VLA 微调的辅助多任务先验数据。同时（可选）跑 GPU 并行采集自己的数据做对比。

### 10.1 总流程

```
T-D.1 装通 ManiSkill 3 + 跑通 PickCube
T-D.2 下载 3 个公开 demo 数据集
T-D.3 解析 hdf5 结构（搞清楚字段命名）
T-D.4 写 ManiSkill → LeRobot 转换器
T-D.5 验证转换数据可被 LeRobot 训练脚本加载
T-D.6 （可选）GPU 并行采集自己的数据
T-D.7 数据集在 Phase 4 的接入方式
```

预计时间：核心 T-D.1~T-D.5 约 2–3 天，含可选项 1 周

---

### T-D.1 装通 ManiSkill 3 + 跑通 PickCube

**目标**：确认 ManiSkill 3 在你机器上能跑

```bash
# 独立 conda 环境
conda create -n maniskill python=3.10 -y
conda activate maniskill

pip install --upgrade mani-skill
python -m mani_skill.utils.download_asset ycb partnet_mobility

# 跑官方 demo
python -m mani_skill.examples.demo_random_action -e "PickCube-v1"
```

**验证**：弹出 viewer 看到 Franka 在动 → ManiSkill 装好

**关键文件**：`requirements-maniskill.txt`（记录版本）

---

### T-D.2 下载 3 个公开 demo 数据集

**目标**：拉取至少 3 个任务的官方 demo

```bash
# 创建数据目录
mkdir -p data/maniskill_demos

# 下载 demo（ManiSkill 提供工具）
python -m mani_skill.utils.download_demo "PickCube-v1"       -o data/maniskill_demos/
python -m mani_skill.utils.download_demo "StackCube-v1"      -o data/maniskill_demos/
python -m mani_skill.utils.download_demo "PickClutterYCB-v1" -o data/maniskill_demos/

# 文件结构应该是：
# data/maniskill_demos/
#   PickCube-v1/
#     panda/
#       motionplanning.h5
#       motionplanning.json
#   StackCube-v1/
#     ...
#   PickClutterYCB-v1/
#     ...
```

**关键说明**：
- ManiSkill 提供的 demo 大部分是 **motion-planning 生成的**（不是人类遥操），动作较干净
- 默认机器人是 Franka Panda，与 SO-ARM 不同 → 仅作多任务先验，不直接训单任务策略
- demo 数量级：PickCube 通常 1000 条，PickClutterYCB 可能更多

**验证**：每个任务目录里都有 `.h5` + `.json` 两个文件，h5 大小通常 100MB–2GB

---

### T-D.3 解析 hdf5 结构

**目标**：搞清楚 hdf5 里有哪些字段，对应 LeRobot 哪些 feature

```python
# data/maniskill_to_lerobot/inspect.py
import h5py

def inspect_demo(h5_path):
    with h5py.File(h5_path, "r") as f:
        print(f"=== {h5_path} ===")
        print(f"Top-level keys: {list(f.keys())}")
        # ManiSkill 通常是 data 下面 traj_0, traj_1, ...
        first_traj_key = list(f["traj_0"].keys()) if "traj_0" in f else list(list(f.values())[0].keys())
        traj = f["traj_0"] if "traj_0" in f else list(f.values())[0]
        print(f"\nFirst trajectory keys:")
        def walk(k, v, indent=2):
            if hasattr(v, "shape"):
                print(f"{' '*indent}{k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"{' '*indent}{k}/")
                for k2, v2 in v.items():
                    walk(k2, v2, indent+2)
        for k, v in traj.items():
            walk(k, v)
        print(f"\nNumber of trajectories: {len(f.keys())}")

inspect_demo("data/maniskill_demos/PickCube-v1/panda/motionplanning.h5")
```

**典型输出**（具体可能因版本变化）：
```
First trajectory keys:
  actions: shape=(80, 8), dtype=float32        # 7 joints + 1 gripper
  obs/
    agent/
      qpos: shape=(81, 9), dtype=float32       # 9 = 7 arm + 2 gripper fingers
      qvel: shape=(81, 9), dtype=float32
    extra/
      tcp_pose: shape=(81, 7), dtype=float32   # ee pose (xyz + quat)
      goal_pos: shape=(81, 3), dtype=float32
    sensor_data/
      base_camera/
        rgb: shape=(81, 128, 128, 3), dtype=uint8
      hand_camera/
        rgb: shape=(81, 128, 128, 3), dtype=uint8
  rewards: shape=(80,), dtype=float32
  success: shape=(80,), dtype=bool
  ...
```

**关键文件**：
- `data/maniskill_to_lerobot/inspect.py`
- 把 inspect 输出存到 `data/maniskill_to_lerobot/schema_<task>.txt` 备查

**步骤**：
- [ ] 跑 inspect.py 看 3 个任务的结构
- [ ] 确认哪些字段对 VLA 训练有用（必要：rgb + qpos + actions + task name；可选：tcp_pose, goal_pos）

**验证**：3 个任务的 schema 文件都已生成，字段含义清楚

---

### T-D.4 写 ManiSkill → LeRobot 转换器

**目标**：把 ManiSkill h5 转成 LeRobot dataset

```python
# data/maniskill_to_lerobot/convert.py
import h5py
import json
import numpy as np
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

INSTRUCTION_POOLS = {
    "PickCube-v1": [
        "pick up the red cube",
        "grab the red block",
        "lift the cube",
        "take the red cube",
    ],
    "StackCube-v1": [
        "stack the red cube on the green cube",
        "put the red block on top of the green one",
        "place the red cube onto the green cube",
    ],
    "PickClutterYCB-v1": [
        "pick up the {target}",
        "grab the {target}",
        "lift the {target}",
    ],
}

def convert_task(task_name: str, h5_path: str, repo_id: str,
                  max_episodes: int | None = None):
    f = h5py.File(h5_path, "r")
    json_meta = json.load(open(h5_path.replace(".h5", ".json")))

    # 探测维度（用第一条 demo）
    first_key = list(f.keys())[0]
    first = f[first_key]
    rgb_shape = first["obs/sensor_data/base_camera/rgb"][0].shape
    state_dim = first["obs/agent/qpos"].shape[1]
    action_dim = first["actions"].shape[1]

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=20,
        features={
            "observation.images.base":  {"dtype": "video", "shape": rgb_shape,
                                          "names": ["height", "width", "channels"]},
            "observation.images.hand":  {"dtype": "video", "shape": rgb_shape,
                                          "names": ["height", "width", "channels"]},
            "observation.state":        {"dtype": "float32", "shape": (state_dim,),
                                          "names": [f"q{i}" for i in range(state_dim)]},
            "action":                   {"dtype": "float32", "shape": (action_dim,),
                                          "names": [f"a{i}" for i in range(action_dim)]},
        },
    )

    instructions = INSTRUCTION_POOLS.get(task_name, [task_name])
    rng = np.random.default_rng(42)

    n_written = 0
    for traj_key in f.keys():
        if max_episodes and n_written >= max_episodes:
            break
        traj = f[traj_key]
        # 仅取成功 episode
        if not bool(traj["success"][-1] if "success" in traj else True):
            continue

        T = traj["actions"].shape[0]
        instr = rng.choice(instructions)
        for t in range(T):
            dataset.add_frame({
                "observation.images.base":  traj["obs/sensor_data/base_camera/rgb"][t],
                "observation.images.hand":  traj["obs/sensor_data/hand_camera/rgb"][t],
                "observation.state":        traj["obs/agent/qpos"][t].astype("float32"),
                "action":                   traj["actions"][t].astype("float32"),
                "task": instr,
            })
        dataset.save_episode()
        n_written += 1
    f.close()
    print(f"Converted {n_written} episodes from {task_name} → {repo_id}")


if __name__ == "__main__":
    convert_task("PickCube-v1",
                 "data/maniskill_demos/PickCube-v1/panda/motionplanning.h5",
                 "local/maniskill_pickcube_v0")
    convert_task("StackCube-v1",
                 "data/maniskill_demos/StackCube-v1/panda/motionplanning.h5",
                 "local/maniskill_stackcube_v0")
    convert_task("PickClutterYCB-v1",
                 "data/maniskill_demos/PickClutterYCB-v1/panda/motionplanning.h5",
                 "local/maniskill_clutter_v0",
                 max_episodes=500)
```

**关键文件**：
- `data/maniskill_to_lerobot/convert.py`
- `data/maniskill_to_lerobot/INSTRUCTION_POOLS` 配置可独立维护

**步骤**：
- [ ] 运行 `python data/maniskill_to_lerobot/convert.py`
- [ ] 等待（每个任务约 10–30 分钟视数据量）
- [ ] 检查生成的 3 个 LeRobot dataset

**验证**：
- 3 个 dataset 都有 ≥ 500 条 episode
- 总时长合理（PickCube ~80 frames/ep @ 20fps = 4 秒/ep）

---

### T-D.5 验证转换数据可被 LeRobot 训练脚本加载

**目标**：确保下游可消费

```python
# data/maniskill_to_lerobot/verify.py
from lerobot.datasets.lerobot_dataset import LeRobotDataset

for repo_id in ["local/maniskill_pickcube_v0", "local/maniskill_stackcube_v0", "local/maniskill_clutter_v0"]:
    ds = LeRobotDataset(repo_id)
    print(f"\n=== {repo_id} ===")
    print(f"episodes: {ds.num_episodes}")
    print(f"frames:   {ds.num_frames}")
    print(f"avg ep:   {ds.num_frames / ds.num_episodes:.1f}")
    sample = ds[0]
    print(f"sample keys: {list(sample.keys())}")
    print(f"task: {sample['task']}")

    # 可视化（弹出 GUI）
    # 命令行：lerobot-dataset-viz --repo-id {repo_id} --episode-index 0
```

**步骤**：
- [ ] 跑 verify.py 看打印
- [ ] 用 `lerobot-dataset-viz` 抽 3 条 episode 目视检查

**验证**：3 个 dataset 全部可加载，无 schema 错误

---

### T-D.6 （可选）GPU 并行采集自己的数据

**目标**：体验 ManiSkill 3 的 GPU 并行采集，与 Genesis 做横向对比

```python
# sim/maniskill/gpu_collect.py
import gymnasium as gym
import mani_skill.envs
import torch
from tqdm import tqdm

N_ENVS = 256

env = gym.make(
    "PickCube-v1",
    num_envs=N_ENVS,
    obs_mode="rgb",
    control_mode="pd_joint_delta_pos",
    sim_backend="gpu",
)
obs, _ = env.reset(seed=0)

# 用 motion planning 求解器或简单 PD 控制
records = []
for step in tqdm(range(200)):
    # 这里用随机 action 占位，实际应该接入 motion planner
    action = torch.randn(env.action_space.shape, device="cuda") * 0.1
    obs, reward, term, trunc, info = env.step(action)
    records.append({
        "rgb": obs["sensor_data"]["base_camera"]["rgb"].cpu().numpy(),
        "state": obs["agent"]["qpos"].cpu().numpy(),
        "action": action.cpu().numpy(),
    })

# 同 T-D.4 风格写 LeRobot
```

**Tip**：ManiSkill 内置 motion planner（`mani_skill.examples.motionplanning.panda`），可直接调用给 expert action。

**关键文件**：`sim/maniskill/gpu_collect.py`

---

### T-D.7 数据集在 Phase 4 的接入方式

**目标**：把转换好的 3 个 ManiSkill dataset 加进 Phase 4 VLA 训练

**两种接入方式**：

**方式 A：多任务联合训练**（推荐）

```yaml
# training/configs/pi05_so101_multitask.yaml
dataset:
  type: multi
  datasets:
    - repo_id: local/so101_pickcube_mixed_v1     # 主：SO-ARM 数据
      weight: 5.0
    - repo_id: local/maniskill_pickcube_v0       # 辅：Franka PickCube
      weight: 1.0
    - repo_id: local/maniskill_stackcube_v0
      weight: 1.0
    - repo_id: local/maniskill_clutter_v0
      weight: 0.5
```

**方式 B：预训练 → 微调两阶段**

```
Stage 1：在 maniskill_* 三个数据集上预训练 SmolVLA（学多任务先验）
Stage 2：在 so101_pickcube_mixed_v1 上微调（学具体本体）
```

**判别使用哪种**：
- 时间紧 / GPU 显存够：用 A
- 想看 ManiSkill 数据带来的"先验提取"独立效果：用 B
- 写 Phase 4 实验报告时**两种都对比**最好

---

### 10.2 常见问题排查

| 症状 | 原因 | 修复 |
|------|------|------|
| h5 加载报错 "no dataset 'success'" | 不同 ManiSkill 版本 schema 差异 | 自适应：`if "success" in traj else assume_all_success` |
| 转换后图像全黑 | obs_mode 不对 | 下载 demo 时用 `obs_mode=rgb` 重新生成 |
| 视频写入慢拖累转换 | LeRobot 默认 image_writer_processes=0 | 设为 4，并行写帧 |
| LeRobot 加载报"feature mismatch" | features dict 类型字段写错 | 严格按 `{"dtype": "video", "shape": (...), "names": [...]}` |
| 多任务训练时 ManiSkill 数据 dominate | weight 没设 | 主数据权重设 5× 以上 |
| Franka 7 关节与 SO-ARM 6 关节不匹配 | action/state 维度差异 | 训练时用动作 head 适配；或两阶段训练 |

---

### 10.3 完成本节后的状态

跑完这一节，你拥有：
- **`data/maniskill_to_lerobot/`** 完整转换器（适用于任何 ManiSkill demo）
- **3 个公开 LeRobot dataset**（PickCube / StackCube / PickClutterYCB），可直接用作 Phase 4 多任务先验
- **Phase 4 多任务训练配置**模板
- （可选）GPU 并行采集脚本

**对项目主线的价值**：
- 不替代主路径，但**显著扩大 Phase 4 训练数据规模**（数千 episode → 数万）
- 给 VLA 提供"操作物体"的通用先验，对 SO-ARM 单任务泛化有正向作用
- 这部分**可以与主路径完全并行推进**，不阻塞 Phase 1–3
