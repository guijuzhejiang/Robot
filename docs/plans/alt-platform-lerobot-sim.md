# 备选平台：LeRobot 自带 sim

> **本文档定位**：HuggingFace LeRobot 生态中 4 个仿真包的搭建与使用指引。其中 `gym-lowcostrobot` **是本项目主路径之前的最快起步路径**（SO-ARM100 原生支持）。

---

## 1. LeRobot 自带 sim 是什么

LeRobot 主仓库 `huggingface/lerobot` 本身**不包含**完整的仿真环境，它只在 `lerobot/common/envs/factory.py` 提供 env 工厂，真正的仿真环境是**几个独立的 pip 包**：

| 仓库 | 维护方 | 机器人 | 主要任务 | 用途 |
|------|--------|--------|----------|------|
| `huggingface/gym-aloha` | HuggingFace | ALOHA 双臂 | TransferCube, InsertPeg | 双臂操作研究 |
| `huggingface/gym-pusht` | HuggingFace | 无臂（推方块）| Push-T | Diffusion Policy 经典基准 |
| `huggingface/gym-xarm` | HuggingFace | xArm | Lift | 单臂操作基准 |
| **`perezjln/gym-lowcostrobot`** | **社区** | **SO-ARM100/101, Koch** | **PickPlace / Stack / Push / Lift / Reach** | **本项目入门主选** |

### 1.1 核心卖点

| 特性 | 说明 |
|------|------|
| **LeRobot 数据格式原生** | env 输出可直接喂 `LeRobotDataset.add_frame()` |
| **gymnasium 标准 API** | reset / step / observation_space 全标准 |
| **MuJoCo 底层** | 物理稳定、可用 `mujoco.viewer` 调试 |
| **SO-ARM 直接支持** | gym-lowcostrobot 内置 SO-ARM100 MJCF |
| **轻量** | 单包安装 < 100MB，对比 Isaac 系列优势明显 |

### 1.2 诚实评估：对本项目的适用性

| 维度 | 现状（2026） |
|------|--------------|
| 安装难度 | ⭐ 极低（一条 pip / git clone）|
| SO-ARM100/101 原生支持 | ✅（`gym-lowcostrobot`）|
| 任务库丰富度 | ⭐⭐⭐ 几个基础任务足够入门 |
| 自定义任务难度 | ⭐⭐⭐ 容易，直接继承 base env |
| 域随机化接口 | ⭐⭐ 不如 robosuite / ManiSkill 完善，要自己加 |
| GPU 并行 | ❌ 单进程（多 worker 要 multiprocessing 自包装）|
| sim2real 案例 | ⭐⭐⭐ HuggingFace 教程级别 |
| LeRobot 衔接成本 | ⭐⭐⭐⭐⭐ 零成本 |

**结论**：
- **Phase 0–1 最快起步路径** —— 1 天内能跑通"仿真 → LeRobot dataset → 训练"闭环
- **不适合 Phase 2 大规模采集**（单进程慢、DR 接口弱），最终要切到 MuJoCo + Menagerie 主路径
- 切换时**模型几何一致**（都来自 mujoco_menagerie），数据格式一致，代价只在 env API

---

## 2. 安装

### 2.1 环境要求

- Linux / macOS（Windows 可用但 OpenGL 渲染需折腾）
- **Python 3.10 推荐，3.12 也可**（详见下方安装顺序）
- MuJoCo **≥ 3.2**（必须现代版，不要 mujoco-py 旧版）
- 已有 LeRobot 环境（Phase 0 已搭建）

### 2.2 哪些包必装、哪些可跳过

| 包 | 对 SO101 项目必要性 | 装哪个 |
|----|--------------------|--------|
| **`gym-lowcostrobot`** | ✅ **必装**（SO-ARM 原生支持）| 必装 |
| `gym-aloha` | ❌ 不需要（ALOHA 双臂）| 可跳过 |
| `gym-pusht` | ❌ 不需要（2D 推方块基准）| 可跳过 |
| `gym-xarm` | ❌ 不需要（xArm）| 可跳过 |

**结论**：先只装 `gym-lowcostrobot`，其他三个**有兴趣再装**。

### 2.3 安装命令（重要：顺序敏感）

**关键原则**：先装现代 `mujoco`，再装 gym 包。否则 pip 解析依赖时可能回退到老 mujoco 版本，触发"MUJOCO_PATH not set"源码构建错误。

```bash
# 步骤 0：进入你的 conda 环境
conda activate <your_env_name>

# 步骤 1：先装现代 mujoco（必须 ≥3.2，有 Python 3.10/3.11/3.12 预编译 wheel）
pip install --upgrade "mujoco>=3.2"
python -c "import mujoco; print(mujoco.__version__)"
# 期望：3.2.x 或更高

# 步骤 2：装 LeRobot 主仓库
git clone https://github.com/huggingface/lerobot ~/lerobot
cd ~/lerobot
pip install -e ".[smolvla,feetech]"
cd -

# 步骤 3：装 gym-lowcostrobot（SO-ARM）
git clone https://github.com/perezjln/gym-lowcostrobot ~/gym-lowcostrobot
cd ~/gym-lowcostrobot
pip install -e .
cd -
```

**可选三件套（仅在确认上面都成功后再装）**：
```bash
pip install gym-pusht     # 不依赖 MuJoCo，最容易成功
pip install gym-aloha     # 可能与 dm_control 老版本冲突
pip install gym-xarm
```

### 2.4 验证

```python
import gymnasium as gym
import gym_lowcostrobot   # 触发环境注册

env = gym.make("LiftCube-v0", render_mode="human")
env.reset()
for _ in range(200):
    env.step(env.action_space.sample())
env.close()
```

弹出 viewer 看到 SO-ARM100 在动 → 成功。

### 2.5 安装故障排查

#### 错误：Python 3.13 没有 mujoco wheel

**根因**：截至 2026 中，部分 wheel 还在补齐 3.13 支持。

**修复**：用 Python 3.10–3.12 conda 环境，不要硬上 3.13。

#### 错误：`conda run -n <env>` 返回错误的 Python 版本

**根因**：conda run 有时不正确切换 Python。

**修复**：直接用绝对路径，或先 `conda activate <env>` 再运行命令：
```bash
/path/to/miniconda3/envs/<env>/bin/python --version
```

#### 错误：`ModuleNotFoundError: No module named 'lerobot.common.datasets'`

**根因**：**LeRobot 0.4 → 0.5.x 重构了模块结构**，把 `lerobot.common.*` 移到了顶层，同时把 `python -m lerobot.scripts.X` 改成了 CLI 入口 `lerobot-x`。本仓库文档使用 **0.5.x 新 API**。

**0.x ↔ 0.5.x API 对照表**：

| 旧 API（0.3.x 及以前）| 新 API（0.5.x，本仓库使用）|
|----------------------|---------------------------|
| `from lerobot.common.datasets.lerobot_dataset import LeRobotDataset` | `from lerobot.datasets.lerobot_dataset import LeRobotDataset` |
| `dataset.add_frame({...}, task="...")` | `dataset.add_frame({..., "task": "..."})` — `task` 进 dict |
| `dataset.consolidate()` | `dataset.finalize()` |
| `python -m lerobot.scripts.visualize_dataset` | `lerobot-dataset-viz` |
| `python -m lerobot.scripts.find_cameras opencv` | `lerobot-find-cameras opencv` |
| `python -m lerobot.scripts.calibrate ...` | `lerobot-calibrate ...` |
| `python -m lerobot.scripts.teleoperate ...` | `lerobot-teleoperate ...` |
| `python -m lerobot.scripts.record ...` | `lerobot-record ...` |
| `python lerobot/scripts/train.py ...` | `lerobot-train ...` |
| `lerobot/scripts/eval.py` | `lerobot-eval` CLI |

**确认你装的是哪个版本**：
```bash
pip show lerobot | grep -i version
# 0.5.x → 用新 API
# 0.4.x 或更早 → 自行参照旧 API 文档（不在本仓库范围）
```

**快速检查 LeRobotDataset 在你的版本里在哪**：
```bash
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; print('NEW api OK')" \
  || python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; print('OLD api')"
```

#### 错误：`FileExistsError: '/home/.../cache/huggingface/lerobot/local/<repo_id>'`

**根因**：`LeRobotDataset.create()` 不接受已存在的目录（参数固定 `exist_ok=False`），目的是防止误覆盖。

**推荐做法：默认续接，显式重建**。本仓库正文示例（§4、§12 T-A.4）都用了一个 helper 函数 `make_or_resume_dataset(repo_id, fps, features, *, reset=False)`：

```python
def make_or_resume_dataset(repo_id, fps, features, *, reset=False):
    """默认续接已有 dataset，reset=True 时强制清空重建。"""
    root = Path.home() / ".cache/huggingface/lerobot" / repo_id
    if reset and root.exists():
        shutil.rmtree(root)
    if root.exists():
        ds = LeRobotDataset(repo_id)
        print(f"Resumed: {ds.num_episodes} existing episodes")
        return ds
    return LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features)
```

**两种使用场景**：

| 场景 | 用法 |
|------|------|
| 调通脚本后批量采集 / 脚本崩了重启 | `reset=False`（默认）。已有 episode 不丢，继续累积到目标数量 |
| 改了 features dict / 想从头采新一批 | `reset=True`。删旧目录，新建空 dataset |

**续接的 3 个注意点**：

1. **续接时 features 参数被忽略**：`LeRobotDataset(repo_id)` 从磁盘 metadata 读 schema，脚本里写的 features dict **无效**。如果你改了 features dict 想生效，必须 `reset=True`。
2. **schema 不一致会在 add_frame 时炸**：续接后第一次 `add_frame` 字段不匹配会报错，提示你哪个字段缺失或类型不对。
3. **崩溃残留的 episode_buffer**：上次脚本崩溃时如果有未 `save_episode()` 也未 `clear_episode_buffer()` 的残留，新一轮第一帧可能挂——遇到时调 `dataset.clear_episode_buffer()` 一次再继续。

**应急清理**：`rm -rf ~/.cache/huggingface/lerobot/local/<repo_id>`

#### 错误：`RepositoryNotFoundError: 401 Client Error ... /api/datasets/local/<repo_id>/refs`（前置 `FileNotFoundError: meta/tasks.parquet`）

**根因**：dataset 目录存在但**损坏**——上次 `LeRobotDataset.create()` 只写了 `meta/info.json`，还没成功 `save_episode()` 至少一次就崩了（没有 episode 就没有 `tasks.parquet`）。第二次跑 `LeRobotDataset(repo_id)` 找不到 `tasks.parquet` → 回退尝试从 HF Hub 拉 → `local/xxx` 在 HF 上不存在 → 401。

**根本原因（必须先解决）**：上一次跑为什么一条 episode 都没存？常见三种：
1. **obs key 写错**：`obs["pixels"]["front"]` vs 实际 `obs["image_front"]` → `add_frame` 字段不匹配
2. **action 维度不匹配**：`action_mode="ee"` 但 features 写了 6 维 → save 时维度报错
3. **成功判定永远 False**：`info.get('is_success', False)` 但 info 是空 dict → 所有 episode 都被 `clear_episode_buffer()`

**修复**：
1. 用 §12 T-A.1 的 inspect 脚本**实测** obs / action / info 的实际 schema
2. 修正 obs key、features.action.shape、成功判定
3. 删损坏目录：`rm -rf ~/.cache/huggingface/lerobot/local/<repo_id>`
4. （已在 §4 的 helper 里固化）`make_or_resume_dataset` 自动检测 `tasks.parquet` 不存在 → 当作损坏目录清理

#### 错误：`TypeError: LeRobotDataset.add_frame() got an unexpected keyword argument 'task'`

**根因**：LeRobot 0.5.x 的 `add_frame` 签名是 `add_frame(frame: dict)` —— **只接受一个 dict 参数**，`task` 必须作为 dict 里的一个 key，不是单独的 kwarg。

**老 API 写法（会报错）**：
```python
dataset.add_frame({
    "observation.images.front": obs["image_front"],
    "action": action.astype("float32"),
}, task="lift the red cube")    # ❌ 旧版风格
```

**新 API 正确写法**：
```python
dataset.add_frame({
    "observation.images.front": obs["image_front"],
    "action": action.astype("float32"),
    "task": "lift the red cube",   # ✅ 放进 dict 里
})
```

把这条加到 §2.5 的「0.x ↔ 0.5.x API 对照表」（隐式条目，新版多了这条）。

#### 噪音（不是错误）：`'NoneType' object is not callable` in `glfw/__init__.py` / `Renderer.__del__`

**根因**：Python 解释器退出时，MuJoCo 的 `Renderer` / `GLContext` 析构函数被调用，但 `glfw` 模块的全局变量已被清理为 None。这是 MuJoCo + Python 解释器关闭顺序问题，**不影响数据正确性**。

**修复**：忽略即可。如果想消除噪音：
```python
# 在脚本最后显式关闭 env，让析构提前发生
env.close()
import gc; gc.collect()
```
或：
```python
import warnings
warnings.filterwarnings("ignore", category=Warning)
```

---

## 3. 最小可运行示例

### 3.1 跑 SO-ARM100 PickPlace（gym-lowcostrobot）

```python
import gymnasium as gym
import gym_lowcostrobot

env = gym.make(
    "PickPlaceCube-v0",
    observation_mode="both",   # "image" / "state" / "both"
    action_mode="joint",        # "joint" / "ee"
    render_mode="human",
)

obs, info = env.reset(seed=42)
# obs 结构（gym-lowcostrobot 实测，扁平 key，**不是嵌套**）：
# obs["arm_qpos"]:    (6,) 关节位置
# obs["arm_qvel"]:    (6,) 关节速度
# obs["cube_pos"]:    (3,) 物体位置（state mode 时）
# obs["image_front"]: (240, 320, 3) RGB
# obs["image_top"]:   (240, 320, 3) RGB
# 注意：info 通常是空 dict {}，没有 "is_success" 字段——成功要自己判断（看 cube z 高度）

for _ in range(300):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

### 3.2 跑 ALOHA 双臂（gym-aloha）

```python
import gymnasium as gym
import gym_aloha

env = gym.make("gym_aloha/AlohaTransferCube-v0", render_mode="human")
obs, _ = env.reset()
for _ in range(400):
    action = env.action_space.sample()   # (14,) 双臂各 7 关节
    obs, _, term, trunc, _ = env.step(action)
    if term or trunc:
        obs, _ = env.reset()
env.close()
```

### 3.3 跑 Push-T（Diffusion Policy 经典任务）

```python
import gymnasium as gym
import gym_pusht

env = gym.make("gym_pusht/PushT-v0", render_mode="human")
obs, _ = env.reset()
for _ in range(200):
    obs, _, term, trunc, _ = env.step(env.action_space.sample())
    if term or trunc:
        obs, _ = env.reset()
```

---

## 4. 数据生成完整示例（与 LeRobot dataset 衔接）

这是它对本项目**最有价值的能力**——零摩擦输出 LeRobot 数据集。

> **续接 vs 重建**：`LeRobotDataset.create()` 不允许目标目录已存在（防止误覆盖）。脚本应**默认续接**已有数据集（脚本崩溃时不丢之前的 episode），仅在 schema 变更或刻意清空时设置 `RESET=True`。下面的 helper 函数封装了这个逻辑。

```python
import shutil
from pathlib import Path
import numpy as np
import gymnasium as gym
import gym_lowcostrobot  # noqa: F401  # 触发 gym.register()，必须保留即使 linter 标黄
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def make_or_resume_dataset(repo_id: str, fps: int, features: dict, *, reset: bool = False) -> LeRobotDataset:
    """
    默认续接已有数据集；reset=True 或目录不存在/损坏时新建。

    损坏检测：只有 meta/info.json 但没有 meta/tasks.parquet → 说明上次 create 跑了但
    一条 episode 都没成功 save_episode()，目录不可续接，自动清理重建。

    重要：续接时 features 参数会被忽略（schema 由磁盘 metadata 决定）。
    如果想换 schema，必须 reset=True 重建。
    """
    root = Path.home() / ".cache/huggingface/lerobot" / repo_id
    meta_tasks = root / "meta" / "tasks.parquet"   # save_episode 后才会生成

    if reset and root.exists():
        print(f"RESET=True, removing {root}")
        shutil.rmtree(root)
    elif root.exists() and not meta_tasks.exists():
        print(f"WARN: corrupted dataset at {root} (no tasks.parquet). Removing.")
        shutil.rmtree(root)

    if root.exists():
        ds = LeRobotDataset(repo_id)
        print(f"Resumed existing dataset: {ds.num_episodes} episodes, {ds.num_frames} frames")
        return ds
    return LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features)


REPO_ID = "local/lowcost_lift_sim_v0"
RESET = False   # 改 features / 想清空时设 True

# 关键：action_mode 必须与下面 policy 的输出维度一致
# - "ee"    → action 4 维 [dx, dy, dz, gripper]
# - "joint" → action 6 维 [q0..q5]（不含 gripper，gripper 在 obs 里独立）
env = gym.make("LiftCube-v0", observation_mode="both", action_mode="ee")

dataset = make_or_resume_dataset(
    repo_id=REPO_ID,
    fps=30,
    reset=RESET,
    features={
        "observation.images.front": {
            "dtype": "video", "shape": (240, 320, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.top": {
            "dtype": "video", "shape": (240, 320, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32", "shape": (6,),
            "names": [f"q{i}" for i in range(6)],
        },
        "action": {                            # ee 模式 = 4 维
            "dtype": "float32", "shape": (4,),
            "names": ["dx", "dy", "dz", "gripper"],
        },
    },
)

class LiftCubePolicy:
    """
    LiftCube 任务的脚本策略 = IK + waypoint 范式的最小实现。

    "IK + waypoint" 是什么意思？
      - waypoint：把任务拆成几个三维空间中的关键位姿（"先到 cube 上方"→"贴近 cube"→"夹紧"→"抬高"）
      - IK：每一步把"当前 ee 位置→下一个 waypoint"的 ee 增量喂给 env，env 内部用逆运动学解关节
        （`action_mode="ee"` 时 env 自带 IK；`action_mode="joint"` 则需自己用 mink / dm_control 求 IK）

    4 阶段：APPROACH (cube 上方) → DESCEND (贴 cube) → GRASP (闭夹爪) → LIFT (抬起)
    """
    def __init__(self):
        self.phase = "APPROACH"
        self.phase_steps = 0
        self.gripper = 1.0   # 1 open, -1 close

    def reset(self):
        self.__init__()

    def _ee_pos(self, env):
        # 用 MJCF 里专门标的 end_effector_site（夹爪末端锚点，比 body 中心精确）
        # 旧版 gym-lowcostrobot 可能用 body("Moving_Jaw")，新版 (2025+) 已改名
        return env.unwrapped.data.site("end_effector_site").xpos.copy()

    def _step_to(self, env, target_xyz, max_speed=0.04):
        """计算把 ee 移向 target 的单步增量（闭环：每帧基于当前 ee 重算）"""
        delta = target_xyz - self._ee_pos(env)
        norm = np.linalg.norm(delta)
        if norm < 1e-3:
            return np.zeros(3), True
        delta = delta / norm * min(norm, max_speed)
        scale = getattr(env.unwrapped, "action_scale", 0.05)
        return np.clip(delta / scale, -1.0, 1.0), False

    def __call__(self, env, obs):
        cube = obs["cube_pos"]
        if self.phase == "APPROACH":
            d, reached = self._step_to(env, cube + np.array([0, 0, 0.05]))
            if reached: self.phase = "DESCEND"
        elif self.phase == "DESCEND":
            d, reached = self._step_to(env, cube + np.array([0, 0, 0.005]))
            if reached: self.phase = "GRASP"; self.phase_steps = 0
        elif self.phase == "GRASP":
            d = np.zeros(3); self.gripper = -1.0; self.phase_steps += 1
            if self.phase_steps > 10: self.phase = "LIFT"
        elif self.phase == "LIFT":
            d, _ = self._step_to(env, cube + np.array([0, 0, 0.10]))
        else:
            d = np.zeros(3)
        return np.concatenate([d, [self.gripper]]).astype(np.float32)


# PickPlace 完整实战见 §12 T-A.2
policy = LiftCubePolicy()
for ep in range(50):
    obs, _ = env.reset(seed=ep)
    policy.reset()
    init_cube_z = float(obs["cube_pos"][2])   # 记录初始高度，用于成功判定
    done = False
    last_cube_z = init_cube_z
    while not done:
        action = policy(env, obs)
        next_obs, reward, term, trunc, info = env.step(action)
        dataset.add_frame({
            "observation.images.front": obs["image_front"],     # 注意是扁平 key
            "observation.images.top":   obs["image_top"],
            "observation.state":        obs["arm_qpos"].astype("float32"),
            "action":                   action.astype("float32"),
            "task": "lift the red cube",
        })
        last_cube_z = float(next_obs["cube_pos"][2])
        obs = next_obs
        done = term or trunc

    # 成功判定：info 是空 dict，自己看 cube 高度（举高 > 5cm 算成功）
    is_success = (last_cube_z - init_cube_z) > 0.05
    if is_success:
        dataset.save_episode()
        print(f"ep {ep}: SUCCESS (cube rose {last_cube_z - init_cube_z:.3f}m)")
    else:
        dataset.clear_episode_buffer()
        print(f"ep {ep}: fail (cube rose {last_cube_z - init_cube_z:.3f}m)")

env.close()
```

### 4.0 LiftCube 脚本抓取的现实局限（先读这个）

**gym-lowcostrobot 的 `LiftCube-v0` 是设计给 RL 学的**（见 `examples/gym_manipulation_sb3.py`），用纯脚本策略实现稳定抓取非常困难。实测发现：

| 限制 | 数值 |
|------|------|
| ee 物理工作空间下限 | z ≈ 0.027（机械臂 IK 在工作空间边界饱和）|
| cube 顶部高度 | z = 0.030（cube 半边长 1.5cm，center z=0.015）|
| 夹爪指尖与 cube 几何关系 | ee_site 在 link_6 末端，**夹爪指尖位置无独立 body**，难精确控制 |
| env `max_episode_steps` | 默认 50（RL 风格），脚本需要 `gym.make(..., max_episode_steps=200+)` 才够 |
| `reset()` 不调 `mj_step` | obs["cube_pos"] 初始 z=0，第一步 step 后才物理 settle 到 z=0.015 |

**实测结果**：脚本策略 50/50 失败，cube z 始终在 0.015m（自然 settle 位置）。

**两种应对**：

| 场景 | 建议 |
|------|------|
| 验证 sim → LeRobot dataset pipeline 可用 | 切到 **`ReachCube-v0`**（只要 ee 接近 cube，脚本 100% 成功）|
| 真正训练 VLA 抓取策略 | **跳过 gym-lowcostrobot LiftCube**，进入：① Phase 0 真机遥操采集，或 ② 主路径 Phase 1 MuJoCo + Menagerie 自定义任务 |

ReachCube 替代示例（仅改成功判定 + 任务名）：

```python
env = gym.make("ReachCube-v0", observation_mode="both", action_mode="ee",
               max_episode_steps=100)
# ...采集循环...

is_success = np.linalg.norm(
    env.unwrapped.data.site("end_effector_site").xpos - obs["cube_pos"]
) < 0.05   # ee 接近 cube 5cm 内算成功
```

### 4.1 "IK + waypoint" 范式拆解

上面 `LiftCubePolicy` 体现了**IK + waypoint 抓取**的 3 个核心要素：

| 要素 | 上面代码里的位置 | 干什么 |
|------|------------------|-------|
| **Waypoint 列表** | `APPROACH → DESCEND → GRASP → LIFT` 4 个阶段 | 把任务拆成"机械臂依次要到的几个 3D 位姿" |
| **IK 求解** | `action_mode="ee"` + `env.step()` 内部 | 把"我想让 ee 到这个 xyz"翻译成"6 个关节该转到哪里" |
| **闭环 step** | `_step_to()` 每帧重算 delta | 每一步基于**当前真实 ee 位置**算下一步，而不是预先规划整条轨迹 |

**两种实现 IK 的方式**：

1. **依赖 env 自带 IK**（最简单，本文默认）
   - 设 `action_mode="ee"`，action 是 ee 增量 `[dx, dy, dz, gripper]`
   - env 内部用数值 IK（雅可比伪逆）把 ee 增量转关节增量
   - 缺点：奇异位姿会跳变，没法定制
2. **自己用 mink 求 IK**（生产级，主路径用）
   - `action_mode="joint"`，自己写 IK：
     ```python
     import mink
     config = mink.Configuration(env.unwrapped.model)
     ee_task = mink.FrameTask("Moving_Jaw", frame_type="body", position_cost=1.0)
     ee_task.set_target(mink.SE3.from_translation(target_xyz))
     vel = mink.solve_ik(config, [ee_task], dt=1/30, solver="quadprog")
     config.integrate_inplace(vel, 1/30)
     joint_target = config.q.copy()
     ```
   - 优点：能加 damping 防奇异；能多任务（如同时约束姿态）；能加关节限位
   - 缺点：要安装 mink，要查 body 名

**Waypoint 设计的常见 pitfall**：
- 直接从 home pose 跳到 cube 上方 → 工作空间外、撞桌 → 加中间 waypoint
- LIFT 高度不够 → cube 还在 cube_pos.z + 5cm，下次 reset 重叠 → 抬到 10cm 以上
- GRASP 闭合时机错 → 用 `np.linalg.norm(cube_pos - ee_pos) < 0.01` 做切阶段判断更鲁棒
- 每个 waypoint 都要**闭环执行**：`while not reached: step toward it`，不能一次输出整段轨迹后开环执行

**完整的 PickPlace（7 阶段，含 place）参考 §12 T-A.2 的 `PickPlacePolicy`**。

### 4.2 可视化数据集

```bash
lerobot-dataset-viz \
    --repo-id local/lowcost_lift_sim_v0 \
    --episode-index 0
```

---

## 5. gym-lowcostrobot 内置任务一览

截至 2026 中：

| 任务 ID | 描述 | 状态空间 | 动作空间 |
|--------|------|----------|----------|
| `LiftCube-v0` | 抓起一个 cube | qpos(6) + cube_pos(3) | (6,) 关节 |
| `PickPlaceCube-v0` | 抓 cube 放进目标位 | + target_pos(3) | (6,) |
| `PushCube-v0` | 把 cube 推到目标 | + target_pos(2) | (6,) |
| `ReachCube-v0` | ee 接近 cube | qpos + cube_pos | (6,) |
| `StackTwoCubes-v0` | cube1 叠到 cube2 上 | 双 cube 位姿 | (6,) |
| `ReachTarget-v0` | ee 到达空间目标点 | 仅 qpos + target | (6,) |

**两种动作模式**：
- `action_mode="joint"`：直接出 6 个关节目标位置
- `action_mode="ee"`：出 ee 位姿目标（内部用 IK 转关节）

---

## 6. 自定义任务（扩展 gym-lowcostrobot）

如果你想加一个 `PickColorCube` 任务（按指令颜色抓），最小骨架：

```python
# my_envs/pick_color_cube.py
from gym_lowcostrobot.envs.base import BaseRobotEnv
import numpy as np

class PickColorCubeEnv(BaseRobotEnv):
    def __init__(self, **kwargs):
        super().__init__(xml_path="my_assets/pick_color.xml", **kwargs)
        self.cube_colors = ["red", "green", "blue"]
        self.target_color = None

    def reset_model(self):
        # 随机化目标颜色 + 三个 cube 位置
        self.target_color = np.random.choice(self.cube_colors)
        for color in self.cube_colors:
            xy = np.random.uniform([-0.1, -0.1], [0.1, 0.1], size=2)
            self.set_body_pos(f"cube_{color}", [*xy, 0.02])
        return self._get_obs()

    def _get_obs(self):
        obs = super()._get_obs()
        obs["language"] = f"pick the {self.target_color} cube"
        return obs

    def _is_success(self):
        cube_pos = self.get_body_pos(f"cube_{self.target_color}")
        return cube_pos[2] > 0.08  # 抬高 8cm 算成功

# 注册
import gymnasium as gym
gym.register(
    id="PickColorCube-v0",
    entry_point="my_envs.pick_color_cube:PickColorCubeEnv",
    max_episode_steps=400,
)
```

---

## 7. 与本项目主路径的衔接

### 7.1 模型一致性

`gym-lowcostrobot` 的 SO-ARM100 MJCF 与 `mujoco_menagerie` 的 `trs_so_arm100/` **同源**（社区从 menagerie 引用），所以：
- 同一关节顺序
- 同一关节限位
- 同一夹爪开合范围

切换到主路径时 **MJCF 直接复用**，自定义任务的逻辑代码也可以保留。

### 7.2 数据格式一致

两边都输出 `LeRobotDataset`，Phase 3 / Phase 4 完全不用区分数据来源。

### 7.3 何时切换

| 状态 | 用 gym-lowcostrobot | 用 MuJoCo + Menagerie 主路径 |
|------|---------------------|------------------------------|
| Phase 0–1 跑通最小闭环 | ✅ | ❌ |
| 写自定义 grasp sampler / motion planner | ⚠️ 可，但要包装 | ✅ 直接 |
| Phase 2 批量采集 1000+ 条 | ❌ 单进程太慢 | ✅ 多进程 |
| 复杂域随机化 | ❌ 接口弱 | ✅ 自由 |
| Phase 3 真机 demo 扩增（MimicGen 移植） | ❌ | ✅ |

---

## 8. 学习资源

| 资源 | URL |
|------|-----|
| LeRobot 主仓库 | `https://github.com/huggingface/lerobot` |
| LeRobot 文档主入口 | `https://huggingface.co/docs/lerobot/index` |
| SO-101 真机教程 | `https://huggingface.co/docs/lerobot/so101` |
| 数据集格式文档 | `https://huggingface.co/docs/lerobot/lerobot_dataset` |
| 示例代码 | `https://github.com/huggingface/lerobot/tree/main/examples` |
| SmolVLA 微调指南 | `https://huggingface.co/docs/lerobot/smolvla` |
| Pi0 + LeRobot 博客 | `https://huggingface.co/blog/pi0` |
| gym-aloha | `https://github.com/huggingface/gym-aloha` |
| gym-pusht | `https://github.com/huggingface/gym-pusht` |
| gym-xarm | `https://github.com/huggingface/gym-xarm` |
| gym-lowcostrobot | `https://github.com/perezjln/gym-lowcostrobot` |
| LeRobot Discord | 主仓库 README 有邀请链接 |

---

## 9. 推荐学习路径

```
0.5 天：装 gym-lowcostrobot，跑通 LiftCube random policy
0.5 天：写最小 LeRobot dataset 导出
1 天：实现 IK + 简单 waypoint 脚本策略
0.5 天：跑 50 条成功 episode 并可视化
0.5 天：（可选）写一个自定义任务
```

总计约 3 天可达"Phase 0/1 起步级别"。

---

## 10. 风险与陷阱

| 风险 | 应对 |
|------|------|
| `import gym_lowcostrobot` 被 linter 标记 "unused"，误删后 `gym.make` 报 UnregisteredEnvError | 该 import 是为执行 `gym.register()` 的副作用，**必须保留**。加 `# noqa: F401` 压制警告 |
| 社区包 API 可能在小版本间变化 | 锁定 `gym-lowcostrobot` 的 commit hash |
| 渲染相机分辨率默认 240×320，与真机 480/640 不一致 | 在 reset 后通过 `env.unwrapped.model.cam_resolution` 改 |
| 内置任务的 reward 不一定适合 IL 数据生成（IL 不用 reward） | 直接用 `info["is_success"]` 做过滤 |
| 域随机化要自己加 | 重写 reset 时改 `mjModel` 的 material rgba / light dir |
| 性能不够（单进程 ~50fps） | 用 `multiprocessing.Pool` 包 N 个 env，每进程独立 viewer-less |
| gym-lowcostrobot 与 LeRobot 主仓库同时升级会版本错配 | 用 `pip freeze > requirements.txt` 锁定整套版本 |

---

## 11. 决策建议

| 你的情况 | 是否用 LeRobot 自带 sim |
|----------|------------------------|
| Phase 0 想最快验证"sim → dataset → 训练"通路 | ✅ 直接用 gym-lowcostrobot |
| Phase 1 起步阶段 | ✅ 起步用，Phase 2 切到主路径 |
| Phase 2 批量采集 | ❌ 切到 MuJoCo + Menagerie 主路径 |
| 想对照 ALOHA / Push-T 已有 baseline | ✅ gym-aloha / gym-pusht 直接复现 |
| 对比"零 setup vs 全自建"开发速度 | ✅ 跑一个对照实验 |

---

## 12. 实战手册：用 gym-lowcostrobot 从零生成 100 条仿真数据

> **目标**：在 0.5 天内得到 100 条成功的 `PickPlaceCube` LeRobot 数据集，可直接用于 Phase 4 SmolVLA 微调测试。

### 12.1 总流程

```
T-A.1 环境与控制模式选定
T-A.2 编写 scripted policy（IK + 4 阶段抓取）
T-A.3 单条 episode 调试与可视化
T-A.4 批量采集 + 成功过滤
T-A.5 数据验证与可视化
T-A.6 域随机化升级（可选）
T-A.7 上传 HuggingFace Hub（可选）
```

预计时间：调试 2–3 小时，生成 100 条约 30 分钟。

---

### T-A.1 环境与控制模式选定

**目标**：搞清楚 obs / action 的具体形状，决定用什么控制模式

**步骤**：
- [ ] 在 `sim/lowcost/00_inspect_env.py` 写：

```python
import gymnasium as gym
import gym_lowcostrobot
import numpy as np

env = gym.make(
    "PickPlaceCube-v0",
    observation_mode="both",   # 同时拿 state + image
    action_mode="ee",          # 用 ee-pose 模式（内置 IK，比 joint 简单）
    render_mode="rgb_array",   # 后续要离屏渲染
)

obs, info = env.reset(seed=0)

print("=== observation_space ===")
for k, v in obs.items():
    if isinstance(v, dict):
        for kk, vv in v.items():
            print(f"  {k}.{kk}: shape={np.asarray(vv).shape}, dtype={np.asarray(vv).dtype}")
    else:
        print(f"  {k}: shape={np.asarray(v).shape}, dtype={np.asarray(v).dtype}")

print("\n=== action_space ===")
print(env.action_space)

print("\n=== info keys ===")
print(list(info.keys()))
```

**期望输出**（具体可能因版本略变化，以你跑出来的为准）：
```
  arm_qpos: shape=(6,) float32
  arm_qvel: shape=(6,) float32
  cube_pos: shape=(3,) float32
  target_pos: shape=(3,) float32
  pixels.front: shape=(240, 320, 3) uint8
  pixels.top:   shape=(240, 320, 3) uint8

action_space: Box(-1.0, 1.0, (4,), float32)   # [dx, dy, dz, dgripper]
```

**为什么选 `action_mode="ee"`**：
- `ee` 模式下 action 是 ee 增量位移 + 夹爪开合，**只有 4 维**，写脚本策略简单
- `joint` 模式需要自己写 IK，先用 ee 把流水线跑通

**验证**：上面打印出来的 keys 与 shape 与你的输出一致

---

### T-A.2 编写 scripted policy（IK + 4 阶段抓取）

**目标**：写一个能稳定完成 PickPlace 的脚本策略（不调 ML）

**核心思路**：用状态读取（cube_pos / target_pos / ee_pos）做四阶段闭环控制

```python
# sim/lowcost/scripted_pickplace.py
import numpy as np

class PickPlacePolicy:
    """
    四阶段：
      0. APPROACH:  ee 移动到 cube 上方 5cm，夹爪张开
      1. DESCEND:   ee 下降到 cube 旁，夹爪张开
      2. GRASP:     夹爪闭合
      3. LIFT:      ee 上升 8cm
      4. TRANSPORT: ee 移动到 target 上方
      5. PLACE:     ee 下降到 target 旁
      6. RELEASE:   夹爪张开
    """
    APPROACH_OFFSET = np.array([0.0, 0.0, 0.05])
    LIFT_OFFSET     = np.array([0.0, 0.0, 0.08])

    def __init__(self, env):
        self.env = env
        self.phase = "APPROACH"
        self.phase_steps = 0
        self.gripper_state = 1.0    # 1 = open, -1 = close

    def reset(self):
        self.phase = "APPROACH"
        self.phase_steps = 0
        self.gripper_state = 1.0

    def _ee_pos(self):
        # gym-lowcostrobot 在 ee 模式下，obs 里可以读 ee 当前位置
        # 也可以从 env.unwrapped 读 mujoco state
        data = self.env.unwrapped.data
        # 用 MJCF 的 end_effector_site（新版 2025+；老版可能叫 body("Moving_Jaw")）
        # 用下面 inspect 命令确认你的版本：
        #   model = env.unwrapped.model
        #   print([model.site(i).name for i in range(model.nsite)])
        return data.site("end_effector_site").xpos.copy()

    def _step_to(self, target_xyz, max_speed=0.04):
        """生成把 ee 移向 target 的单步 action"""
        cur = self._ee_pos()
        delta = target_xyz - cur
        norm = np.linalg.norm(delta)
        if norm < 1e-4:
            return np.zeros(3), True
        delta = delta / norm * min(norm, max_speed)
        # ee 模式 action 单位通常已 normalize 到 [-1, 1]，需要按 action_scale 反向缩放
        scale = self.env.unwrapped.action_scale if hasattr(self.env.unwrapped, "action_scale") else 0.05
        return np.clip(delta / scale, -1.0, 1.0), False

    def __call__(self, obs):
        cube_pos = obs["cube_pos"]
        target_pos = obs["target_pos"]

        if self.phase == "APPROACH":
            goal = cube_pos + self.APPROACH_OFFSET
            d, reached = self._step_to(goal)
            if reached:
                self.phase = "DESCEND"

        elif self.phase == "DESCEND":
            goal = cube_pos + np.array([0, 0, 0.005])   # 紧贴 cube
            d, reached = self._step_to(goal)
            if reached:
                self.phase = "GRASP"
                self.phase_steps = 0

        elif self.phase == "GRASP":
            d = np.zeros(3)
            self.gripper_state = -1.0     # 闭合
            self.phase_steps += 1
            if self.phase_steps > 10:
                self.phase = "LIFT"

        elif self.phase == "LIFT":
            goal = cube_pos + self.LIFT_OFFSET
            d, reached = self._step_to(goal)
            if reached:
                self.phase = "TRANSPORT"

        elif self.phase == "TRANSPORT":
            goal = target_pos + self.APPROACH_OFFSET
            d, reached = self._step_to(goal)
            if reached:
                self.phase = "PLACE"

        elif self.phase == "PLACE":
            goal = target_pos + np.array([0, 0, 0.01])
            d, reached = self._step_to(goal)
            if reached:
                self.phase = "RELEASE"
                self.phase_steps = 0

        elif self.phase == "RELEASE":
            d = np.zeros(3)
            self.gripper_state = 1.0
            self.phase_steps += 1
            if self.phase_steps > 10:
                self.phase = "DONE"

        else:
            d = np.zeros(3)

        return np.concatenate([d, [self.gripper_state]]).astype(np.float32)
```

**关键文件**：
- `sim/lowcost/scripted_pickplace.py`

**注意**：
- `body("Moving_Jaw").xpos` 的 body 名要跟你版本的 MJCF 对齐。先在 viewer 里打开 MJCF 找到 ee 对应的 body 名
- `action_scale` 是版本相关字段，找不到时直接用经验值 0.05
- 这是最小可工作版本，**不追求美观，先跑通**

---

### T-A.3 单条 episode 调试与可视化

**目标**：让脚本策略至少**成功 1 条**，然后再批量跑

```python
# sim/lowcost/01_single_episode.py
import gymnasium as gym
import gym_lowcostrobot
import imageio
from scripted_pickplace import PickPlacePolicy

env = gym.make("PickPlaceCube-v0", observation_mode="both", action_mode="ee", render_mode="rgb_array")
policy = PickPlacePolicy(env)

obs, info = env.reset(seed=42)
policy.reset()

frames = []
for t in range(500):
    action = policy(obs)
    obs, reward, term, trunc, info = env.step(action)
    frames.append(env.render())
    if term or trunc:
        break

print(f"success: {info.get('is_success', False)}, phase: {policy.phase}, steps: {t+1}")
imageio.mimsave("episode_debug.mp4", frames, fps=30)
env.close()
```

**步骤**：
- [ ] 跑 5 个不同 seed（0, 1, 2, 3, 4），看 success 率
- [ ] 失败的 episode 打开 mp4 看是哪一阶段卡住
- [ ] 调阈值：`APPROACH_OFFSET` / GRASP 步数 / max_speed

**调试 tips**：
- DESCEND 过深 → cube 被推走 → 把 `+0.005` 改成 `+0.02`
- GRASP 太早闭合 → ee 还没到位 → 用 `np.linalg.norm(cube_pos - ee_pos) < 0.01` 作为切阶段条件
- LIFT 后 cube 掉了 → 夹爪闭合时间太短，从 10 步加到 20 步

**验证**：seed 0–4 中至少 3 条成功（≥60%）

---

### T-A.4 批量采集 + 成功过滤

**目标**：跑 N 次 episode，只保留成功的写入 LeRobot dataset

```python
# sim/lowcost/02_batch_collect.py
import gymnasium as gym
import gym_lowcostrobot
import numpy as np
from tqdm import tqdm
from scripted_pickplace import PickPlacePolicy
# 复用 §4 的 helper（也可以放到 sim/lowcost/utils.py 里 import）
from utils import make_or_resume_dataset

REPO_ID = "local/lowcost_pickplace_v0"
N_TARGET_SUCCESS = 100   # 目标成功数（续接模式下：累积到这个数为止）
MAX_ATTEMPTS = 300       # 单次运行上限：避免成功率太低无限跑
MAX_STEPS_PER_EP = 500
RESET = False            # 改 features / 想清空时设 True

env = gym.make("PickPlaceCube-v0", observation_mode="both", action_mode="ee", render_mode="rgb_array")
policy = PickPlacePolicy(env)

dataset = make_or_resume_dataset(
    repo_id=REPO_ID,
    fps=30,
    reset=RESET,
    features={
        "observation.images.front": {"dtype": "video", "shape": (240, 320, 3),
                                      "names": ["height", "width", "channels"]},
        "observation.images.top":   {"dtype": "video", "shape": (240, 320, 3),
                                      "names": ["height", "width", "channels"]},
        "observation.state":        {"dtype": "float32", "shape": (6,),
                                      "names": [f"q{i}" for i in range(6)]},
        "action":                   {"dtype": "float32", "shape": (4,),
                                      "names": ["dx", "dy", "dz", "gripper"]},
    },
)

# 续接模式：从已有 episode 数继续累积
n_total = dataset.num_episodes
print(f"Starting from {n_total} existing episodes, target = {N_TARGET_SUCCESS}")

INSTRUCTIONS = [
    "pick up the cube and place it on the target",
    "move the cube to the target location",
    "put the cube on the goal",
    "transfer the cube to the marker",
    "place the cube at the target spot",
]

n_new = 0    # 本次 run 新增的成功 episode
n_attempt = 0
pbar = tqdm(total=N_TARGET_SUCCESS, initial=n_total)
for ep in range(MAX_ATTEMPTS):
    if n_total >= N_TARGET_SUCCESS:
        break
    n_attempt += 1
    seed = (n_total + ep) * 7 + 11   # 用累积 episode 偏移 seed，避免重复采到同样的初始
    obs, info = env.reset(seed=seed)
    policy.reset()
    instruction = np.random.choice(INSTRUCTIONS)

    for t in range(MAX_STEPS_PER_EP):
        action = policy(obs)
        next_obs, _, term, trunc, info = env.step(action)
        dataset.add_frame({
            "observation.images.front": obs["pixels"]["front"],
            "observation.images.top":   obs["pixels"]["top"],
            "observation.state":        obs["arm_qpos"].astype("float32"),
            "action":                   action.astype("float32"),
            "task": instruction,
        })
        obs = next_obs
        if term or trunc:
            break

    if info.get("is_success", False):
        dataset.save_episode()
        n_total += 1
        n_new += 1
        pbar.update(1)
    else:
        dataset.clear_episode_buffer()
pbar.close()

print(f"\nThis run: +{n_new} new episodes (success rate {n_new/max(n_attempt,1):.1%})")
print(f"Dataset total: {n_total} episodes")
env.close()
```

**步骤**：
- [ ] 先跑 `N_TARGET_SUCCESS=10` 验证 pipeline 走通
- [ ] 看 success rate，若 < 40% 回 T-A.3 继续调脚本
- [ ] 调好后改成 `N_TARGET_SUCCESS=100`，约 20–30 分钟跑完

**验证**：
- 100 条 episode 保存到 `~/.cache/huggingface/lerobot/local/lowcost_pickplace_v0/`
- 成功率打印 ≥ 60%

---

### T-A.5 数据验证与可视化

**目标**：确认数据真的能用，不是有 bug 的"假成功"

**步骤**：
- [ ] 用 LeRobot 自带可视化工具：
  ```bash
  lerobot-dataset-viz \
      --repo-id local/lowcost_pickplace_v0 \
      --episode-index 0
  ```
- [ ] 抽 5 条 episode 看完整动作
- [ ] 检查统计：
  ```python
  from lerobot.datasets.lerobot_dataset import LeRobotDataset
  ds = LeRobotDataset("local/lowcost_pickplace_v0")
  print(f"episodes: {ds.num_episodes}")
  print(f"frames:   {ds.num_frames}")
  print(f"avg ep length: {ds.num_frames / ds.num_episodes:.1f}")
  # 看每条 episode 的语言指令分布
  tasks = [ds[i]["task"] for i in range(0, ds.num_frames, 50)]
  from collections import Counter
  print(Counter(tasks))
  ```

**验证**：
- 平均 episode 长度 100–300 frames（合理）
- 5 种语言指令大致均匀分布
- 目视检查：无穿模、无关节抖动到极限

---

### T-A.6 域随机化升级（可选，但强烈建议）

**目标**：让生成的数据具备 sim2real 所需的视觉多样性

gym-lowcostrobot 没原生 DR 接口，但底层是 MuJoCo，可以直接改 `mjModel`：

```python
# sim/lowcost/randomize.py
import numpy as np

def randomize_scene(env, rng):
    model = env.unwrapped.model

    # 1. 随机化桌面颜色
    table_geom_id = model.geom("table").id   # geom 名以你的 MJCF 为准
    model.geom_rgba[table_geom_id] = [*rng.uniform(0.3, 0.9, size=3), 1.0]

    # 2. 随机化 cube 颜色
    cube_geom_id = model.geom("cube").id
    color_options = {
        "red":   [0.9, 0.1, 0.1, 1.0],
        "green": [0.1, 0.8, 0.1, 1.0],
        "blue":  [0.1, 0.3, 0.9, 1.0],
    }
    color_name = rng.choice(list(color_options.keys()))
    model.geom_rgba[cube_geom_id] = color_options[color_name]

    # 3. 随机化光源
    if model.nlight > 0:
        model.light_diffuse[0] = rng.uniform(0.5, 1.0, size=3)
        model.light_pos[0] = [rng.uniform(-0.5, 0.5),
                              rng.uniform(-0.5, 0.5),
                              rng.uniform(0.8, 1.5)]

    return color_name
```

集成到 `02_batch_collect.py`：

```python
rng = np.random.default_rng(seed)
obs, info = env.reset(seed=seed)
color_name = randomize_scene(env, rng)
instruction = f"pick up the {color_name} cube and place it on the target"
# ... 继续脚本策略
```

**验证**：连续 reset 10 次截图，桌面色 / cube 色 / 光照明显不同

---

### T-A.7 上传 HuggingFace Hub（可选）

**目标**：远端备份 + Phase 4 训练时方便拉取

```bash
# 先登录
huggingface-cli login

# 推送（私有 repo）
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('local/lowcost_pickplace_v0')
ds.push_to_hub('YOUR_HF_USERNAME/lowcost_pickplace_v0', private=True)
"
```

**验证**：在 `https://huggingface.co/datasets/YOUR_HF_USERNAME/lowcost_pickplace_v0` 看到数据集

---

### 12.2 整套流程一键脚本（参考）

把上面整合成 `sim/lowcost/run.py`：

```python
# sim/lowcost/run.py
import argparse
from collect import collect_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="PickPlaceCube-v0")
    parser.add_argument("--num-success", type=int, default=100)
    parser.add_argument("--repo-id", default="local/lowcost_pickplace_v0")
    parser.add_argument("--with-dr", action="store_true", help="enable domain randomization")
    parser.add_argument("--push-to-hub", default=None, help="HF user/repo name, optional")
    args = parser.parse_args()

    collect_dataset(args)
```

运行：
```bash
python sim/lowcost/run.py --num-success 100 --with-dr
python sim/lowcost/run.py --num-success 500 --with-dr --push-to-hub YOUR_USERNAME/lowcost_pickplace_v1
```

---

### 12.3 常见问题排查

| 症状 | 可能原因 | 修复 |
|------|---------|------|
| 成功率 < 20% | DESCEND 阶段把 cube 推开 | 减小下降速度 / 提高 GRASP 触发阈值 |
| GRASP 后 LIFT 时 cube 掉 | 夹爪闭合时间不够 | GRASP 步数从 10 加到 20–30 |
| `KeyError: "Invalid name 'Moving_Jaw'..."` 或类似 body 找不到 | 不同版本 gym-lowcostrobot 的 MJCF body 名不同；2025+ 新版改用 `end_effector_site`（site 而非 body）| 跑这段查名：`print([env.unwrapped.model.body(i).name for i in range(env.unwrapped.model.nbody)])` 与 `print([env.unwrapped.model.site(i).name for i in range(env.unwrapped.model.nsite)])`；优先用 site，没 site 才退 body |
| 跑得慢（< 30 FPS） | 渲染拖累 | `render_mode=None`，最后只在保存 episode 时渲一遍 |
| 内存涨 | 长 episode 帧没清 | `del frames` 显式释放；用 `image_writer_processes=4` |
| 推 HF Hub 卡 | 视频文件大 | 用 `dataset.finalize()` 后再 push |

---

### 12.4 完成本节后的状态

跑完这一节，你拥有：
- 一个可调用的 PickPlace scripted policy（`sim/lowcost/scripted_pickplace.py`）
- 100 条带语言指令、含视觉的 LeRobot 数据集
- （可选）域随机化版本数据集
- 可上传到 HF Hub 的远端备份
- 完整的"sim → dataset"工程经验，可平移到主路径（MuJoCo + Menagerie）

**这套数据已经可以直接喂 Phase 4 的 SmolVLA 微调** —— 当作 Phase 0–1 的最早可训练数据集使用。后续主路径 Phase 2 生成的数据，会与这批 v0 合并训练。
