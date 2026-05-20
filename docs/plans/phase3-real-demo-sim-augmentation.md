# Phase 3：LeIsaac 遥操采集 + IsaacLab Mimic 扩增

**目标**：用真实 SO-101 leader 在 LeIsaac 仿真里遥操 follower，采 30–50 条种子 demo，用 IsaacLab Mimic 扩增到 1.5K–5K 条，转 LeRobot v3 格式供 VLA 微调。

**周期**：1–2 周

**任务**：把红 cube 放进 plate。详细规约见 [README.md](README.md)。

---

## 选用此方案的依据

- **LeIsaac 是 LeRobot 官方 EnvHub 收录的仿真环境**（不是第三方）：[LeRobot 官方 EnvHub LeIsaac 文档](https://huggingface.co/docs/lerobot/envhub_leisaac)
- **IsaacLab Mimic 是 NVIDIA 在 IsaacLab 框架内重写的 MimicGen**（独立于 `mimicgen.github.io` 原版）：[IsaacLab 增广模仿学习](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/augmented_imitation.html)
- **LeIsaac 自带 `LeIsaac-SO101-LiftCube-Mimic-v0` 任务示例 + 全套 mimic 脚本**：[LeIsaac 文档首页](https://lightwheelai.github.io/leisaac/)
- **LeRobot 仓库不带 dataset augmentation**，扩增只能通过 LeIsaac × IsaacLab Mimic（成熟）或 LeIsaac × Cosmos（视频生成，新）两条路；本 Phase 走 Mimic

> 之前文档里的 MuJoCo 自实现 + 自写 `MG_EnvInterface` 路线已**全部作废**。残留代码（`sim/collectors/`、`data/mimicgen_official/`、`data/converters/lerobot_to_robomimic.py`）跑通 Phase 3 后可统一删除。
>
> 备用 VLA 方案见 [alt-platform-groot-n15.md](alt-platform-groot-n15.md)（NVIDIA × HF 官方背书的 GR00T N1.5 + SO-101 微调路线）。

---

## 整体流程

```
LeIsaac LiftCube-Mimic 健康检查（验证官方链路可跑）
              │
              ▼
基于 lift_cube/ copy 出 pick_place_red/（加 plate + 改 success）
              │
              ▼
遥操录制 30–50 条种子 HDF5  ──►  IsaacLab Mimic annotate ──►  generate_dataset 扩增
                                                                       │
                                                                       ▼
                                                          isaaclab2lerobot HDF5 → LeRobot v3
                                                                       │
                                                                       ▼
                                                          LeRobot 训练 ACT / Diffusion / VLA
```

---

## T3.1 在 LiftCube-Mimic 上跑通官方链路（端到端验证）

**先用官方任务把链路跑通，再去改 PickPlaceRed**。如果 LiftCube 链路有问题，新任务必然也有问题。

```bash
conda activate py312_cu121
cd /home/zzg/workspace/pycharm/leisaac

# 1) 列出注册任务，确认 LiftCube-Mimic-v0 在
ls scripts/                                          # 看官方脚本目录
ls source/leisaac/leisaac/tasks/lift_cube/           # 看任务实现

# 2) 录 3–5 条 LiftCube 种子（GUI 出现后 b 开始/n 通过/r 丢弃/q 退出）
#    [脚本作用] teleop_se3_agent.py = leisaac 遥操主入口
#    启动 Isaac Sim → 加载指定 task → 接 leader 设备 → 实时把 leader 关节/姿态
#    映射成 sim follower 的 action → 每一步收集 obs+action+image 写 HDF5。
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-SO101-LiftCube-v0 \
    --teleop_device=so101leader --port=/dev/ttyACM0 \
    --num_envs=1 --device=cuda --enable_cameras \
    --record --dataset_file=./datasets/liftcube_seeds.hdf5

# 3) 走官方 mimic 三段：eef 转换 → annotate → generate
#    具体脚本路径以仓库实际为准；首次跑前 ls scripts/mimic/ 或 find . -name "generate_dataset*"

# [脚本作用] eef_action_process.py = 关节空间 action ↔ 末端空间 action 转换
# 录制时 action 是 6 关节角；mimic 算法需要 EEF 6-DoF 位姿+夹爪做"对象中心
# 轨迹变换"。本步用 FK 把每条 demo 的 joint trajectory 转成 EEF trajectory，
# 写回 HDF5 的 `actions_eef` 字段。
# 标志位（二选一必填）：
#   --to_ik    : 关节 → EEF（喂给 mimic 的方向，本节用这个）
#   --to_joint : EEF → 关节（mimic 输出后回放/训练用）
# --headless 跑无窗口加速。
python scripts/mimic/eef_action_process.py \
    --input_file=./datasets/liftcube_seeds.hdf5 \
    --output_file=./datasets/liftcube_seeds_eef.hdf5 \
    --to_ik --headless

# [脚本作用] annotate_demos.py = 自动切分 subtask 边界
# 读取 ObservationsCfg.SubtaskCfg 里定义的 ObsTerm（如 pick_cube → mdp.object_grasped），
# 在 timeline 上找 0→1 的 edge 作为该 subtask 的"完成点"，写进 HDF5 的
# `datagen_info/subtask_term_signals/<name>` 数组。后续 generate 步骤按这些
# 边界把每条 demo 切成多段，对每段独立做位姿变换。
python scripts/mimic/annotate_demos.py \
    --task=LeIsaac-SO101-LiftCube-Mimic-v0 \
    --input_file=./datasets/liftcube_seeds_eef.hdf5 \
    --output_file=./datasets/liftcube_seeds_annotated.hdf5

# [脚本作用] generate_dataset.py = IsaacLab Mimic 扩增主引擎
# 对每个 SubTaskConfig：① 选最近邻种子轨迹 ② 按当前 object_ref 位姿做刚体变换
# ③ 拼接子段+插值过渡 ④ 在新 init state 下并行回放（num_envs 个并行 env）
# ⑤ 用 termination 判断成功并保留。N 条种子 ≈ 出 N×10 条扩增 demo（成功率 ~70%）。
python scripts/mimic/generate_dataset.py \
    --task=LeIsaac-SO101-LiftCube-Mimic-v0 \
    --input_file=./datasets/liftcube_seeds_annotated.hdf5 \
    --output_file=./datasets/liftcube_mimic.hdf5 \
    --num_envs=4 --device=cuda

# 4) 转 LeRobot v3 格式
# [脚本作用] isaaclab2lerobot.py = HDF5 → LeRobot v3 parquet/mp4
# 读取 IsaacLab HDF5（每条 episode 一个 group），重新组帧 + 视频编码成
# LeRobot v3 标准布局：data/*.parquet + videos/*.mp4 + meta/info.json。
# 输出 repo_id 写到 ~/.cache/huggingface/lerobot/<repo_id>/，可直接被
# LeRobotDataset 加载训练。
python scripts/convert/isaaclab2lerobot.py \
    --input ./datasets/liftcube_mimic.hdf5 \
    --output-repo-id local/liftcube_mimic_v0
```

**验证**：

- 步骤 2 弹出 Isaac Sim 窗口、leader 同步 follower、录的 episode 数 ≥ 3
- 步骤 3 generate 输出文件大小约为种子 × 10（成功率 ~70%）
- 步骤 4 LeRobot dataset 含 `observation.images.front`、`observation.state`、`action`

> **重要**：T3.2 之后所有改动都假设这条链路已通。如果步骤 3 任何脚本不存在或参数不一致，**先查 LeIsaac 仓库最新 README 而不是改我的命令**——LeIsaac 0.4.0 之后的脚本命名/参数可能微调。

参考：[IsaacLab Mimic Teleop 官方教程](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html)

---

## T3.2 基于 lift_cube/ 改出 pick_place_red/

LeIsaac `lift_cube` 任务的实际结构（实测）：

```
leisaac/source/leisaac/leisaac/tasks/lift_cube/
├── __init__.py                   # 注册 v0 / DigitalTwin / Mimic / Direct 四变体
├── lift_cube_env_cfg.py          # LiftCubeEnvCfg(SingleArmTaskEnvCfg)
├── lift_cube_mimic_env_cfg.py    # LiftCubeMimicEnvCfg(LiftCubeEnvCfg, MimicEnvCfg)
└── mdp/{observations,terminations}.py
```

关键事实：
- 基类是 `SingleArmTaskEnvCfg`（`leisaac.tasks.template`），已预置 SO-101 follower + teleop 接入 + `subtask_terms` ObsGroup
- cube 实体名固定为 `"cube"`（不要改）
- 场景是 USD `TABLE_WITH_CUBE_CFG` + `parse_usd_and_create_subassets`
- 现成 success 条件用 `mdp.cube_height_above_base`

### 6 步实操

```bash
LEISAAC=/home/zzg/workspace/pycharm/leisaac/source/leisaac/leisaac

# Step 1: copy 整个 lift_cube 目录作起点
cp -r $LEISAAC/tasks/lift_cube $LEISAAC/tasks/pick_place_red
```

**Step 2**：改 `pick_place_red/__init__.py` —— 把 `LiftCube` 全替换成 `PickPlaceRed`：

```python
gym.register(
    id="LeIsaac-SO101-PickPlaceRed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.pick_place_red_env_cfg:PickPlaceRedEnvCfg"},
)
gym.register(
    id="LeIsaac-SO101-PickPlaceRed-Mimic-v0",
    entry_point="leisaac.enhance.envs:ManagerBasedRLLeIsaacMimicEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.pick_place_red_mimic_env_cfg:PickPlaceRedMimicEnvCfg"},
)
```

**Step 3**：改 `pick_place_red_env_cfg.py` —— rename 类，在 `__post_init__` 里加 plate，把 success term 指向新函数：

```python
from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils

# class TerminationsCfg: success 改成
success = DoneTerm(
    func=mdp.cube_on_plate,
    params={
        "cube_cfg": SceneEntityCfg("cube"),
        "plate_cfg": SceneEntityCfg("plate"),
        "xy_tol": 0.05,
        "z_tol": 0.025,
    },
)

# class PickPlaceRedEnvCfg.__post_init__ 末尾加 plate
self.scene.plate = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Plate",
    spawn=sim_utils.CylinderCfg(
        radius=0.06, height=0.01,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9,0.9,0.92)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.10, 0.005)),
)
```

**Step 4**：在 `mdp/terminations.py` 末尾追加 `cube_on_plate`：

```python
def cube_on_plate(env, cube_cfg, plate_cfg, xy_tol=0.05, z_tol=0.025):
    cube = env.scene[cube_cfg.name]
    plate = env.scene[plate_cfg.name]
    xy_dist = torch.linalg.norm(cube.data.root_pos_w[:,:2] - plate.data.root_pos_w[:,:2], dim=-1)
    z_gap = torch.abs(cube.data.root_pos_w[:,2] - plate.data.root_pos_w[:,2])
    return (xy_dist < xy_tol) & (z_gap < z_tol)
```

**Step 5**：改 `pick_place_red_mimic_env_cfg.py` —— rename 类 + 改 datagen name：

```python
# 把所有 LiftCube 改 PickPlaceRed
class PickPlaceRedMimicEnvCfg(PickPlaceRedEnvCfg, MimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.datagen_config.name = "pick_place_red_leisaac_task_v0"
        # 其余 SubTaskConfig 保持原状（subtask 1 是 pick_cube，subtask 2 末段不设 term_signal）
```

**Step 6**：注册到 tasks 索引：

```bash
echo "from . import pick_place_red  # noqa: F401" \
    >> $LEISAAC/tasks/__init__.py
```

### 验证

```bash
cd /home/zzg/workspace/pycharm/leisaac
# [脚本作用] 同 T3.1：teleop_se3_agent.py 是 leisaac 遥操主入口。
# 这里不带 --record，纯启动 sim 验证场景能加载、success 条件能触发。
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-SO101-PickPlaceRed-v0 \
    --teleop_device=so101leader --port=/dev/ttyACM0 \
    --num_envs=1 --device=cuda --enable_cameras
```

应当看到 SO-101 + 红 cube + 白圆盘 plate；遥操放 cube 到 plate 上时 success 信号亮起。

---

## T3.2.5 `teleop_se3_agent.py` 参数详解

### 脚本整体作用

`scripts/environments/teleoperation/teleop_se3_agent.py` 是 leisaac 的**遥操主入口**，整条 T3 数据采集链路从这里开始。它做这几件事：

```
1. 通过 AppLauncher 启动 Isaac Sim（带或不带窗口）
2. parse_env_cfg(--task) 加载注册的 gym env（如 PickPlaceRed-v0）
3. 根据 --teleop_device 实例化输入设备：
     so101leader  → 打开 USB serial 串口读 leader 关节角
     keyboard     → 订阅 carb keyboard 事件
     gamepad      → 订阅手柄事件
4. 进入主循环（每 1/--step_hz 秒一次）：
     输入设备 → action 张量 → env.step(action)
                                 ↓
                              Isaac Sim 物理 + 渲染
                                 ↓
                              obs (state + 相机帧) + reward + done
5. 带 --record 时：每帧写 HDF5；按 B/N/R 控制 episode 边界
6. Ctrl+C 或 Q 退出 → 关闭 SimulationApp、保存 HDF5
```

简单说：**leader 关节角 → IK → sim follower action → 渲染 → 存帧**，一条完整 sim2real 数据闭环。下面是它的 CLI 参数（按用途分组）：

### 必填 / 核心

| 参数 | 默认 | 说明 |
|---|---|---|
| `--task` | `None` | gym env id，例如 `LeIsaac-SO101-PickPlaceRed-v0` |
| `--teleop_device` | `keyboard` | 输入设备类型，见下表 |
| `--num_envs` | `1` | 并行 env 数（遥操固定 1，扩增时可>1） |
| `--device` | `cuda` | 计算设备（仿真物理 + 渲染） |
| `--enable_cameras` | flag | 启用相机渲染（录数据**必须开**） |

### `--teleop_device` 可选值

| 值 | 适用 | 备注 |
|---|---|---|
| `keyboard` | 无硬件试玩 | SE(3) 增量控制，本机键盘 |
| `gamepad` | 无硬件试玩 | Xbox/PS 手柄 |
| `so101leader` | **正式录种子用** | 物理 SO-101 leader 臂（USB serial） |
| `bi-so101leader` | 双臂任务 | 左右各一根 leader 臂 |
| `lekiwi-keyboard` / `lekiwi-gamepad` / `lekiwi-leader` | LeKiwi 移动平台 | 当前 PickPlaceRed 不用 |

### 物理 leader 臂相关

| 参数 | 默认 | 说明 |
|---|---|---|
| `--port` | `/dev/ttyACM0` | leader 臂 USB serial 端口（`ls /dev/ttyACM*` 查） |
| `--left_arm_port` / `--right_arm_port` | `ttyACM0` / `ttyACM1` | bi-so101leader 双臂端口 |
| `--remote_endpoint` | `None` | ZMQ 远程 leader（leader 在另一台机器上跑 `so101_joint_state_server.py`） |
| `--recalibrate` | flag | 强制重做标定（默认读 `~/.cache/huggingface/lerobot/.../leader.json`） |

### 录数据 / 数据集导出

| 参数 | 默认 | 说明 |
|---|---|---|
| `--record` | flag | 启用录制（按 `b` 才真正开始写） |
| `--dataset_file` | `./datasets/dataset.hdf5` | 输出 HDF5 路径 |
| `--resume` | flag | 续写到已有 HDF5，不重置 episode_index |
| `--num_demos` | `0` (∞) | 录到第 N 条自动退出 |
| `--step_hz` | `60` | 仿真步进 Hz（IsaacLab 默认 60） |
| `--use_lerobot_recorder` | flag | 用 LeRobot 的 recorder 直接写 LeRobot v3（跳过 HDF5→v3 转换步骤） |
| `--lerobot_dataset_repo_id` | `None` | 配合上一项，LeRobot 数据集 repo_id |
| `--lerobot_dataset_fps` | `30` | LeRobot 数据集帧率（与 `--step_hz` 解耦：record 端可下采样） |

### 其他

| 参数 | 默认 | 说明 |
|---|---|---|
| `--sensitivity` | `1.0` | 输入灵敏度倍数，键盘建议 0.5–2.0，手感太冲就调小 |
| `--seed` | `None` | env 初始化种子（每次 reset 内部仍按 DR 重新采样） |
| `--quality` | flag | 高质量渲染（更真但更慢，正式录 demo 用） |
| `--headless` (AppLauncher) | flag | 无窗口运行（**遥操不用**，扩增/批跑用） |

### GUI / 录制热键（在 Isaac Sim 窗口聚焦下生效）

| 键 | 作用 |
|---|---|
| `B` | begin episode 录制（带 `--record` 时） |
| `N` | next：标当前 episode **成功** + 写 HDF5 + reset |
| `R` | reset：**丢弃**当前 episode（失败 demo 不录） |
| `Q` | 退出脚本 |

---

## T3.2.6 无硬件试玩（`--teleop_device=keyboard`）

leader 臂还没接上时，用键盘验证整个链路（场景加载 + IK + success 触发 + reset）。**这条不录数据**，纯熟悉操作。

### 启动命令

```bash
cd /home/zzg/workspace/pycharm/leisaac
# [脚本作用] 同 T3.1：teleop_se3_agent.py 是 leisaac 遥操主入口。
# 这里 --teleop_device=keyboard 让脚本走 SO101Keyboard 设备类，不接物理串口，
# 用本机键盘事件生成 action（仍调用同样的 IK 控制器 → follower）。
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-SO101-PickPlaceRed-v0 \
    --teleop_device=keyboard \
    --num_envs=1 --device=cuda --enable_cameras \
    --sensitivity=1.0
```

### 键位映射（`SO101Keyboard`）

控制目标是末端 frame `gripper`，**增量姿态控制**，每按一次/持续按一次进行一次微小位移：

| 键 | 动作 | 维度 |
|---|---|---|
| `W` / `S` | **前进 / 后退**（z 方向） | gripper 局部 z 轴 |
| `Q` / `E` | **抬起 / 下落**（x 方向） | gripper 局部 x 轴（竖直方向） |
| `A` / `D` | **左转身 / 右转身** | 整臂 shoulder_pan 关节 |
| `I` / `K` | **俯仰 down / up**（pitch） | gripper 局部 y 轴旋转 |
| `J` / `L` | **偏航 left / right**（yaw） | gripper 局部 z 轴旋转 |
| `U` / `O` | **开 / 合夹爪** | 夹爪关节 |

> ⚠️ 方向是**末端局部坐标系**，不是世界坐标。仿真启动后，**先用鼠标点一下 Isaac Sim 视口让窗口聚焦**。如果按了一个键发现机械臂动错方向，先 `A`/`D` 把臂转身到面对你、再用 W/S/Q/E 比较直观。

---

## T3.2.7 `pick_place_red/` 目录详解 + 新增任务模板

### 当前目录结构

```
leisaac/source/leisaac/leisaac/tasks/pick_place_red/
├── __init__.py                       # gym.register 入口
├── pick_place_red_env_cfg.py         # 普通 RL env 配置（v0）
├── pick_place_red_mimic_env_cfg.py   # Mimic 扩增专用配置（-Mimic-v0）
└── mdp/
    ├── __init__.py                   # 把 observations + terminations re-export
    ├── observations.py               # 自定义 obs 函数（object_grasped）
    └── terminations.py               # 自定义 success 函数（cube_on_plate）
```

整个 `tasks/` 下的子目录由 [tasks/__init__.py](../../../../leisaac/source/leisaac/leisaac/tasks/__init__.py) 用 `isaaclab_tasks.utils.import_packages` 自动发现并执行——**新加目录不需要手动 import**。

### 每个文件的作用

#### `__init__.py` — gym 注册入口

```python
import gymnasium as gym

gym.register(
    id="LeIsaac-SO101-PickPlaceRed-v0",                              # 任务唯一 id
    entry_point="isaaclab.envs:ManagerBasedRLEnv",                   # IsaacLab 通用 RL env 类
    disable_env_checker=True,                                        # 跳过 gym 默认 obs/action 形状校验
    kwargs={"env_cfg_entry_point":                                   # cfg 类位置，惰性加载
            f"{__name__}.pick_place_red_env_cfg:PickPlaceRedEnvCfg"},
)

gym.register(
    id="LeIsaac-SO101-PickPlaceRed-Mimic-v0",
    entry_point="leisaac.enhance.envs:ManagerBasedRLLeIsaacMimicEnv", # Mimic 专用 env，多了 datagen hooks
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point":
            f"{__name__}.pick_place_red_mimic_env_cfg:PickPlaceRedMimicEnvCfg"},
)
```

- **v0** 给遥操录种子用（`teleop_se3_agent.py --task=...-v0`）
- **-Mimic-v0** 给 `annotate_demos.py` / `generate_dataset.py` 用（必须是 Mimic env 才有 `datagen_config` 字段）

#### `pick_place_red_env_cfg.py` — 主环境配置

继承链：`PickPlaceRedEnvCfg → SingleArmTaskEnvCfg (leisaac.tasks.template) → ManagerBasedRLEnvCfg (isaaclab.envs)`

文件内 4 个 `@configclass`：

| 类 | 角色 |
|---|---|
| `PickPlaceRedSceneCfg` | 场景：桌面 USD（`TABLE_WITH_CUBE_CFG`）+ front 相机 + DomeLight |
| `ObservationsCfg` | 观测组：继承 `SingleArmObservationsCfg`，额外加 `SubtaskCfg.pick_cube`（mimic 切分需要） |
| `TerminationsCfg` | 成功条件：`success = DoneTerm(func=mdp.cube_on_plate, ...)` |
| `PickPlaceRedEnvCfg` | 顶层组装：拼场景+obs+termination，`__post_init__` 里加 plate + 域随机化 |

关键代码位（行号会随编辑漂移，看 IDE 跳转）：
- 场景类：[L29](../../../../leisaac/source/leisaac/leisaac/tasks/pick_place_red/pick_place_red_env_cfg.py#L29)
- success 条件指向 `mdp.cube_on_plate`：[L93](../../../../leisaac/source/leisaac/leisaac/tasks/pick_place_red/pick_place_red_env_cfg.py#L93)
- 加 plate 的代码：[L127](../../../../leisaac/source/leisaac/leisaac/tasks/pick_place_red/pick_place_red_env_cfg.py#L127)
- 域随机化（cube 位姿 + 相机抖动）：[L146](../../../../leisaac/source/leisaac/leisaac/tasks/pick_place_red/pick_place_red_env_cfg.py#L146)

#### `pick_place_red_mimic_env_cfg.py` — Mimic 扩增配置

继承链：`PickPlaceRedMimicEnvCfg → PickPlaceRedEnvCfg + MimicEnvCfg`（多继承）

**核心是 `__post_init__` 里两块东西**：

1. **`datagen_config`**：生成超参（成功率门槛、随机种子、subtask 选源策略等），直接 copy LiftCube 的够用
2. **`subtask_configs["so101_follower"]`**：subtask 列表，按时序定义把一条 demo 切成几段：
   - subtask 1：`object_ref="cube"`、`subtask_term_signal="pick_cube"` → 接近+抓取阶段
   - subtask 2：`object_ref="plate"`、`subtask_term_signal=None` → 放置阶段（末段不设 term，跑到 termination 自然结束）

> mimic 扩增是**按 subtask 独立做"位姿变换+轨迹拼接"**——subtask 1 的"以 cube 为中心"意味着每条新 demo cube 落在新位置时，整段抓取轨迹会刚体变换跟过去；subtask 2 同理跟 plate 走。SubTaskConfig 字段含义见文档末尾 [需要先理解的两个概念](#1-subtaskconfig-是什么) 小节。

#### `mdp/observations.py`

只定义一个函数 `object_grasped(env, robot_cfg, ee_frame_cfg, object_cfg)` → 返回 `Tensor[bool]`，被 `ObservationsCfg.SubtaskCfg.pick_cube` 这个 ObsTerm 调用。逻辑：**末端相对 cube 距离 < 2cm 且夹爪关节闭合度 > 阈值** → grasped。

mimic `annotate_demos.py` 读取这个 obs 在 timeline 上找 0→1 edge 作为 subtask 1 的完成点。

#### `mdp/terminations.py`

两个函数：
- `cube_height_above_base(...)` ← LiftCube 留下的，我们不用
- `cube_on_plate(env, cube_cfg, plate_cfg, xy_tol, z_tol)` ← **PickPlaceRed 真正的 success 判定**

被 `TerminationsCfg.success` 引用，每 step 评估，返回 True 时 env 终止（自动 reset + 标记成功）。

#### `mdp/__init__.py`

```python
from isaaclab.envs.mdp import *                # 拿所有 IsaacLab 通用 MDP 函数
from leisaac.enhance.envs.mdp import *         # 拿 leisaac 扩展的 MDP 函数（如 se3 控制）
from .observations import *                    # 把我们的 object_grasped 抛出
from .terminations import *                    # 把 cube_on_plate / cube_height_above_base 抛出
```

后续在 cfg 里写 `mdp.cube_on_plate` 就是这条路径解析的。

### 使用流程串联

```
1. 你改 pick_place_red_env_cfg.py（改场景/物体/DR/success 容差）
                                 ↓
2. python -c "import leisaac; ..."  ← 触发 tasks/__init__.py 自动 import
                                 ↓
3. gym.envs.registry 里出现 LeIsaac-SO101-PickPlaceRed-v0
                                 ↓
4. teleop_se3_agent.py --task=...-v0 启动 sim → 录种子 HDF5
                                 ↓
5. annotate_demos.py --task=...-Mimic-v0 读 ObservationsCfg.SubtaskCfg 切边界
                                 ↓
6. generate_dataset.py --task=...-Mimic-v0 按 subtask_configs 扩增
                                 ↓
7. isaaclab2lerobot.py 转 LeRobot v3
```

---

### 新增任务（如 PickPlaceBlueCube / StackTwoCubes）

按以下 5 步走，可以无脑复用本任务的脚手架：

#### Step 1: 复制目录

```bash
LEISAAC=/home/zzg/workspace/pycharm/leisaac/source/leisaac/leisaac
cp -r $LEISAAC/tasks/pick_place_red $LEISAAC/tasks/<new_task_name>
cd $LEISAAC/tasks/<new_task_name>
```

例如 `<new_task_name>` = `stack_cubes` 或 `place_blue_cube`。

#### Step 2: 改文件名

```bash
mv pick_place_red_env_cfg.py        <new_task_name>_env_cfg.py
mv pick_place_red_mimic_env_cfg.py  <new_task_name>_mimic_env_cfg.py
```

#### Step 3: 改 `__init__.py`

`pick_place_red` 全替换成 `<new_task_name>`，`PickPlaceRed` 全替换成 `<NewTaskName>`（驼峰），`id` 改成 `LeIsaac-SO101-<NewTaskName>-v0` 和 `LeIsaac-SO101-<NewTaskName>-Mimic-v0`。

#### Step 4: 改 `<new_task_name>_env_cfg.py`

按需修改下面这些点（其他保持原样即可）：

| 要改的东西 | 改哪 |
|---|---|
| 场景 USD（换桌面、换房间） | `from leisaac.assets.scenes.<...> import <SCENE>_CFG, <SCENE>_USD_PATH`，替换两处引用 |
| 加新物体（如第二个 cube、bowl） | `__post_init__` 里仿照 plate 那段写 `self.scene.<name> = RigidObjectCfg(...)` |
| 改 success 条件 | 在 `mdp/terminations.py` 写新函数，`TerminationsCfg.success = DoneTerm(func=mdp.<new_fn>, ...)` 引用它 |
| 改 subtask（新动作流） | 在 `mdp/observations.py` 写新 ObsTerm 用的函数，`ObservationsCfg.SubtaskCfg` 里加对应字段 |
| 改任务描述 | `task_description: str = "..."` |
| 改物体随机化范围 / 加新物体的 DR | `__post_init__` 末尾 `domain_randomization(self, random_options=[...])` 添加 `randomize_object_uniform("<name>", pose_range={...})` |

#### Step 5: 改 `<new_task_name>_mimic_env_cfg.py`

```python
from .<new_task_name>_env_cfg import <NewTaskName>EnvCfg

class <NewTaskName>MimicEnvCfg(<NewTaskName>EnvCfg, MimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.datagen_config.name = "<new_task_name>_leisaac_task_v0"   # 唯一标识
        # ... datagen_config 其他字段沿用 ...

        subtask_configs = []
        # subtask 1
        subtask_configs.append(SubTaskConfig(
            object_ref="<物体名>",                    # 这一段以谁为参考帧
            subtask_term_signal="<ObsTerm 名字>",    # 何时切到下一段
            ...
        ))
        # subtask 2, 3, ...
        self.subtask_configs["so101_follower"] = subtask_configs
```

**SubTaskConfig 数量决定扩增灵活度**：1 段=只随物体位姿变换；3 段=三个动作阶段独立组合，最灵活但需要 3 个 ObsTerm 信号定义。

#### Step 6: 验证

```bash
conda activate py311_leisaac
cd /home/zzg/workspace/pycharm/leisaac

# 注册检查
python -c "
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import leisaac, gymnasium as gym
ids = [e for e in gym.envs.registry.keys() if '<NewTaskName>' in e]
print(ids)
assert f'LeIsaac-SO101-<NewTaskName>-v0' in ids
app.close()
"

# 场景视觉检查（不接 leader）
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-SO101-<NewTaskName>-v0 \
    --teleop_device=keyboard \
    --num_envs=1 --device=cuda --enable_cameras
```

通过后 T3.3–T3.6 流程完全复用，只换任务名 / HDF5 文件名。

---

### 常见坑

1. **`from . import mdp` 找不到新函数** → 检查 `mdp/__init__.py` 是否有 `from .terminations import *`（star import 才会捞新函数）
2. **`SceneEntityCfg("<name>")` 报 KeyError** → 物体没注册到场景，确认 `__post_init__` 里写了 `self.scene.<name> = ...` 且名字一致
3. **mimic generate 全失败** → ObsTerm 信号在录的种子里从未触发 0→1，去 HDF5 里 dump `datagen_info/subtask_term_signals/<name>` 看 timeline
4. **新任务 USD 资产没下载** → 同 T3.2 验证遇到的"scene.usd not found"，按那一节方法从 HF 拉
5. **改了 cfg 但仿真还是老的** → IsaacLab 会缓存 USD 解析结果，重启进程而不是热重载

---

## T3.3 标定 leader（LeRobot 官方）

```bash
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=leader
```

校准 JSON 自动写到 `~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/leader.json`。LeIsaac 内部自动读取这个文件做 leader→sim follower 映射，无需手写其他配置。

---

## T3.4 录种子 demo

**多样性 checklist**（30–50 条覆盖到）：

- cube 初始 xy 至少落在 5×5 网格的不同格子（DR 已经会随机化，但保证人为路径不雷同）
- 抓取角度：正面、侧面、斜 45°
- 抓取高度：直上直下、由高处下降、由侧面靠近
- 失败案例不录（让 IsaacLab Mimic 只学好行为）

```bash
cd /home/zzg/workspace/pycharm/leisaac

# [脚本作用] 同 T3.1：teleop_se3_agent.py 是 leisaac 遥操主入口。
# 带 --record 后开始把每帧 obs+action 写到 --dataset_file 指定的 HDF5，
# 用 B 开始 / N 通过 / R 丢弃 / Q 退出 控制 episode 边界。
# smoke 5 条（先试录、确认链路通）
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-SO101-PickPlaceRed-v0 \
    --teleop_device=so101leader --port=/dev/ttyACM0 \
    --num_envs=1 --device=cuda --enable_cameras \
    --record --dataset_file=./datasets/pickplace_smoke.hdf5

# 正式 30–50 条（要写入不同 HDF5 避免和 smoke 混）
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-SO101-PickPlaceRed-v0 \
    --teleop_device=so101leader --port=/dev/ttyACM0 \
    --num_envs=1 --device=cuda --enable_cameras \
    --record --dataset_file=./datasets/pickplace_seeds.hdf5
```

GUI 键位：`b` begin、`n` next（标成功 + reset）、`r` reset（丢弃）、`q` 退出。

### 录完后审计 HDF5 内容

每次录完种子建议**先扫一遍 HDF5 结构**，确认相机/action/state 都齐了、帧数合理、失败 demo 数量在预期内。用 [`data/converters/inspect_hdf5.py`](../../data/converters/inspect_hdf5.py)：

```bash
cd /home/zzg/workspace/pycharm/Robot
python -m data.converters.inspect_hdf5 \
    /home/zzg/workspace/pycharm/leisaac/datasets/pickplace_seeds.hdf5
```

输出包含 4 部分：

1. **ROOT 元数据**：文件大小、顶层 attrs、children
2. **DEMOS SUMMARY**：每条 demo 的 `success` / `num_samples` / `obs` 键列表
3. **BYTES BY KEY**：每类数据占多少 MB（一眼看出谁在吃空间）
4. **DETAIL TREE + SAMPLE VALUES**：挑第一条 success demo 完整展开 + 关键数组（actions / joint_pos / ee_frame_state）取首帧、min/max 看是否合理

**正常的预期**（PickPlaceRed v0）：

| 检查项 | 期望 |
|---|---|
| `data/demo_X/obs/front` | shape=(N, 512, 512, 3) uint8 |
| `data/demo_X/obs/wrist` | shape=(N, 512, 512, 3) uint8 ← **加了 wrist 相机后** |
| `data/demo_X/actions` | shape=(N, 6) float32，前 5 维关节 ±π 范围，第 6 维夹爪 0-1 |
| `data/demo_X/obs/ee_frame_state` | shape=(N, 7) float32，前 3 维 xyz，后 4 维 quat (wxyz) |
| `data/demo_X/states/rigid_object/cube/root_pose` | shape=(N, 7)，cube 全程轨迹 |
| `data/demo_X/states/rigid_object/plate/root_pose` | shape=(N, 7)，plate 全程轨迹 |
| `data/demo_X/initial_state/...` | shape=(1, ...)，reset 时的 state |
| `attrs.success` | True 占比 ≥ 80%（多 R 丢失败的） |
| `attrs.num_samples` | 单条 demo 200-600 帧（10-30 秒 × 20 Hz） |

**常见异常**：

| 现象 | 含义 / 修法 |
|---|---|
| `obs/front` 缺失 | 录制时没加 `--enable_cameras` |
| `obs/wrist` 缺失 | wrist 相机没在 SceneCfg 里注册成功，看 import 报错 |
| `num_samples=0` 的 demo 占大头 | 录的时候 B/N 太快、没真步进 sim，参考 T3.2.6 操作序列 |
| 单条 demo > 1000 帧 | 操作太慢，目标 ≤600 帧/条 |
| `actions` 值域全 0 | leader 没接上 / 键盘焦点不在 viewport |

**`inspect_hdf5` 只打印 shape/sample，看不到画面**。要肉眼确认 front/wrist 视频内容、轨迹是否连贯、夹爪何时闭合，用 [`data/converters/extract_hdf5_demos.py`](../../data/converters/extract_hdf5_demos.py) 把每条 demo 拆成可播放的 mp4 + PNG + CSV：

```bash
cd /home/zzg/workspace/pycharm/Robot
/home/zzg/miniconda3/envs/py312_cu121/bin/python \
    -m data.converters.extract_hdf5_demos \
    --input  /home/zzg/workspace/pycharm/leisaac/datasets/pickplace_seeds.hdf5 \
    --output /home/zzg/workspace/pycharm/leisaac/datasets/pickplace_seeds_extracted \
    --fps 15

# 常用变体：
#   --demo demo_0   只抽一条（先小验证再全量）
#   --no-video      只生成首末帧 PNG + CSV（不写 mp4，秒级完成）
```

> 注意走 `py312_cu121` 解释器：脚本用到 `imageio + imageio_ffmpeg`，只在该环境装了。

每条 demo 产出目录 `pickplace_seeds_extracted/demo_X/` 内含：

| 文件 | 用途 |
|---|---|
| `front.mp4` / `wrist.mp4` | 拖播放器看动作流畅度、视角、wrist 临近时方块是否清晰 |
| `front_first.png` / `front_last.png` | 起止画面 — 起始 cube/plate 在 DR 范围内？最后 cube 是否落到 plate 上？ |
| `wrist_first.png` / `wrist_last.png` | wrist 拍到桌面（正常）还是天花板/全黑（pose 错） |
| `actions.csv` | 每行一帧 6 维动作 — 方差太小或全零 = leader 没驱动 |
| `obs_joint_pos.csv` / `obs_ee_frame_state.csv` | 关节角 / 末端位姿，可对照 `actions` 看 IK 是否正常 |
| `meta.json` | `attrs.success` / `num_samples` / 各 dataset shape |

**肉眼快速验真清单**：

1. `front_first.png`：cube 在 ±7.5 cm 方框内，plate 在 (0.2,-0.35) ±5 cm 邻域
2. `wrist_first.png`：能看到桌面 + 夹爪指尖（不是天花板 / 黑屏）
3. 播 `front.mp4`：轨迹连贯、夹爪逼近 cube → 闭合 → 抬起 → 移到 plate 上方 → 张开
4. `meta.json["attrs"]["success"]` = True 的 demo 占大头
5. `actions.csv` 第 6 维（夹爪）应在 0–1 之间且有明显跳变（grasp/release 时刻）

### HDF5 体积控制

`--record` **默认 `EXPORT_ALL`**——失败 demo 也会写入。

**实测数据成分**（4 条 demo / 4045 帧 / 640×480 / 30FPS）：

| 数据键 | 占比 |
|---|---|
| `obs/front`（RGB 图像） | **99.95%**（3555 MB） |
| 其他全部（joint/action/state） | 0.05%（1.6 MB） |

结论：体积**完全被相机帧主导**，joint/action/state 全是 float32 小数组可忽略。压体积只能从图像下手。

`pick_place_red_env_cfg.py` 当前配置：
- **分辨率 512×512**（匹配 SmolVLA 原生输入，相比 640×480 减 ~2.3×）
- **15 Hz 相机**（覆盖 Pi0.5/SmolVLA/ACT 的 10–15 Hz 训练率 + GR00T N1.5 的 20 Hz default；camera FPS 不影响 mimic 重放——mimic 只用 EEF action 不用相机）
- **lzf 压缩**默认开（[teleop_se3_agent.py:246](../../source/leisaac/scripts/environments/teleoperation/teleop_se3_agent.py#L246)，对 RGB 压缩比 ~2×）

体积估算（40 秒/条 × 30 FPS = 1200 帧/条）：

```
单帧:  512 × 512 × 3 = 786 KB
单 demo raw: 1200 × 786 KB ≈ 920 MB
lzf 压缩后: ≈ 460 MB
30 条种子 demo: ≈ 14 GB
```

> 还是不小。如果磁盘吃紧，**录的时候控制 demo 时长**（理想 10–15 秒/条而不是 40 秒，加快 demo 节奏）。10 秒/条的 30 条种子 ≈ 3.5 GB，更合理。

**录到失败 demo 的处理**：

- ✅ **推荐**：录制时主动按 `R` 丢弃失败 episode（buffer 直接丢，HDF5 不写）
- 兜底：录完后跑过滤脚本剔掉 `success=False` 的：

```bash
cd /home/zzg/workspace/pycharm/Robot
python -m data.converters.filter_failed_demos \
    --input  /home/zzg/workspace/pycharm/leisaac/datasets/pickplace_seeds.hdf5 \
    --output /home/zzg/workspace/pycharm/leisaac/datasets/pickplace_seeds_clean.hdf5

# 后续 mimic pipeline 用 _clean.hdf5
```

脚本：[`data/converters/filter_failed_demos.py`](../../data/converters/filter_failed_demos.py)。

**如果还是太大的下一步降级方向**：

- 降分辨率到 256×256（VLA 训练时 SmolVLA 内部会 resize，原生 512 更多是供后续 fine-tune 用，256 也能跑）
- 降相机 FPS：`update_period=1/15.0`（mimic 重放需重新验证时序）
- 减少种子数量（30 → 15）让 mimic 多扩增承担多样性

---

## T3.5 用 IsaacLab Mimic 扩增

照搬 T3.1 步骤 3 的官方三段，把任务名 + 数据文件名换成 PickPlaceRed。每个脚本的作用见 T3.1 注释，这里只贴命令：

> ⚠️ **`--enable_cameras` 必加**：annotate / generate 都会加载完整 `PickPlaceRedMimicEnvCfg`，SceneCfg 含 `front` + `wrist` 两个 `TiledCameraCfg`，Isaac Sim 启动时会强制要求该 flag，否则报 `RuntimeError: A camera was spawned without the --enable_cameras flag`。`--headless` 控制有无 GUI 窗口，跟相机渲染是独立的两件事。
>
> ⚠️ **务必跑过 `filter_failed_demos` 拿到 `_clean.hdf5` 再喂 eef_action_process**：mimic pipeline 不容忍 `num_samples=0` 的空 demo（FK / annotate 都会崩）。

```bash
# [脚本作用] eef_action_process.py = 关节 action → EEF action（mimic 算法要的格式）
python scripts/mimic/eef_action_process.py \
    --input_file=./datasets/pickplace_seeds_clean.hdf5 \
    --output_file=/media/zzg/GJ_disk01/data/leisaac/datasets/pickplace_seeds_eef.hdf5 \
    --to_ik --headless --enable_cameras

# [脚本作用] annotate_demos.py = 自动标 subtask 完成边界（基于 ObsTerm 信号）
# ⚠️ --auto 必加：不传则进入"逐 episode 手动按 N/S/Q 标注"模式（18 条 = 按 18 次），
# 加上 --auto 让脚本调用 mdp.object_grasped 在 timeline 上自动找 0→1 edge。
python scripts/mimic/annotate_demos.py \
    --task=LeIsaac-SO101-PickPlaceRed-Mimic-v0 \
    --input_file=/media/zzg/GJ_disk01/data/leisaac/datasets/pickplace_seeds_eef.hdf5 \
    --output_file=/media/zzg/GJ_disk01/data/leisaac/datasets/pickplace_seeds_annotated.hdf5 \
    --auto --headless --enable_cameras

# [脚本作用] generate_dataset.py = mimic 主引擎，对象中心轨迹变换 + 并行回放
# --generation_num_trials 直接决定输出 episode 数（覆盖 cfg 里默认的 10）
# 2 路相机 × N 个 env 同时渲染：21GB 显存可用 num_envs=8；显存吃紧降到 4/2
python scripts/mimic/generate_dataset.py \
    --task=LeIsaac-SO101-PickPlaceRed-Mimic-v0 \
    --input_file=/media/zzg/GJ_disk01/data/leisaac/datasets/pickplace_seeds_annotated.hdf5 \
    --output_file=/media/zzg/GJ_disk01/data/leisaac/datasets/pickplace_mimic.hdf5 \
    --generation_num_trials=500 \
    --num_envs=4 --device=cuda --headless --enable_cameras
```

参考：[IsaacLab Mimic datagen API](https://isaac-sim.github.io/IsaacLab/main/source/api/lab_mimic/isaaclab_mimic.datagen.html)

---

## T3.6 转 LeRobot v3 格式

```bash
# [脚本作用] isaaclab2lerobot.py = IsaacLab HDF5 → LeRobot v3
# 重新组帧 + 视频编码到 ~/.cache/huggingface/lerobot/<repo_id>/
# 输出含 data/*.parquet + videos/*.mp4 + meta/info.json，可直接被 LeRobotDataset 加载。
python scripts/convert/isaaclab2lerobot.py \
    --input ./datasets/pickplace_mimic.hdf5 \
    --output-repo-id local/so101_pickplace_mimic_v0
```

**校验**：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
d = LeRobotDataset("local/so101_pickplace_mimic_v0")
print("episodes:", d.num_episodes)
print("features:", list(d.features.keys()))
assert "observation.images.front" in d.features
assert "action" in d.features
```

---

## 验收

| 检查项 | 期望 |
|---|---|
| `LeIsaac-SO101-PickPlaceRed-v0` / `-Mimic-v0` 注册成功 | ✅ |
| `pickplace_seeds.hdf5` episodes ≥ 30 | ✅ |
| `pickplace_mimic.hdf5` episodes ≈ seeds × 7（成功率约 70%） | ✅ |
| LeRobot dataset 含 RGB、state、action | ✅ |
| 可在 LeRobot 训练脚本里直接喂入（dataset key 全对得上） | ✅ |

---

## 关键参考链接

| 文档 | 用途 |
|---|---|
| [LeRobot EnvHub × LeIsaac](https://huggingface.co/docs/lerobot/envhub_leisaac) | LeRobot 官方收录说明（确认本方案是官方路径） |
| [LeIsaac 文档首页](https://lightwheelai.github.io/leisaac/) | LeIsaac 安装、任务、teleop |
| [LeIsaac × Cosmos](https://lightwheelai.github.io/leisaac/docs/tutorials/cosmos_tutorial/) | 备选扩增方案（视频生成 + IDM），未来扩规模时考虑 |
| [IsaacLab 增广模仿学习](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/augmented_imitation.html) | IsaacLab Mimic 的 NVIDIA 官方说明 |
| [IsaacLab Mimic Teleop 教程](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html) | annotate + generate_dataset 完整示例 |
| [IsaacLab Mimic API](https://isaac-sim.github.io/IsaacLab/main/source/api/lab_mimic/isaaclab_mimic.datagen.html) | SubTaskConfig、selection_strategy 参数详解 |
| [MimicGen 原论文](https://arxiv.org/pdf/2310.17596) | 理解为什么这种"对象中心轨迹拼接"能扩增数据 |

---

## 需要先理解的两个概念（决定你能改对 PickPlaceRedMimicEnvCfg）

### 1. SubTaskConfig 是什么

IsaacLab Mimic 的核心思想：把一条完整 demo 切成几个**对象中心子任务**（subtask），扩增时对每个子任务独立做"在不同初始位置下的轨迹变换 + 拼接"。

每个 `SubTaskConfig` 字段：

- `object_ref`：本子任务围绕哪个物体（如 `"cube"`）
- `subtask_term_signal`：什么条件下本子任务算完成（对应 `ObsTerm` 名字，如 `"pick_cube"`）
- `subtask_term_offset_range`：在 term 信号触发点前后多少帧切断（增加随机性）
- `selection_strategy`：扩增时从哪条种子选轨迹（`nearest_neighbor_object` 最常用）
- `action_noise`：扩增轨迹叠加的高斯噪声幅度

LiftCube 用了 2 个 subtask：① pick_cube（term=`pick_cube`，对应 `mdp.object_grasped`）② 末段 lift（无 term，靠 termination 收尾）。PickPlaceRed 沿用这两个就够——拼起来等同于"先抓 cube，再放到 plate 上"。

### 2. 为什么不需要写 `MG_EnvInterface`

原版 MimicGen 要求宿主环境实现 `MG_EnvInterface`（`get_object_poses`、`target_pose_to_action` 等）。IsaacLab Mimic 用另一套机制：

- 物体位姿通过 IsaacLab 的 `scene[name].data.root_pos_w` 自动拿到
- 子任务完成信号通过 `ObsTerm` 写进 `subtask_terms` 观测组，被 annotate_demos.py 自动抽取
- 动作回放通过 IsaacLab 的 IK 控制器自动处理

所以**只要 `LiftCubeMimicEnvCfg` 能跑，照搬就是 `PickPlaceRedMimicEnvCfg`** —— 这是路径 B 比自写 MuJoCo + MimicGen 省事一个数量级的根本原因。
