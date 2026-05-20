# 备选平台：ManiSkill 3

> **定位**：作为 **Phase 4 VLA 多任务预训练数据源**（其公开 demo 数据集），以及 GPU 并行数据生成对比。
>
> **不替换主路径**——SO-ARM 不是 ManiSkill 原生支持，自己 portage 性价比低。

---

## ManiSkill 3 是什么

- **维护**：UC San Diego HaoSu Lab
- **GitHub**：[haosulab/ManiSkill](https://github.com/haosulab/ManiSkill)
- **官方文档**：[maniskill.readthedocs.io](https://maniskill.readthedocs.io/)
- **底层物理**：SAPIEN（自研，GPU 加速）

| 特性 | 说明 |
|------|------|
| **GPU 并行** | SAPIEN 原生 GPU 并行，30000+ FPS 数据生成 |
| **任务库丰富** | 30+ benchmark 任务（PickCube/Stack/Peg-Insert/Mobile Manipulation 等）|
| **多机器人** | Franka / xArm / WidowX / Fetch 内置；SO-ARM 需自导 |
| **多模态观测** | RGB / 深度 / 点云 / segmentation 全支持 |
| **Demo 数据集** | 数百 K 条公开示教数据，方便 VLA 预训练 |

### 对本项目的适用性

| 维度 | 现状 |
|------|------|
| SO-ARM 原生支持 | ❌ 需要自己导入 URDF + 写 robot adapter |
| GPU 数据生成速度 | ⭐⭐⭐⭐⭐ 比 MuJoCo MJX 略快、比 Genesis 慢但稳 |
| 生态成熟度 | ⭐⭐⭐⭐ 比 Genesis 成熟，论文广泛使用 |
| sim2real 案例 | ⭐⭐⭐⭐ 学界大量 ManiSkill→真机案例 |
| LeRobot 衔接 | ⚠️ 需自写 dataset writer |
| **Demo 数据集对 Phase 4 VLA 预训练价值** | ⭐⭐⭐⭐⭐ 最主要价值 |

---

## 安装

```bash
conda create -n maniskill python=3.10 -y
conda activate maniskill
pip install --upgrade mani-skill

# 下载资产
python -m mani_skill.utils.download_asset all
# 或选择性：python -m mani_skill.utils.download_asset ycb partnet_mobility

# 验证
python -m mani_skill.examples.demo_random_action -e "PickCube-v1"
```

弹出 viewer 看到 Franka 在动 → 成功。

---

## 公开 Demo 数据集（最有用的部分）

ManiSkill 维护一批高质量 demo，对 VLA 预训练有用。

```bash
# 下载某任务的 demo
python -m mani_skill.utils.download_demo "PickCube-v1"
# 默认存到 ~/.maniskill/demos/PickCube-v1/
```

**任务库**：[maniskill.readthedocs.io/en/latest/tasks/index.html](https://maniskill.readthedocs.io/en/latest/tasks/index.html)

**对本项目的用法**：
- 把多个 ManiSkill 公开 demo 转成 LeRobot 格式
- 在 Phase 4 微调 SmolVLA / Pi-0.5 时作为**多任务预训练 / 辅助任务数据**
- 不直接用于 SO-ARM 单任务训练（机器人不同）

---

## LeRobot 数据格式桥接

```python
# data/converters/maniskill_to_lerobot.py
import h5py
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def convert_maniskill_demo(h5_path, dataset, task_desc):
    with h5py.File(h5_path, "r") as f:
        for ep_key in f.keys():
            ep = f[ep_key]
            T = len(ep["actions"])
            for t in range(T):
                dataset.add_frame({
                    "observation.images.front": ep["observations/sensor_data/base_camera/rgb"][t],
                    "observation.state":        ep["observations/agent/qpos"][t].astype("float32"),
                    "action":                   ep["actions"][t].astype("float32"),
                    "task": task_desc,
                })
            dataset.save_episode()
```

**注意**：
- ManiSkill 的 action 维度（Franka 7-D）与 SO-101 6-D 不同——预训练时这两类数据要么走 per-robot head（OpenVLA 风格），要么用 padding（Pi-0 风格）
- 图像分辨率可能与本项目不一致（默认 128×128），训练时统一 resize

---

## GPU 并行采集（次要用途）

如果想自己采，单卡 RTX 4090 上约 25K FPS：

```python
import gymnasium as gym, mani_skill.envs, torch

env = gym.make(
    "PickCube-v1",
    num_envs=512,                # 512 并行
    obs_mode="rgb",
    control_mode="pd_joint_delta_pos",
    sim_backend="gpu",
)
obs, _ = env.reset(seed=0)
for step in range(1000):
    action = torch.randn(env.action_space.shape, device="cuda")
    obs, reward, term, trunc, info = env.step(action)
```

---

## 如果非要导入 SO-ARM100

ManiSkill 的 robot 系统比 MuJoCo 复杂——需要 robot adapter 类：

```python
# robots/so_arm100.py
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.registration import register_agent
from mani_skill.agents.controllers import PDJointPosControllerConfig

@register_agent()
class SoArm100(BaseAgent):
    uid = "so_arm100"
    urdf_path = "assets/menagerie/trs_so_arm100/so_arm100.urdf"

    @property
    def _controller_configs(self):
        return dict(
            pd_joint_pos=dict(
                arm=PDJointPosControllerConfig(
                    joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
                    lower=-3.14, upper=3.14, stiffness=200, damping=20,
                    normalize_action=False,
                ),
                gripper=PDJointPosControllerConfig(
                    joint_names=["gripper"], lower=0.0, upper=0.04, stiffness=200, damping=20,
                ),
            ),
        )
```

然后 `gym.make("PickCube-v1", robot_uids="so_arm100", ...)`。

> **注意**：mujoco_menagerie 提供 MJCF 不是 URDF。需要从原始 SO-ARM100 仓库拉 URDF，或用 `mjcf_to_urdf` 工具转。**不推荐这条路**——SO-101 主线已经在 MuJoCo 和 LeIsaac 跑通。

---

## 关键参考链接

| 资源 | URL |
|------|-----|
| GitHub | https://github.com/haosulab/ManiSkill |
| 文档 | https://maniskill.readthedocs.io/ |
| 任务列表 | https://maniskill.readthedocs.io/en/latest/tasks/index.html |
| 公开 demo 数据集列表 | https://maniskill.readthedocs.io/en/latest/user_guide/datasets/datasets.html |

---

## 风险与陷阱

| 风险 | 应对 |
|------|------|
| ManiSkill demo action 与 SO-101 维度不同 | 按 VLA 选择走 padding 或 per-robot head；详见 [phase4 §数据流水线分支](phase4-vla-finetuning.md) |
| 图像分辨率不一致 | 训练时 resize 到 SO-101 统一分辨率（640×480）|
| GPU 模式吃显存（512 并行）| 先 64 并行起步，看 nvidia-smi |
| ManiSkill 版本升级 API 变 | 锁定 pip 版本 |

---

## 决策建议

| 你的情况 | 是否用 ManiSkill |
|----------|-----------------|
| Phase 4 想做 VLA 多任务预训练 | ✅ 转其公开 demo 作辅助数据 |
| Phase 6 想做"数据生成平台对比" | ✅ 与 Genesis / MuJoCo 三方对比 |
| 想给 SO-ARM 加新任务 | ❌ 用 MuJoCo / LeIsaac 更顺 |
| 只想最快真机部署 | ❌ 不要分心 |
