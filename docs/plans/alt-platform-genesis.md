# 备选平台：Genesis

> **定位**：作为 **Phase 6 GPU 加速数据生成对比实验** 的候选平台。**不替换主路径**（Phase 1/2 MuJoCo + Phase 3 LeIsaac）。
>
> Phase 5 真机验证完成后，若想研究"GPU 大规模合成数据是否能压过 LeIsaac Mimic 扩增"，Genesis 是首选对比对象（速度优势明显）。

---

## Genesis 是什么

- **发布**：2024 年 12 月，CMU / 上海 AI Lab / UMD / MIT / 北大 / Stanford 等联合发起
- **GitHub**：[Genesis-Embodied-AI/Genesis](https://github.com/Genesis-Embodied-AI/Genesis)
- **官网**：[genesis-embodied-ai.github.io](https://genesis-embodied-ai.github.io/)

| 特性 | 说明 |
|------|------|
| **极高速度** | 单卡 RTX4090 官方称可达 4300 万 FPS（远超 MuJoCo MJX / Isaac Sim）|
| **多物理引擎** | 刚体 / 软体 / 流体 / 颗粒物 统一框架 |
| **可微物理** | 原生支持梯度反传，对策略学习友好 |
| **Pythonic API** | 比 MuJoCo XML / Isaac Sim USD 上手简单 |
| **跨平台 GPU** | 支持 NVIDIA / Apple Silicon / CPU |

### 对本项目的适用性

| 维度 | 现状 |
|------|------|
| SO-ARM100/101 原生支持 | ❌ 需要自己导入 URDF/MJCF |
| 生态成熟度 | ⚠️ 比 MuJoCo / Isaac Sim 弱一截 |
| sim2real 公开案例（SO-ARM） | ⚠️ 很少 |
| 数据生成速度 | ⭐⭐⭐⭐⭐ 真的比 MuJoCo 快很多 |
| 与 LeRobot 衔接 | ⚠️ 需自写 dataset writer |
| Phase 6 研究价值 | ⭐⭐⭐⭐ "GPU 大规模数据 vs Mimic 扩增"对比 |

---

## 安装

```bash
# 独立 conda 环境（避免与 LeRobot 主环境冲突）
conda create -n genesis python=3.11 -y
conda activate genesis

# pip 直装（推荐）
pip install genesis-world

# 或从源码（拿最新功能）
git clone https://github.com/Genesis-Embodied-AI/Genesis && cd Genesis && pip install -e .

# 验证
python -c "import genesis as gs; gs.init(); print('Genesis OK')"
```

**要求**：Linux / macOS / Windows；Python 3.10–3.12；CUDA 11.8+ 或 12.x；16GB+ 内存。

---

## 最小示例

### Hello World

```python
import genesis as gs

gs.init(backend=gs.gpu)
scene = gs.Scene(show_viewer=True, sim_options=gs.options.SimOptions(dt=1/60))

scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

scene.build()
for _ in range(1000):
    scene.step()
```

### SO-ARM100 接入

Genesis 没有官方 SO-ARM 资产，**复用 mujoco_menagerie 的 MJCF**：

```python
scene.add_entity(gs.morphs.MJCF(file="assets/menagerie/trs_so_arm100/so_arm100.xml"))
```

> Genesis 对 MJCF 支持还在迭代，actuator / equality constraint 可能不被识别——第一次接入要测试，必要时简化 MJCF。

### GPU 并行（杀手锏）

```python
import genesis as gs, torch
gs.init(backend=gs.gpu)

scene = gs.Scene(show_viewer=False)
scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(gs.morphs.MJCF(file="..."))
cube = scene.add_entity(gs.morphs.Box(size=(0.04,)*3))
scene.build(n_envs=1024, env_spacing=(1.0, 1.0))  # 一次开 1024 个并行 env

target = torch.zeros((1024, 7), device="cuda")
for _ in range(1000):
    robot.set_dofs_position(target)
    scene.step()
```

**何时值得用**：
- Phase 2 想生成 100K+ 条数据（MuJoCo 跑要几天，Genesis 可能几小时）
- Phase 6 对比实验：同任务，MuJoCo / LeIsaac Mimic / Genesis 各生成 10K 条，看哪个训出的 VLA 真机表现更好

---

## 与本项目主路径的衔接

### LeRobot 数据格式桥接

Genesis 没有原生 LeRobot writer，自写桥接：

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

### Phase 6 对比实验设计

```
Phase 3 主路径：LeIsaac × IsaacLab Mimic   → 1500 条 → VLA → 真机评估
Phase 6 对比 A：MuJoCo + 主路径 Phase 2     → 10000 条 → VLA → 真机评估
Phase 6 对比 B：Genesis GPU 并行            → 10000 条 → VLA → 真机评估

度量：相同评估 setup 下三组 VLA 的真机成功率、训练时间、数据生成时间
```

---

## 关键参考链接

| 资源 | URL |
|------|-----|
| GitHub | https://github.com/Genesis-Embodied-AI/Genesis |
| 官方文档 | https://genesis-world.readthedocs.io/ |
| 示例集 | `Genesis/examples/` 目录 |
| 社区论坛 | GitHub Discussions |

---

## 风险与陷阱

| 风险 | 应对 |
|------|------|
| MJCF 兼容性不完整（actuator / sensor 解析失败） | 简化 MJCF；或在 Genesis 里重新定义 actuator |
| 渲染与 MuJoCo 视觉风格差异大 → 策略迁移性差 | 视觉 DR 调到与主路径一致 |
| 1024 并行内存爆 | 先 64 并行起步逐步扩 |
| API 在版本间变化（新平台） | 锁定 pip 版本；记录 `requirements.txt` |
| 公开 sim2real 案例少 | 视为实验，结果不及预期不要硬走 |

---

## 决策建议

| 你的情况 | 是否用 Genesis |
|----------|---------------|
| Phase 1–5 主线 | ❌ 用 MuJoCo / LeIsaac |
| Phase 6 想做"数据生成平台对比"研究 | ✅ 主要候选 |
| Phase 2 想生成 50K+ 数据加速实验 | ✅ 值得 1 周尝试 |
| 有论文 / 公开课题需求 | ✅ 是个热点，写出来好讲故事 |
| 只想最快部署到真机 | ❌ 不要分心 |
