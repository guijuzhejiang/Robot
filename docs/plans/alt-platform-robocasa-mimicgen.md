# RoboCasa / MimicGen（概念参考）

> **状态**：原本作为 Phase 3 数据扩增的实操蓝图，**Phase 3 已改走 LeIsaac × IsaacLab Mimic 官方链路**（见 [phase3-real-demo-sim-augmentation.md](phase3-real-demo-sim-augmentation.md)），本文档降级为**算法原理参考**：理解 MimicGen 类方法为什么能"少量种子 → 大数据集"。
>
> **不再作为实操指南**——原版 MimicGen 官方不支持 SO-101，自建集成需要 2–4 周（[研究结论](https://mimicgen.github.io/docs/introduction/overview.html)，官方只支持 Panda/Sawyer/IIWA/UR5e）。

---

## 体系关系图

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
              DexMimicGen（灵巧手扩展）

   RoboCasa（NVIDIA）
   100+ 厨房场景任务（基于 robosuite + MimicGen）
```

| 组件 | 维护方 | 角色 |
|------|--------|------|
| **robosuite** | ARISE Initiative | 底层物理 + 任务环境 |
| **robomimic** | ARISE Initiative | IL 训练框架 + hdf5 数据格式 |
| **MimicGen** | NVIDIA | **数据扩增核心算法** |
| **RoboCasa** | NVIDIA | 大规模厨房任务库（本项目用不上）|

---

## MimicGen 核心思想（必读）

这是本项目数据扩增逻辑的算法源头——LeIsaac × IsaacLab Mimic 是 NVIDIA 把同样思想在 IsaacLab 框架里独立重写的版本。

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
- **物理可行**（不是合成图像）
- 自动覆盖物体姿态分布
- 与原 demo 的"风格"接近，VLA 学到的策略迁移性好

**算法核心伪代码**（来自 `mimicgen/datagen/data_generator.py` 简化）：

```python
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
            for q in new_joint_traj:
                obs, _, _, _ = self.env.step(self.controller(q))
                full_traj.append(obs)

        # 4. 物理验证
        if self.env.is_success():
            return full_traj
        return None
```

---

## 为什么不直接用 MimicGen 训 SO-101

**原版 MimicGen 官方支持的机械臂**：Panda、Sawyer、IIWA、UR5e（[官方文档](https://mimicgen.github.io/docs/introduction/overview.html)），**全部基于 robosuite 框架**。

**SO-101 + 原版 MimicGen 不存在官方/成熟集成**：
- NVlabs/mimicgen 仓库：无 SO-101 代码或文档
- LeRobot/HuggingFace：SO-101 走的是 Isaac Sim 路线（GR00T N1.5 + LeIsaac × IsaacLab Mimic），与原版 MimicGen 独立
- robosuite (ARISE-Initiative)：无 SO-101 robot model
- 社区：无 stars > 50 的 SO-101 + robosuite/MimicGen 项目

**自建集成工作量** ≈ 2–4 周（按 [datagen_custom 文档](https://mimicgen.github.io/docs/tutorials/datagen_custom.html)）：
1. 为 SO-101 制作 MJCF 模型 + 注册为 robosuite `RobotModel`
2. 实现 robomimic env wrapper（MimicGen 前置依赖）
3. 实现 `MG_EnvInterface` 5 个方法
4. 写任务 `MG_Config`

**结论**：与其自建，不如直接走 NVIDIA 在 IsaacLab 框架内重写的 IsaacLab Mimic——LeIsaac 已经把它集成到 SO-101 的 `LeIsaac-SO101-LiftCube-Mimic-v0`，详见 [phase3](phase3-real-demo-sim-augmentation.md)。

---

## 关键参考链接（学习算法用）

| 资源 | 用途 |
|---|---|
| [mimicgen.github.io 官方文档](https://mimicgen.github.io/docs/introduction/overview.html) | 算法 overview + 官方支持任务列表 |
| [NVlabs/mimicgen GitHub](https://github.com/NVlabs/mimicgen) | 源码（`mimicgen/datagen/data_generator.py` 是算法核心）|
| [robomimic MimicGen datasets](https://robomimic.github.io/docs/datasets/mimicgen.html) | 官方扩增产物 hdf5 下载（Stack/Square/Threading 等）|
| [MimicGen 原论文 (CoRL 2023)](https://arxiv.org/pdf/2310.17596) | 理论详解 |
| [IsaacLab Mimic 官方教程](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html) | NVIDIA 在 IsaacLab 内的重写版本（本项目实际使用）|

---

## 跑官方 demo 的最小路径（仅用于学习算法）

如果你想亲手跑一遍原版 MimicGen 理解流水线，独立 conda 环境装：

```bash
conda create -n robocasa python=3.10 -y
conda activate robocasa
pip install robosuite robomimic
git clone https://github.com/NVlabs/mimicgen && cd mimicgen && pip install -e .

# 下载官方 source demos（每任务 10 条）
python mimicgen/scripts/download_datasets.py --tasks stack square threading

# 跑 Stack 任务扩增（10 → 1000 demo，约 30 分钟）
python mimicgen/scripts/generate_dataset.py \
    --config mimicgen/exps/templates/robosuite/stack.json \
    --num-trajs 1000 --auto-remove-exp

# 回放看效果
python mimicgen/scripts/playback_dataset.py \
    --dataset datasets/generated/stack_d0/demo.hdf5 --n 5 --render
```

看完 `mimicgen/exps/templates/robosuite/stack.json`（subtask 定义 + object_ref + term_signal）和 `mimicgen/datagen/data_generator.py` 这两个文件，你就完全理解了 MimicGen 的工作机制——LeIsaac Mimic 是同一套思路在 IsaacLab API 下的重新实现。
