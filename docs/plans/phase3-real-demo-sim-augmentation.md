# Phase 3：真机 demo 采集与 sim 扩增

**周期**：2 周
**前置依赖**：Phase 0（真机控制）+ Phase 2（仿真数据 pipeline）完成
**目标**：在真机上采集 100 条高质量 **PickPlaceRed** demo，并把这些 demo 作为"种子"用 MimicGen 思路在仿真中扩增到 5K–10K 条，最终形成可用于 VLA 微调的混合数据集

> **核心任务**：把红 cube 放进 plate。详细规约见 [README.md](README.md)。
> **命名约定**：仿真侧文件统一使用颜色无关命名 `data/instructions/pick_place.txt`、`assets/scenes/pick_place.xml` 等，便于扩展到其他颜色任务。

---

## 代码入口（快速开始）

> 所有命令在仓库根目录 `/home/zzg/workspace/pycharm/Robot` 下、`conda activate py312_cu121` 后执行。
> Phase 3 主流水线：**真机 demo → 分段 → MimicGen 替换场景重放 → 混合数据集**。

| 想做的事 | 一行命令 | 产出 |
|---------|---------|------|
| **冒烟测试整个 MimicGen 流水线（不依赖真机，用脚本策略合成种子）** | `python -m data.mimicgen_adapter.augment --from-sim-seeds 5 --output-repo-id local/so101_sim_mimicgen_smoke --n-per-demo 10` | dataset + 控制台显示成功率统计 |
| 真机 demo 采集（T3.2，需要硬件，本仓库未实现录制 GUI——使用 LeRobot 官方工具） | `python -m lerobot.scripts.control_robot record --robot.type so101 --control.fps 30 ...`（参数见 LeRobot 文档） | `local/so101_real_pickplace_v0` |
| **正式扩增（T3.5，从真机种子展开到 5K-10K 条仿真 episode）** | `python -m data.mimicgen_adapter.augment --source-repo-id local/so101_real_pickplace_v0 --output-repo-id local/so101_sim_mimicgen_v1 --n-per-demo 50` | `local/so101_sim_mimicgen_v1`（注：`--source-repo-id` 路径目前为 `NotImplementedError`，需先实现真机 demo→SegmentedDemo 反演；smoke 路径 `--from-sim-seeds` 可用） |
| 合并 real + sim_mimicgen + sim_scripted 三源（T3.6） | `python -m data.converters.merge_datasets --source local/so101_real_pickplace_v0:real --source local/so101_sim_mimicgen_v1:sim_mimicgen --source local/so101_pickplace_v1:sim_scripted --output-repo-id local/so101_pickplace_mixed_v1` | 单一 dataset，含 `source` tag |
| 语言指令 ×K 扩展（T3.7） | `python -m data.converters.expand_instructions --source-repo-id local/so101_real_pickplace_v0 --output-repo-id local/so101_real_pickplace_v0_langx3 --instructions data/instructions/pick_place.txt --copies 3` | `..._langx3` dataset |
| 扩增质量审计（T3.8，含 per-source 统计） | `python -m eval.audit_dataset --repo-id local/so101_pickplace_mixed_v1 --n-sample 100 --out-dir runs/audit_mixed` | 控制台 + markdown 报告 |

**入口与任务清单的对应关系**：

| Task | 实现文件（库） | 由哪个 CLI 串起来 |
|------|--------------|------------------|
| T3.1 标定模板 | [`configs/world_frame.yaml`](../../configs/world_frame.yaml)、[`configs/cameras.yaml`](../../configs/cameras.yaml) | 人工填值；被 `object_tracker` 读取 |
| T3.2 真机录制 | 依赖 LeRobot 上游 `control_robot record`（本仓库无 GUI） | LeRobot 上游 CLI |
| T3.3 子任务分段器 | [`data/mimicgen_adapter/segmenter.py`](../../data/mimicgen_adapter/segmenter.py) | 被 `augment` 内部调用 |
| T3.4 多锚点重放器 | [`data/mimicgen_adapter/replayer.py`](../../data/mimicgen_adapter/replayer.py) | 同上 |
| T3.5 扩增编排 | [`data/mimicgen_adapter/augment.py`](../../data/mimicgen_adapter/augment.py) | `python -m data.mimicgen_adapter.augment ...`（独立 CLI） |
| T3.6 混合数据集 | [`data/converters/merge_datasets.py`](../../data/converters/merge_datasets.py) | `python -m data.converters.merge_datasets ...` |
| T3.7 指令扩展 | [`data/converters/expand_instructions.py`](../../data/converters/expand_instructions.py) | `python -m data.converters.expand_instructions ...` |
| T3.8 扩增审计 | [`eval/audit_dataset.py`](../../eval/audit_dataset.py)（含 per-source 表） | `python -m eval.audit_dataset ...` |
| 公用：HSV + 3D 反投影 tracker | [`data/mimicgen_adapter/object_tracker.py`](../../data/mimicgen_adapter/object_tracker.py) | 被 replayer/真机回放路径使用 |
| 公用：dataclass 定义 | [`data/mimicgen_adapter/types.py`](../../data/mimicgen_adapter/types.py) | — |

> **已知限制**：见 [docs/implementation-status.md](../implementation-status.md) 中 L-7（plate 透视椭圆）/ L-8（脚本策略低成功率级联到 MimicGen 回放）/ L-9（真机 demo 反演到 SegmentedDemo 未实现）。在 L-9 解决之前，正式扩增路径只能用 `--from-sim-seeds` smoke 模式。
>
> **TL;DR**：Phase 3 当前可一键跑通的路径是 `augment --from-sim-seeds N --output-repo-id ... --n-per-demo K`，但因 L-1 / L-8 上游问题，实际产出 episode 数会显著少于 `N×K`。

---

## 关键技术与工具

| 工具 | 用途 | 备注 |
|------|------|------|
| LeRobot record | 真机 demo 采集 | Phase 0 已用 |
| MimicGen（参考实现） | 数据扩增核心算法 | `https://github.com/NVlabs/mimicgen` |
| robosuite / 自实现 | 子任务分段 + 重放 | 我们用轻量自实现 |
| LeRobot dataset merge | 合并真机 + sim 数据 | LeRobot 原生支持 |
| trimesh + numpy | 物体姿态变换 | `pip install trimesh` |

**MimicGen 核心思想**（必须理解）：
1. 给定一条人类 demo（end-effector 在世界系下的位姿序列）
2. 按"物体交互边界"切成 N 段：**对 pick-place 任务来说 = approach / grasp / transport / place**
3. 在新场景里，把对应物体的新姿态变换矩阵作用于各段：
   - `approach` + `grasp` 段：以 cube 新姿态为锚
   - `transport` 段：插值连接
   - `place` 段：以 plate 新姿态为锚
4. 再用 IK 重放并验证物理可行性，成功则保存
5. 一条种子可扩增到 10–100 条

**单锚点对的注意事项**：transport 段只需保证不撞机械臂自身或桌面（无干扰物）。但当 cube 与 plate 极近时可能直接 0 段 transport。

---

## 任务清单

### T3.1 真机演示采集前的环境标定

**目标**：保证真机和仿真坐标系可对齐

**步骤**：
- [ ] 在桌面贴 ChArUco 板，记录桌面 (0,0,0) 相对机械臂 base 的 transform
- [ ] 把同样的 transform 写入仿真 MJCF（更新 `assets/scenes/pick_place.xml`）
- [ ] 把真机 front camera 的 extrinsic 量到，写入 `configs/cameras.yaml`

**关键文件**：
- `configs/world_frame.yaml`：桌面 / camera / robot base 的统一坐标
- `assets/scenes/pick_place.xml`：仿真同步更新

**参考**：
- OpenCV ChArUco 标定教程

**验证**：在真机上把 cube 放在某点，读 camera → 反推 cube 在 base 系坐标；同样位置在仿真里渲染，两边视角下 cube 位置一致

---

### T3.2 真机 PickPlaceRed demo 采集

**目标**：采集 100 条高质量真机 demo

**步骤**：
- [ ] 准备：1 红 cube（边长 3cm）+ 1 plate（直径 12cm）
- [ ] 每条 demo 前手动**重新摆放**：cube 与 plate 在工作区不同位置随机散布
- [ ] **多样性要求**：100 条 demo 中 cube 位置覆盖工作区左/右/前/后各 ≥ 20 条，每种 cube yaw（0/π/2/π/3π/2）各 ≥ 20 条
- [ ] 用 leader-follower 遥操，每条 demo 15s 以内
- [ ] 命令：`lerobot-record --robot.type=so101_follower --teleop.type=so101_leader --dataset.repo_id=local/so101_real_pickplace_v0 --dataset.num_episodes=100 --dataset.single_task="put the red cube on the plate"`
- [ ] 失败 demo 当场删除（cube 没进 plate / 抓不起来等）

**关键文件**：
- `data/real_demos/so101_real_pickplace_v0/`

**验证**：
- 100 条成功 episode
- cube 位置 + yaw 覆盖均匀
- 100% episode 末态：cube 在 plate 上
- 平均时长 8–15 秒

**采集 tips**：
- 不要每次都从同一个 home pose 开始
- 不要全部用同一种抓取角度（用 yaw 多样性反映 cube 4 重对称）
- transport 阶段保持合理高度
- place 时夹爪松开要果断，不要在 plate 上方犹豫

---

### T3.3 子任务分段器

**目标**：把每条 demo 自动切成 sub-segments（pick-place 有 4 段）

**步骤**：
- [ ] 在 `data/mimicgen_adapter/segmenter.py` 实现：
  - 输入：一条 episode（ee 轨迹 + 夹爪开合状态 + 物体姿态 from sim 或推断）
  - 边界检测：
    - 夹爪**闭合**点（pick）→ approach / grasp 边界
    - ee 高度上升超过阈值 → grasp / transport 边界
    - 夹爪**张开**点（place）→ transport / place_release 边界
  - 输出：`[approach, grasp, transport, place_release]` 4 段
  - 每段标注 **anchor object**：approach & grasp → cube；transport → 无锚点（自由插值）；place_release → plate
- [ ] 真机数据缺少物体姿态 → 用红色掩膜（HSV）+ plate 颜色/形状掩膜从 front camera 估算 2D 位置 → 投影回桌面平面

**关键文件**：
- `data/mimicgen_adapter/segmenter.py`
- `data/mimicgen_adapter/object_tracker.py`：从图像追踪红 cube + plate 位置

**参考**：
- MimicGen 论文 §3.2 "object-centric subtask segmentation"
- MimicGen 代码 `mimicgen/datagen/data_generator.py`

**验证**：可视化抽 10 条 demo 的分段结果，pick / place 两个夹爪事件边界都在 ±0.2s 内

---

### T3.4 仿真重放器（多锚点）

**目标**：在仿真里 replay 真机 demo 的各段，支持多锚点变换

**步骤**：
- [ ] 在 `data/mimicgen_adapter/replayer.py` 实现：
  - 输入：分好段的 demo + 新场景（新 cube / 新 plate 姿态）
  - 各段独立应用变换：
    - `approach` / `grasp`：`T_new = T_new_cube × T_old_cube^-1 × T_segment`
    - `transport`：起点接上一段终点，终点接下一段起点（B-spline 插值或线性 + 高度抬升）
    - `place_release`：`T_new = T_new_plate × T_old_plate^-1 × T_segment`
  - 用 IK 解关节序列，在仿真中执行
  - 检查：碰撞、关节限位、成功标志
- [ ] 失败的扩增样本自动丢弃，失败原因记录

**关键文件**：
- `data/mimicgen_adapter/replayer.py`

**参考**：
- MimicGen `mimicgen/datagen/datagen_utils.py`

**验证**：拿真机 demo，在仿真随机化 100 次新场景（新 cube/plate 位置），成功 replay ≥ 50 条

---

### T3.5 扩增 pipeline 整合

**目标**：把 segmenter + replayer 串成自动 pipeline

**步骤**：
- [ ] 在 `data/mimicgen_adapter/augment.py` 实现：
  ```
  for real_demo in real_dataset:
      segments = segment(real_demo)
      for trial in range(N_per_demo):
          new_scene = randomize_scene()
          new_episode = []
          for seg in segments:
              new_seg = replay(seg, new_scene)
              if new_seg is None: break
              new_episode.append(new_seg)
          if check_success(new_episode):
              save_to_lerobot(new_episode, task=real_demo.task)
  ```
- [ ] N_per_demo 默认 100（100 真机 × 100 = 10K 上限，扣除失败率约 5K 实际）
- [ ] 全程开仿真侧 DR（Phase 2 模块复用）

**关键文件**：
- `data/mimicgen_adapter/augment.py`

**验证**：100 条真机 demo 扩增到 ≥ 5000 条仿真 episode

---

### T3.6 混合数据集组装

**目标**：把真机 + 仿真 + 纯仿真（Phase 2）合并

**步骤**：
- [ ] 在 `data/converters/merge_datasets.py` 实现：
  - 输入：N 个 LeRobotDataset
  - 输出：合并后 dataset，保留每个 episode 的 `source` 字段（"real" / "sim_mimicgen" / "sim_scripted"）
  - 比例建议（v1）：100 real + 5K mimicgen + 5K scripted = 10.1K 条
- [ ] 推 HF Hub：`<your_id>/so101_pickplace_mixed_v1`

**关键文件**：
- `data/converters/merge_datasets.py`
- `data/lerobot/so101_pickplace_mixed_v1/`

**参考**：
- LeRobot dataset merging：`https://huggingface.co/docs/lerobot/lerobot_dataset#merging-datasets`

**验证**：
- 合并后 dataset 可被 LeRobot 训练脚本加载
- `source` 字段分布符合预期

---

### T3.7 语言指令扩展与对齐

**目标**：真机指令与仿真指令分布一致

**步骤**：
- [ ] 把 Phase 2 的 `data/instructions/pick_place.txt`（内容是 red 任务指令）复用
- [ ] 给真机 demo 后处理：每条 episode 随机选 3 种指令变体生成 3 个语言副本（仅复制 task 字段，frame 数据共享）
- [ ] 仿真扩增数据沿用相同指令池
- [ ] **颜色锚定一致**：所有 instruction 都以 "red" 为目标，不混入其他颜色（避免语义噪声）

**关键文件**：
- `data/instructions/pick_place.txt`
- `data/converters/expand_instructions.py`

**验证**：dataset 内 task 字段直方图，真机 / sim 分布相近

---

### T3.8 扩增数据质量审计

**目标**：避免 MimicGen 引入"物理合法但不像真"的轨迹

**步骤**：
- [ ] 用 `eval/audit_dataset.py`（Phase 2 已写）增强：
  - 加入 source-aware 抽检
  - 关节速度直方图（mimicgen 数据不应明显不同于真机）
  - ee 轨迹平滑度
- [ ] 人工抽检 50 条 mimicgen 数据

**关键文件**：
- `eval/audit_dataset.py`（扩展）

**验证**：mimicgen 数据中明显异常 episode 比例 < 5%

---

## 验收标准（全部满足后进入 Phase 4）

- [ ] 100 条高质量真机 demo（cube 位置 + yaw 均匀覆盖）
- [ ] 真机-仿真坐标系标定一致
- [ ] MimicGen pipeline 成功扩增 ≥ 5000 条仿真 episode
- [ ] 合并 dataset 总量 ≥ 10K 条，多指令变体覆盖
- [ ] 质量审计通过率 ≥ 95%
- [ ] dataset 已推送 HF Hub

---

## 风险与陷阱

| 风险 | 应对 |
|------|------|
| 真机标定误差导致仿真重放失败率高 | 多采几条 ChArUco 标定 sample 求平均；标定误差 > 5mm 重做 |
| 真机 demo 里物体姿态估计不准 | 在采集时让 cube 严格只在桌面平面上，z 已知；只估 xy + yaw |
| MimicGen replay 大多失败（< 30% 成功率） | 分析失败段：通常是 lift 段 IK 在边角奇异 → 收紧采样范围；或加 retry 不同 yaw |
| 真机和 sim 视觉差异过大导致 VLA 学不到 sim2real 一致表征 | 加重视觉 DR，特别是相机抖动；考虑 real2sim 重建桌面纹理 |
| MimicGen 数据 dominates 训练 | 训练时给 real episode 5–10× 采样权重 |

---

## 输出物

- 100 条真机 demo（LeRobot 格式）
- 5K+ MimicGen 扩增 sim 数据
- 混合 dataset（v1）已推 HF Hub
- 数据扩增工具链（segmenter + replayer + augment）
