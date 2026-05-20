---
title: SO-101 Top-Down Grasp 调参与踩坑全记录
date: 2026-05-16
context: Phase 1 L-1 修复（0% → 15%）+ Phase 2 完整 pick-101 迁移（15% → 93.8%）
---

# SO-101 Top-Down Grasp 调参与踩坑全记录

> 把脚本 oracle policy 从 0% 抓取成功率提升到 93.8% 的整个调试过程。**写给未来的自己 / 任何接手 SO-101 manipulation 的人**——避免我们走过的弯路。

## TL;DR — 两句话教训

> **(Phase 1)** SO-101 这种 5-DoF 小臂做 top-down 抓取，**不能让 IK 自由解全 5 个关节**——必须先选定让夹爪朝下的 wrist 配置（wrist_flex + wrist_roll），**lock 这两个关节**，IK 只解前 3 个（shoulder_pan / shoulder_lift / elbow_flex）来满足 XYZ 位置。
>
> **(Phase 2)** 光抄方法不够，**模型本身也要抄**。pick-101 的 `so101_new_calib.xml` 多了 graspframe / 双 fingertip 站点 / 1.25 mm 指垫 collision geom + `cone="elliptic"` 摩擦——这些是稳定夹持的物理基础。menagerie 的 `so101.xml` 没有这些，再怎么调 IK 也只能到 26%。把整套模型搬过来 + finger_width_offset + yaw 量化 + 腕部按 cube 位置同步对齐，从 15% 一路推到 93.8%。

---

# Phase 1 教训：把方法论搞对（0% → 15%）

详细错误分析见 git 历史（commit 之前的版本），核心 6 个错误浓缩成结论：

| # | 错误 | 教训 |
|---|------|------|
| 1 | 用 mink 的 5-DoF position-only IK 自由解全 5 个关节 | 5-DoF position-only 有 2-DoF 零空间，ee 朝向**没有任何保证**——夹爪斜着、横着随机方向闭合 |
| 2 | 让 IK 解完 5 关节后用 `data.ctrl[3,4] = force_wrist` 覆盖 | 约束必须**喂进 IK solver 里面**，覆盖 IK 输出会让 shoulder/elbow 解失效；**8/8 episode 全部 joint_limit** |
| 3 | 以为 `ee+z · world+z = +1` 就是夹爪朝下 | jaws 张开口朝**上**根本无法抓桌面上的 cube；正确是 `ee+z · world-z = +1`。**选好"朝下"配置后用 viewer 眼睛确认**，不要依赖 dot product 符号 |
| 4 | home keyframe 的 `shoulder_lift = -1.8` 超出 ctrlrange `[-1.745, 1.745]` | mujoco 自动 clip 到 -1.745 立刻触发 `at_limit`；30 步内被 `joint_limit_streak` 提前终止。**写 keyframe 一定对照每个 actuator 的 ctrlrange，留 ≥0.05 rad 余量** |
| 5 | `sim/randomization/camera_pose.py` 有模块级 `_DEFAULT_FRONT_POS` hardcode | DR 每次 reset 把相机偷偷拉到 stale 位置，MJCF 设计被默默推翻。**DR 模块应该从 MJCF 读初始值缓存** |
| 6 | 在 mink FrameTask 加 `orientation_cost` 强行让 ee 朝下 | 5-DoF 无法同时满足 3-D position + 3-D orientation；trade-off 让关节被推到 ctrlrange 边缘——所有 episode `joint_limit`。**5-DoF 要么 position-only（朝向失控），要么 lock 关节让 IK 维度匹配** |

## 关键的"啊哈"时刻

1. **用户上传的 SO-101 静态图**：清楚显示夹爪朝下，证明 5-DoF 能实现这个姿态——根本不是关节自由度问题，是 IK 使用方式问题
2. 找到 [pick-101 仓库](https://github.com/ggand0/pick-101) + 博客 [ggando.com/blog/so101-rl-lift](https://ggando.com/blog/so101-rl-lift/)，`tests/test_topdown_pick.py:116` 一行 `locked_joints = [3, 4]` 道出核心
3. **DLS Jacobian IK 70 行胜过 mink QP 框架**——pick-101 那 70 行原生支持 `locked_joints`，mink freezing joints 要绕路。**简单方案胜出**
4. **不同 SO-101 MJCF 的关节约定不通用**：pick-101 用 `wrist_flex=π/2, wrist_roll=π/2` 让 ee 朝下；同样数值在 menagerie 上让 ee 朝 world +y。**抄数字不行，要抄方法**。menagerie 上正确值是 `wrist_flex=0, wrist_roll=-2.74`

## Phase 1 修复总结

- IK：[`sim/controllers/ik_dls.py`](../sim/controllers/ik_dls.py)——70 行 DLS Jacobian，原生 `locked_joints` 支持
- Home keyframe：`q=[0, -1.4, 1.4, 0, -2.74]` 让 `ee+z · world-z = 0.94`，所有关节远离 ctrlrange 极限
- Env hook：`self.locked_joints: list[int] | None = None`；`_apply_ee_action` 传给 IK
- Policy：grasp 阶段 `env.locked_joints = [3, 4]`

---

# Phase 2 教训：把成功率从 15% 推到 93.8%

### 错误 7：以为 menagerie SO101 = pick-101 SO101，只抄方法不抄模型

**为什么错**：模型本身不一样：

| 项 | menagerie `so101.xml` | pick-101 `so101_new_calib.xml` |
|---|---|---|
| `gripperframe` 位置 | `(0.012, ~0, -0.098)` 偏 X | `(0, 0, -0.098)` 正中心 |
| `graspframe` / fingertip 站点 | ❌ 无 | ✅ 用于 reach reward / 指尖中点 |
| `static/moving_finger_pad` geom | ❌ 无 | ✅ **1.25 mm 接触贴片，多点接触稳定夹持** |
| 摩擦/接触 | `friction="1 0.005 0.0001"` 默认 | `friction="0.5 0.05 0.001"` + `cone="elliptic"` + `noslip_iterations="3"` + `solref="0.001 1"` |

没有指垫 + elliptic 摩擦锥 → 夹爪闭合时 cube 从指缝滑出；没有 fingertip 站点 → 没法做指尖中点计算。

**正确做法**：把 pick-101 整个 `models/so101/` 目录原封不动搬到 `assets/so101_pick101/`，scene 文件 `<include>` 直接复用。**别只抄数字、别只抄方法、整套抄过来**。

### 错误 8：以为 gripperframe 就是两指中心，漏了 `FINGER_WIDTH_OFFSET = -0.015`

pick-101 的 `gripperframe` 是 TCP（10cm forward 虚拟点），沿 jaw spread 方向偏向 static finger ~12mm。不补偿就会用 static fingertip 直接撞 cube 顶面。`test_topdown_pick.py:113` 的 `finger_width_offset = -0.015` 看起来多余实则关键。**读他人代码时每个魔法数字都要查清来由**。

### 错误 9：每个 tick 从 obs 重读 cube 位置去追 → 追逐环

cube 被 finger 轻碰移位 1-2 mm，policy 立刻把目标挪过去，gripper 跟着追又碰到——cube 在 DESCEND 期间被推走 2-3 cm。

**正确做法**：pick-101 `test_topdown_pick.py:139` settle 后**一次性 snapshot** `actual_cube_pos`，整个 PICK 流程用固定值。我们的 policy 第一次被调时 snapshot `cube_anchor`/`plate_anchor`。

### 错误 10：gripper action 用 delta 而非 absolute

旧 env "action[3] * 0.2 加到 ctrl[5]" delta 接口。pick-101 grasp 是**渐进闭合 + 接触后再紧握**——需要绝对值平滑斜坡（0.3 → -0.8，250 个 mj_step）。delta 模式 5 步就饱和。

**正确做法**：env 加 `gripper_action_mode = "absolute"`，pick-101 那种 `(a+1)/2 * range + lo` 映射。

### 错误 11：500 Hz IK 反而让成功率变差

观察到 pick-101 裸 mj_step loop 100% 成功，把 env.step 改成"每个 mj_step 都跑 IK"。**结果 16 seed 从 3/16 (100Hz) 掉到 1/16 (500Hz)**。

**为什么**：sts3215 严重欠阻尼（ζ ≈ 0.26）。500 Hz IK 比 actuator 设定时间快 40 倍，actuator 跟不上、来不及衰减振荡，反而抖得更厉害。

**正确做法**：保留 100Hz env.step 单次 IK + actuator 自然 settle；用 `ee_target_override` 旁路让 oracle 提供**绝对世界系目标**，IK gain=0.5 自然产生指数衰减。

### 错误 12：DESCEND 一够 REACH_THR 就立刻进 GRASP

欠阻尼 actuator 抵达 target 时仍有 3-4 mm 过冲 + 30-40 ms 衰减。policy 在到达阈值瞬间切 GRASP，闭合 jaw 时 ee 还在振荡——static fingertip 短暂插进 cube 下沿、把 cube 顶飞。

**正确做法**：加 `PHASE_HOLD_STEPS = 15`，每个 phase 至少 15 个 ctrl tick（150 ms）才允许切换，**即使 _reached 早已 True**。

### 错误 13：cube yaw 随机 ±π 然后纳闷为什么 grasp 时好时坏

locked-wrist 的 jaws 永远沿世界 +y 闭合；cube 转 45° 后呈现对角线面，**两个平行 jaw 根本咬不住对角**。

**正确做法**：3 cm cube 4-fold 对称，yaw 量化到 `{0, π/2, π, 3π/2}`。**这一项把成功率从 ~60% 提到 ~92%**。

### 错误 14：以为 locked wrist + 中心工作空间就行了

把 cube y 收到 (-0.04, 0.04) 拿到 92%，以为搞定。但当 shoulder_pan 旋转去够 cube 时整个 wrist 跟着转——jaws 不再沿世界 Y。`FINGER_WIDTH_OFFSET` 沿世界 Y 的假设在 cube 偏离 y=0 时不准。

**正确做法**：reset 时按 cube 位置预先调整 wrist_roll：

```python
α = atan2(cube.y, cube.x)
wrist_roll = π/2 - α            # 让 shoulder_pan + wrist_roll 总是 π/2
qpos[4] = ctrl[4] = wrist_roll  # 一次设到位、整个 episode 锁住
```

shoulder_pan IK 解出 α 时，总 z 旋转 = π/2，jaws 沿世界 Y。工作空间才能拉回完整 (-0.08, 0.08)。

### 错误 15：PLACE_DESCEND 假设腕部仍沿世界 Y

PICK 时 wrist_roll = π/2 - α_cube。TRANSPORT shoulder_pan 换到指向 plate（α_plate），wrist_roll 锁着不变。总 z 旋转 = π/2 + (α_plate - α_cube)，jaws 已经不沿世界 Y。`FINGER_WIDTH_OFFSET` 沿世界 Y 加进去就偏到盘子外。

**正确做法**：PLACE_DESCEND 用**闭环反馈**：

```python
desired_cube = (plate.x, plate.y, plate_top + cube_half + offset)
target_ee = ee_current + (desired_cube - cube_obs)  # 每 tick 重算
```

无需知道腕部转了多少度——cube 离目标差多少 ee 就移多少，自洽稳定。TRANSPORT 不能用这种闭环（cube 离 plate 太远 → 单 tick 跳整个 delta → IK 被推过关节限位），TRANSPORT 走直接 plate xy 目标，靠 PLACE_DESCEND 精修。

### 错误 16：LeRobot dataset schema 写死 240×320

`data/converters/sim_to_lerobot.py` 的 `DEFAULT_FEATURES` 把分辨率硬编进 schema；env 渲染分辨率改了 schema 没改就崩。**正确做法**：`build_features(img_height, img_width)` 工厂动态拼 shape，CLI 加 `--img-height` / `--img-width` 透传。MJCF `<global offwidth="1920" offheight="1080"/>` 把 framebuffer 上限设到 1080p。

---

## Phase 2 修复（最终配置）

- **整套 pick-101 模型搬过来**：`assets/so101_pick101/`（14 STL + `so101_new_calib.xml` 未改）
- **env 三条 oracle 旁路**（`sim/envs/base.py`）：`ee_target_override` / `gripper_action_override` / `gripper_action_mode='absolute'`
- **wrist_roll 按 cube 位置同步**（`sim/envs/pick_place.py::_post_reset`）：`wrist_roll = clip(π/2 - atan2(cube.y, cube.x), 限位)`
- **policy 用 cube_anchor + finger_width_offset + 闭环 PLACE**（`sim/scripted_policies/pick_place.py`）：`FINGER_WIDTH_OFFSET = -0.015`、`PHASE_HOLD_STEPS = 15`、首次 snapshot anchor
- **分辨率参数化**：`build_features(img_height, img_width)` + `--img-height/--img-width` CLI 透传

---

## 进步轨迹

| 版本 | 改动 | success rate | 夹爪垂直 | 工作空间 (y) |
|---|---|---:|---|---|
| 原版 | mink position-only IK | 0%/50 | ❌ | (-0.12, +0.12) |
| v5 | 3cm cube + closed-start gripper | 5%/50 | ❌ | (-0.10, +0.10) |
| v6 | DESCEND_Z=0.020 + 修 home keyframe 越界 bug | 26%/50 | ❌（但抓到了） | (-0.10, +0.10) |
| v16 (pick-101 风格 IK) | DLS Jacobian + locked wrist + down-facing home | 15%/50 | ✅ | (-0.05, +0.05) |
| **v17 (整套 pick-101 模型)** | so101_new_calib.xml + 指垫 + elliptic 摩擦 + finger_width_offset | 38% | ✅ | (-0.08, +0.08) |
| v18 | + cube/plate snapshot 锚定 + absolute gripper | 44% | ✅ | (-0.08, +0.08) |
| v19 | + PHASE_HOLD_STEPS | 59% | ✅ | (-0.08, +0.08) |
| v20 (方案 A) | + cube yaw 量化 + 工作空间收到 ±0.04 | **92.2%** | ✅ | (-0.04, +0.04) |
| **v21 (方案 B)** | + wrist_roll = π/2 − atan2(cube.y, cube.x) + 闭环 PLACE_DESCEND | **93.8%** | ✅ | **(-0.08, +0.08)** 恢复全范围 |

**关键拐点**：v16→v17 整套模型搬过来 +23pp；v19→v20 cube yaw 量化（一行代码）+33pp；v20→v21 方案 B 把工作空间放大 4 倍同时保住 ~94%。

---

## 给未来的 checklist（黄金 22 条）

下次在 SO-101（或任何小型 5-DoF 臂）实现 grasp 时按这个走，能少走 80% 弯路。

### Phase 1: 把方法论搞对

1. [ ] 找一个**社区已经验证 work 的参考实现**（gym-lowcostrobot, pick-101 等），先读 README + grasp test 脚本
2. [ ] 在 MuJoCo viewer 里**手动滑动每个关节**，确认每个关节怎么影响 ee 朝向；记下 "ee 朝下"对应的 (shoulder_lift, elbow_flex, wrist_flex, wrist_roll) 组合
3. [ ] 在 home keyframe **直接设到那个姿态**（不指望 IK 帮你转过去）
4. [ ] IK 用 **locked_joints 模式**，只在剩下的关节空间解 position；DLS Jacobian 70 行够用，不需要 mink
5. [ ] 验证 home keyframe 的每个 ctrl **至少离 ctrlrange 边界 0.05 rad**
6. [ ] 用 viewer 或单帧渲染**眼睛确认**夹爪真的朝下，不要只看 dot product
7. [ ] cube 尺寸跟随社区标准（gym-lowcostrobot 3cm），不要随手定 4cm 这种"看起来挺大的"数字
8. [ ] DR 模块**不要 hardcode 默认值**——首次调用从 model 读快照存起来
9. [ ] 第一次能跑通后**先跑 50 episode** 看稳定 success rate，不要只看 4 episode 就下结论
10. [ ] success 后 **hold 至少 1 秒**再 terminate，让视频录到完整 place + retract

### Phase 2: 把成功率从 15% 推到 90%+

11. [ ] **整套模型一起搬**，不只搬 IK 方法：参考库的 `models/` 目录直接拷过来。menagerie 缺指垫/fingertip 站点/摩擦锥配置，再调 IK 也只能到 ~25%
12. [ ] 找到参考库的**每一个"魔法常量"**并搬过来——`FINGER_WIDTH_OFFSET = -0.015` 这种看起来多余的偏移往往是关键
13. [ ] **快照** cube/plate 起始位置，整个 pick 流程只用快照值，**绝不**每个 tick 重读 obs（追逐环会把 cube 推走）
14. [ ] gripper action 用 **absolute 模式**（不要 delta）；scripted oracle 的 contact-detection 抓紧需要平滑斜坡
15. [ ] **不要盲目提高 IK 频率**：欠阻尼 actuator（ζ < 0.5）下 500 Hz IK 比 100 Hz IK 更糟糕——actuator 跟不上目标抖动。先实测 100 Hz baseline 再考虑
16. [ ] 每个 phase 加 `PHASE_HOLD_STEPS`（≥ 15 个 ctrl tick），让 actuator 完成过冲衰减再切下一阶段
17. [ ] cube yaw **量化到对称角度**（3 cm cube → `{0, π/2, π, 3π/2}`），不要全 `[-π, π]` 随机——locked-wrist 的平行 jaw 咬不住对角面。这一项往往是 30+ 个百分点
18. [ ] 工作空间想拉大时，给 wrist_roll **按 cube 位置同步对齐**：`wrist_roll = π/2 − atan2(cube.y, cube.x)`，把"shoulder_pan 摆过去后 jaw 仍沿世界 Y"的不变量保住
19. [ ] PLACE 阶段**不要假设腕部朝向**——cube 被夹住后跟着 jaw 转，用 `target_ee = ee + (desired_cube − cube_obs)` 闭环反馈才稳健
20. [ ] LeRobot dataset 的图像 feature shape **要和 env 渲染分辨率联动**：用工厂函数 `build_features(img_height, img_width)`，别在模块顶部写死 `(240, 320, 3)`
21. [ ] MJCF 的 `<global offwidth="..." offheight="..."/>` 决定 offscreen renderer 的上限——想录 720p/1080p 视频时记得提前设
22. [ ] 跑诊断时**实测 64+ seeds**，不要 8/16 就下结论——cube yaw 这种确定性 bug 在小样本里看起来像随机故障

---

## 参考资料

- ggando.com 博客（核心）：<https://ggando.com/blog/so101-rl-lift/>
- pick-101 仓库：<https://github.com/ggand0/pick-101>
  - `tests/test_topdown_pick.py` — 标准 4 阶段 pick 流程
  - `src/controllers/ik_controller.py` — DLS Jacobian IK with locked_joints
- gym-lowcostrobot：<https://github.com/perezjln/gym-lowcostrobot>
- mujoco_menagerie SO-101：<https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101>

---

## 后续待办

仍可继续打磨：
- [ ] 剩余 ~6% 失败是 `joint_limit`，下游表象其实是 GRASP 没抓住——加 LIFT 后的 grasp-check（cube_z 是否真的离地），失败立刻 abort 走 retract
- [ ] cube 摩擦从 `0.5 0.05 0.001` 调到 `1 0.05 0.001` 可能减少滑出指间
- [ ] `wrist_roll_alignment` 在 cube y 极端时被 clip 到 ctrlrange 边界——可提前过滤这些位置或留死区
- [ ] 接入实际机械臂时验证 sim → real 的腕部 alignment 是否能照搬（实际伺服阻尼可能不同）
