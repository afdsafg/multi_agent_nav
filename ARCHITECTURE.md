# Graph-grounded Predictive Navigation — 架构文档

## 目录

1. [概述](#1-概述)
2. [项目结构](#2-项目结构)
3. [核心数据结构](#3-核心数据结构)
4. [Workflow 主流程](#4-workflow-主流程)
5. [五 Agent 链](#5-五-agent-链)
6. [AVU 四级 Grounding](#6-avu-四级-grounding)
7. [VVD 视角选择](#7-vvd-视角选择)
8. [Feedback 机制](#8-feedback-机制)
9. [Fallback 机制](#9-fallback-机制)
10. [配置参数](#10-配置参数)

---

## 1. 概述

基于诊断.md 9 节改造的 GOAT-Bench 导航系统。核心思想：**Graph-grounded Predictive Navigation** — 用场景图锚定预测，多层 grounding 防幻觉，跨 subtask 记忆复用几何知识。

三种任务类型：
- **object**：找指定类别物体
- **description**：找匹配自然语言描述的特定物体
- **image**：找与参考图相同的特定实例

---

## 2. 项目结构

```
predqa/
├── run_goatbench_evaluation.py   # 入口：episode/subtask 主循环
├── cfg/eval_goatbench.yaml       # 配置
├── src/
│   ├── explore_multi_agent.py    # 五 Agent 链 + prompt + parser
│   ├── query_vlm.py              # AVU 四级 grounding + VVD 调用
│   ├── tsdf_planner.py           # TSDF 地图 + frontier 管理 + 导航
│   ├── memory_structures.py      # 三层记忆数据结构
│   ├── utils.py                  # VVD / 视角选择 / 工具函数
│   └── conceptgraph/slam/utils.py # 点云反投影
```

---

## 3. 核心数据结构

### 3.1 三层记忆 (`memory_structures.py`)

| 层 | 类 | 范围 | 生命周期 |
|---|---|---|---|
| Global Scene Memory | `scene.all_observations/objects/edges` | episode 级 | 跨 subtask 保留 |
| Subtask Working Memory | `SubtaskWorkingMemory` | subtask 级 | 每个 subtask reset |
| Decision Memory | `frontier_registry` | 跨 subtask | 保留几何，标记 STALE |

### 3.2 TargetCandidate

VLM 确认的目标候选，四级状态机：

```
VISUAL_ONLY → GROUNDED_3D → (nav success → released)
                ↓
          NEED_CLOSER_VIEW → (max attempts → released)
                ↓
           REJECTED (VLM 否决)
```

- `candidate_id`：C000, C001...
- `image_path`：证据图路径
- `camera_pose`：4x4 世界→相机矩阵
- `pinned`：True 时 Image Manager 不可丢弃
- `closer_view_attempts`：近距离观察失败计数

### 3.3 FrontierState

稳定 frontier 身份，跨 subtask 保留几何：

| 状态 | 含义 |
|---|---|
| `ACTIVE` | 可选 |
| `EXPLORED` | 已探索无新信息 |
| `BLOCKED` | 阻塞 |
| `STALE` | 跨 subtask 保留但需重新评估 |

- `selected_count`：被选次数（≥ `max_frontier_reselect` 硬排除）
- `reached_count`：到达次数
- `last_result`：NO_NEW_INFO / LED_TO_ROOM / FOUND_CANDIDATE

### 3.4 FeedbackEvent

失败闭环反馈，注入 prompt：

| 类型 | 触发 | suggested_fix |
|---|---|---|
| `TASK_CHECK_FAIL` | task_check 否决 | rotate / reject candidate |
| `TASK_CHECK_PASS` | task_check 通过 | (空) |
| `AVU_FAIL` | AVU 全级失败 | pin image, try aliases |
| `AVU_VISUAL_ONLY` | L4 evidence-pose | re-observe from closer |
| `FRONTIER_NO_INFO` | frontier 无新信息 | select different frontier |
| `PLANNER_STALE` | plan 连续不变 | revise plan |
| `WRONG_INSTANCE` | 错误实例 | reject, search others |

---

## 4. Workflow 主流程

`run_goatbench_evaluation.py` 主循环：

```
for episode in episodes:
    working_memory = SubtaskWorkingMemory()
    for subtask in episode.subtasks:
        # 1. subtask 初始化
        working_memory.reset_for_new_subtask(subtask_id, question)
        high_level_plan = None
        image_pool = None

        for step in range(max_steps):
            # 2. 观察环境（ego views + depth + scene graph）
            # 3. 更新 TSDF frontier map
            tsdf_planner.update_frontier_map(...)
            # 4. 五 Agent 链决策
            target_type, target_index, reason, _, class_name = explore_multi_agent(step)
            # 5. 执行决策
            if target_type == "image":
                # Answerer 确认目标 → AVU grounding → VVD 视角 → 导航
                target_type, obj_pos, _, _ = query_vlm_for_response_end(step)
                navigate_to(obj_pos)
            elif target_type == "frontier":
                navigate_to_frontier(target_index)
                working_memory.mark_frontier_selected(fid)
            elif target_type == "stop":
                break

            # 6. 到达后 task_check
            if target_type == "image" and nav_success:
                vlm_response = task_check(last_5_steps)
                if vlm_response == "yes":
                    working_memory.release_after_navigation(cid)
                    break  # subtask 成功
                else:
                    working_memory.reject_candidate_by_vlm(img_path, reason)
                    # 继续探索
```

### 关键路径

1. **subtask 开始**：`reset_for_new_subtask` 清 pinned/candidates/feedback/plan，frontier_registry 标记 STALE（保留几何）
2. **每步**：Image Manager → Frontier Manager → Answerer → Planner → Executor
3. **Answerer 确认** → `query_vlm_for_response_end` AVU grounding → VVD 选视角 → 导航
4. **导航到达** → `task_check`（VLM 看最近 5 步 ego views 判断是否到达目标）
5. **task_check 通过** → `release_after_navigation` → subtask 结束
6. **task_check 失败** → `reject_candidate_by_vlm` → 继续探索

---

## 5. 五 Agent 链

`explore_multi_agent.py` L1089-1452。每步顺序执行：

### 5.1 Image Manager

**职责**：过滤图片池，保留与目标相关的图片

**Prompt** (`format_image_manager_prompt` L123)：
- SYS: "你是 MEMORY MANAGEMENT AGENT，保守保留——有 ANY 潜在相关性就保留"
- USER: Question + Available Images (b64 + connected objects) + Output Format

**Pinned 保护**：active candidate 的图片强制保留，Manager 只过滤非 pinned 图片

**Fallback** (L1221)：VLM 解析失败 → 保留 pinned + 最新 `max_pool - pinned` 张非 pinned

**触发条件**：非 pinned 图片 > 3 张时才运行

### 5.2 Frontier Manager

**职责**：过滤 frontier，移除已探索且无关的方向

**Prompt** (`format_frontier_manager_prompt` L216)：
- 每个显示为 `F_XXX (display N) | status=ACTIVE | selected=N | reached=N`
- 保留条件：同时满足"已确认访问"且"明确无关"——否则保留

**Fallback** (L1277)：VLM 解析失败 → 启发式打分 `_frontier_heuristic_score` 取 top_k=3

**打分器** (`_frontier_heuristic_score` L1057)：
```
score = novelty + info_gain - repeat_penalty - failed_branch_penalty
```

### 5.3 Answerer

**职责**：判断当前图片池是否已包含目标

**三态决策** (`format_answerer_prompt` L382)：
- `NOT_FOUND`：目标不可见，继续探索
- `CANDIDATE_VISIBLE`：可能可见但不确定（小/遮挡/边缘）→ 触发 AVU grounding
- `TARGET_CONFIRMED`：清晰可见可识别 → 触发 AVU grounding

**TARGET_CONFIRMED 条件**：
- 目标主体直接可见（非"可能在房间里"）
- 基于视觉证据（非常识猜测）
- 小物体能指出具体区域
- 属性问题的属性本身可见

**解析** (`parse_answerer_response` L876)：返回 `(decision, idx, class_name)`

**输出格式**：
```
Decision: NOT_FOUND | CANDIDATE_VISIBLE | TARGET_CONFIRMED
Image: <i>
Target phrase: <class>
Visibility:
  directly_visible: yes | no
  central_enough: yes | no
  partially_occluded: yes | no
  approximate_location: <text>
  confidence: <0.0-1.0>
Need action: move closer | rotate | ground with AVU | none
```

### 5.4 High-Level Planner

**职责**：生成长期探索计划（XML checklist）

**Prompt** (`format_high_level_planner_prompt` L523)：
- 分解目标为子任务（去某房间、找某物体、观察某属性）
- 生成并行预测分支
- 输出 `<update_todo_list><todos>...</todos></update_todo_list>`

**Stale 检测** (L1337)：plan 连续 `planner_stale_threshold` 步不变 → 注入 WARNING 强制 replan

**Progress Signals 注入** (L1350)：每步注入 current_pose / candidate grounding status / recent feedback / stale_plan_count

### 5.5 Executor

**职责**：选择具体 frontier 或停止探索

**硬约束过滤** (L1390)：`get_valid_frontier_ids` 排除：
- status = EXPLORED 或 BLOCKED
- selected_count ≥ `max_frontier_reselect`
- 在 `recent_window` 步内选过的

**Prompt** (`format_executor_prompt` L732)：
- frontier 显示为 `F_XXX (display N) | status=ACTIVE | selected=N | reached=N | DO NOT SELECT`
- History Decisions：避免重复选择

**解析** (`parse_executor_frontier_id` L960)：解析 `Next Step: Frontier F_XXX` 或 `Stop Exploration`

**无有效 frontier** → `("stop", None)` → subtask 失败

---

## 6. AVU 四级 Grounding

`query_vlm.py` L555-780。当 Answerer 返回 CANDIDATE_VISIBLE / TARGET_CONFIRMED 时触发。

**目标**：将 VLM 的视觉确认转化为 3D 可导航坐标，防止 VLM 幻觉导致导航到错误位置。

### 流程

```
Answerer 确认 Image (img_path, class_name)
    ↓
创建 TargetCandidate (VISUAL_ONLY, pinned)
    ↓
L1: YOLO(target_phrase) on target_image
    成功 → SAM mask → 反投影 → VVD
    失败 ↓
L2: YOLO(aliases)  (别名词表，如 ficus tree ↔ potted plant)
    成功 → SAM mask → 反投影 → VVD
    失败 ↓
L3: YOLO(_GENERIC_CLASSES) + CLIP rerank
    CLIP score ≥ _CLIP_RERANK_THRESH → SAM mask → 反投影 → VVD
    失败 ↓
L4: evidence-pose navigation (导航到拍证据图的相机位置)
    + feedback: AVU_VISUAL_ONLY
```

### 反投影 (`detections_to_obj_pcd_and_bbox`)

`conceptgraph/slam/utils.py` L1337：
1. `batch_mask_depth_to_points_colors`：depth + cam_K + mask → 相机坐标系点云
2. `trans_pose @ points`：变换到世界坐标系
3. DBSCAN 去噪 + downsample
4. 返回 `{pcd, bbox}`

### VVD 调用

L760-780：用反投影点云调 `Visibility_based_Viewpoint_Decision` 选最佳观察视角

### 各级失败处理

| 级别 | 失败动作 |
|---|---|
| L1 | record_attempt(1, False) → 尝试 L2 |
| L2 | record_attempt(2, False) → 尝试 L3 |
| L3 | record_attempt(3, False) → 尝试 L4 |
| L4 | record_attempt(4, False) → 返回 cam_pose[:3,3] + feedback |

---

## 7. VVD 视角选择

`utils.py` L51-87 `Visibility_based_Viewpoint_Decision`

### 流程

```
1. bbox_center = target_points.mean(axis=0)   # 点云质心
2. candidate_viewpoints = generate_candidate_viewpoints(bbox_center, radius=0.75, pts)
   → 在质心周围 0.75m 圆上生成 20 个候选视角
3. filtered_viewpoints = tsdf_planner.mask_true_point(candidates)
   → 过滤掉不可达的候选点（TSDF unoccupied 检查）
4. LOS check: tsdf_planner.is_line_of_sight_clear(vp, bbox_center)
   → Bresenham 线遍历 occupied 栅格，墙遮挡的候选点剔除
5. search_pool = los_passed (无 LOS 通过 → search_pool=[] → 走 fallback)
6. for vp in search_pool:
     compute_visibility(vp, target_points, scene_points_tree)
     → 物体点云遮挡检查（家具级细粒度）
   → 选 visibility_score 最高的
7. best_viewpoint is None → fallback: get_near_true_point
   → snap 到最近可达点，再选 visibility 最高的
```

### 坐标系

- `generate_candidate_viewpoints` 输出 `[habitat_x, ground_height, habitat_z]`
- `bbox_center` 是 `[x, y_height, z_depth]`
- LOS 检查用 2D 俯视投影 `(x, z)`
- `vp[1] += 1.5` 临时抬高到相机高度做 visibility 计算

### `is_line_of_sight_clear` (`tsdf_planner.py` L884)

Bresenham 线遍历 `self.occupied` 栅格：
- 两点转 voxel 坐标
- 逐 voxel 检查 `occupied[row, col]`
- 任一 voxel occupied → False（视线被挡）
- 越界 → False

---

## 8. Feedback 机制

### 8.1 反馈注入

`feedback_prompt_block()` 生成最近 4 条 feedback 文本，注入 Answerer 和 Planner prompt：

```
Recent Failure Feedback:
- Step 5 [TASK_CHECK_FAIL]: agent in laundry room, not target → mark candidate rejected
- Step 3 [AVU_VISUAL_ONLY]: VLM saw 'potted plant' but YOLO grounded nothing → re-observe
Consider this feedback when deciding the next action. Do not repeat actions that led to the same failure.
```

### 8.2 suggest_fix_for 映射

`memory_structures.py` L488-505：

| feedback type | suggested_fix |
|---|---|
| TASK_CHECK_FAIL (facing/view) | rotate toward target / use VVD viewpoint |
| TASK_CHECK_FAIL (other) | mark candidate rejected, find other instances |
| AVU_FAIL / AVU_VISUAL_ONLY | pin image, try aliases / class-agnostic / navigate closer |
| FRONTIER_NO_INFO | select a different frontier |
| PLANNER_STALE | revise plan: mark branch completed/failed or add new branch |
| WRONG_INSTANCE | reject candidate, search for other instances |

### 8.3 Progress Signals

`progress_signals_block()` 每步注入 Planner prompt：
```
Progress Signals:
- current pose: [x, y, z]
- candidate grounding status: C000=GROUNDED_3D
- last frontier: F_008 → NO_NEW_INFO
- recent feedback: TASK_CHECK_FAIL at step 5
- stale_plan_count: 1
```

---

## 9. Fallback 机制

### 9.1 Image Manager Fallback

**触发**：VLM 返回无效 / 解析失败
**行为**：保留 pinned + 最新 `max_pool - len(pinned)` 张非 pinned 图片

### 9.2 Frontier Manager Fallback

**触发**：VLM 返回无效 / 解析失败
**行为**：启发式打分 `_frontier_heuristic_score`，取 top_k=3

### 9.3 Executor 无有效 Frontier

**触发**：`get_valid_frontier_ids` 返回空（全部 EXPLORED/BLOCKED/超限/recent）
**行为**：`return ("stop", None)` → subtask 失败

### 9.4 AVU L4 Evidence-pose

**触发**：L1-L3 全部失败
**行为**：返回 `cam_pose[:3, 3]`（拍证据图的相机位置），agent 导航到该位置重新观察

### 9.5 VVD LOS 全失败

**触发**：所有候选视角到目标质心的视线都被墙挡
**行为**：`search_pool = []` → `get_near_true_point` snap 到最近可达点 → 选 visibility 最高的

### 9.6 VVD 无候选点

**触发**：`mask_true_point` 返回空（质心周围 0.75m 全不可达）
**行为**：`get_near_true_point` snap 到最近可达点

### 9.7 Planner Stale

**触发**：plan 连续 `planner_stale_threshold`（默认 2）步不变
**行为**：注入 WARNING 强制 replan

### 9.8 Candidate 释放条件

| 条件 | 方法 | 触发点 |
|---|---|---|
| GROUNDED_3D + 导航成功 | `release_after_navigation` | task_check 通过 |
| VLM 否决 (task_check 失败) | `reject_candidate_by_vlm` | task_check 否决 |
| N 次 closer-view 失败 | `check_closer_view_limit` | 每步开始检查 |

---

## 10. 配置参数

`cfg/eval_goatbench.yaml` 关键参数：

### 导航

| 参数 | 值 | 说明 |
|---|---|---|
| `success_distance` | 0.25 | 成功距离阈值 |
| `dicision_radius` | 0.75 | VVD 候选视角半径 |

### AVU

| 参数 | 值 | 说明 |
|---|---|---|
| `AVU_conf_threshold` | 0.1 | YOLO 置信度阈值 |

### 工作记忆

| 参数 | 值 | 说明 |
|---|---|---|
| `max_frontier_reselect` | 2 | frontier 最大选择次数 |
| `frontier_recent_window` | 3 | 近期排除窗口 |
| `max_pool_size` | 6 | 图片池软上限 |
| `candidate_max_closer_view_attempts` | 3 | candidate 近距离观察失败上限 |
| `planner_stale_threshold` | 2 | plan 不变步数阈值 |

### 相机

| 参数 | 值 | 说明 |
|---|---|---|
| `camera_height` | 1.5 | 相机高度 |
| `camera_tilt_deg` | -30 | 相机俯角 |
| `img_width/height` | 1280 | 图像尺寸 |
| `hfov` | 120 | 水平视场角 |

### TSDF

| 参数 | 值 | 说明 |
|---|---|---|
| `tsdf_grid_size` | 0.1 | 栅格大小 |
| `explored_depth` | 1.7 | 探索深度 |

### Scene Graph

| 参数 | 值 | 说明 |
|---|---|---|
| `mask_conf_threshold` | 0.95 | mask 置信度 |
| `merge_overlap_thresh` | 0.7 | 合并重叠阈值 |
| `obj_min_detections` | 3 | 物体最小检测次数 |
