# 耦合 Pred-EQA Workflow 到 MSGNav GOAT-Bench 评测

> 日期: 2026-07-03
> 工作区: MSGNav (基于 Pred-EQA 移植多Agent协作 workflow)

## 一、目标

在 MSGNav 的 GOAT-Bench 评测上，将原 KSS→单次VLM查询 的反应式决策，替换为 Pred-EQA 的多Agent协作 workflow（Snapshot Manager + Frontier Manager + Answerer + High-Level Planner + Executor）。

**复用 MSGNav**: M3DSG 场景图、KSS、AVU、VVD、TSDF导航、task_check、Scene/TSDFPlanner 类、模型加载、主循环骨架。
**移植 Pred-EQA**: 4个agent prompt + 解析逻辑 + 高层计划TODO + TextLongTermMemory。

## 二、核心设计决策

| 决策 | 方案 |
|------|------|
| Answerer 输出 | `image i`（i 为 snapshot 池索引→img_path），触发 MSGNav AVU+VVD+导航+task_check |
| snapshot prompt | 每图附 M3DSG 关联物体（ID+类名），来自 `scene.img_to_edge` 反查 |
| snapshot 来源 | 原始视角图（KSS 历史边图像 + 每步渲染视角图），不裁剪 |
| KSS 注入频率 | 仅 subtask 的 step0 注入图像边作为 snapshot 初始池 |
| 高层计划 | 跨 subtask 继承，subtask 切换时 prompt 加"新subtask+任务"提示让其整理记忆 |
| 高层计划格式 | 复用 Pred-EQA XML `<update_todo_list>` + `plan_extraction_utils.py` 解析器 |
| Forced Answerer | 不引入，步数耗尽判该 subtask 失败，进入下一 subtask |
| 长期记忆 | 移植 `TextLongTermMemory`，跨 subtask 持久（高层计划+step summary） |

## 三、新 Workflow（每 subtask）

```
subtask 开始:
  ├─ 清空 snapshot_manager 状态 (snapshot池、filtered_snapshots)
  ├─ 清空 frontier_manager 状态 (已访问frontier记录)
  ├─ 高层计划继承 (TextLongTermMemory 不清空)
  └─ prompt 提示: "新subtask: <task_type>/<question>, 请整理记忆"

step 0:
  ├─ 观察环境 (7视角) → M3DSG增量构建 → TSDF更新
  ├─ KSS → 图像边 processed_images → 注入 snapshot 池 (附关联物体)
  ├─ 当前所有 frontier 图 → 注入 frontier_manager
  └─ 进入 agent 链

每步 (step >= 0):
  ├─ 1. Snapshot Manager [当 snapshot数>3]
  │   ├─ 池 = KSS图像边(step0) ∪ 跨步累积视角图
  │   ├─ VLM 判相关性 → "Retain Snapshots: {i,...}"
  │   └─ 过滤, 记录到 TextLongTermMemory
  │
  ├─ 2. Frontier Manager [当 frontier数>1]
  │   ├─ VLM 判哪些frontier已探索且无关 → "Retain Frontiers: {i,...}"
  │   └─ 过滤, 记录到 TextLongTermMemory
  │
  ├─ 3. Answerer
  │   ├─ VLM 判证据是否足够
  │   ├─ 足够 → "Image i, <class>" → 走 AVU+VVD+导航+task_check
  │   └─ 不足 → "Continue Exploration"
  │
  ├─ 4. [IF Continue] High-Level Planner + Low-Level Executor
  │   ├─ Planner: 生成/更新 XML TODO list (跨subtask继承)
  │   ├─ Executor: 选 frontier → "Frontier i"
  │   └─ 记录到 TextLongTermMemory
  │
  ├─ 5. Step Summary → TextLongTermMemory
  │
  └─ 6. 导航执行 (MSGNav agent_step) → task_check (MSGNav)
      ├─ Yes → subtask 成功
      └─ No → 记入 CLR 历史, 继续

步数耗尽 → 判失败, 进入下一 subtask
```

## 四、模块对应

| 新模块 | 来源 | 位置 |
|--------|------|------|
| `explore_multi_agent()` | 新写 (替代 `explore_two_step`) | `src/explore_multi_agent.py` |
| Snapshot Manager prompt | 移植 Pred-EQA `format_manage_prompt` | `src/explore_multi_agent.py` |
| Frontier Manager prompt | 移植 Pred-EQA `format_plan_manager_prompt` | `src/explore_multi_agent.py` |
| Answerer prompt | 改写 Pred-EQA `format_answer_prompt` (输出 image i) | `src/explore_multi_agent.py` |
| High-Level Planner prompt | 移植 Pred-EQA `format_high_level_plan_prompt` + subtask切换提示 | `src/explore_multi_agent.py` |
| Low-Level Executor prompt | 移植 Pred-EQA `format_explore_prompt` | `src/explore_multi_agent.py` |
| Step Summary prompt | 移植 Pred-EQA step summary | `src/explore_multi_agent.py` |
| TODO 解析 | 移植 Pred-EQA `plan_extraction_utils.py` | `src/plan_extraction_utils.py` |
| TextLongTermMemory | 移植 Pred-EQA `long_term_memory.py` | `src/long_term_memory.py` |
| SceneIntegration | 移植 Pred-EQA `scene_integration.py` | `src/scene_integration.py` |

## 五、关键数据流

### snapshot 池结构

```python
snapshot_pool = [
    {
        "img_path": "0-view_0.png",       # scene.all_observations 的 key
        "img_b64": <base64>,              # resized RGB
        "connected_objects": [(5,"chair"), (8,"table")],  # M3DSG 反查
        "source": "kss_edge" | "egocentric",
        "step": 0,
    },
    ...
]
```

### Answerer 输出 → AVU 映射

```
Answerer: "Image 3, espresso machine"
  → snapshot_pool[3].img_path
  → scene.all_observations[img_path] (RGB)
  → scene.all_depths[img_path] (depth)
  → scene.all_cam_poses[img_path] (cam_pose)
  → MSGNav AVU: set_classes(["espresso machine"]) + 低阈值重检测
  → SAM → 点云 → VVD → 导航 → task_check
```

### subtask 切换 prompt 注入

```
[原有 XML TODO list]

--- NEW SUBTASK ---
Previous subtask completed/failed. New subtask:
Task type: {goal_type} (object|description|image)
Question: {question}
Please review and reorganize your memory and plan for this new target.
```

## 六、GOAT-Bench 三类 question 适配

| goal_type | question 格式 | Answerer prompt 引导 |
|-----------|---------------|---------------------|
| object | "Can you find the {category}?" | "Find the {category} object" |
| description | "Could you find the object exactly described as '{lang_desc}'?" | "Find the object matching this description" |
| image | "Identify the target object shown near the center of the reference image." | (附 reference image) "Find the same object in environment" |

## 七、实现步骤

### Step 1: 移植基础模块
- 复制 `Pred-EQA/src/long_term_memory.py` → `MSGNav/src/long_term_memory.py`
- 复制 `Pred-EQA/src/plan_extraction_utils.py` → `MSGNav/src/plan_extraction_utils.py`
- 适配 import 路径

### Step 2: 写 `src/explore_multi_agent.py`
- 移植 5 个 prompt 函数（Snapshot Mgr / Frontier Mgr / Answerer / High-Level Planner / Executor）
- 改写 Answerer 输出格式为 `Image i, <class>`
- 加 subtask 切换 prompt 段
- 写 `explore_multi_agent()` 主入口：snapshot 池维护 + agent 链 + 响应解析

### Step 3: 改 `src/query_vlm.py`
- 新增 `query_vlm_multi_agent()` 函数
- Answerer 输出 `Image i` → 映射 img_path → 复用现有 AVU 代码块
- Executor 输出 `Frontier i` → 复用现有 frontier 处理

### Step 4: 改 `run_goatbench_evaluation.py`
- subtask 开始: 初始化 TextLongTermMemory (跨subtask) + 清空 snapshot/frontier manager 状态
- step0: KSS 图像边注入 snapshot 池
- 每步: 调 `query_vlm_multi_agent()` 替代 `query_vlm_for_response()`
- 保留 task_check 逻辑

### Step 5: 配置与测试
- `cfg/eval_goatbench.yaml` 加 `use_multi_agent: true` 开关
- 单 episode 冒烟测试

## 八、风险与简化

- **VLM 调用次数**: 每步 4-5 次 VLM 调用 (vs MSGNav 原 1-2 次)。可接受，GOAT-Bench 评测非实时。
- **snapshot 池膨胀**: Snapshot Manager 阈值 >3 触发裁剪，但 KSS 初始注入可能>3。step0 若 KSS 图像边>3 张，立即触发 Manager 裁剪。
- **高层计划跨subtask失效风险**: subtask 目标差异大时旧计划误导。prompt 的"新subtask"提示让其整理，降级保护：若 Executor 无法解析计划则退化为纯 frontier 选择。
