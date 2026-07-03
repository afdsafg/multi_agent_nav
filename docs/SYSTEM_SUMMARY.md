# MSGNav Multi-Agent 系统总结

## 1. 系统架构

### 1.1 五 Agent 链（explore_multi_agent.py）

每步按顺序调用：

| Agent | 职责 | 输入 | 输出 |
|-------|------|------|------|
| **Image Manager** | 裁剪 image pool（保留与问题相关视角） | pool (List[Dict]), question, high_level_plan | `Retain Images: {i,...}` → 裁剪后 pool |
| **Frontier Manager** | 裁剪 frontier（淘汰已探索+无关方向） | frontier_imgs, pool, question | `Retain Frontiers: {i,...}` → 裁剪后 frontiers |
| **Answerer** | 判断是否找到目标 | pool, question, image_goal | `Image i, <class>` 或 `Continue Exploration` |
| **High-Level Planner** | 维护 todo list（跨步） | question, history, plan | `<update_todo_list>...</update_todo_list>` |
| **Executor** | 选 frontier 执行导航 | frontier_imgs, history, plan | `Frontier i` 或 `Stop Exploration` |

### 1.2 辅助模块

| 模块 | 作用 | 触发时机 |
|------|------|----------|
| **KSS** | step0 注入初始 image pool（7 视角全景） | 仅 step0（`is_new_subtask=True`） |
| **AVU** | Answerer 选 `Image i, class` 后，YOLO 在该图检测 class → SAM 分割 → 重建点云 → VVD 选观察点 | Answerer 返回 image target 时 |
| **VVD** | (1) 导航阶段：基于可见性选最佳观察点（正对目标）(2) 最终验证：VLM 判断是否"really reached" | (1) AVU 后 (2) target_arrived 后 |
| **task_check** | 取最后 5 步 egocentric，VLM 判 yes/no | target_arrived 且非 frontier |

### 1.3 视角渲染

- **step0**：7 视角（`extra_view_phase_2=6` + 主视角）
- **step1+**：3 视角（`extra_view_phase_1=2` + 主视角）
- 每步渲染图保存到 `output_dir/<subtask_id>/views/<global_step>-view_<idx>.png`

### 1.4 数据流

```
step0: KSS(7视角) → pool
step1+: 渲染3视角 → add_egocentric_to_pool(pool) → 
        Image Manager(裁剪pool) → Frontier Manager(裁剪frontier) →
        Answerer(判answerable?) → 
          ├─ answerable: AVU(YOLO检测) → VVD(选观察点) → 导航
          └─ not answerable: High-Level Planner(更新plan) → Executor(选frontier) → 导航
        → (若target_arrived) task_check(yes/no) → break or continue
```

---

## 2. Workflow（每步执行流程）

```
run_goatbench_evaluation.py 主循环:
  for each subtask:
    for each step (max 50):
      (1) Observe: 渲染视角，更新场景图(YOLO+SAM+点云)
      (2) Update Memory
      (3) Update Frontier: 更新 TSDF occupancy，提取 frontier
      (4) Query VLM (multi-agent chain):
          query_vlm_multi_agent(step_dict, cfg):
            a. KSS (仅step0): 注入初始pool
            b. add_egocentric_to_pool: 追加3张egocentric
            c. Image Manager: pool>3时裁剪
            d. Frontier Manager: frontier>1时裁剪
            e. Answerer: 判断answerable
               ├─ answerable + image target: AVU(YOLO) → VVD → 返回导航点
               └─ not answerable: Planner → Executor → 返回frontier
      (5) Planner navigate: tsdf_planner.agent_step(导航到目标点)
      (6) Check arrival: 
          ├─ target_arrived + 非frontier: task_check(yes/no)
          │   ├─ yes: break (subtask成功)
          │   └─ no: continue
          └─ 记录 his_decision
    评估: agent_subtask_distance < success_distance(0.25m) → success
```

### 跨 subtask 交接

- **pool**: 依赖 KSS 覆盖（未显式清空，可能泄漏）
- **high_level_plan**: 跨 subtask 延续（新 subtask 提示"Previous subtask completed/failed"）
- **History Decisions**: 每 subtask 重置（不跨 subtask 累积）
- **scene graph / all_observations**: 跨 subtask 累积（同场景）

---

## 3. 已修复的问题

### 3.1 KSS 每步运行（已修复）
- **问题**: KSS 在 step1+ 也运行，重复注入
- **修复**: `query_vlm.py` KSS 仅在 `is_new_subtask=True` 时运行

### 3.2 Image Manager pool 膨胀（已修复）
- **问题**: 每步 +7 egocentric，解析失败时 pool 不缩，膨胀到 31+ 张
- **修复**: (1) `extra_view_phase_1: 6→2`（step1+ 3 视角）(2) 正则修复（见下）

### 3.3 Image Manager 正则匹配失败（已修复）
- **问题1**: VLM 回应引用 prompt 指令"retain images that..."，旧正则 `[:：]?` 可选 → 误匹配描述句
- **修复1**: `[:：]` 必须有冒号，group 改 `[\d,\s]+` 只匹配数字
- **问题2**: `r'\s*\}}?'` 误写为必须一个 `}`，无花括号输出 `Retain Images: 0,1,2,3` 失败
- **修复2**: `\}}?` → `\}?`

### 3.4 其他修复
- ollama 改可选 import
- `retrieve_by_type` 按 timestamp 降序
- 删除 `filtered_snapshots` dead attribute
- 保存每步视角图到 `views/` 目录

---

## 4. 待解决的问题

### 4.1 ★★★ YOLO 类别限制导致 Answerer 正确判断被否决（最严重）

**现象**: Answerer 看到 ficus tree / carpet → 判 answerable → YOLO 检测不到（不在 HM3D 200 类）→ `No objects detected` → fallback 随机 frontier → Answerer 失去线索 → 徘环

**案例**:
- episode specific 1: carpet（多解，Answerer 选了对的图但 YOLO 无 carpet 类）
- episode specific 32: ficus tree（Answerer 看到 green plant，YOLO 无 ficus 类）

**根因**: 
- Answerer 的"对象"定义（视觉可见）与 YOLO 的"对象"定义（检测类别）不一致
- YOLO 否决后无反馈机制：Image Manager 清掉该图，Answerer 忘记曾看到目标

**影响**: 直接导致任务失败 + 徘徊浪费 step

### 4.2 ★★★ Executor 重复探索同一 frontier

**现象**: subtask 1 选 Frontier 0 共 6 次，History Decisions 记录"already explored"但未强制排除

**根因**: 
- History Decisions 仅提示，无硬约束
- Executor prompt 未强制排除已选 frontier 索引
- frontier 索引跨步不稳定（裁剪后索引变化），History 引用索引可能失效

### 4.3 ★★ Answerer 误判 answerable（目标未实际可见）

**现象**: step2 Answerer 判 `Image 1, refrigerator`，但冰箱"not fully visible"/"likely just outside frame"

**根因**: 
- Answerer prompt 允许推断（"if ANY potential visual relevance"）
- 无"目标必须可见且居中"的硬约束
- connected objects: none 时仍判 answerable（无 detection 支撑）

### 4.4 ★★ Image Manager 清理导致线索丢失

**现象**: Answerer 选中 `2-view_2.png`（含 ficus）→ YOLO 否决 → Image Manager 下一步裁掉该图 → Answerer 再也看不到 ficus

**根因**: 
- Image Manager 不知道某图曾被 Answerer 选中
- 无"Answerer 选中但检测失败"的保护机制

### 4.5 ★★ 跨 subtask pool/plan 未显式清空

**现象**: 
- pool 依赖 KSS 覆盖（可能泄漏旧 subtask 视角）
- high_level_plan 未随 target 切换重置（subtask 4 仍引用 kitchen cabinet 计划）

### 4.6 ★ VVD 复核结果不反馈给 multi-agent

**现象**: task_check 返回 no（如"agent 仅相邻未正对"），但 reason 不传给 Image Manager / Answerer

**影响**: multi-agent 链不知道为何失败，只通过 History Decisions 知道 yes/no

### 4.7 ★ Frontier Manager fallback 保留全部

**现象**: 2 次 `no valid retain response`，fallback 保留全部 frontier → 冗余探索

### 4.8 ★ Planner 滞后

**现象**: todo list 5 步不变，仍报"at threshold"，未随 agent 位置推进

---

## 5. 已确认的非问题

- **VVD 正对目标**: 非 GOAT-Bench 要求（成功标准纯距离 0.25m），是 MSGNav 为提高 task_check 通过率自加
- **task_check**: 非 GOAT-Bench 要求，是额外 VLM 确认
- **GOAT-Bench 多解**: 题目特性，非系统 bug（同类别物体多处，目标位置不唯一）

---

## 6. 当前代码版本

- **HEAD**: `c32f335` (feat: 保存每步渲染视角图)
- **关键文件**:
  - `run_goatbench_evaluation.py`: 主循环，AVU/VVD 调用，task_check
  - `src/query_vlm.py`: `query_vlm_multi_agent` 入口
  - `src/explore_multi_agent.py`: 5-agent 链 + parse 函数
  - `src/explore_utils.py`: `call_openai_api`, `task_check`, `format_end_prompt`
  - `src/tsdf_planner.py`: 导航 + VVD 朝向计算
  - `cfg/eval_goatbench.yaml`: 配置（`extra_view_phase_1: 2`, `save_visualization: true`）

- **服务器**: `/root/multi_agent_nav`，Python 3.9 (`/root/miniconda3/envs/3dmem/bin/python`)
- **API**: 阿里云百炼 `qwen3-vl-flash`，max_tokens=4096
- **CLIP**: ViT-B-32 + laion2b_s34b_b79k，Offline 模式
