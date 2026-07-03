# MSGNav: 多模态3D场景图驱动的零样本具身导航 —— 核心创新、方法与Workflow详解

> 论文: *MSGNav: Unleashing the Power of Multi-modal 3D Scene Graph for Zero-Shot Embodied Navigation*  
> 录用: **CVPR 2026** | arXiv: 2511.10376v5  
> 源代码: `/src/` 目录下完整实现, 基于 3D-Mem 和 ConceptGraphs 构建

---

## 一、项目总览

**MSGNav** 是一个零样本（zero-shot）VLM引导的具身导航系统。Agent被放置在一个未知的室内3D环境中，需要通过自然语言描述、物体类别或参考图像找到指定的目标物体。系统不进行任何训练或微调，完全依赖预训练模型（YOLOWorld、SAM、CLIP、GPT-4o/Qwen-VL）实现视觉感知、场景理解与导航决策。

**核心思想:** 用动态分配的图像替代传统3D场景图中的纯文本关系边，构建**多模态3D场景图（M3DSG）**，保留视觉证据以增强VLM对场景的上下文理解。

**文件:** `README.md:1` (项目说明), `src/explore_utils.py:1-1003` (VLM交互), `src/query_vlm.py:108` (主决策接口)

---

## 二、核心创新点

### 创新1：多模态3D场景图 (Multi-modal 3D Scene Graph, M3DSG)

**问题背景:** 传统3D场景图（如ConceptGraphs）将物体间关系压缩为纯文本标签（"top", "beside"），导致：
- **构建昂贵:** 需要频繁调用MLLM进行关系推理
- **视觉信息丢失:** 将丰富的视觉观察转化为文本后，丧失了关键视觉证据
- **词汇受限:** 超出预设词汇表的新类别无法被表达

**解决方案:** M3DSG用动态分配的真实图像替代文本关系边，每条边存储的是记录了两个物体同时出现（co-occur）的RGB-D图像集合。

**数据结构定义** (`src/multimodal_3d_scene_graph.py:56`):

```
M3DSG = (O, E)
  O = {o_i}: 物体节点集合
    o_i = {ID, 类别, 3D位置, BBox, 语义掩码, 点云, CLIP特征, 房间标签}
  E = {e_j}: 图像边集合
    e_j: (ID_x, ID_y) → {I_1, I_2, ...}  # 图像集合作为边
  H: 图像→物体对的映射（反向索引）
```

**增量构建** (`src/multimodal_3d_scene_graph.py:update_scene_graph`, 约L400-700):

1. **物体更新 (Object Update):** 每帧通过YOLOWorld检测200类物体 → SAM分割 → CLIP提取特征 → 深度反投影生成3D点云 → 空间相似度+视觉相似度匹配到已有物体 → 合并点云/BBox/类别投票
2. **边更新 (Edge Update):** 对同一帧中空间距离 < θ (3.5m) 的物体对建立/更新边，将当前图像追加到对应图像集合中，**无需VLM查询**

**代码实现:**
- 物体匹配与合并: `src/conceptgraph/slam/mapping.py` (空间相似度 `overlap` + 视觉相似度 `CLIP cosine`, 聚合公式 `sim_sum = (1+phys_bias)*spatial + (1-phys_bias)*visual`)
- 边构建: `src/multimodal_3d_scene_graph.py` 中 `update_scene_graph_edges()`, 阈值 `edge_dist_threshold = 3.5m`
- 周期性清理: `periodic_cleanup_objects()` 每20帧去噪/过滤/合并物体

### 创新2：关键子图选择 (Key Subgraph Selection, KSS)

**问题背景:** 随着探索进行，场景图不断增长，Token消耗剧增（原始可达数万tokens），VLM推理成本高昂。

**方法: 压缩-聚焦-剪枝 三步流程** (`src/explore_utils.py:157-202`, `related_object_KSS` + `edge_pruning_KSS`):

1. **压缩 (Compress):** 将完整场景图简化为邻接表 S^，每个物体仅保留 ID + 类别 + 邻居列表

2. **聚焦 (Focus):** 将压缩后的邻接表输入VLM，由VLM选出与导航目标语义最相关的 top-k 个物体 O_rel（默认k=20）
   - 代码: `get_prefiltering_objs()` → `format_prefiltering_prompt()`, 输出为排序后的物体ID列表 (`src/explore_utils.py:666-690`)

3. **剪枝 (Pruning):** **贪心动态图像分配算法** (Greedy Dynamic Allocation, Algorithm 1)
   - 收集O_rel及其1-hop邻居的所有关联边
   - 使用**最大堆优先队列**（heapq），每次选择覆盖最多未覆盖边的图像
   - 更新剩余边的覆盖率，迭代直到所有边被覆盖
   - 代码: `edge_pruning_KSS()` (`src/explore_utils.py:724-780`, L753 `heapq.heappop(heap)`)
   - **效果:** Token成本降低 >95%，平均每次查询仅需约4张图像

**文件:** `src/explore_utils.py:157-202` (KSS主函数), `src/explore_utils.py:724-780` (贪心剪枝算法)

### 创新3：自适应词汇更新 (Adaptive Vocabulary Update, AVU)

**问题背景:** 零样本导航依赖YOLOWorld等开放词汇检测器，但仍受限于预设词汇表（HM3D-200类）。VLM可能识别出词汇表中不存在的类别（如"espresso machine"），需要动态扩展检测能力。

**方法** (`src/query_vlm.py:179-275`, `src/explore_utils.py:207-351`):

1. **触发条件:** VLM回复格式为 `"Image i, j"`，其中 `j` 是VLM预测的目标类别名（可能是预设词汇之外的类别）
2. **执行流程:**
   - 从 `scene.all_observations` 中取出图像 i
   - 将YOLOWorld的检测类别**临时设置为仅目标类别j**（而非完整的200类）
   - 使用**更低的置信度阈值** `AVU_conf_threshold`(HM3D:0.01, GOAT:0.1) 进行重检测
   - 对最高置信度的检测结果用SAM分割 → 深度反投影生成3D点云
   - 通过VVD选择最佳导航视点
3. **词汇表更新:** V_t = V_{t-1} ∪ V̂_t

**代码:** `src/query_vlm.py:195-203` (AVU重检测核心), `src/query_vlm.py:247-253` (AVU中的VVD调用)

### 创新4：闭环推理 (Closed-Loop Reasoning, CLR)

**问题背景:** 传统方法仅记忆场景本身，忽略了决策历史的重要性——Agent可能重复选择之前已被验证为错误的物体/图像。

**方法** (`src/explore_utils.py:207-351`):

1. **决策记忆 M_t:** 维护历史决策仓库，记录每一步的：
   - 目标类型（object/image/frontier）
   - 目标ID/索引
   - 是否正确（由后续验证反馈）

2. **提示注入:** 在VLM Prompt的 `History Decision` 部分显式列出所有历史错误决策，指示VLM避免重复选择：
   ```
   "Choosing those incorrect objects or images again is prohibited"
   ```

3. **公式:** R_t, V̂_t = VLM(S^k, M_t, F, g, t)

**代码:** `src/explore_utils.py:322-345` (CLR提示构建), `Prompt_with_AVU_and_CLR()` 中的 `history_decision` 参数

### 创新5：基于可见性的视点决策 (Visibility-based Viewpoint Decision, VVD)

**问题背景 (Last-mile problem):** 即使VLM正确定位了目标物体的3D位置，选择最近的可行走点作为导航终点往往会导致较差的观察视点（太近、被遮挡、角度不合适），致使Agent虽然"到达"但在最终验证中失败。

**方法** (`src/utils.py:51-78`, Algorithm 2):

1. **候选视点生成:** 在目标bbox中心周围的圆上均匀采样20个点（半径 `dicision_radius = 0.75m`），每个候选点高度=Agent摄像头高度(1.5m)
   - 代码: `generate_candidate_viewpoints()` (`src/utils.py:9-18`)

2. **可行走性过滤:** 通过TSDF占用网格过滤不可行走的候选点

3. **可见性评分:** 对每个候选视点→目标的射线路径采样（最多1000个目标点），通过**KD-Tree查询场景中所有其他物体点云**检测遮挡
   - 可见性 = 无遮挡的目标点比例
   - 代码: `is_point_visible()` (`src/utils.py:22-37`), `compute_visibility()` (`src/utils.py:39-48`)

4. **最佳选择:** 选择可见性评分最高的候选视点作为导航目标

5. **降级策略:** 若无可行走候选点（`best_viewpoint is None`），则使用 `get_near_true_point()` 寻找最近的可行走点；若仍失败则退化为 `select_navigation_corner()` 选择bbox最近角点
   - 代码: `src/utils.py:66-74`, `src/query_vlm.py:39-106`

**文件:** `src/utils.py:9-78` (VVD完整实现)

---

## 三、系统架构与完整Workflow

### 3.1 系统架构概览

```
输入: 室内3D场景(HM3D) + 导航目标(自然语言/图像/类别) + Episode JSON
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│                  MSGNav 导航系统                          │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │  VFMs感知层 │→│ M3DSG场景图  │→│  VLM推理决策层 │  │
│  │  YOLOWorld  │  │ (物体+图像边)│  │  KSS→AVU→CLR   │  │
│  │  SAM        │  │ 增量构建     │  │                │  │
│  │  CLIP       │  │ 周期维护     │  │                │  │
│  └─────────────┘  └──────────────┘  └───────┬────────┘  │
│                                               │           │
│  ┌─────────────┐                     ┌───────▼────────┐  │
│  │  TSDF规划层 │◄────────────────────│  VVD视点决策   │  │
│  │  体积融合   │   导航目标点          │  可见性优化   │  │
│  │  前沿探索   │                     │                │  │
│  └─────────────┘                     └────────────────┘  │
│                                                          │
输出: 成功/失败 + SPL + 轨迹可视化
└─────────────────────────────────────────────────────────┘
```

### 3.2 单步完整Workflow

每个导航步骤的完整流程（`src/run_hm3d_evaluation.py` / `src/run_goatbench_evaluation.py` 的 `main()` 循环）:

```
STEP t (循环直到 max_step 或找到目标):

┌── 1. 观察环境 (Observe Surroundings) ─────────────────────┐
│  - 计算 N 个观察角度 (1个正面 + extra_view_phase_1=6个侧视) │
│  - 对每个角度:                                              │
│    a. scene.get_observation(pts, angle) → RGB + Depth       │
│       │ 文件: src/multimodal_3d_scene_graph.py:get_observation│
│    b. scene.update_scene_graph(...):                        │
│       · YOLOWorld 200类检测 (yolov8x-world.pt)              │
│       · SAM 分割 (sam_l.pt, bbox提示)                       │
│       · BACTH CLIP特征提取 (ViT-H-14)                       │
│       · 过滤: IoU>0.9的重叠框、低置信度、背景类              │
│       · 深度反投影 → 3D点云 → 体素下采样(0.025m)             │
│       · DBSCAN去噪 (eps=0.1, min_pts=10)                    │
│       · 空间+视觉相似度匹配 → 合并/分配物体ID                │
│       · 更新图像边 (co-visible objects, dist < 3.5m)        │
│    c. tsdf_planner.integrate(...) → 更新TSDF体素网格(0.1m)  │
│       │ 文件: src/tsdf_base.py:TSDFPlannerBase.integrate    │
│  - scene.periodic_cleanup_objects():                        │
│    · 每20帧: DBSCAN去噪、过滤(<3次检测的物体)、合并相似物体  │
└─────────────────────────────────────────────────────────────┘
                              │
┌── 2. 更新记忆 (Update Memory) ─────────────────────────────┐
│  - 收集 agent 周围 obj_include_dist(3.5m) 内的物体          │
│  - scene.del_unused_scene_graph_edges() → 清理孤立边        │
└─────────────────────────────────────────────────────────────┘
                              │
┌── 3. 更新前沿 (Update Frontier) ───────────────────────────┐
│  - 从TSDF体素计算占用/可达/已探索地图                        │
│  - 检测前沿边界 (已探索自由空间 vs 未探索空间的交界)         │
│  - DBSCAN聚类前沿像素 → KMeans分割大角度前沿(>150°)          │
│  - 跨步IoU匹配: 保留/分割/合并前沿                           │
│  - 为每个前沿拍摄RGB图像作为VLM输入                          │
│  │ 文件: src/tsdf_planner.py:update_frontier_map            │
└─────────────────────────────────────────────────────────────┘
                              │
┌── 4. VLM查询决策 (Query VLM) ──────────────────────────────┐
│  query_vlm_for_response() → explore_two_step():             │
│  │ 文件: src/query_vlm.py:108, src/explore_utils.py:783     │
│                                                              │
│  Phase 1: 场景图推理                                         │
│    ├─ KSS 预过滤: VLM排序 top-20 相关物体                     │
│    │   get_prefiltering_objs() → format_prefiltering_prompt │
│    ├─ KSS 贪心剪枝: 最小图像集覆盖所有相关边                  │
│    │   edge_pruning_KSS() (最大堆+贪心set-cover)             │
│    ├─ 构建 Prompt (AVU+CLR):                                 │
│    │   · Objects Attribution (物体ID+类别+3D坐标+房间)       │
│    │   · Relationship Attribution (边+图像引用)               │
│    │   · Image List (选出的最小图像集合, 约4张)               │
│    │   · Egocentric Views (周围观察)                         │
│    │   · History Decision (CLR错误决策历史)                  │
│    │   · Frontier Images (备用)                              │
│    ├─ VLM API 调用 (GPT-4o / Qwen-VL):                      │
│    │   call_openai_api() (5次重试, 指数退避)                 │
│    └─ 解析响应:                                              │
│        · "Object i"    → 选物体节点                          │
│        · "Image i, j"  → 触发AVU重检测                       │
│        · "Continue Exploration" → 进入Phase 2                │
│                                                              │
│  Phase 2: 前沿探索 (如 Phase 1 返回 Continue Exploration)    │
│    └─ format_exploreonly_prompt() → 仅展示前沿图像            │
│       └─ VLM 选择最有希望的前沿 "Frontier i"                  │
│                                                              │
│  解析并执行 (query_vlm.py:query_vlm_for_response):           │
│    ├─ "object i":                                            │
│    │   → 获取bbox中心 → Visibility_based_Viewpoint_Decision │
│    │   → 降级: select_navigation_corner(最近角点)            │
│    ├─ "image i, j" (AVU):                                    │
│    │   → YOLOWorld重检测(仅类别j, 低阈值) → SAM分割          │
│    │   → 深度反投影3D点云 → VVD选择最佳视点                  │
│    └─ "frontier i":                                          │
│        → 前沿位置 + 方向作为导航目标                          │
│                                                              │
│  tsdf_planner.set_next_navigation_point(...)                │
└─────────────────────────────────────────────────────────────┘
                              │
┌── 5. 导航执行 (Navigate) ──────────────────────────────────┐
│  tsdf_planner.agent_step(...):                              │
│  │ 文件: src/tsdf_planner.py:agent_step                     │
│  - 通过 navmesh 计算 geodesic 最短路径                       │
│  - Phase 1(探索): 沿路径走 max_dist_from_cur_phase_1 米     │
│  - Phase 2(接近): 沿路径走 max_dist_from_cur_phase_2 米     │
│  - 调整导航点避开障碍物(adjust_navigation_point)            │
│  - 返回新位置、角度、target_arrived标志                      │
└─────────────────────────────────────────────────────────────┘
                              │
┌── 6. 成功验证 (Success Check) ─────────────────────────────┐
│  若 target_arrived 且 target_type != "frontier":            │
│    query_vlm_for_response_end() → task_check():             │
│      · 展示最后5步的周围图像 + 目标问题                      │
│      · VLM 回答 "Yes" → 导航成功, break                     │
│      · VLM 回答 "No"  → 记录失败尝试, 继续导航               │
│  更新 CLR 决策历史 (his_decision)                           │
└─────────────────────────────────────────────────────────────┘
                              │
┌── 7. 日志记录 ─────────────────────────────────────────────┐
│  - 记录步数、移动距离                                        │
│  - 可选保存 俯视图 + 前沿可视化                              │
│  - 每10个episode保存一次部分结果                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 关键算法伪代码

#### KSS贪心图像分配 (对应论文 Algorithm 1)

```
输入: 场景图 S=(O,E), 相关物体 O_rel
1. 初始化关键边 E^k = ∅, 关键物体 O^k = O_rel, 未覆盖边集 U = ∅
2. for o in O:  # 过滤相关边
     if ∃o_r ∈ O_rel 且 (ID_o, ID_or) ∈ dom(E):
       O^k.add(o), U.add((ID_o, ID_or))
3. 构造最大堆: 按每张图像覆盖的未覆盖边数量排序
4. while U ≠ ∅:
     取出覆盖最多未覆盖边的图像 I*
     将 I* 覆盖的边加入 E^k
     U 移除已覆盖的边
     更新堆(邻接图像增益-1)
5. return S^k = (O^k, E^k)
```

代码: `src/explore_utils.py:724-780`

#### VVD可见性决策 (对应论文 Algorithm 2)

```
输入: 目标物体ō, 场景点云 PC, 半径集 R, 采样数 K, 相机高度 h, 遮挡阈值 τ
1. 计算目标中心 c = mean(PC_ō)
2. 生成候选视点 V_c: 在R内以K个角度均匀采样, Z=h
3. 过滤 V_c: 仅保留可行走点
4. for each v_i ∈ V_c:
     对目标点云随机采样最多1000点
     对每条射线(v_i→p)采样中间点, 通过KD-Tree查询场景点云距离
     若所有采样点距离 > τ → 该点可见
     S_vi = 可见点数 / 总采样点数
5. v_best = argmax S_vi
```

代码: `src/utils.py:22-76`

---

## 四、关键模块与文件对应

| 模块 | 文件 | 核心类/函数 | 作用 |
|------|------|-------------|------|
| **M3DSG场景图** | `src/multimodal_3d_scene_graph.py` | `Scene` (L56) | 场景图核心: 增量构建、物体维护、边管理 |
| **TSDF规划** | `src/tsdf_base.py` | `TSDFPlannerBase` | 体素融合 (Numba加速) |
| **前沿探索** | `src/tsdf_planner.py` | `TSDFPlanner`, `Frontier` | 前沿检测/匹配、导航步进 |
| **VLM交互** | `src/explore_utils.py` | `explore_two_step()` (L783) | 两阶段VLM查询: 场景图→前沿 |
| | | `call_openai_api()` (L75) | API调用 (GPT-4o/Qwen-VL, 重试x5) |
| | | `Key_Subgraph_Selection()` (L157) | KSS: 压缩→聚焦→剪枝 |
| | | `edge_pruning_KSS()` (L724) | KSS贪心图像分配算法 |
| | | `Prompt_with_AVU_and_CLR()` (L207) | AVU+CLR完整Prompt构建 |
| | | `get_prefiltering_objs()` (L666) | KSS预过滤VLM调用 |
| | | `task_check()` (L945) | 最终成功验证 (Yes/No) |
| **VLM响应处理** | `src/query_vlm.py` | `query_vlm_for_response()` (L108) | 解析VLM输出、AVU执行、VVD调用 |
| | | `query_vlm_for_response_end()` (L324) | 最终验证 |
| | | `select_navigation_corner()` (L39) | VVD降级策略 |
| **VVD视点决策** | `src/utils.py` | `Visibility_based_Viewpoint_Decision()` (L51) | 可见性优化的最佳视点选择 |
| | | `generate_candidate_viewpoints()` (L9) | 候选视点圆采样 |
| | | `compute_visibility()` (L39) | 射线遮挡可见性评分 |
| **Habitat接口** | `src/habitat.py` | `make_semantic_cfg()` | 模拟器初始化 |
| | | `pos_habitat_to_normal()` | 坐标系转换 |
| **几何工具** | `src/geom.py` | `get_cam_intr()`, `IoU()` | 相机内参、几何计算 |
| **ConceptGraphs** | `src/conceptgraph/slam/mapping.py` | `compute_spatial_similarities()` | 空间相似度 (overlap) |
| | | `compute_visual_similarities()` | 视觉相似度 (CLIP cosine) |
| | `src/conceptgraph/slam/utils.py` | `detections_to_obj_pcd_and_bbox()` | 深度反投影生成3D点云 |
| | `src/conceptgraph/utils/model_utils.py` | `compute_clip_features_batched()` | 批处理CLIP特征提取 |
| **数据分析** | `src/dataset_utils.py` | `prepare_goatbench_navigation_goals()` | GOAT-Bench episode解析 |
| **日志/评估** | `src/logger_goatbench.py` | `Logger` | 结果聚合、SR/SPL计算 |
| **配置** | `cfg/eval_goatbench.yaml` | - | 完整超参数 |
| | `cfg/eval_hm3d.yaml` | - | HM3D配置 |
| **运行入口** | `run_goatbench_evaluation.py` | `main()` | GOAT-Bench评测主循环 |
| | `run_hm3d_evaluation.py` | `main()` | HM3D评测主循环 |
| | `start_multiprocess.py` | - | 多GPU并行调度 |

---

## 五、关键超参数与配置

| 参数 | GOAT-Bench | HM3D | 说明 | 文件位置 |
|------|------------|------|------|----------|
| `success_distance` | 0.25m | 1.0m | 导航成功距离阈值 | `cfg/eval_*.yaml` |
| `dicision_radius` | 0.75m | 0.75m | VVD候选视点圆半径 | `cfg/eval_*.yaml` |
| `AVU_conf_threshold` | 0.1 | 0.01 | AVU重检测置信度阈值 | `cfg/eval_*.yaml` |
| `top_k_categories` | 20 | 20 | KSS预过滤保留物体数 | `cfg/eval_*.yaml` |
| `tsdf_grid_size` | 0.1m | 0.1m | TSDF体素分辨率 | `cfg/eval_*.yaml` |
| `explored_depth` | 1.7m | 1.7m | 相机最大探索深度 | `cfg/eval_*.yaml` |
| `obj_include_dist` | 3.5m | 3.5m | 场景图物体包含距离 | `cfg/eval_*.yaml` |
| `edge_dist_threshold` | 3.5m | 3.5m | 边构建距离阈值 | `cfg/concept_graph_default.yaml` |
| `sim_threshold` | 1.2 | 1.2 | 物体匹配合并阈值 | `cfg/eval_*.yaml` |
| `denoise_interval` | 20步 | 20步 | 去噪间隔 | `cfg/eval_*.yaml` |
| `merge_interval` | 20步 | 20步 | 合并间隔 | `cfg/eval_*.yaml` |

---

## 六、模型依赖

| 模型 | 用途 | 权重文件 |
|------|------|----------|
| **YOLOWorld** | 开放词汇物体检测 (200类) | `yolov8x-world.pt` |
| **SAM** | 实例分割 | `sam_l.pt` |
| **OpenCLIP ViT-H-14** | 视觉特征提取、相似度匹配 | 自动下载 |
| **GPT-4o / Qwen-VL-Max** | VLM推理决策、场景理解 | API调用 |

---

## 七、评测数据集

| 数据集 | 指标 | 难度 |
|--------|------|------|
| **GOAT-Bench** | Success Rate, SPL (0.25m阈值) | 多模态目标 (object/description/image), 终身导航 |
| **HM3D-ObjNav** | Success Rate, SPL (1.0m阈值) | 物体类别目标, 200类词汇表 |

---

## 八、技术创新总结

```
┌──────────────────────────────────────────────────────────────────┐
│                    MSGNav 技术创新矩阵                              │
├──────────────┬───────────────────────┬────────────────────────────┤
│   创新点      │    解决的问题           │     技术方案                 │
├──────────────┼───────────────────────┼────────────────────────────┤
│ M3DSG        │ 文本关系边丢失视觉信息   │ 图像边替代文本边             │
│ (场景图)     │ 构建昂贵(vs MLLM查询)   │ 0额外推理成本               │
│              │ 词汇受限               │ 视觉证据保留支持动态扩展     │
├──────────────┼───────────────────────┼────────────────────────────┤
│ KSS          │ 场景图膨胀 → VLM超限   │ 压缩→VLM聚焦→贪心剪枝       │
│ (子图选择)   │ Token成本过高           │ Token减少>95%, 平均4图/查询 │
├──────────────┼───────────────────────┼────────────────────────────┤
│ AVU          │ 预设词汇无法覆盖新类别   │ VLM驱动按需重检测           │
│ (词汇更新)   │                        │ 低阈值单类检测+3D定位       │
├──────────────┼───────────────────────┼────────────────────────────┤
│ CLR          │ 重复选择已失败的决策     │ 决策记忆+反馈提示            │
│ (闭环推理)   │                        │                            │
├──────────────┼───────────────────────┼────────────────────────────┤
│ VVD          │ "Last-mile": 到达≠看到  │ 候选视点圆采样+射线遮挡评估    │
│ (视点决策)   │                        │ 选择最高可见性视点            │
└──────────────┴───────────────────────┴────────────────────────────┘
```
