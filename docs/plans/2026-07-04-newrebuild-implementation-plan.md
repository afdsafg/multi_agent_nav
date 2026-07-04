# newrebuild.md 完整重构实施计划

## Summary

准入结论：当前范围内执行。实施目标是“完整重构 + 直接替换”，不做 feature flag 兜底；实施工作应在当前 worktree `/home/afdsafg/下载/new/pre_msg/MSGNav/.codebuddy/worktrees/predqa` 执行。

当前基线是 `65e9f06`。核心改动是把现有每步串行 `Image Manager -> Frontier Manager -> Answerer -> High-Level Planner -> Executor` 替换为共享 typed state、typed events、SpatialBranch、HypothesisBranch、BEV、Candidate Controller 驱动的导航流程。

## Key Changes

- 状态契约：在 `src/memory_structures.py` 中新增/重整 `SpatialBranchRecord`、`BranchTaskState`、`HypothesisBranch`、`Anchor` union、`FrontierInstance`、`StepOutcome`、`TypedEvent`、`ExploreIntent`、`VisualApproachIntent`、`TargetViewpointIntent`、`NavigationResult`、`NavigationMode`。
- Frontier 职责调整：`src/tsdf_planner.py` 不再把 frontier ID 当长期记忆；当前 frontier 只作为瞬时 `FrontierInstance`，长期空间连续性由 `SpatialBranchRecord` 承担。禁用旧 `_reassociate_stale_frontiers` 作为主语义依据。
- 新增纯代码模块：新增 `src/spatial_context.py` 渲染 BEV；新增 `src/event_engine.py` 负责事件检测、trigger、debounce；新增 `src/candidate_controller.py` 负责 VISUAL_APPROACH、TARGET_APPROACH、VERIFY 状态机。
- Agent 流程替换：`src/explore_multi_agent.py` 删除 VLM Frontier Manager 调用；Image Manager 只在 `WORKING_MEMORY_OVER_BUDGET` 等事件触发；High-Level Planner 改为 Hypothesis Manager，且只在规定事件触发；Executor 每个 EXPLORE step 输出 JSON action。
- 主循环替换：`run_goatbench_evaluation.py` 使用 typed intent/navigation mode，不再用 `target_type == "image"` 或 evidence-pose 到达触发验证；只有 `TargetViewpointIntent` 到达后进入 VERIFY。
- VLM 查询整合：`src/query_vlm.py` 只做流程桥接和工具调用调度；AVU/VVD 大段逻辑迁移到 Candidate Controller；保留 KSS、M3DSG、TSDF、VVD 能力，但职责边界重新划分。

## Public Interfaces

- Answerer 输出 JSON：`decision: NOT_FOUND | CANDIDATE_VISIBLE | TARGET_CONFIRMED | ANSWER_READY`，并包含 `candidate`、`evidence_updates`、`evidence_conflict`。
- Hypothesis Manager 输出 JSON：`updates` 和 `new_hypotheses`，是唯一允许修改 `HypothesisBranch` 的 writer。
- Executor 输出 JSON：`frontier_id`、`spatial_branch_id`、`hypothesis_id`、`action_mode`、`reason_code`；允许的 action mode 为 `CONTINUE_SPATIAL_BRANCH`、`SWITCH_SPATIAL_BRANCH`、`REVISIT_SPATIAL_BRANCH`。
- Candidate Controller 输入/输出强类型 intent：`ExploreIntent`、`VisualApproachIntent`、`TargetViewpointIntent`；VERIFY 输出 `SUCCESS | WRONG_INSTANCE | POOR_VIEW | TARGET_NOT_VISIBLE`。
- Subtask reset：清空 working memory、hypotheses、candidates、BranchTaskState、EventEngine task scope；保留 M3DSG、TSDF、all observations、SpatialBranchRecord。

## Implementation Steps

1. 先落地 typed state 和单元测试：完成 memory dataclass、enum、JSON 序列化/反序列化、prompt summary helper，并保证旧 `TargetCandidate` 基本字段可迁移。
2. 接入 BranchTracker：从当前 `tsdf_planner.frontiers` 生成 `FrontierInstance`，维护 branch spine、active tips、progress、split/merge v1、BranchTaskState 的 `NEW/ADVANCING/STALLED/REVISITING`。
3. 接入 BEV：每步用 TSDF observed/free/occupied、当前 pose、recent high-level decision poses、frontier labels `F_xxx/Bxx` 渲染空间上下文；不显示 GT target、GT room、全量 scene graph object。
4. 接入 EventEngine：生成 spatial、semantic、candidate、memory events；按 `(event.type, entity_id, active_hypothesis_id)` debounce；只在 `newrebuild.md` 指定 trigger 下调用 Hypothesis Manager 或 Memory Manager。
5. 重写 `explore_multi_agent` 主流程：每步执行 Answerer/Evidence Assessor、事件路由、必要时 Hypothesis Manager、eligibility filter、Executor；不再每步调用 Image Manager 和 High-Level Planner。
6. 抽离 Candidate Controller：实现 L1 target phrase、L2 alias、L3 class-agnostic + CLIP proposal、visual ray approach、GROUNDED_3D 后 VVD、TargetViewpoint VERIFY；禁止 evidence-pose/visual-approach 到达后直接 task check。
7. 改主循环：用 `NavigationMode` 和 typed intent 驱动 `set_next_navigation_point` 适配层；生成 `StepOutcome`；VERIFY 成功结束 subtask，失败按事件回到 EXPLORE/VISUAL_APPROACH/TARGET_APPROACH。
8. 清理旧路径：旧 VLM Frontier Manager、旧 task_check 主路径、旧自然语言 high-level plan 状态机不再被新流程调用；非 multi-agent legacy 函数可保留但不能影响新路径。

## Test Environment 补充

- 所有测试必须在远端服务器执行：`root@8.157.94.238 -p 52137`。
- SSH 密码：`6a08777c-0bcb-493d-8146-b058fdf25564`。
- 明确要求：本轮实施临时忽略凭据明文写入文档的安全问题，实施者可以直接使用上述密码进行远端测试操作。
- 使用 `sshpass` 进行 SSH 和远程命令操作；本机只做代码编辑和静态检查，不作为测试环境。
- 不在本机运行 `pytest`、仿真、GOAT/HM3D 评估或依赖 GPU/数据集的验证。
- 服务器上已经有绑定相同远程仓库的项目文件夹。实施者应在本地完成代码修改并提交/推送后，在服务器项目目录中通过 `git fetch`、`git checkout`、`git pull` 或等价 git 命令同步代码；不要用手工复制文件作为主要同步方式。
- 远端测试命令示例：

```bash
sshpass -p '6a08777c-0bcb-493d-8146-b058fdf25564' ssh -o StrictHostKeyChecking=no -p 52137 root@8.157.94.238 \
  'cd <remote_repo_path> && git fetch && git checkout predqa && git pull && python -m pytest tests'
```

- 验收标准：只有远端服务器上的测试结果算有效；本机测试缺失不视为已验证。

## Implementation Status

- 已补齐 typed state contract：`SpatialBranchRecord`、`BranchTaskState`、`HypothesisBranch`、`StepOutcome` 扩展到 newrebuild 字段，并保持旧字段兼容。
- 已升级 `BranchTracker`：维护 active tips/frontier history/progress，支持多信号 branch association、split/merge v1，`STALLED/REVISITING` 只作为上下文而不是硬排除。
- 已接通 Answerer evidence：`evidence_updates` 会生成 `HYPOTHESIS_SUPPORTED`、`HYPOTHESIS_CONTRADICTED`、`HYPOTHESIS_TEST_COMPLETED` 等 typed events；无 active hypothesis 时显式发出 `NO_ACTIVE_HYPOTHESIS`。
- 已移除 Candidate Controller 的 evidence-pose fallback：没有可用 visual-ray approach 时返回 `GROUNDING_FAILED` 和 `CANDIDATE_REJECTED`，不再把历史相机位姿当作导航目标。
- 已替换主循环 VERIFY：`run_goatbench_evaluation.py` 不再导入/调用 `query_vlm_for_response_end`，只在 `TargetViewpointIntent` 到达后调用 `query_vlm_for_verify` 并处理 typed `VerifyStatus`。
- 已补测试：结构 roundtrip、event routing、Answerer evidence routing、Candidate rejection、CLIP gate、parser 和 runner 主路径断言。

## Test Plan

- 结构测试：dataclass 默认值、subtask reset、Anchor union、typed intent、StepOutcome、event debounce、Hypothesis single-writer 约束。
- BranchTracker 测试：新 frontier 建 branch、frontier 重新聚类但 branch 保持、连续低 progress 进入 STALLED、回环进入 REVISITING、closed branch 被 eligibility filter 排除。
- Candidate 测试：L3 proposal 不修改 Ultralytics result；CLIP gate 使用 raw similarity/margin/depth；VISUAL_APPROACH 不触发 VERIFY；只有 GROUNDED_3D 才进入 TARGET_APPROACH。
- Agent parser 测试：Answerer/Hypothesis Manager/Executor JSON 正常解析；格式错误时安全降级为 NOT_FOUND 或 STOP，不随机选 frontier。
- 主循环回归：mock VLM 跑一条 EXPLORE -> CANDIDATE_VISIBLE -> VISUAL_APPROACH -> GROUNDED_3D -> TARGET_APPROACH -> VERIFY_SUCCESS；再跑 WRONG_INSTANCE 和 POOR_VIEW 分支。
- 验证命令：优先在远端运行 `python -m pytest tests`；若环境缺少 pytest，则提供 `python tests/test_rebuild_structures.py` 风格的无依赖测试入口，并同样只在远端执行。

## Assumptions

- 按“直接替换”执行，不新增 feature flag；旧 multi-agent 串行链不会作为运行时兜底。
- KSS、M3DSG、TSDF integration、YOLO/SAM、VVD 继续复用现有能力，只重分职责和状态流。
- 第一版 branch split/merge 只实现 v1 简化逻辑：持续 2-3 步分歧才 split，merge 只做 alias/merged_into 标记，不实现复杂 DAG。
- VLM 平均调用目标：普通 EXPLORE step 约 2 次，事件步按需增加 Hypothesis Manager 或 Memory Manager。
