import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.branch_tracker import BranchTracker, BranchTrackerConfig
from src.event_engine import EventEngine, EventEngineConfig
from src.memory_structures import (
    AnswererDecision,
    BranchTaskState,
    BranchTaskStatus,
    CandidateAnchor,
    EventType,
    ExecutorActionMode,
    ExploreIntent,
    F_ACTIVE,
    FrontierInstance,
    FrontierState,
    HypothesisBranch,
    HypothesisStatus,
    ImageAnchor,
    NavStatus,
    NavTargetKind,
    NavigationMode,
    NavigationResult,
    S_REJECTED,
    SpatialBranchRecord,
    StepOutcome,
    SubtaskWorkingMemory,
    TargetCandidate,
    TargetViewpointIntent,
    TypedEvent,
    VerifyStatus,
    VisualApproachIntent,
    anchor_from_dict,
    should_enter_verify,
)


class FakeFrontier:
    def __init__(self, frontier_id, position, region=None, orientation=None):
        self.frontier_id = frontier_id
        self.position = np.asarray(position)
        self.orientation = np.asarray(orientation if orientation is not None else [1.0, 0.0])
        self.region = np.asarray(region if region is not None else [True, True])
        self.image = None


class FakeTSDFPlanner:
    def __init__(self, frontiers):
        self.frontiers = frontiers
        self.unexplored = np.ones((16, 16), dtype=bool)
        self.unoccupied = np.zeros((16, 16), dtype=bool)
        self.occupied = np.zeros((16, 16), dtype=bool)
        self.frontier_map = np.zeros((16, 16), dtype=int)

    def voxel2habitat(self, position):
        arr = np.asarray(position, dtype=float)
        if arr.shape[0] == 2:
            return np.array([arr[0], 0.0, arr[1]], dtype=float)
        return arr.astype(float)

    def habitat2voxel(self, position):
        arr = np.asarray(position, dtype=float)
        if arr.shape[0] >= 3:
            return np.array([round(arr[0]), round(arr[2]), 0], dtype=int)
        return np.array([round(arr[0]), round(arr[1]), 0], dtype=int)


def _install_explore_utils_stub():
    old_module = sys.modules.get("src.explore_utils")
    module = types.ModuleType("src.explore_utils")
    module.call_openai_api = lambda *args, **kwargs: None
    module.encode_tensor2base64 = lambda value: value
    module.resize_image = lambda image, *args, **kwargs: image
    module.format_question = lambda question, *args, **kwargs: question
    sys.modules["src.explore_utils"] = module
    return old_module


def test_dataclass_json_roundtrip_anchor_intents_and_step_outcome():
    branch = SpatialBranchRecord(
        spatial_branch_id="B001",
        frontier_ids=[7],
        spine=[np.array([1.0, 0.0, 2.0])],
        active_tip_frontier_id=7,
    )
    restored_branch = SpatialBranchRecord.from_dict(json.loads(json.dumps(branch.to_dict())))
    assert restored_branch.frontier_ids == [7]
    assert restored_branch.spine == [[1.0, 0.0, 2.0]]

    image_anchor = anchor_from_dict({"kind": "image", "image_path": "1-view_0.png"})
    candidate_anchor = anchor_from_dict(
        {"kind": "candidate", "candidate_id": "C001", "target_phrase": "red chair"}
    )
    assert isinstance(image_anchor, ImageAnchor)
    assert isinstance(candidate_anchor, CandidateAnchor)

    hypothesis = HypothesisBranch(
        hypothesis_id="H001",
        description="Target is likely past the north branch.",
        anchors=[image_anchor, candidate_anchor],
        status=HypothesisStatus.ACTIVE,
        confidence=0.45,
    )
    restored_hypothesis = HypothesisBranch.from_dict(
        json.loads(json.dumps(hypothesis.to_dict()))
    )
    assert restored_hypothesis.status is HypothesisStatus.ACTIVE
    assert [type(anchor) for anchor in restored_hypothesis.anchors] == [
        ImageAnchor,
        CandidateAnchor,
    ]

    explore = ExploreIntent(
        frontier_id=7,
        spatial_branch_id="B001",
        action_mode=ExecutorActionMode.SWITCH_SPATIAL_BRANCH,
        reason_code="TEST",
    )
    visual = VisualApproachIntent(
        candidate_id="C001",
        image_path="1-view_0.png",
        target_phrase="chair",
        approach_xyz=np.array([1.0, 0.0, 2.0]),
    )
    viewpoint = TargetViewpointIntent(
        candidate_id="C001",
        image_path="1-view_0.png",
        target_phrase="chair",
        target_xyz=np.array([2.0, 0.0, 3.0]),
    )
    assert VisualApproachIntent.from_dict(visual.to_dict()).mode is NavigationMode.VISUAL_APPROACH
    assert TargetViewpointIntent.from_dict(viewpoint.to_dict()).mode is NavigationMode.TARGET_APPROACH

    outcome = StepOutcome(
        step=3,
        mode=NavigationMode.EXPLORE,
        intent=explore,
        navigation_result=NavigationResult(
            mode=NavigationMode.EXPLORE,
            success=True,
            target_arrived=True,
            frontier_id=7,
            step=3,
        ),
        events=[
            TypedEvent(
                event_id="e1",
                type=EventType.FRONTIER_REACHED,
                step=3,
                entity_id="F_007",
            )
        ],
        answerer_decision=AnswererDecision.NOT_FOUND,
        verification_result=VerifyStatus.POOR_VIEW,
    )
    restored_outcome = StepOutcome.from_dict(json.loads(json.dumps(outcome.to_dict())))
    assert restored_outcome.intent.frontier_id == 7
    assert restored_outcome.events[0].type is EventType.FRONTIER_REACHED
    assert restored_outcome.verification_result is VerifyStatus.POOR_VIEW


def test_subtask_reset_clears_task_scope_but_preserves_spatial_branches():
    memory = SubtaskWorkingMemory()
    memory.upsert_spatial_branch(
        SpatialBranchRecord(spatial_branch_id="B001", frontier_ids=[1])
    )
    memory.upsert_branch_task_state(
        BranchTaskState(
            spatial_branch_id="B001",
            status=BranchTaskStatus.ADVANCING,
            selected_count=2,
        )
    )
    memory.set_hypotheses_from_manager(
        [
            HypothesisBranch(
                hypothesis_id="H001",
                description="Old hypothesis",
                anchors=[],
            )
        ]
    )
    memory.add_typed_event(
        TypedEvent(event_id="e1", type=EventType.CANDIDATE_VISIBLE, step=1)
    )
    memory.frontier_registry[1] = FrontierState(
        frontier_id=1,
        centroid=np.array([0.0, 0.0, 0.0]),
        area=4.0,
        view_yaw=0.0,
        first_seen_step=0,
        last_seen_step=1,
        selected_count=2,
        reached_count=1,
        status=F_ACTIVE,
        last_result="NO_NEW_INFO",
    )

    memory.reset_for_new_subtask("task-2", "Find the table")

    assert "B001" in memory.spatial_branches
    assert memory.branch_task_states == {}
    assert memory.hypotheses == {}
    assert memory.target_candidates == {}
    assert memory.typed_events == []
    assert memory.recent_frontier_ids == []
    frontier_state = memory.frontier_registry[1]
    assert frontier_state.status == "STALE"
    assert frontier_state.selected_count == 0
    assert frontier_state.reached_count == 0


def test_hypothesis_single_writer_constraint():
    memory = SubtaskWorkingMemory()
    memory.set_hypotheses_from_manager(
        [
            HypothesisBranch(
                hypothesis_id="H001",
                description="Valid manager update",
                anchors=[],
                writer="HypothesisManager",
            )
        ]
    )
    assert "H001" in memory.hypotheses

    try:
        memory.set_hypotheses_from_manager(
            [
                HypothesisBranch(
                    hypothesis_id="H002",
                    description="Invalid writer",
                    anchors=[],
                    writer="Answerer",
                )
            ]
        )
    except ValueError as exc:
        assert "HypothesisManager" in str(exc)
    else:
        raise AssertionError("Expected non-HypothesisManager writer to be rejected")


def test_event_engine_debounce_and_routing():
    engine = EventEngine(EventEngineConfig(debounce_steps=2, memory_pool_budget=1))
    first = engine.emit(EventType.HYPOTHESIS_SUPPORTED, step=4, entity_id="H001")
    duplicate = engine.emit(EventType.HYPOTHESIS_SUPPORTED, step=5, entity_id="H001")
    later = engine.emit(EventType.HYPOTHESIS_SUPPORTED, step=6, entity_id="H001")

    assert first is not None
    assert duplicate is None
    assert later is not None

    memory_events = engine.detect_memory_events(pool_size=3, step=6)
    routing = engine.route([first] + memory_events)
    assert routing.call_hypothesis_manager is True
    assert routing.call_memory_manager is True
    assert EventType.HYPOTHESIS_SUPPORTED.value in routing.reasons
    assert EventType.WORKING_MEMORY_OVER_BUDGET.value in routing.reasons


def test_answerer_evidence_updates_route_to_hypothesis_manager():
    memory = SubtaskWorkingMemory()
    engine = EventEngine(EventEngineConfig(debounce_steps=1))
    events = engine.detect_answerer_events(
        AnswererDecision.NOT_FOUND,
        step=7,
        evidence_updates=[
            {"hypothesis_id": "H001", "result": "SUPPORT", "observed_cues": ["red chair"]},
            {"hypothesis_id": "H002", "result": "CONTRADICT", "missing_expected_cues": ["lamp"]},
            {"hypothesis_id": "H003", "result": "TEST_COMPLETED"},
        ],
        working_memory=memory,
    )

    assert [event.type for event in events] == [
        EventType.HYPOTHESIS_SUPPORTED,
        EventType.HYPOTHESIS_CONTRADICTED,
        EventType.HYPOTHESIS_TEST_COMPLETED,
    ]
    assert engine.route(events).call_hypothesis_manager is True
    assert [event.type for event in memory.typed_events] == [event.type for event in events]


def test_branch_tracker_sync_and_eligibility_filter():
    memory = SubtaskWorkingMemory()
    tracker = BranchTracker(BranchTrackerConfig(match_distance_m=0.25))
    planner = FakeTSDFPlanner(
        [
            FakeFrontier(10, [0.0, 0.0, 0.0], region=[True, True, True]),
            FakeFrontier(11, [5.0, 0.0, 0.0], region=[True]),
        ]
    )

    instances, events = tracker.sync_frontiers(
        planner,
        memory,
        step=1,
        current_position=np.array([0.0, 0.0, 0.0]),
    )

    assert {inst.frontier_id for inst in instances} == {10, 11}
    assert all(inst.spatial_branch_id for inst in instances)
    assert {event.type for event in events} == {EventType.SPATIAL_BRANCH_CREATED}

    branch_for_10 = memory.get_branch_for_frontier(10)
    branch_for_11 = memory.get_branch_for_frontier(11)
    assert branch_for_10 is not None
    assert branch_for_11 is not None

    memory.branch_task_states[branch_for_10.spatial_branch_id].status = BranchTaskStatus.CLOSED
    branch_for_11.merged_into = "B999"
    memory.upsert_spatial_branch(branch_for_11)
    memory.frontier_registry[11] = FrontierState(
        frontier_id=11,
        centroid=np.array([5.0, 0.0, 0.0]),
        area=1.0,
        view_yaw=0.0,
        first_seen_step=1,
        last_seen_step=1,
        selected_count=3,
    )

    eligible = tracker.eligible_frontier_instances(
        instances,
        memory,
        recent_ids=[10],
        recent_window=3,
        max_reselect=2,
    )
    assert eligible == []

    memory.branch_task_states[branch_for_10.spatial_branch_id].status = BranchTaskStatus.ADVANCING
    branch_for_11.merged_into = None
    memory.upsert_spatial_branch(branch_for_11)
    eligible = tracker.eligible_frontier_instances(
        instances,
        memory,
        recent_ids=[10],
        recent_window=3,
        max_reselect=2,
    )
    assert [inst.frontier_id for inst in eligible] == []


def test_branch_tracker_stalled_revisiting_split_and_merge_v1():
    tracker = BranchTracker(
        BranchTrackerConfig(
            match_distance_m=0.25,
            stalled_progress_epsilon=0.05,
            stalled_after_steps=2,
            revisit_distance_m=0.4,
            split_distance_m=0.4,
            split_tip_distance_m=1.0,
            split_after_steps=2,
            merge_distance_m=0.5,
        )
    )
    memory = SubtaskWorkingMemory()
    planner = FakeTSDFPlanner([FakeFrontier(1, [0.0, 0.0, 0.0])])
    tracker.sync_frontiers(planner, memory, step=1, current_position=[5.0, 0.0, 5.0])
    branch = memory.get_branch_for_frontier(1)
    assert branch is not None

    planner.frontiers = [FakeFrontier(1, [0.01, 0.0, 0.01])]
    tracker.sync_frontiers(planner, memory, step=2, current_position=[5.0, 0.0, 5.0])
    planner.frontiers = [FakeFrontier(1, [0.01, 0.0, 0.01])]
    _instances, events = tracker.sync_frontiers(
        planner, memory, step=3, current_position=[5.0, 0.0, 5.0]
    )
    state = memory.branch_task_states[branch.spatial_branch_id]
    assert state.status is BranchTaskStatus.STALLED
    assert EventType.SPATIAL_BRANCH_STALLED in {event.type for event in events}

    planner.frontiers = [FakeFrontier(1, [1.5, 0.0, 0.0])]
    _instances, events = tracker.sync_frontiers(
        planner, memory, step=4, current_position=[0.0, 0.0, 0.0]
    )
    state = memory.branch_task_states[branch.spatial_branch_id]
    assert state.status is BranchTaskStatus.REVISITING
    assert EventType.SPATIAL_BRANCH_REVISITING in {event.type for event in events}

    planner.frontiers = [FakeFrontier(2, [0.05, 0.0, 0.0])]
    instances, _events = tracker.sync_frontiers(
        planner, memory, step=5, current_position=[5.0, 0.0, 5.0]
    )
    assert instances[0].spatial_branch_id == branch.spatial_branch_id
    planner.frontiers = [FakeFrontier(2, [0.05, 0.0, 0.0])]
    instances, _events = tracker.sync_frontiers(
        planner, memory, step=6, current_position=[5.0, 0.0, 5.0]
    )
    split_branch = memory.get_branch_for_frontier(2)
    assert split_branch is not None
    assert split_branch.spatial_branch_id != branch.spatial_branch_id
    assert f"split_from:{branch.spatial_branch_id}" in split_branch.aliases
    assert instances[0].spatial_branch_id == split_branch.spatial_branch_id

    merge_memory = SubtaskWorkingMemory()
    merge_tracker = BranchTracker(BranchTrackerConfig(match_distance_m=0.25, merge_distance_m=0.5))
    b1 = SpatialBranchRecord(
        spatial_branch_id="B001",
        frontier_ids=[10],
        spine=[[0.0, 0.0, 0.0]],
        active_tip_frontier_id=10,
    )
    b2 = SpatialBranchRecord(
        spatial_branch_id="B002",
        frontier_ids=[20],
        spine=[[0.2, 0.0, 0.1]],
        active_tip_frontier_id=20,
    )
    merge_memory.upsert_spatial_branch(b1)
    merge_memory.upsert_spatial_branch(b2)
    merge_planner = FakeTSDFPlanner([FakeFrontier(20, [0.25, 0.0, 0.1])])
    merge_tracker.sync_frontiers(
        merge_planner, merge_memory, step=2, current_position=[3.0, 0.0, 3.0]
    )
    assert merge_memory.spatial_branches["B001"].merged_into == "B002"
    assert "merged_into:B002" in merge_memory.spatial_branches["B001"].aliases


def test_bev_renderer_uses_tsdf_state_and_frontier_labels():
    from src.spatial_context import render_bev_context

    planner = FakeTSDFPlanner([])
    planner.unexplored[:, :] = True
    planner.unexplored[1:8, 1:8] = False
    planner.unoccupied[2:7, 2:7] = True
    planner.occupied[4, 4] = True
    frontier = FrontierInstance(
        frontier_id=10,
        position=[5.0, 0.0, 5.0],
        spatial_branch_id="B001",
    )

    bev = render_bev_context(
        planner,
        current_pose=[2.0, 0.0, 2.0],
        current_yaw=0.0,
        frontier_instances=[frontier],
        recent_decision_poses=[[3.0, 0.0, 3.0]],
    )

    assert bev.image.shape == (512, 512, 3)
    assert "F_010/B001" in bev.labels
    assert len(np.unique(bev.image.reshape(-1, 3), axis=0)) > 1


def test_verify_gate_only_allows_reached_target_viewpoint():
    candidate = TargetCandidate(
        candidate_id="C001",
        subtask_id="task",
        image_path="1-view_0.png",
        source_step=1,
        camera_pose=np.eye(4),
        view_yaw=0.0,
        target_phrase="chair",
    )
    target_intent = TargetViewpointIntent(
        candidate_id="C001",
        image_path="1-view_0.png",
        target_phrase="chair",
        target_xyz=[1.0, 0.0, 2.0],
    )
    visual_intent = VisualApproachIntent(
        candidate_id="C001",
        image_path="1-view_0.png",
        target_phrase="chair",
        approach_xyz=[1.0, 0.0, 2.0],
    )
    candidate.nav_target_kind = NavTargetKind.VIEWPOINT_POSE
    candidate.nav_status = NavStatus.REACHED

    assert should_enter_verify(True, target_intent, candidate) is True
    assert should_enter_verify(False, target_intent, candidate) is False
    assert should_enter_verify(True, visual_intent, candidate) is False
    assert should_enter_verify(True, ExploreIntent(frontier_id=1), candidate) is False
    candidate.nav_target_kind = NavTargetKind.VISUAL_APPROACH_POSE
    assert should_enter_verify(True, target_intent, candidate) is False


def test_candidate_controller_rejects_when_visual_approach_unavailable():
    from src.candidate_controller import CandidateController

    memory = SubtaskWorkingMemory()
    candidate = memory.get_or_create_candidate(
        image_path="1-view_0.png",
        target_phrase="chair",
        source_step=1,
        camera_pose=np.eye(4),
        view_yaw=0.0,
    )
    controller = CandidateController(SimpleNamespace(AVU_conf_threshold=0.1, dicision_radius=1.0))
    result = controller._build_visual_approach_result(
        scene=SimpleNamespace(),
        tsdf_planner=SimpleNamespace(),
        working_memory=memory,
        candidate=candidate,
        img_path="1-view_0.png",
        target_phrase="chair",
        cam_pose=np.eye(4),
        view_yaw=0.0,
        l3_boxes=None,
        l3_best_idx=None,
        step_index=1,
    )

    assert result.intent is None
    assert candidate.status == S_REJECTED
    assert candidate.nav_target_kind is None
    assert candidate.nav_status is NavStatus.FAILED
    assert should_enter_verify(True, result.intent, candidate) is False
    assert {event.type for event in result.events} == {
        EventType.GROUNDING_FAILED,
        EventType.CANDIDATE_REJECTED,
    }


def test_candidate_clip_gate_uses_raw_margin_depth_and_does_not_mutate_boxes():
    import torch
    import src.candidate_controller as controller_module
    from src.candidate_controller import CandidateController

    class FakeBoxes:
        def __init__(self, xyxy, conf):
            self.xyxy = torch.as_tensor(xyxy, dtype=torch.float32)
            self.conf = torch.as_tensor(conf, dtype=torch.float32)

        def __len__(self):
            return int(self.xyxy.shape[0])

    class FakeResult:
        def __init__(self, xyxy, conf):
            self.boxes = FakeBoxes(xyxy, conf)

    class FakeDetectionModel:
        def __init__(self, boxes):
            self.boxes = boxes
            self.classes = []

        def set_classes(self, classes):
            self.classes = list(classes)

        def predict(self, image_rgb, conf=0.0, verbose=False):
            if self.classes == controller_module.GENERIC_CLASSES:
                return [FakeResult(self.boxes, [0.8, 0.7])]
            return []

    class FakeObjClasses:
        def get_classes_arr(self):
            return ["chair"]

    boxes = np.array([[1, 1, 4, 4], [6, 6, 10, 10]], dtype=float)
    original_boxes = boxes.copy()
    scene = SimpleNamespace(
        detection_model=FakeDetectionModel(boxes),
        obj_classes=FakeObjClasses(),
        clip_model=object(),
        clip_tokenizer=object(),
        clip_preprocess=object(),
    )
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    depth = np.ones((12, 12), dtype=float)
    candidate = TargetCandidate(
        candidate_id="C001",
        subtask_id="task",
        image_path="1-view_0.png",
        source_step=1,
        camera_pose=np.eye(4),
        view_yaw=0.0,
        target_phrase="chair",
    )

    def fake_clip(_model, _tokenizer, _preprocess, crop, _prompt):
        return np.array([0.9 if crop.shape[0] == 3 else 0.2], dtype=float)

    module_name = "src.conceptgraph.utils.model_utils"
    old_model_utils = sys.modules.get(module_name)
    fake_model_utils = types.ModuleType(module_name)
    fake_model_utils.clip_recognition = fake_clip
    sys.modules[module_name] = fake_model_utils
    try:
        controller = CandidateController(
            SimpleNamespace(AVU_conf_threshold=0.1, dicision_radius=1.0),
            clip_rerank_thresh=0.5,
            clip_margin_thresh=0.3,
            clip_depth_min_ratio=0.8,
        )
        ground2d, l3_boxes, best_idx = controller._ground_2d(
            scene,
            image,
            "chair",
            ["chair"],
            candidate,
            depth_array=depth,
        )
        assert ground2d is not None
        assert best_idx == 0
        assert ground2d.raw_score == 0.9
        assert abs(ground2d.rank_score - 0.7) < 1e-6
        assert np.array_equal(l3_boxes, original_boxes)

        zero_depth_candidate = TargetCandidate(
            candidate_id="C002",
            subtask_id="task",
            image_path="1-view_0.png",
            source_step=1,
            camera_pose=np.eye(4),
            view_yaw=0.0,
            target_phrase="chair",
        )
        rejected, _l3, rejected_idx = controller._ground_2d(
            scene,
            image,
            "chair",
            ["chair"],
            zero_depth_candidate,
            depth_array=np.zeros_like(depth),
        )
        assert rejected is None
        assert rejected_idx == 0
        assert "depth 0.00" in zero_depth_candidate.grounding_attempts[-1].reason
    finally:
        if old_model_utils is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = old_model_utils


def test_agent_json_parsers_fail_safe_without_random_frontier():
    old_explore_utils = _install_explore_utils_stub()
    old_explore_multi_agent = sys.modules.get("src.explore_multi_agent")
    sys.modules.pop("src.explore_multi_agent", None)
    try:
        import src.explore_multi_agent as explore_multi_agent

        assert explore_multi_agent.parse_answerer_response("not json") == (
            "NOT_FOUND",
            None,
            None,
        )
        assert explore_multi_agent.parse_answerer_response(
            '{"decision": "CANDIDATE_VISIBLE", "candidate": {"image": 2, "target_phrase": "lamp"}}'
        ) == ("CANDIDATE_VISIBLE", 2, "lamp")

        stop_intent, reason = explore_multi_agent.parse_executor_json_response(
            '{"frontier_id": "abc", "action_mode": "BOGUS"}'
        )
        assert stop_intent.mode is NavigationMode.STOP
        assert stop_intent.action_mode is ExecutorActionMode.STOP
        assert stop_intent.frontier_id is None

        malformed_intent, _ = explore_multi_agent.parse_executor_json_response(
            "select whatever seems best"
        )
        assert malformed_intent.mode is NavigationMode.STOP
        assert malformed_intent.reason_code == "EXECUTOR_PARSE_FAIL"

        hypotheses, _ = explore_multi_agent.parse_hypothesis_manager_response(
            '{"new_hypotheses": [{"id": "H003", "summary": "Check branch B003", '
            '"anchors": [{"kind": "spatial_branch", "spatial_branch_id": "B003"}]}]}',
            step_index=9,
        )
        assert len(hypotheses) == 1
        assert hypotheses[0].hypothesis_id == "H003"
        assert hypotheses[0].writer == "HypothesisManager"
    finally:
        if old_explore_multi_agent is None:
            sys.modules.pop("src.explore_multi_agent", None)
        else:
            sys.modules["src.explore_multi_agent"] = old_explore_multi_agent
        if old_explore_utils is None:
            sys.modules.pop("src.explore_utils", None)
        else:
            sys.modules["src.explore_utils"] = old_explore_utils


def test_goat_runner_uses_rebuild_path_without_legacy_flag_fallback():
    runner = (ROOT / "run_goatbench_evaluation.py").read_text()

    assert "use_multi_agent" not in runner
    assert "query_vlm_for_response(" not in runner
    assert "query_vlm_multi_agent(" in runner


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
