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
    FrontierAnchor,
    FrontierInstance,
    FrontierState,
    Grounding2D,
    HypothesisBranch,
    HypothesisStatus,
    ImageAnchor,
    NavStatus,
    NavTargetKind,
    NavigationMode,
    NavigationResult,
    S_GROUNDED_3D,
    S_NEED_CLOSER_VIEW,
    SpatialBranchAnchor,
    SpatialBranchRecord,
    StepOutcome,
    SubtaskWorkingMemory,
    TargetCandidate,
    TargetViewpointIntent,
    TypedEvent,
    VerifyStatus,
    VisualApproachIntent,
    anchor_from_dict,
    anchor_from_value,
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
        self.frontier_registry = {}
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

    def mark_frontier_selected(self, frontier_id):
        self.last_selected_frontier_id = frontier_id


def _install_explore_utils_stub():
    old_module = sys.modules.get("src.explore_utils")
    module = types.ModuleType("src.explore_utils")
    module.call_openai_api = lambda *args, **kwargs: None
    module.encode_tensor2base64 = lambda value: value
    module.resize_image = lambda image, *args, **kwargs: image

    def format_question_stub(step, *args, **kwargs):
        if isinstance(step, dict):
            return step.get("question", ""), step.get("image")
        return step, None

    module.format_question = format_question_stub
    module.explore_two_step = lambda *args, **kwargs: None
    module.Key_Subgraph_Selection = lambda *args, **kwargs: (
        "",
        None,
        [],
        [],
        [],
        {},
        [],
    )
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


def test_anchor_parser_accepts_hypothesis_manager_shorthand_strings():
    branch_anchor = anchor_from_value("B001")
    assert isinstance(branch_anchor, SpatialBranchAnchor)
    assert branch_anchor.spatial_branch_id == "B001"

    branch_anchor_with_prefix = anchor_from_value("spatial_branch:B002")
    assert isinstance(branch_anchor_with_prefix, SpatialBranchAnchor)
    assert branch_anchor_with_prefix.spatial_branch_id == "B002"

    frontier_anchor = anchor_from_value("F_001")
    assert isinstance(frontier_anchor, FrontierAnchor)
    assert frontier_anchor.frontier_id == 1

    frontier_anchor_from_dict = anchor_from_value(
        {"kind": "frontier", "frontier_id": "F_003", "spatial_branch_id": "B001"}
    )
    assert isinstance(frontier_anchor_from_dict, FrontierAnchor)
    assert frontier_anchor_from_dict.frontier_id == 3
    assert frontier_anchor_from_dict.spatial_branch_id == "B001"

    hypothesis = HypothesisBranch(
        hypothesis_id="H001",
        claim="Check branch B001.",
        anchor="B001",
        anchors="B001",
    )
    assert isinstance(hypothesis.anchor, SpatialBranchAnchor)
    assert hypothesis.linked_spatial_branches == ["B001"]


def test_subtask_reset_clears_task_scope_but_preserves_spatial_branches():
    memory = SubtaskWorkingMemory()
    memory.upsert_spatial_branch(
        SpatialBranchRecord(spatial_branch_id="B001", frontier_ids=[1], geometric_status="OPEN")
    )
    memory.upsert_spatial_branch(
        SpatialBranchRecord(spatial_branch_id="B002", frontier_ids=[2], geometric_status="CLOSED")
    )
    memory.upsert_spatial_branch(
        SpatialBranchRecord(
            spatial_branch_id="B003",
            frontier_ids=[3],
            geometric_status="OPEN",
            merged_into="B001",
        )
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
    assert memory.branch_task_states["B001"].status is BranchTaskStatus.NEW
    assert memory.branch_task_states["B001"].subtask_id == "task-2"
    assert "B002" not in memory.branch_task_states
    assert "B003" not in memory.branch_task_states
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
    first = engine.emit(
        EventType.HYPOTHESIS_SUPPORTED,
        step=4,
        entity_id="H001",
        payload={"score": 1},
        severity="info",
    )
    duplicate = engine.emit(
        EventType.HYPOTHESIS_SUPPORTED,
        step=5,
        entity_id="H001",
        payload={"score": 1},
        severity="info",
    )
    changed = engine.emit(
        EventType.HYPOTHESIS_SUPPORTED,
        step=5,
        entity_id="H001",
        payload={"score": 2},
        severity="info",
    )
    severe = engine.emit(
        EventType.HYPOTHESIS_SUPPORTED,
        step=5,
        entity_id="H001",
        payload={"score": 2},
        severity="warning",
    )
    later = engine.emit(
        EventType.HYPOTHESIS_SUPPORTED,
        step=7,
        entity_id="H001",
        payload={"score": 2},
        severity="info",
    )

    assert first is not None
    assert duplicate is None
    assert changed is not None
    assert severe is not None
    assert later is not None

    memory_events = engine.detect_memory_events(pool_size=3, step=6)
    routing = engine.route([first] + memory_events)
    assert routing.call_hypothesis_manager is True
    assert routing.call_memory_manager is True
    assert EventType.HYPOTHESIS_SUPPORTED.value in routing.reasons
    assert EventType.WORKING_MEMORY_OVER_BUDGET.value in routing.reasons


def test_pending_typed_events_are_consumed_once_and_ttl_bound():
    memory = SubtaskWorkingMemory()
    event = TypedEvent(
        event_id="candidate-rejected-1",
        type=EventType.CANDIDATE_REJECTED,
        step=10,
        entity_id="C001",
        ttl_steps=4,
    )
    memory.add_typed_event(event)

    assert memory.pop_pending_typed_events(current_step=11) == [event]
    memory.mark_events_consumed([event])
    assert memory.pop_pending_typed_events(current_step=12) == []

    expired = TypedEvent(
        event_id="expired",
        type=EventType.WRONG_INSTANCE,
        step=1,
        ttl_steps=2,
    )
    memory.add_typed_event(expired)
    assert memory.pop_pending_typed_events(current_step=4) == []


def test_spatial_events_route_only_when_linked_and_over_threshold():
    engine = EventEngine(
        EventEngineConfig(
            spatial_stalled_steps_threshold=2,
            spatial_revisit_reversal_threshold=2,
            spatial_revisit_overlap_threshold=0.7,
        )
    )
    memory = SubtaskWorkingMemory()
    memory.upsert_branch_task_state(
        BranchTaskState(
            spatial_branch_id="B001",
            status=BranchTaskStatus.STALLED,
            steps_without_progress=2,
        )
    )
    stalled_event = TypedEvent(
        event_id="stalled",
        type=EventType.SPATIAL_BRANCH_STALLED,
        step=3,
        entity_id="B001",
    )

    assert engine.route([stalled_event], working_memory=memory).call_hypothesis_manager is False

    memory.set_hypotheses_from_manager(
        [
            HypothesisBranch(
                hypothesis_id="H001",
                claim="Check branch B001.",
                linked_spatial_branches=["B001"],
                status=HypothesisStatus.ACTIVE,
            )
        ]
    )
    assert engine.route([stalled_event], working_memory=memory).call_hypothesis_manager is True

    memory.branch_task_states["B001"].steps_without_progress = 1
    assert engine.route([stalled_event], working_memory=memory).call_hypothesis_manager is False

    memory.branch_task_states["B001"] = BranchTaskState(
        spatial_branch_id="B001",
        status=BranchTaskStatus.REVISITING,
        reversal_count=1,
        recent_region_overlap=0.6,
    )
    revisit_event = TypedEvent(
        event_id="revisit",
        type=EventType.SPATIAL_BRANCH_REVISITING,
        step=4,
        entity_id="B001",
    )
    assert engine.route([revisit_event], working_memory=memory).call_hypothesis_manager is False
    memory.branch_task_states["B001"].reversal_count = 2
    assert engine.route([revisit_event], working_memory=memory).call_hypothesis_manager is True


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


def test_memory_manager_routes_on_answerer_conflict_and_hypothesis_revised():
    engine = EventEngine(EventEngineConfig())
    conflict = TypedEvent(
        event_id="conflict",
        type=EventType.ANSWERER_EVIDENCE_CONFLICT,
        step=1,
    )
    revised = TypedEvent(
        event_id="revised",
        type=EventType.HYPOTHESIS_REVISED,
        step=1,
        entity_id="hypothesis_store",
    )

    routing = engine.route([conflict, revised])

    assert routing.call_memory_manager is True
    assert routing.call_hypothesis_manager is True
    assert EventType.ANSWERER_EVIDENCE_CONFLICT.value in routing.reasons
    assert EventType.HYPOTHESIS_REVISED.value in routing.reasons


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
    assert [inst.frontier_id for inst in eligible] == [10, 11]


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


def test_candidate_controller_keeps_candidate_on_grounding_failure():
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
    assert candidate.status == S_NEED_CLOSER_VIEW
    assert candidate.pinned is True
    assert "1-view_0.png" in memory.pinned_ids
    assert candidate.nav_target_kind is None
    assert candidate.nav_status is NavStatus.FAILED
    assert should_enter_verify(True, result.intent, candidate) is False
    assert [event.type for event in result.events] == [EventType.GROUNDING_FAILED]


def test_candidate_controller_stores_grounded_artifacts():
    import torch
    import src.candidate_controller as controller_module
    from src.candidate_controller import CandidateController

    class FakeCfg(dict):
        def __getattr__(self, name):
            return self[name]

    class FakePcd:
        def __init__(self, points):
            self.points = np.asarray(points, dtype=float)

    class FakeMasks:
        def __init__(self):
            self.data = torch.ones((1, 3, 3), dtype=torch.bool)

    class FakeSamResult:
        masks = FakeMasks()

    class FakeSamPredictor:
        def predict(self, *args, **kwargs):
            return [FakeSamResult()]

    memory = SubtaskWorkingMemory()
    candidate = memory.get_or_create_candidate(
        image_path="1-view_0.png",
        target_phrase="chair",
        source_step=1,
        camera_pose=np.eye(4),
        view_yaw=0.0,
    )
    scene = SimpleNamespace(
        device="cpu",
        sam_predictor=FakeSamPredictor(),
        all_depths={"1-view_0.png": np.ones((3, 3), dtype=float)},
        intrinsics=np.eye(4),
        cfg_cg=FakeCfg(
            min_points_threshold=1,
            spatial_sim_type="bbox",
            obj_pcd_max_points=10,
            downsample_voxel_size=0.01,
            dbscan_remove_noise=False,
            dbscan_eps=0.1,
            dbscan_min_points=1,
        ),
        objects={1: {"pcd": FakePcd([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]])}},
    )

    def fake_detect_to_obj(**kwargs):
        return [{"pcd": FakePcd([[2.0, 0.0, 2.0], [2.5, 0.0, 2.0]])}]

    old_concept = controller_module._conceptgraph_slam_utils
    old_utils = sys.modules.get("src.utils")
    fake_utils = types.ModuleType("src.utils")
    fake_utils.Visibility_based_Viewpoint_Decision = (
        lambda *args, **kwargs: np.array([3.0, 0.0, 3.0])
    )
    controller_module._conceptgraph_slam_utils = lambda: (
        lambda spatial_sim_type, pcd: SimpleNamespace(),
        lambda pcd, **kwargs: pcd,
        fake_detect_to_obj,
    )
    sys.modules["src.utils"] = fake_utils
    try:
        controller = CandidateController(
            SimpleNamespace(AVU_conf_threshold=0.1, dicision_radius=1.0)
        )
        result = controller._build_target_viewpoint_result(
            scene=scene,
            tsdf_planner=SimpleNamespace(),
            working_memory=memory,
            candidate=candidate,
            img_path="1-view_0.png",
            target_phrase="chair",
            target_image=np.zeros((3, 3, 3), dtype=np.uint8),
            cam_pose=np.eye(4),
            ground2d=Grounding2D(
                source="class_agnostic_clip",
                phrase="chair",
                bbox_xyxy=np.array([0, 0, 2, 2]),
            ),
            pts=np.array([0.0, 0.0, 0.0]),
            step_index=1,
        )
    finally:
        controller_module._conceptgraph_slam_utils = old_concept
        if old_utils is None:
            sys.modules.pop("src.utils", None)
        else:
            sys.modules["src.utils"] = old_utils

    stored = memory.target_candidates[candidate.candidate_id]
    assert result.intent is not None
    assert stored.status == S_GROUNDED_3D
    assert stored.bbox_xyxy == [0, 0, 2, 2]
    assert stored.mask is not None
    assert stored.target_pointcloud == [[2.0, 0.0, 2.0], [2.5, 0.0, 2.0]]
    assert stored.grounding_source == "class_agnostic_clip"


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


def test_hypothesis_parser_accepts_spec_partial_updates_and_auto_ids():
    old_explore_utils = _install_explore_utils_stub()
    old_explore_multi_agent = sys.modules.get("src.explore_multi_agent")
    sys.modules.pop("src.explore_multi_agent", None)
    try:
        import src.explore_multi_agent as explore_multi_agent

        existing = {
            "H001": HypothesisBranch(
                hypothesis_id="H001",
                claim="The target may be past branch B001.",
                expected_cues=["microwave"],
                contradiction_cues=["branch is unrelated"],
                linked_spatial_branches=["B001"],
                status=HypothesisStatus.ACTIVE,
                created_step=3,
            )
        }
        hypotheses, reason = explore_multi_agent.parse_hypothesis_manager_response(
            json.dumps(
                {
                    "updates": [
                        {
                            "hypothesis_id": "H001",
                            "decision": "REJECT",
                            "reason": "Covered the branch with no microwave.",
                        }
                    ],
                    "new_hypotheses": [
                        {
                            "claim": "Try the open branch toward the kitchen.",
                            "expected_cues": ["appliances"],
                            "next_test": "advance until a new room is exposed",
                        }
                    ],
                    "reason": "branch evidence changed",
                }
            ),
            step_index=9,
            existing_hypotheses=existing,
        )

        assert reason == "branch evidence changed"
        assert len(hypotheses) == 2
        rejected = next(h for h in hypotheses if h.hypothesis_id == "H001")
        created = next(h for h in hypotheses if h.hypothesis_id != "H001")
        assert rejected.status is HypothesisStatus.REJECTED
        assert rejected.claim == existing["H001"].claim
        assert rejected.created_step == 3
        assert "Covered the branch with no microwave." in rejected.negative_evidence
        assert created.hypothesis_id == "H002"
        assert created.claim == "Try the open branch toward the kitchen."
    finally:
        if old_explore_multi_agent is None:
            sys.modules.pop("src.explore_multi_agent", None)
        else:
            sys.modules["src.explore_multi_agent"] = old_explore_multi_agent
        if old_explore_utils is None:
            sys.modules.pop("src.explore_utils", None)
        else:
            sys.modules["src.explore_utils"] = old_explore_utils


def test_hypothesis_parser_accepts_server_shorthand_anchor_response():
    old_explore_utils = _install_explore_utils_stub()
    old_explore_multi_agent = sys.modules.get("src.explore_multi_agent")
    sys.modules.pop("src.explore_multi_agent", None)
    try:
        import src.explore_multi_agent as explore_multi_agent

        hypotheses, reason = explore_multi_agent.parse_hypothesis_manager_response(
            json.dumps(
                {
                    "updates": [],
                    "new_hypotheses": [
                        {
                            "claim": (
                                "The refrigerator is located in the area indicated by "
                                "frontier F_001 of spatial branch B001."
                            ),
                            "expected_cues": [
                                "large rectangular appliance with vertical doors"
                            ],
                            "contradiction_cues": ["non-kitchen environment"],
                            "anchor": "B001",
                            "anchors": ["B001"],
                            "next_test": "Advance along frontier F_001.",
                            "linked_spatial_branches": ["B001"],
                            "confidence": 0.5,
                        }
                    ],
                    "reason": "No active hypotheses exist.",
                }
            ),
            step_index=79,
        )

        assert reason == "No active hypotheses exist."
        assert len(hypotheses) == 1
        hypothesis = hypotheses[0]
        assert hypothesis.hypothesis_id == "H001"
        assert isinstance(hypothesis.anchor, SpatialBranchAnchor)
        assert hypothesis.anchor.spatial_branch_id == "B001"
        assert hypothesis.linked_spatial_branches == ["B001"]
        assert hypothesis.writer == "HypothesisManager"
    finally:
        if old_explore_multi_agent is None:
            sys.modules.pop("src.explore_multi_agent", None)
        else:
            sys.modules["src.explore_multi_agent"] = old_explore_multi_agent
        if old_explore_utils is None:
            sys.modules.pop("src.explore_utils", None)
        else:
            sys.modules["src.explore_utils"] = old_explore_utils


def test_deduplicate_image_pool_removes_exact_snapshot_duplicates():
    old_explore_utils = _install_explore_utils_stub()
    old_explore_multi_agent = sys.modules.get("src.explore_multi_agent")
    sys.modules.pop("src.explore_multi_agent", None)
    try:
        import src.explore_multi_agent as explore_multi_agent

        pool = [
            {"img_path": "a.png", "img_b64": "img-a"},
            {"img_path": "a.png", "img_b64": "img-a-new"},
            {"img_path": "b.png", "img_b64": "img-a"},
            {"img_path": "c.png", "img_b64": "img-c"},
            {"img_path": None, "img_b64": "img-c"},
            {"img_path": "d.png", "img_b64": None},
            {"img_path": "e.png", "img_b64": None},
        ]

        explore_multi_agent.deduplicate_image_pool(pool)

        assert [(snap.get("img_path"), snap.get("img_b64")) for snap in pool] == [
            ("a.png", "img-a"),
            ("c.png", "img-c"),
            ("d.png", None),
            ("e.png", None),
        ]
    finally:
        if old_explore_multi_agent is None:
            sys.modules.pop("src.explore_multi_agent", None)
        else:
            sys.modules["src.explore_multi_agent"] = old_explore_multi_agent
        if old_explore_utils is None:
            sys.modules.pop("src.explore_utils", None)
        else:
            sys.modules["src.explore_utils"] = old_explore_utils


def test_grounding_failure_falls_through_without_hypothesis_replan():
    old_explore_utils = _install_explore_utils_stub()
    old_explore_multi_agent = sys.modules.get("src.explore_multi_agent")
    sys.modules.pop("src.explore_multi_agent", None)
    try:
        import src.explore_multi_agent as explore_multi_agent

        memory = SubtaskWorkingMemory()
        memory.set_hypotheses_from_manager(
            [
                HypothesisBranch(
                    hypothesis_id="H001",
                    claim="Check the current branch.",
                    status=HypothesisStatus.ACTIVE,
                )
            ]
        )
        planner = FakeTSDFPlanner([FakeFrontier(10, [1.0, 0.0, 2.0])])
        scene = SimpleNamespace(objects={}, img_to_edge={})
        step = {
            "question": "Find the chair",
            "task_type": "object",
            "image": None,
            "CLR": {},
            "scene": scene,
            "tsdf_planner": planner,
            "working_memory": memory,
            "image_pool": [
                {
                    "img_path": "1-view_0.png",
                    "img_b64": "candidate-img",
                    "connected_objects": [],
                    "source": "egocentric",
                    "step": 1,
                }
            ],
            "frontier_imgs": ["frontier-img"],
            "processed_images": {},
            "image_map_reverse": {},
            "step_index": 1,
            "current_position": np.array([0.0, 0.0, 0.0]),
        }
        calls = []

        def fake_call(sys_prompt, content):
            joined = "\n".join(
                part[0] if isinstance(part, tuple) else str(part)
                for part in content
            )
            calls.append((sys_prompt, joined))
            if "HYPOTHESIS MANAGER" in sys_prompt:
                raise AssertionError("GROUNDING_FAILED alone should not route Hypothesis Manager")
            if "frontier_id" in joined and "Available Frontier IDs" in joined:
                return json.dumps(
                    {
                        "frontier_id": 10,
                        "spatial_branch_id": None,
                        "hypothesis_id": None,
                        "action_mode": "CONTINUE_SPATIAL_BRANCH",
                        "reason_code": "TEST",
                        "reason": "continue after rejected candidate",
                    }
                )
            return json.dumps(
                {
                    "decision": "CANDIDATE_VISIBLE",
                    "candidate": {"image": 0, "target_phrase": "chair"},
                    "evidence_updates": [],
                    "evidence_conflict": False,
                    "visibility": {
                        "directly_visible": "yes",
                        "central_enough": "no",
                        "partially_occluded": "yes",
                        "confidence": 0.4,
                    },
                    "reason": "plausible chair",
                    }
                )

        class GroundingFailureController:
            def __init__(self, cfg):
                self.cfg = cfg

            def handle_visible_candidate(self, **kwargs):
                return SimpleNamespace(
                    intent=None,
                    navigation_goal=None,
                    reason="grounding failed",
                    events=[
                        TypedEvent(
                            event_id="grounding-failed",
                            type=EventType.GROUNDING_FAILED,
                            step=kwargs["step_index"],
                            entity_id="C001",
                        )
                    ],
                )

        old_call = explore_multi_agent.call_openai_api
        old_controller = explore_multi_agent.CandidateController
        explore_multi_agent.call_openai_api = fake_call
        explore_multi_agent.CandidateController = GroundingFailureController
        try:
            result = explore_multi_agent.explore_multi_agent(
                step,
                SimpleNamespace(max_pool_size=6),
                verbose=False,
            )
        finally:
            explore_multi_agent.call_openai_api = old_call
            explore_multi_agent.CandidateController = old_controller

        assert result[0] == "frontier"
        assert result[1] == 0
        assert any(
            event.type is EventType.GROUNDING_FAILED
            for event in step["typed_events"]
        )
        assert not any("HYPOTHESIS MANAGER" in sys_prompt for sys_prompt, _ in calls)
    finally:
        if old_explore_multi_agent is None:
            sys.modules.pop("src.explore_multi_agent", None)
        else:
            sys.modules["src.explore_multi_agent"] = old_explore_multi_agent
        if old_explore_utils is None:
            sys.modules.pop("src.explore_utils", None)
        else:
            sys.modules["src.explore_utils"] = old_explore_utils


def test_answerer_conflict_triggers_memory_manager_in_rebuild_flow():
    old_explore_utils = _install_explore_utils_stub()
    old_explore_multi_agent = sys.modules.get("src.explore_multi_agent")
    sys.modules.pop("src.explore_multi_agent", None)
    try:
        import src.explore_multi_agent as explore_multi_agent

        memory = SubtaskWorkingMemory()
        memory.set_hypotheses_from_manager(
            [
                HypothesisBranch(
                    hypothesis_id="H001",
                    claim="Check the open branch.",
                    status=HypothesisStatus.ACTIVE,
                )
            ]
        )
        planner = FakeTSDFPlanner([FakeFrontier(10, [1.0, 0.0, 2.0])])
        step = {
            "question": "Find the microwave",
            "task_type": "object",
            "image": None,
            "CLR": {},
            "scene": SimpleNamespace(objects={}, img_to_edge={}),
            "tsdf_planner": planner,
            "working_memory": memory,
            "image_pool": [
                {
                    "img_path": "keep.png",
                    "img_b64": "keep-img",
                    "connected_objects": [],
                    "source": "kss",
                    "step": 1,
                },
                {
                    "img_path": "drop.png",
                    "img_b64": "drop-img",
                    "connected_objects": [],
                    "source": "kss",
                    "step": 1,
                },
            ],
            "frontier_imgs": ["frontier-img"],
            "processed_images": {},
            "image_map_reverse": {},
            "step_index": 5,
            "current_position": np.array([0.0, 0.0, 0.0]),
        }
        calls = []

        def fake_call(sys_prompt, content):
            joined = "\n".join(
                part[0] if isinstance(part, tuple) else str(part)
                for part in content
            )
            calls.append((sys_prompt, joined))
            if "MEMORY MANAGEMENT AGENT" in sys_prompt:
                return "Retain Images: {0}"
            if "HYPOTHESIS MANAGER" in sys_prompt:
                return json.dumps(
                    {
                        "updates": [
                            {
                                "hypothesis_id": "H001",
                                "decision": "KEEP",
                                "reason": "conflict noted",
                            }
                        ],
                        "new_hypotheses": [],
                        "reason": "conflict routed",
                    }
                )
            if "frontier_id" in joined and "Available Frontier IDs" in joined:
                return json.dumps(
                    {
                        "frontier_id": 10,
                        "spatial_branch_id": None,
                        "hypothesis_id": "H001",
                        "action_mode": "CONTINUE_SPATIAL_BRANCH",
                        "reason_code": "TEST",
                        "reason": "continue",
                    }
                )
            return json.dumps(
                {
                    "decision": "NOT_FOUND",
                    "candidate": {"image": None, "target_phrase": None},
                    "evidence_updates": [],
                    "evidence_conflict": True,
                    "reason": "conflicting visual evidence",
                }
            )

        old_call = explore_multi_agent.call_openai_api
        explore_multi_agent.call_openai_api = fake_call
        try:
            result = explore_multi_agent.explore_multi_agent(
                step,
                SimpleNamespace(max_pool_size=6),
                verbose=False,
            )
        finally:
            explore_multi_agent.call_openai_api = old_call

        assert result[0] == "frontier"
        assert result[3] == 1
        assert [snap["img_path"] for snap in step["image_pool"]] == ["keep.png"]
        assert any(
            "MEMORY MANAGEMENT AGENT" in sys_prompt for sys_prompt, _ in calls
        )
        assert any(
            event.type is EventType.ANSWERER_EVIDENCE_CONFLICT
            for event in step["typed_events"]
        )
        assert any(
            event.type is EventType.HYPOTHESIS_REVISED
            for event in step["typed_events"]
        )
    finally:
        if old_explore_multi_agent is None:
            sys.modules.pop("src.explore_multi_agent", None)
        else:
            sys.modules["src.explore_multi_agent"] = old_explore_multi_agent
        if old_explore_utils is None:
            sys.modules.pop("src.explore_utils", None)
        else:
            sys.modules["src.explore_utils"] = old_explore_utils


def test_recent_frontier_remains_eligible_for_executor():
    old_explore_utils = _install_explore_utils_stub()
    old_explore_multi_agent = sys.modules.get("src.explore_multi_agent")
    sys.modules.pop("src.explore_multi_agent", None)
    try:
        import src.explore_multi_agent as explore_multi_agent

        memory = SubtaskWorkingMemory()
        memory.recent_frontier_ids = [10]
        planner = FakeTSDFPlanner([FakeFrontier(10, [1.0, 0.0, 2.0])])
        step = {
            "question": "Find the chair",
            "task_type": "object",
            "image": None,
            "CLR": {},
            "scene": SimpleNamespace(objects={}, img_to_edge={}),
            "tsdf_planner": planner,
            "working_memory": memory,
            "image_pool": [],
            "frontier_imgs": ["frontier-img"],
            "processed_images": {},
            "image_map_reverse": {},
            "step_index": 3,
            "current_position": np.array([0.0, 0.0, 0.0]),
        }
        calls = []

        def fake_call(sys_prompt, content):
            joined = "\n".join(
                part[0] if isinstance(part, tuple) else str(part)
                for part in content
            )
            calls.append((sys_prompt, joined))
            if "HYPOTHESIS MANAGER" in sys_prompt:
                return json.dumps(
                    {
                        "updates": [],
                        "new_hypotheses": [
                            {
                                "id": "H001",
                                "summary": "Check another branch.",
                                "status": "ACTIVE",
                            }
                        ],
                        "reason": "route no-frontier event",
                    }
                )
            if "frontier_id" in joined and "Available Frontier IDs" in joined:
                return json.dumps(
                    {
                        "frontier_id": 10,
                        "spatial_branch_id": None,
                        "hypothesis_id": None,
                        "action_mode": "CONTINUE_SPATIAL_BRANCH",
                        "reason_code": "TEST",
                        "reason": "recent frontier is still eligible",
                    }
                )
            return json.dumps(
                {
                    "decision": "NOT_FOUND",
                    "candidate": {"image": None, "target_phrase": None},
                    "evidence_updates": [],
                    "evidence_conflict": False,
                    "reason": "nothing visible",
                }
            )

        old_call = explore_multi_agent.call_openai_api
        explore_multi_agent.call_openai_api = fake_call
        try:
            result = explore_multi_agent.explore_multi_agent(
                step,
                SimpleNamespace(max_pool_size=6),
                verbose=False,
            )
        finally:
            explore_multi_agent.call_openai_api = old_call

        assert result[0] == "frontier"
        assert result[1] == 0
        hypothesis_prompts = [
            prompt for sys_prompt, prompt in calls
            if "HYPOTHESIS MANAGER" in sys_prompt
        ]
        assert any("NO_ACTIVE_HYPOTHESIS" in prompt for prompt in hypothesis_prompts)
        assert not any("NO_ELIGIBLE_FRONTIER" in prompt for prompt in hypothesis_prompts)
        assert not any(
            event.type is EventType.NO_ELIGIBLE_FRONTIER
            for event in step["typed_events"]
        )
    finally:
        if old_explore_multi_agent is None:
            sys.modules.pop("src.explore_multi_agent", None)
        else:
            sys.modules["src.explore_multi_agent"] = old_explore_multi_agent
        if old_explore_utils is None:
            sys.modules.pop("src.explore_utils", None)
        else:
            sys.modules["src.explore_utils"] = old_explore_utils


def test_no_frontier_routes_hypothesis_manager_before_stop():
    old_explore_utils = _install_explore_utils_stub()
    old_explore_multi_agent = sys.modules.get("src.explore_multi_agent")
    sys.modules.pop("src.explore_multi_agent", None)
    try:
        import src.explore_multi_agent as explore_multi_agent

        memory = SubtaskWorkingMemory()
        planner = FakeTSDFPlanner([])
        step = {
            "question": "Find the chair",
            "task_type": "object",
            "image": None,
            "CLR": {},
            "scene": SimpleNamespace(objects={}, img_to_edge={}),
            "tsdf_planner": planner,
            "working_memory": memory,
            "image_pool": [],
            "frontier_imgs": [],
            "processed_images": {},
            "image_map_reverse": {},
            "step_index": 3,
            "current_position": np.array([0.0, 0.0, 0.0]),
        }
        calls = []

        def fake_call(sys_prompt, content):
            joined = "\n".join(
                part[0] if isinstance(part, tuple) else str(part)
                for part in content
            )
            calls.append((sys_prompt, joined))
            if "HYPOTHESIS MANAGER" in sys_prompt:
                return json.dumps(
                    {
                        "updates": [],
                        "new_hypotheses": [
                            {
                                "id": "H001",
                                "summary": "Check another branch.",
                                "status": "ACTIVE",
                            }
                        ],
                        "reason": "route no-frontier event",
                    }
                )
            return json.dumps(
                {
                    "decision": "NOT_FOUND",
                    "candidate": {"image": None, "target_phrase": None},
                    "evidence_updates": [],
                    "evidence_conflict": False,
                    "reason": "nothing visible",
                }
            )

        old_call = explore_multi_agent.call_openai_api
        explore_multi_agent.call_openai_api = fake_call
        try:
            result = explore_multi_agent.explore_multi_agent(
                step,
                SimpleNamespace(max_pool_size=6),
                verbose=False,
            )
        finally:
            explore_multi_agent.call_openai_api = old_call

        assert result[0] == "stop"
        hypothesis_prompts = [
            prompt for sys_prompt, prompt in calls
            if "HYPOTHESIS MANAGER" in sys_prompt
        ]
        assert any("NO_ACTIVE_HYPOTHESIS" in prompt for prompt in hypothesis_prompts)
        assert any("NO_ELIGIBLE_FRONTIER" in prompt for prompt in hypothesis_prompts)
        assert any(
            event.type is EventType.NO_ELIGIBLE_FRONTIER
            for event in step["typed_events"]
        )
    finally:
        if old_explore_multi_agent is None:
            sys.modules.pop("src.explore_multi_agent", None)
        else:
            sys.modules["src.explore_multi_agent"] = old_explore_multi_agent
        if old_explore_utils is None:
            sys.modules.pop("src.explore_utils", None)
        else:
            sys.modules["src.explore_utils"] = old_explore_utils


def test_query_vlm_multi_agent_preserves_typed_intent_and_spatial_metadata():
    old_explore_utils = _install_explore_utils_stub()
    old_explore_multi_agent = sys.modules.get("src.explore_multi_agent")
    old_query_vlm = sys.modules.get("src.query_vlm")
    sys.modules.pop("src.explore_multi_agent", None)
    sys.modules.pop("src.query_vlm", None)
    try:
        import src.query_vlm as query_vlm

        expected_intent = VisualApproachIntent(
            candidate_id="C001",
            image_path="1-view_0.png",
            target_phrase="chair",
            approach_xyz=[1.0, 0.0, 2.0],
        )
        captured_step = {}
        evidence_updates = [
            {"hypothesis_id": "H001", "result": "SUPPORT", "observed_cues": ["chair"]}
        ]

        def fake_explore(step, cfg, verbose=False):
            captured_step.update(step)
            step["evidence_updates"] = evidence_updates
            return (expected_intent, [1.0, 0.0, 2.0], "approach", 0, "1-view_0.png")

        old_explore = query_vlm.explore_multi_agent
        query_vlm.explore_multi_agent = fake_explore
        try:
            subtask_metadata = {
                "question": "Find the chair",
                "task_type": "object",
                "class": "chair",
                "image": None,
                "CLR": {},
                "is_new_subtask": False,
                "current_step": 5,
                "current_yaw": 1.25,
                "decision_pose_history": [[0.0, 0.0, 0.0]],
                "working_memory": SubtaskWorkingMemory(),
            }
            scene = SimpleNamespace(
                objects={},
                edges=[],
                img_to_edge={},
                all_observations={},
                image_pool=[],
            )
            planner = FakeTSDFPlanner([FakeFrontier(10, [1.0, 0.0, 2.0])])
            planner.frontiers[0].feature = np.zeros((2, 2, 3), dtype=np.uint8)
            cfg = SimpleNamespace(
                prompt_h=16,
                prompt_w=16,
                use_full_obj_list=False,
                egocentric_views=False,
                prefiltering=False,
                top_k_categories=1,
                use_room_filter=False,
                use_ollama=False,
            )

            result = query_vlm.query_vlm_multi_agent(
                subtask_metadata=subtask_metadata,
                scene=scene,
                tsdf_planner=planner,
                rgb_egocentric_views=[],
                cfg=cfg,
                pts=np.array([2.0, 0.0, 3.0]),
                verbose=False,
            )
        finally:
            query_vlm.explore_multi_agent = old_explore

        assert result == (expected_intent, [1.0, 0.0, 2.0], 0, "1-view_0.png")
        assert subtask_metadata["navigation_intent"] is expected_intent
        assert subtask_metadata["last_evidence_updates"] == evidence_updates
        assert captured_step["current_yaw"] == 1.25
        assert captured_step["recent_decision_poses"] == [[0.0, 0.0, 0.0]]
        assert np.array_equal(captured_step["current_position"], np.array([2.0, 0.0, 3.0]))
    finally:
        if old_query_vlm is None:
            sys.modules.pop("src.query_vlm", None)
        else:
            sys.modules["src.query_vlm"] = old_query_vlm
        if old_explore_multi_agent is None:
            sys.modules.pop("src.explore_multi_agent", None)
        else:
            sys.modules["src.explore_multi_agent"] = old_explore_multi_agent
        if old_explore_utils is None:
            sys.modules.pop("src.explore_utils", None)
        else:
            sys.modules["src.explore_utils"] = old_explore_utils


def test_verify_failure_transition_helper_modes():
    import importlib

    stubbed = {}

    def install_stub(name, module):
        stubbed[name] = sys.modules.get(name)
        sys.modules[name] = module

    omega = types.ModuleType("omegaconf")
    omega.OmegaConf = SimpleNamespace()
    install_stub("omegaconf", omega)
    install_stub("open_clip", types.ModuleType("open_clip"))
    ultralytics = types.ModuleType("ultralytics")
    ultralytics.SAM = object
    ultralytics.YOLOWorld = object
    install_stub("ultralytics", ultralytics)

    habitat = types.ModuleType("src.habitat")
    habitat.pose_habitat_to_tsdf = lambda *args, **kwargs: None
    install_stub("src.habitat", habitat)
    geom = types.ModuleType("src.geom")
    geom.get_cam_intr = lambda *args, **kwargs: None
    geom.get_scene_bnds = lambda *args, **kwargs: None
    install_stub("src.geom", geom)
    tsdf = types.ModuleType("src.tsdf_planner")
    tsdf.TSDFPlanner = object
    tsdf.Frontier = type("Frontier", (), {})
    install_stub("src.tsdf_planner", tsdf)
    scene_graph = types.ModuleType("src.multimodal_3d_scene_graph")
    scene_graph.Scene = object
    install_stub("src.multimodal_3d_scene_graph", scene_graph)
    utils = types.ModuleType("src.utils")
    utils.resize_image = lambda image, *args, **kwargs: image
    utils.calc_agent_subtask_distance = lambda *args, **kwargs: 999.0
    utils.get_pts_angle_goatbench = lambda *args, **kwargs: (None, None)
    install_stub("src.utils", utils)
    dataset = types.ModuleType("src.dataset_utils")
    dataset.prepare_goatbench_navigation_goals = lambda *args, **kwargs: None
    install_stub("src.dataset_utils", dataset)
    query = types.ModuleType("src.query_vlm")
    query.query_vlm_for_verify = lambda *args, **kwargs: (VerifyStatus.SUCCESS, "")
    query.query_vlm_multi_agent = lambda *args, **kwargs: None
    install_stub("src.query_vlm", query)
    ltm = types.ModuleType("src.long_term_memory")
    ltm.TextLongTermMemory = object
    install_stub("src.long_term_memory", ltm)
    logger = types.ModuleType("src.logger_goatbench")
    logger.Logger = object
    install_stub("src.logger_goatbench", logger)
    old_runner = sys.modules.get("run_goatbench_evaluation")
    sys.modules.pop("run_goatbench_evaluation", None)

    try:
        runner = importlib.import_module("run_goatbench_evaluation")
        candidate = TargetCandidate(
            candidate_id="C001",
            subtask_id="task",
            image_path="1-view_0.png",
            source_step=1,
            camera_pose=np.eye(4),
            view_yaw=0.0,
            target_phrase="chair",
            status=S_GROUNDED_3D,
            nav_goal_xyz=[2.0, 0.0, 3.0],
        )
        intent = TargetViewpointIntent(
            candidate_id="C001",
            image_path="1-view_0.png",
            target_phrase="chair",
            target_xyz=[2.0, 0.0, 3.0],
        )
        metadata = {}

        runner._apply_verify_failure_transition(
            metadata, VerifyStatus.POOR_VIEW, intent, candidate
        )
        assert metadata["navigation_mode"] is NavigationMode.TARGET_APPROACH
        assert isinstance(metadata["navigation_intent"], TargetViewpointIntent)
        assert metadata["navigation_intent"].reason_code == "VERIFY_POOR_VIEW_RETRY"
        assert candidate.nav_status is NavStatus.PLANNED

        runner._apply_verify_failure_transition(
            metadata, VerifyStatus.TARGET_NOT_VISIBLE, intent, candidate
        )
        assert metadata["navigation_mode"] is NavigationMode.EXPLORE
        assert metadata["navigation_intent"] is None
        assert candidate.status == S_NEED_CLOSER_VIEW
        assert candidate.nav_status is NavStatus.FAILED
    finally:
        sys.modules.pop("run_goatbench_evaluation", None)
        if old_runner is not None:
            sys.modules["run_goatbench_evaluation"] = old_runner
        for name, old_module in stubbed.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def test_goat_runner_uses_rebuild_path_without_legacy_flag_fallback():
    runner = (ROOT / "run_goatbench_evaluation.py").read_text()

    assert "use_multi_agent" not in runner
    assert "query_vlm_for_response(" not in runner
    assert "query_vlm_multi_agent(" in runner
    assert "decision_pose_history" in runner
    assert "current_yaw" in runner
    assert "_record_navigation_events" in runner
    assert "detect_navigation_events" in runner
    assert "_apply_verify_failure_transition" in runner
    assert "VERIFY_POOR_VIEW_RETRY" in runner
    assert "NavigationMode.TARGET_APPROACH" in runner
    assert "VerifyStatus.TARGET_NOT_VISIBLE" in runner



if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
