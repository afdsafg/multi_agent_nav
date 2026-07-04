import json
import sys
import types
from pathlib import Path

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
    FrontierState,
    HypothesisBranch,
    HypothesisStatus,
    ImageAnchor,
    NavigationMode,
    NavigationResult,
    SpatialBranchRecord,
    StepOutcome,
    SubtaskWorkingMemory,
    TargetViewpointIntent,
    TypedEvent,
    VerifyStatus,
    VisualApproachIntent,
    anchor_from_dict,
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

    def voxel2habitat(self, position):
        arr = np.asarray(position, dtype=float)
        if arr.shape[0] == 2:
            return np.array([arr[0], 0.0, arr[1]], dtype=float)
        return arr.astype(float)


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
    first = engine.emit(EventType.CANDIDATE_VISIBLE, step=4, entity_id="C001")
    duplicate = engine.emit(EventType.CANDIDATE_VISIBLE, step=5, entity_id="C001")
    later = engine.emit(EventType.CANDIDATE_VISIBLE, step=6, entity_id="C001")

    assert first is not None
    assert duplicate is None
    assert later is not None

    memory_events = engine.detect_memory_events(pool_size=3, step=6)
    routing = engine.route([first] + memory_events)
    assert routing.call_hypothesis_manager is True
    assert routing.call_memory_manager is True
    assert EventType.CANDIDATE_VISIBLE.value in routing.reasons
    assert EventType.WORKING_MEMORY_OVER_BUDGET.value in routing.reasons


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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
