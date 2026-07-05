import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.memory_structures import (
    FB_TASK_CHECK_FAIL,
    FrontierState,
    SubtaskWorkingMemory,
)


def _install_explore_utils_stub():
    old_module = sys.modules.get("src.explore_utils")
    module = types.ModuleType("src.explore_utils")
    module.call_openai_api = lambda *args, **kwargs: None
    module.encode_tensor2base64 = lambda value: value
    module.resize_image = lambda image, *args, **kwargs: image
    module.format_question = lambda step: (step.get("question", ""), step.get("image"))
    module.Key_Subgraph_Selection = lambda *args, **kwargs: None
    sys.modules["src.explore_utils"] = module
    return old_module


class FakeTSDFPlanner:
    def __init__(self, frontier_ids=None):
        if frontier_ids is None:
            frontier_ids = [1]
        self.frontiers = [
            SimpleNamespace(frontier_id=frontier_id) for frontier_id in frontier_ids
        ]
        self.frontier_registry = {
            frontier_id: FrontierState(
                frontier_id=frontier_id,
                centroid=np.array([0.0, 0.0, 0.0]),
                area=10.0,
                view_yaw=0.0,
                first_seen_step=0,
                last_seen_step=3,
            )
            for frontier_id in frontier_ids
        }
        self.selected = []

    def get_valid_frontier_ids(self, **kwargs):
        raise AssertionError("frontier filtering should not be called")

    def mark_frontier_selected(self, frontier_id):
        self.selected.append(frontier_id)


def test_feedback_prompt_uses_numeric_current_step_not_step_dict():
    old_explore_utils = _install_explore_utils_stub()
    old_explore_multi_agent = sys.modules.get("src.explore_multi_agent")
    sys.modules.pop("src.explore_multi_agent", None)
    try:
        explore_multi_agent = importlib.import_module("src.explore_multi_agent")

        responses = iter(
            [
                "Decision: NOT_FOUND\n",
                "<update_todo_list>\n<todos>\n[ ] Explore the frontier\n</todos>\n</update_todo_list>",
                "Next Step: Frontier F_001",
            ]
        )
        explore_multi_agent.call_openai_api = lambda *args, **kwargs: next(responses)

        working_memory = SubtaskWorkingMemory()
        working_memory.add_feedback(
            step=1,
            type_=FB_TASK_CHECK_FAIL,
            reason="wrong instance",
            created_step=1,
        )
        planner = FakeTSDFPlanner()
        step = {
            "question": "Find the chair",
            "task_type": "object",
            "image": None,
            "egocentric_imgs": [],
            "frontier_imgs": ["frontier-image"],
            "processed_images": {},
            "image_map_reverse": {},
            "scene": SimpleNamespace(img_to_edge={}, objects={}),
            "is_new_subtask": False,
            "step_index": 3,
            "high_level_plan": None,
            "image_pool": [
                {
                    "img_path": "0-view_0.png",
                    "img_b64": "image",
                    "connected_objects": [],
                    "source": "egocentric",
                    "step": 0,
                }
            ],
            "CLR": {},
            "tsdf_planner": planner,
            "working_memory": working_memory,
            "episode_memory": None,
            "current_position": np.array([0.0, 0.0, 0.0]),
        }
        cfg = SimpleNamespace(
            candidate_max_closer_view_attempts=3,
            max_pool_size=6,
            planner_stale_threshold=2,
            max_frontier_reselect=2,
            frontier_recent_window=3,
        )

        target_type, target_index, _, _, _ = explore_multi_agent.explore_multi_agent(
            step,
            cfg,
        )

        assert target_type == "frontier"
        assert target_index == 0
        assert planner.selected == [1]
    finally:
        if old_explore_multi_agent is None:
            sys.modules.pop("src.explore_multi_agent", None)
        else:
            sys.modules["src.explore_multi_agent"] = old_explore_multi_agent
        if old_explore_utils is None:
            sys.modules.pop("src.explore_utils", None)
        else:
            sys.modules["src.explore_utils"] = old_explore_utils


def test_current_frontier_options_pass_to_planner_and_executor_without_status_fields():
    old_explore_utils = _install_explore_utils_stub()
    old_explore_multi_agent = sys.modules.get("src.explore_multi_agent")
    sys.modules.pop("src.explore_multi_agent", None)
    try:
        explore_multi_agent = importlib.import_module("src.explore_multi_agent")

        calls = []
        responses = iter(
            [
                "Decision: NOT_FOUND\n",
                "<update_todo_list>\n<todos>\n[ ] Explore all frontiers\n</todos>\n</update_todo_list>",
                "Next Step: Frontier F_003",
            ]
        )

        def fake_call(_sys_prompt, content, *args, **kwargs):
            text = "\n".join(part for item in content for part in item if isinstance(part, str))
            calls.append(text)
            return next(responses)

        explore_multi_agent.call_openai_api = fake_call

        planner = FakeTSDFPlanner(frontier_ids=[1, 2, 3])
        planner.frontier_registry[1].reached_count = 2
        planner.frontier_registry[2].status = "EXPLORED"
        planner.frontier_registry[3].selected_count = 9
        step = {
            "question": "Find the chair",
            "task_type": "object",
            "image": None,
            "egocentric_imgs": [],
            "frontier_imgs": ["frontier-1", "frontier-2", "frontier-3"],
            "processed_images": {},
            "image_map_reverse": {},
            "scene": SimpleNamespace(img_to_edge={}, objects={}),
            "is_new_subtask": False,
            "step_index": 3,
            "high_level_plan": None,
            "image_pool": [
                {
                    "img_path": "0-view_0.png",
                    "img_b64": "image",
                    "connected_objects": [],
                    "source": "egocentric",
                    "step": 0,
                }
            ],
            "CLR": {},
            "tsdf_planner": planner,
            "working_memory": SubtaskWorkingMemory(),
            "episode_memory": None,
            "current_position": np.array([0.0, 0.0, 0.0]),
        }
        cfg = SimpleNamespace(
            candidate_max_closer_view_attempts=3,
            max_pool_size=6,
            planner_stale_threshold=2,
            max_frontier_reselect=2,
            frontier_recent_window=3,
        )

        target_type, target_index, _, _, _ = explore_multi_agent.explore_multi_agent(
            step,
            cfg,
        )

        assert target_type == "frontier"
        assert target_index == 2
        assert planner.selected == [3]
        assert len(calls) == 3
        for call in calls:
            assert "Retain Frontiers" not in call
            assert "DO NOT SELECT" not in call
            assert "status=" not in call
            assert "selected=" not in call
            assert "reached=" not in call
        planner_prompt = calls[1]
        executor_prompt = calls[2]
        assert "Current Selectable Frontiers" in planner_prompt
        assert "Current Selectable Frontiers" in executor_prompt
        assert "Frontier F_001 (display 0):" in planner_prompt
        assert "Frontier F_002 (display 1):" in planner_prompt
        assert "Frontier F_003 (display 2):" in planner_prompt
        assert "Frontier F_001 (display 0):" in executor_prompt
        assert "Frontier F_002 (display 1):" in executor_prompt
        assert "Frontier F_003 (display 2):" in executor_prompt
    finally:
        if old_explore_multi_agent is None:
            sys.modules.pop("src.explore_multi_agent", None)
        else:
            sys.modules["src.explore_multi_agent"] = old_explore_multi_agent
        if old_explore_utils is None:
            sys.modules.pop("src.explore_utils", None)
        else:
            sys.modules["src.explore_utils"] = old_explore_utils


def test_executor_can_stop_when_no_current_frontier_options_exist():
    old_explore_utils = _install_explore_utils_stub()
    old_explore_multi_agent = sys.modules.get("src.explore_multi_agent")
    sys.modules.pop("src.explore_multi_agent", None)
    try:
        explore_multi_agent = importlib.import_module("src.explore_multi_agent")

        calls = []
        responses = iter(
            [
                "Decision: NOT_FOUND\n",
                "<update_todo_list>\n<todos>\n[x] No current frontier options\n</todos>\n</update_todo_list>",
                "Stop Exploration",
            ]
        )

        def fake_call(_sys_prompt, content, *args, **kwargs):
            text = "\n".join(part for item in content for part in item if isinstance(part, str))
            calls.append(text)
            return next(responses)

        explore_multi_agent.call_openai_api = fake_call

        planner = FakeTSDFPlanner(frontier_ids=[])
        step = {
            "question": "Find the chair",
            "task_type": "object",
            "image": None,
            "egocentric_imgs": [],
            "frontier_imgs": [],
            "processed_images": {},
            "image_map_reverse": {},
            "scene": SimpleNamespace(img_to_edge={}, objects={}),
            "is_new_subtask": False,
            "step_index": 3,
            "high_level_plan": None,
            "image_pool": [
                {
                    "img_path": "0-view_0.png",
                    "img_b64": "image",
                    "connected_objects": [],
                    "source": "egocentric",
                    "step": 0,
                }
            ],
            "CLR": {},
            "tsdf_planner": planner,
            "working_memory": SubtaskWorkingMemory(),
            "episode_memory": None,
            "current_position": np.array([0.0, 0.0, 0.0]),
        }
        cfg = SimpleNamespace(
            candidate_max_closer_view_attempts=3,
            max_pool_size=6,
            planner_stale_threshold=2,
            max_frontier_reselect=2,
            frontier_recent_window=3,
        )

        target_type, target_index, _, _, _ = explore_multi_agent.explore_multi_agent(
            step,
            cfg,
        )

        assert target_type == "stop"
        assert target_index is None
        assert planner.selected == []
        assert len(calls) == 3
        assert "No current selectable frontiers available" in calls[1]
        assert "No current selectable frontiers available" in calls[2]
    finally:
        if old_explore_multi_agent is None:
            sys.modules.pop("src.explore_multi_agent", None)
        else:
            sys.modules["src.explore_multi_agent"] = old_explore_multi_agent
        if old_explore_utils is None:
            sys.modules.pop("src.explore_utils", None)
        else:
            sys.modules["src.explore_utils"] = old_explore_utils


def test_description_prompts_use_candidate_when_context_is_uncertain():
    old_explore_utils = _install_explore_utils_stub()
    old_explore_multi_agent = sys.modules.get("src.explore_multi_agent")
    sys.modules.pop("src.explore_multi_agent", None)
    try:
        explore_multi_agent = importlib.import_module("src.explore_multi_agent")

        answerer_sys, _ = explore_multi_agent.format_answerer_prompt(
            question="Could you find the object described as being beside a fixture?",
            pool=[],
            task_type="description",
            image_goal=None,
            high_level_plan=None,
        )
        planner_sys, _ = explore_multi_agent.format_high_level_planner_prompt(
            question="Could you find the object described as being beside a fixture?",
            task_type="description",
            pool=[],
            frontier_options=[],
            high_level_plan_prev=None,
            is_new_subtask=False,
        )

        assert "contextual constraints for disambiguation" in answerer_sys
        assert "use CANDIDATE_VISIBLE instead of NOT_FOUND" in answerer_sys
        assert "context/relationship is not fully verified" in answerer_sys
        assert "agent can only navigate and observe" in planner_sys
        assert "opening, moving, manipulating, or interacting" in planner_sys
        assert "finding a better viewpoint" in planner_sys
    finally:
        if old_explore_multi_agent is None:
            sys.modules.pop("src.explore_multi_agent", None)
        else:
            sys.modules["src.explore_multi_agent"] = old_explore_multi_agent
        if old_explore_utils is None:
            sys.modules.pop("src.explore_utils", None)
        else:
            sys.modules["src.explore_utils"] = old_explore_utils
