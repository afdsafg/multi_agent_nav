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
    def __init__(self):
        self.frontiers = [SimpleNamespace(frontier_id=1)]
        self.frontier_registry = {
            1: FrontierState(
                frontier_id=1,
                centroid=np.array([0.0, 0.0, 0.0]),
                area=10.0,
                view_yaw=0.0,
                first_seen_step=0,
                last_seen_step=3,
            )
        }
        self.selected = []

    def get_valid_frontier_ids(self, **kwargs):
        return [1]

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
