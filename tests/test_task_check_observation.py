import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task_check_observation import (
    build_arrival_check_angles,
    collect_arrival_task_check_views,
)


class FakeScene:
    def __init__(self):
        self.calls = []

    def get_observation(self, pts, angle=None, rotation=None):
        self.calls.append((np.array(pts), angle, rotation))
        value = int(round(angle * 10)) % 255
        image = np.zeros((4, 6, 3), dtype=np.uint8) + value
        return {"color_sensor": image}, np.eye(4)


def test_build_arrival_check_angles_keeps_center_heading_last():
    angles = build_arrival_check_angles(
        angle=1.0,
        extra_views=2,
        extra_view_angle_deg=30,
    )

    assert len(angles) == 3
    assert np.isclose(angles[-1], 1.0)
    assert np.isclose(angles[0], 1.0 - np.pi / 6)
    assert np.isclose(angles[1], 1.0 + np.pi / 6)


def test_collect_arrival_task_check_views_uses_fresh_arrival_pose():
    scene = FakeScene()
    cfg = SimpleNamespace(
        extra_view_phase_2=2,
        extra_view_angle_deg_phase_2=30,
        prompt_h=8,
        prompt_w=10,
    )
    pts = np.array([1.0, 2.0, 3.0])

    views = collect_arrival_task_check_views(scene, pts, angle=1.0, cfg=cfg)

    assert len(views) == 3
    assert all(view.shape == (8, 10, 3) for view in views)
    assert len(scene.calls) == 3
    assert all(np.array_equal(call[0], pts) for call in scene.calls)
    assert np.isclose(scene.calls[-1][1], 1.0)
