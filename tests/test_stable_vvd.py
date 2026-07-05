import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_utils_import_stubs():
    if "habitat_sim" not in sys.modules:
        habitat_sim = types.ModuleType("habitat_sim")
        habitat_sim_utils = types.ModuleType("habitat_sim.utils")
        habitat_sim_common = types.ModuleType("habitat_sim.utils.common")
        habitat_sim_common.quat_to_angle_axis = lambda *args, **kwargs: None
        habitat_sim_common.quat_from_coeffs = lambda *args, **kwargs: None
        sys.modules["habitat_sim"] = habitat_sim
        sys.modules["habitat_sim.utils"] = habitat_sim_utils
        sys.modules["habitat_sim.utils.common"] = habitat_sim_common
    sys.modules.setdefault("quaternion", types.ModuleType("quaternion"))
    sys.modules.setdefault("open3d", types.ModuleType("open3d"))


_install_utils_import_stubs()

from src.utils import Visibility_based_Viewpoint_Decision


class FakeTSDFPlanner:
    """Planner fixture with a wall behind the target and a dangerous global snap."""

    def __init__(self):
        self.unoccupied = np.ones((41, 41), dtype=bool)
        self.island = self.unoccupied.copy()

    def habitat2voxel(self, point):
        point = np.asarray(point, dtype=float)
        return np.array([round(point[0] * 4) + 20, round(point[2] * 4) + 20])

    def get_near_true_point(self, viewpoints):
        snapped = []
        for viewpoint in viewpoints:
            viewpoint = np.asarray(viewpoint, dtype=float)
            if viewpoint[2] < 0:
                snapped.append(np.array([viewpoint[0], viewpoint[1], -0.9]))
            else:
                snapped.append(np.array([viewpoint[0], viewpoint[1], viewpoint[2]]))
        return np.asarray(snapped)

    def is_line_of_sight_clear(self, p1_habitat, p2_habitat):
        p1_habitat = np.asarray(p1_habitat, dtype=float)
        p2_habitat = np.asarray(p2_habitat, dtype=float)
        return p1_habitat[2] >= -0.05 and p2_habitat[2] >= -0.05


def _target_points():
    xs = np.linspace(-0.18, 0.18, 5)
    ys = np.linspace(0.15, 1.1, 4)
    zs = np.linspace(-0.12, 0.12, 5)
    return np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=float)


def test_vvd_keeps_viewpoint_on_evidence_side():
    planner = FakeTSDFPlanner()
    target_points = _target_points()
    pts = np.array([0.0, 0.0, 1.6])
    evidence_position = np.array([0.0, 0.7, 1.4])

    viewpoint = Visibility_based_Viewpoint_Decision(
        target_points,
        target_points,
        pts,
        planner,
        radius_factor=0.8,
        evidence_position=evidence_position,
    )

    assert viewpoint is not None
    assert viewpoint[2] > 0.0


def test_vvd_is_deterministic_for_same_evidence():
    planner = FakeTSDFPlanner()
    target_points = _target_points()
    pts = np.array([0.0, 0.0, 1.6])
    evidence_position = np.array([0.0, 0.7, 1.4])

    first = Visibility_based_Viewpoint_Decision(
        target_points,
        target_points,
        pts,
        planner,
        radius_factor=0.8,
        evidence_position=evidence_position,
    )
    second = Visibility_based_Viewpoint_Decision(
        target_points,
        target_points,
        pts,
        planner,
        radius_factor=0.8,
        evidence_position=evidence_position,
    )

    assert np.allclose(first, second)
