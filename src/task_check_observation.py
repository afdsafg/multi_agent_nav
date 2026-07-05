"""Arrival-time observations for final task verification."""

from typing import List

import numpy as np
from PIL import Image


def _resize_image(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    return np.array(Image.fromarray(image).resize((target_w, target_h)))


def build_arrival_check_angles(
    angle: float,
    extra_views: int,
    extra_view_angle_deg: float,
) -> List[float]:
    """Build a surrounding-view sweep centered on the arrival heading.

    The centered main heading is returned last, matching the main exploration
    observation order so the simulator ends on the nominal target-facing yaw.
    """
    total_views = 1 + int(extra_views)
    angle_increment = float(extra_view_angle_deg) * np.pi / 180.0
    angles = [
        float(angle) + angle_increment * (i - total_views // 2)
        for i in range(total_views)
    ]
    main_angle = angles.pop(total_views // 2)
    angles.append(main_angle)
    return angles


def collect_arrival_task_check_views(scene, pts, angle: float, cfg) -> List[np.ndarray]:
    """Collect fresh verification views after reaching a candidate viewpoint."""
    angles = build_arrival_check_angles(
        angle,
        getattr(cfg, "extra_view_phase_2", 6),
        getattr(cfg, "extra_view_angle_deg_phase_2", 40),
    )
    views: List[np.ndarray] = []
    for ang in angles:
        obs, _ = scene.get_observation(pts, angle=ang)
        views.append(_resize_image(obs["color_sensor"], cfg.prompt_h, cfg.prompt_w))
    return views
