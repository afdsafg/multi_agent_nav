"""BEV spatial context rendering for rebuild prompts.

The renderer is intentionally data-limited: it only accepts occupancy/free/
observed TSDF maps, the agent pose, recent decision poses, and current frontier
labels. It does not read GT target, GT room, or full scene graph objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import numpy as np

from src.memory_structures import FrontierInstance


@dataclass
class BEVContext:
    image: np.ndarray
    labels: List[str] = field(default_factory=list)


@dataclass
class BEVRenderConfig:
    unknown_color: Tuple[int, int, int] = (245, 245, 245)
    observed_color: Tuple[int, int, int] = (206, 232, 211)
    free_color: Tuple[int, int, int] = (215, 215, 215)
    occupied_color: Tuple[int, int, int] = (35, 35, 35)
    frontier_color: Tuple[int, int, int] = (178, 42, 166)
    decision_color: Tuple[int, int, int] = (35, 104, 196)
    agent_color: Tuple[int, int, int] = (22, 160, 220)
    output_size: Optional[int] = 512


def render_bev_context(
    tsdf_planner,
    current_pose=None,
    current_yaw: Optional[float] = None,
    frontier_instances: Optional[Iterable[FrontierInstance]] = None,
    recent_decision_poses: Optional[Iterable] = None,
    config: Optional[BEVRenderConfig] = None,
) -> BEVContext:
    """Render a top-down BEV image from planner state.

    Args:
        tsdf_planner: object exposing unexplored/unoccupied/occupied maps and
            habitat2voxel for pose conversion.
        current_pose: current Habitat pose (x, y, z), optional.
        current_yaw: current yaw in radians, optional.
        frontier_instances: typed current frontiers, optional.
        recent_decision_poses: recent Habitat poses chosen by high-level logic.
        config: render colors and output size.
    """
    cfg = config or BEVRenderConfig()
    shape = _infer_shape(tsdf_planner)
    canvas = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    canvas[:, :] = cfg.unknown_color

    unexplored = _safe_bool_map(getattr(tsdf_planner, "unexplored", None), shape)
    unoccupied = _safe_bool_map(getattr(tsdf_planner, "unoccupied", None), shape)
    occupied = _safe_bool_map(getattr(tsdf_planner, "occupied", None), shape)
    observed = ~unexplored

    canvas[observed] = cfg.observed_color
    canvas[unoccupied] = cfg.free_color
    canvas[occupied] = cfg.occupied_color

    labels: List[str] = []
    if recent_decision_poses:
        for pose in recent_decision_poses:
            pt = _pose_to_voxel_xy(tsdf_planner, pose)
            if pt is not None:
                _draw_disk(canvas, pt, 2, cfg.decision_color)

    if frontier_instances:
        for inst in frontier_instances:
            pt = _pose_to_voxel_xy(tsdf_planner, inst.position)
            if pt is None:
                pt = _xy_from_any(inst.position)
            if pt is None:
                continue
            _draw_disk(canvas, pt, 3, cfg.frontier_color)
            label = f"F_{inst.frontier_id:03d}/{inst.spatial_branch_id or 'B?'}"
            labels.append(label)
            _draw_label(canvas, pt, label, cfg.frontier_color)

    agent_pt = _pose_to_voxel_xy(tsdf_planner, current_pose) if current_pose is not None else None
    if agent_pt is not None:
        _draw_disk(canvas, agent_pt, 4, cfg.agent_color)
        if current_yaw is not None:
            # TSDF map indices are row/col. Use a short arrow in image coords.
            direction = np.array([np.cos(current_yaw), np.sin(current_yaw)])
            end = np.round(agent_pt + direction * 8.0).astype(int)
            _draw_line(canvas, agent_pt, end, cfg.agent_color)

    canvas = np.flipud(canvas)
    if cfg.output_size:
        canvas = _resize_nearest(canvas, cfg.output_size)
    return BEVContext(image=canvas, labels=labels)


def _infer_shape(tsdf_planner) -> Tuple[int, int]:
    for attr in ("unoccupied", "occupied", "unexplored", "frontier_map"):
        arr = getattr(tsdf_planner, attr, None)
        if arr is not None:
            return tuple(np.asarray(arr).shape[:2])
    vol_dim = getattr(tsdf_planner, "_vol_dim", None)
    if vol_dim is not None:
        return tuple(vol_dim[:2])
    return (64, 64)


def _safe_bool_map(value, shape: Tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.zeros(shape, dtype=bool)
    arr = np.asarray(value)
    if arr.shape[:2] != shape:
        out = np.zeros(shape, dtype=bool)
        h = min(shape[0], arr.shape[0])
        w = min(shape[1], arr.shape[1])
        out[:h, :w] = arr[:h, :w].astype(bool)
        return out
    return arr.astype(bool)


def _pose_to_voxel_xy(tsdf_planner, pose) -> Optional[np.ndarray]:
    if pose is None:
        return None
    try:
        vox = tsdf_planner.habitat2voxel(np.asarray(pose, dtype=float))
        return np.asarray(vox[:2], dtype=int)
    except Exception:
        return _xy_from_any(pose)


def _xy_from_any(value) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return None
    if arr.ndim == 0 or arr.shape[0] < 2:
        return None
    if arr.shape[0] >= 3:
        return np.round(arr[[0, 2]]).astype(int)
    return np.round(arr[:2]).astype(int)


def _draw_disk(image: np.ndarray, xy: np.ndarray, radius: int, color) -> None:
    x, y = int(xy[0]), int(xy[1])
    h, w = image.shape[:2]
    for yy in range(max(0, y - radius), min(w, y + radius + 1)):
        for xx in range(max(0, x - radius), min(h, x + radius + 1)):
            if (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2:
                image[xx, yy] = color


def _draw_line(image: np.ndarray, start: np.ndarray, end: np.ndarray, color) -> None:
    x0, y0 = [int(v) for v in start[:2]]
    x1, y1 = [int(v) for v in end[:2]]
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    h, w = image.shape[:2]
    while True:
        if 0 <= x0 < h and 0 <= y0 < w:
            image[x0, y0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _draw_label(image: np.ndarray, xy: np.ndarray, label: str, color) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    # Input image uses row/col; PIL expects col/row.
    row, col = int(xy[0]), int(xy[1])
    draw.text((col + 4, row + 4), label, fill=tuple(int(c) for c in color))
    image[:, :] = np.asarray(pil)


def _resize_nearest(image: np.ndarray, output_size: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == output_size and w == output_size:
        return image
    ys = np.linspace(0, h - 1, output_size).astype(int)
    xs = np.linspace(0, w - 1, output_size).astype(int)
    return image[ys][:, xs]
