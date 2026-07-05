import numpy as np
from PIL import Image
import habitat_sim
from habitat_sim.utils.common import quat_to_angle_axis, quat_from_coeffs
import quaternion
from scipy.spatial import KDTree
import logging
import math
import open3d as o3d


def build_visual_approach_pose(
    bbox_xyxy,
    depth_array,
    intrinsics,
    cam_pose,
    tsdf_planner,
    desired_standoff: float = 1.2,
    mask=None,
):
    """Visual Candidate Approach v0 (report §Visual Candidate Approach).

    Build a navigation pose by advancing along the bbox-center depth ray,
    then snapping to a navigable cell. Returns an ApproachPose or None
    (caller falls back to evidence-pose). None when:
      - LOS check from cam to snapped point blocked by wall
    depth invalid (valid_ratio < 0.25) no longer returns None; falls back
    to fixed-step advance along bearing (report requirement).
    """
    from src.memory_structures import ApproachPose

    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy[:4]]
    H, W = depth_array.shape[:2]
    x1, x2 = max(0, x1), min(W, x2)
    y1, y2 = max(0, y1), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    u = (x1 + x2) / 2.0
    v = (y1 + y2) / 2.0

    if mask is not None:
        depth_vals = depth_array[mask]
        valid = depth_vals[depth_vals > 0]
        valid_ratio = len(valid) / max(int(mask.sum()), 1)
    else:
        depth_crop = depth_array[y1:y2, x1:x2]
        valid = depth_crop[depth_crop > 0]
        area = max(depth_crop.size, 1)
        valid_ratio = len(valid) / area

    if valid_ratio < 0.25:
        # depth无效：沿 bearing 固定步长逼近（报告要求不放弃）
        z = None
        advance = 0.8
    else:
        z = float(np.quantile(valid, 0.30))
        advance = float(np.clip(z - desired_standoff, 0.5, 1.5))

    # pixel ray -> camera frame -> world frame (horizontal plane)
    K = np.asarray(intrinsics, dtype=float)
    ray_cam = np.linalg.inv(K) @ np.array([u, v, 1.0])
    ray_cam = ray_cam / max(np.linalg.norm(ray_cam), 1e-6)
    cam_R = np.asarray(cam_pose)[:3, :3]
    cam_t = np.asarray(cam_pose)[:3, 3]
    ray_world = cam_R @ ray_cam
    ray_world[1] = 0.0  # zero out vertical, keep bearing
    norm = np.linalg.norm(ray_world)
    if norm < 1e-6:
        return None
    ray_world = ray_world / norm

    raw_xyz = cam_t + advance * ray_world

    snapped = tsdf_planner.get_near_true_point(np.array([raw_xyz]))
    if snapped is None or len(snapped) == 0:
        return None
    xyz = np.asarray(snapped[0], dtype=float)

    # LOS check: if wall blocks cam -> snapped point, reject
    if hasattr(tsdf_planner, "is_line_of_sight_clear"):
        if not tsdf_planner.is_line_of_sight_clear(cam_t, xyz):
            logging.info(
                f"[VisualApproach] LOS blocked cam->{xyz}, fallback to evidence-pose"
            )
            return None

    yaw = float(math.atan2(ray_world[2], ray_world[0]))
    _z_str = "none" if z is None else f"{z:.2f}"
    logging.info(
        f"[VisualApproach] approach pose={xyz} yaw={yaw:.2f} "
        f"valid_ratio={valid_ratio:.2f} z={_z_str} advance={advance:.2f}"
    )
    return ApproachPose(
        xyz=xyz,
        yaw=yaw,
        source="visual_ray",
        valid_depth_ratio=float(valid_ratio),
    )


def generate_candidate_viewpoints(bbox_center, radius, pts, num_points=20):
    """Generate candidate viewpoints around the target bounding box center on a circle."""
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    viewpoints = []
    for angle in angles:
        x = bbox_center[0] + radius * np.cos(angle)### x0 + r*cos(theta)
        y = bbox_center[2] + radius * np.sin(angle)### y0 + r*sin(theta)
        z = pts[1]  # Z remains constant
        viewpoints.append(np.array([x, z, y]))
    return np.array(viewpoints)


def _unit_xz(vec):
    arr = np.asarray(vec, dtype=float).copy()
    arr[1] = 0.0
    norm = np.linalg.norm(arr[[0, 2]])
    if norm < 1e-6:
        return None
    return arr / norm


def _rotate_xz(unit_vec, angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            c * unit_vec[0] - s * unit_vec[2],
            0.0,
            s * unit_vec[0] + c * unit_vec[2],
        ],
        dtype=float,
    )


def _is_navigable_habitat(tsdf_planner, point):
    if getattr(tsdf_planner, "unoccupied", None) is None:
        return True
    pos = tsdf_planner.habitat2voxel(point)[:2]
    x, y = int(pos[0]), int(pos[1])
    H, W = tsdf_planner.unoccupied.shape
    if x < 0 or x >= H or y < 0 or y >= W:
        return False
    if not bool(tsdf_planner.unoccupied[x, y]):
        return False
    island = getattr(tsdf_planner, "island", None)
    if island is not None and island.shape == tsdf_planner.unoccupied.shape:
        return bool(island[x, y])
    return True


def _snap_local_navigable(tsdf_planner, raw_point, max_snap_dist=0.45):
    raw_point = np.asarray(raw_point, dtype=float)
    if _is_navigable_habitat(tsdf_planner, raw_point):
        return raw_point, 0.0
    if not hasattr(tsdf_planner, "get_near_true_point"):
        return None, None
    snapped = tsdf_planner.get_near_true_point(np.array([raw_point]))
    if snapped is None or len(snapped) == 0:
        return None, None
    snapped = np.asarray(snapped[0], dtype=float)
    snap_dist = float(np.linalg.norm(snapped[[0, 2]] - raw_point[[0, 2]]))
    if snap_dist > max_snap_dist:
        return None, None
    if not _is_navigable_habitat(tsdf_planner, snapped):
        return None, None
    return snapped, snap_dist


def _target_surface_point(target_points, bbox_center, viewpoint, min_offset=0.12):
    """Approximate the visible target surface instead of aiming LOS at the
    bbox center, which is often inside an occupied object voxel."""
    direction = _unit_xz(np.asarray(viewpoint, dtype=float) - bbox_center)
    if direction is None:
        return bbox_center
    offsets = target_points - bbox_center
    projections = offsets[:, 0] * direction[0] + offsets[:, 2] * direction[2]
    if projections.size == 0:
        offset = min_offset
    else:
        offset = max(float(np.percentile(projections, 90)), min_offset)
    surface = np.asarray(bbox_center, dtype=float).copy()
    surface[0] += direction[0] * offset
    surface[2] += direction[2] * offset
    return surface


def _is_los_clear_to_target_surface(tsdf_planner, viewpoint, target_points, bbox_center):
    if not hasattr(tsdf_planner, "is_line_of_sight_clear"):
        return True
    surface = _target_surface_point(target_points, bbox_center, viewpoint)
    return bool(tsdf_planner.is_line_of_sight_clear(viewpoint, surface))


def _generate_evidence_side_viewpoints(
    bbox_center,
    pts,
    radius,
    evidence_position=None,
):
    center = np.asarray(bbox_center, dtype=float)
    floor_y = float(pts[1])
    if evidence_position is not None:
        side_dir = _unit_xz(np.asarray(evidence_position, dtype=float) - center)
    else:
        side_dir = _unit_xz(np.asarray(pts, dtype=float) - center)
    if side_dir is None:
        return None, generate_candidate_viewpoints(center, radius, pts)

    radii = sorted(set(float(r) for r in [
        max(0.55, radius * 0.75),
        max(0.70, radius),
        max(0.90, radius * 1.25),
        max(1.10, radius * 1.55),
    ]))
    angles_deg = [0, -15, 15, -30, 30, -45, 45, -60, 60]
    viewpoints = []

    if evidence_position is not None:
        evidence_ground = np.asarray(evidence_position, dtype=float).copy()
        evidence_ground[1] = floor_y
        viewpoints.append(evidence_ground)

    for r in radii:
        for angle_deg in angles_deg:
            direction = _rotate_xz(side_dir, math.radians(angle_deg))
            viewpoints.append(center + direction * r)
            viewpoints[-1][1] = floor_y

    return side_dir, np.asarray(viewpoints, dtype=float)


def is_point_visible(viewpoint, target_point, scene_points_tree, threshold=0.05):
    """Check if a target point is visible from a viewpoint considering scene occlusion."""
    direction = target_point - viewpoint
    view_distance = np.linalg.norm(direction)
    if view_distance < 1e-6:
        return True
    direction /= view_distance

    num_samples = min(1000, int(view_distance / threshold) + 1)
    sample_points = np.array([
        viewpoint + t * direction 
        for t in np.linspace(3 * threshold, view_distance - 3 * threshold, num_samples)
    ])
    
    # Batch query nearest neighbors for all sampled points
    distances, indices = scene_points_tree.query(sample_points, k=1)
    # Check if any sampled point is too close to scene points (occlusion)
    return not np.any(distances < threshold)

def compute_visibility(viewpoint, target_points, scene_points_tree):
    """Compute visibility of target points from a given viewpoint considering scene occlusion."""
    visible_count = 0

    for target_point in target_points:
        visible = is_point_visible(viewpoint, target_point, scene_points_tree)
        if visible:
            visible_count += 1

    return visible_count / target_points.shape[0]


def Visibility_based_Viewpoint_Decision(
    target_points,
    scene_points,
    pts,
    tsdf_planner,
    radius_factor,
    evidence_position=None,
):
    target_points = np.asarray(target_points, dtype=float)
    scene_points = np.asarray(scene_points, dtype=float)
    if target_points.shape[0] == 0:
        return None
    if scene_points.shape[0] == 0:
        scene_points = target_points
    if target_points.shape[0] > 1000:
        sample_idx = np.linspace(
            0,
            target_points.shape[0] - 1,
            num=1000,
            dtype=int,
        )
        target_points = target_points[sample_idx]
    scene_points_tree = KDTree(scene_points)
    bbox_center = target_points.mean(axis=0)
    best_visibility = 0
    best_viewpoint = None
    side_dir, candidate_viewpoints = _generate_evidence_side_viewpoints(
        bbox_center,
        pts,
        radius_factor,
        evidence_position=evidence_position,
    )
    logging.info(
        f"[VVD] bbox_center={bbox_center}, n_candidates={len(candidate_viewpoints)}, "
        f"evidence_side={None if side_dir is None else side_dir[[0, 2]]}"
    )

    scored_candidates = []
    for raw_vp in candidate_viewpoints:
        snapped_vp, snap_dist = _snap_local_navigable(tsdf_planner, raw_vp)
        if snapped_vp is None:
            continue
        if side_dir is not None:
            cand_dir = _unit_xz(snapped_vp - bbox_center)
            if cand_dir is None:
                continue
            # Keep viewpoints on the same visible side as the evidence camera.
            if float(np.dot(cand_dir[[0, 2]], side_dir[[0, 2]])) < 0.25:
                continue
        if not _is_los_clear_to_target_surface(
            tsdf_planner,
            snapped_vp,
            target_points,
            bbox_center,
        ):
            continue
        vp_for_visibility = snapped_vp.copy()
        vp_for_visibility[1] += 1.5
        visibility_score = compute_visibility(
            vp_for_visibility,
            target_points,
            scene_points_tree,
        )
        dist_to_target = float(
            np.linalg.norm(snapped_vp[[0, 2]] - bbox_center[[0, 2]])
        )
        desired_dist = max(float(radius_factor), 0.75)
        distance_penalty = abs(dist_to_target - desired_dist)
        travel_penalty = 0.05 * float(
            np.linalg.norm(snapped_vp[[0, 2]] - np.asarray(pts)[[0, 2]])
        )
        snap_penalty = 0.5 * float(snap_dist or 0.0)
        side_bonus = 0.0
        if side_dir is not None:
            cand_dir = _unit_xz(snapped_vp - bbox_center)
            side_bonus = 0.15 * float(np.dot(cand_dir[[0, 2]], side_dir[[0, 2]]))
        score = visibility_score + side_bonus - 0.25 * distance_penalty - travel_penalty - snap_penalty
        scored_candidates.append((score, visibility_score, snapped_vp))

    logging.info(
        f"[VVD] stable candidates after nav/side/LOS filters: {len(scored_candidates)}"
    )
    if scored_candidates:
        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        _, best_visibility, best_viewpoint = scored_candidates[0]

    if best_viewpoint is None and evidence_position is not None:
        # Last safe fallback: return the evidence camera floor position if it
        # is navigable and still has LOS to the target surface. This avoids
        # snapping to the opposite side of a wall.
        evidence_ground = np.asarray(evidence_position, dtype=float).copy()
        evidence_ground[1] = float(pts[1])
        snapped_vp, _ = _snap_local_navigable(tsdf_planner, evidence_ground)
        if snapped_vp is not None and _is_los_clear_to_target_surface(
            tsdf_planner,
            snapped_vp,
            target_points,
            bbox_center,
        ):
            best_viewpoint = snapped_vp
            best_visibility = 0.0
            logging.info("[VVD] fallback to evidence camera pose")

    logging.info(f"[VVD] best_viewpoint={best_viewpoint}, visibility={best_visibility:.3f}")
    return best_viewpoint

def resize_image(image, target_h, target_w):
    # image: np.array, h, w, c
    image = Image.fromarray(image)
    image = image.resize((target_w, target_h))
    return np.array(image)


def find_center_in_room(centers, confidences, xyxy, class_ids, rooms):
    if len(confidences) > 0:
        sorted_indices = np.argsort(confidences)[::-1]
        class_ids = class_ids[sorted_indices]
        confidences = confidences[sorted_indices]
        xyxy = xyxy[sorted_indices]
    room_label = []
    room_conf = []
    for center in centers:   
        find_room = False
        x, y = center
        for idx in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[idx]
            if x1 <= x <= x2 and y1 <= y <= y2:
                room_label.append(rooms[class_ids[idx]])
                room_conf.append(confidences[idx])
                find_room = True
                break
        if not find_room:
            room_label.append('unknown')
    return room_label, room_conf

def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, "RGBA image has 4 channels."

    rgb = np.zeros((row, col, 3), dtype="float32")
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype="float32") / 255.0

    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype="uint8")


def get_pts_angle_aeqa(init_pts, init_quat):
    pts = np.asarray(init_pts)

    init_quat = quaternion.quaternion(*init_quat)
    angle, axis = quat_to_angle_axis(init_quat)
    angle = angle * axis[1] / np.abs(axis[1])

    return pts, angle


def get_pts_angle_goatbench(init_pos, init_rot):
    pts = np.asarray(init_pos)

    init_quat = quat_from_coeffs(init_rot)
    angle, axis = quat_to_angle_axis(init_quat)
    angle = angle * axis[1] / np.abs(axis[1])

    return pts, angle

def calc_agent_subtask_distance(curr_pts, viewpoints, pathfinder):
    # calculate the distance to the nearest view point
    path = habitat_sim.MultiGoalShortestPath()
    path.requested_start = curr_pts
    path.requested_ends = viewpoints
    # np.save(f'/hsun/0625/3D-Mem/vis/pos_{curr_pts[1]}.npy', np.array(viewpoints))
    found_path = pathfinder.find_path(path)
    if found_path:
        distance = path.geodesic_distance
    else:
        distance = 10.0
    return distance
