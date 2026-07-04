"""Candidate grounding and typed navigation intent controller.

This module owns the VISUAL_APPROACH -> TARGET_APPROACH -> VERIFY state
boundary. VLM code identifies a visible candidate; this controller decides
whether the next navigation target is a visual approach pose or a grounded
target viewpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import logging

import numpy as np
import torch

from src.memory_structures import (
    EventType,
    FB_AVU_FAIL,
    FB_AVU_VISUAL_ONLY,
    Grounding2D,
    NavigationMode,
    NavStatus,
    NavTargetKind,
    S_GROUNDED_3D,
    S_REJECTED,
    SubtaskWorkingMemory,
    TargetCandidate,
    TargetViewpointIntent,
    TypedEvent,
    VisualApproachIntent,
    get_aliases,
)


GENERIC_CLASSES = ["object", "item", "thing", "furniture", "appliance"]
DEFAULT_CLIP_RERANK_THRESH = 0.20
DEFAULT_CLIP_MARGIN_THRESH = 0.0
DEFAULT_CLIP_DEPTH_MIN_RATIO = 0.0


@dataclass
class CandidateControllerResult:
    intent: Optional[object] = None
    candidate: Optional[TargetCandidate] = None
    events: List[TypedEvent] = field(default_factory=list)
    reason: str = ""

    @property
    def navigation_goal(self):
        if isinstance(self.intent, VisualApproachIntent):
            return self.intent.approach_xyz
        if isinstance(self.intent, TargetViewpointIntent):
            return self.intent.target_xyz
        return None


class CandidateController:
    def __init__(
        self,
        cfg,
        clip_rerank_thresh: float = DEFAULT_CLIP_RERANK_THRESH,
        clip_margin_thresh: Optional[float] = None,
        clip_depth_min_ratio: Optional[float] = None,
    ) -> None:
        self.cfg = cfg
        self.clip_rerank_thresh = clip_rerank_thresh
        self.clip_margin_thresh = (
            float(clip_margin_thresh)
            if clip_margin_thresh is not None
            else float(
                getattr(cfg, "clip_rerank_margin_threshold", DEFAULT_CLIP_MARGIN_THRESH)
            )
        )
        self.clip_depth_min_ratio = (
            float(clip_depth_min_ratio)
            if clip_depth_min_ratio is not None
            else float(
                getattr(cfg, "clip_depth_min_ratio", DEFAULT_CLIP_DEPTH_MIN_RATIO)
            )
        )

    def handle_visible_candidate(
        self,
        scene,
        tsdf_planner,
        working_memory: Optional[SubtaskWorkingMemory],
        img_path: str,
        target_phrase: str,
        pts,
        step_index: int,
    ) -> CandidateControllerResult:
        if not target_phrase:
            return CandidateControllerResult(reason="missing target phrase")
        if img_path not in scene.all_observations:
            return CandidateControllerResult(reason=f"image {img_path} unavailable")

        target_image = scene.all_observations[img_path]
        cam_pose = scene.all_cam_poses[img_path]
        view_yaw = _cam_pose_to_yaw(cam_pose)
        aliases = get_aliases(target_phrase)
        candidate = self._get_or_create_candidate(
            working_memory, img_path, target_phrase, step_index, cam_pose, view_yaw
        )
        events = [
            self._event(
                EventType.CANDIDATE_VISIBLE,
                step_index,
                candidate.candidate_id,
                {"image_path": img_path, "target_phrase": target_phrase},
            )
        ]

        logging.info(
            f"[CandidateController] candidate={candidate.candidate_id} "
            f"phrase='{target_phrase}' aliases={aliases}"
        )

        try:
            ground2d, l3_boxes, l3_best_idx = self._ground_2d(
                scene,
                target_image,
                target_phrase,
                aliases,
                candidate,
                scene.all_depths.get(img_path) if hasattr(scene, "all_depths") else None,
            )
            if ground2d is None:
                result = self._build_visual_approach_result(
                    scene=scene,
                    tsdf_planner=tsdf_planner,
                    working_memory=working_memory,
                    candidate=candidate,
                    img_path=img_path,
                    target_phrase=target_phrase,
                    cam_pose=cam_pose,
                    view_yaw=view_yaw,
                    l3_boxes=l3_boxes,
                    l3_best_idx=l3_best_idx,
                    step_index=step_index,
                )
                result.events = events + result.events
                self._store_events(working_memory, result.events)
                return result

            result = self._build_target_viewpoint_result(
                scene=scene,
                tsdf_planner=tsdf_planner,
                working_memory=working_memory,
                candidate=candidate,
                img_path=img_path,
                target_phrase=target_phrase,
                target_image=target_image,
                cam_pose=cam_pose,
                ground2d=ground2d,
                pts=pts,
                step_index=step_index,
            )
            result.events = events + result.events
            self._store_events(working_memory, result.events)
            return result
        except Exception as exc:
            logging.info(f"[CandidateController] grounding failed: {exc}")
            candidate.record_attempt(4, False, f"exception: {exc}")
            result = self._reject_grounding_result(
                working_memory,
                candidate,
                img_path,
                step_index,
                f"AVU exception: {exc}",
                feedback_type=FB_AVU_FAIL,
            )
            result.events = events + result.events
            self._store_events(working_memory, result.events)
            return result

    def _ground_2d(
        self,
        scene,
        target_image,
        target_phrase: str,
        aliases: List[str],
        candidate: TargetCandidate,
        depth_array=None,
    ) -> Tuple[Optional[Grounding2D], Optional[np.ndarray], Optional[int]]:
        l3_boxes = None
        l3_best_idx = None
        result = _yolo_detect(
            scene, [target_phrase], target_image, self.cfg.AVU_conf_threshold
        )
        if result is not None:
            logging.info(f"[CandidateController] L1 YOLO('{target_phrase}')")
            candidate.record_attempt(1, True, "YOLO target phrase")
            return _grounding_from_result(result, target_phrase, "yolo_target"), None, None

        candidate.record_attempt(1, False, "no YOLO box for target phrase")
        for alias in aliases:
            if alias.lower() == target_phrase.lower():
                continue
            result = _yolo_detect(
                scene, [alias], target_image, self.cfg.AVU_conf_threshold
            )
            if result is not None:
                logging.info(f"[CandidateController] L2 YOLO alias '{alias}'")
                candidate.record_attempt(2, True, f"alias '{alias}'")
                return _grounding_from_result(result, alias, "yolo_alias"), None, None

        candidate.record_attempt(2, False, "no alias detected")
        result = _yolo_detect(
            scene, GENERIC_CLASSES, target_image, self.cfg.AVU_conf_threshold
        )
        if result is None:
            candidate.record_attempt(3, False, "no generic class detection")
            return None, None, None

        l3_boxes = result.boxes.xyxy.cpu().numpy()
        best_idx, best_score, best_margin = _clip_rerank_bbox(
            scene, target_image, l3_boxes, target_phrase, aliases
        )
        l3_best_idx = best_idx
        best_depth_ratio = _bbox_valid_depth_ratio(depth_array, l3_boxes[best_idx])
        if (
            best_score < self.clip_rerank_thresh
            or best_margin < self.clip_margin_thresh
            or best_depth_ratio < self.clip_depth_min_ratio
        ):
            logging.info(
                f"[CandidateController] L3 CLIP rejected "
                f"(score={best_score:.3f}, margin={best_margin:.3f}, "
                f"depth={best_depth_ratio:.2f})"
            )
            candidate.record_attempt(
                3,
                False,
                (
                    f"CLIP score {best_score:.3f}, margin {best_margin:.3f}, "
                    f"depth {best_depth_ratio:.2f}"
                ),
            )
            return None, l3_boxes, l3_best_idx

        logging.info(
            f"[CandidateController] L3 class-agnostic + CLIP accepted "
            f"(score={best_score:.3f}, margin={best_margin:.3f}, "
            f"depth={best_depth_ratio:.2f})"
        )
        candidate.record_attempt(
            3,
            True,
            (
                f"CLIP score {best_score:.3f}, margin {best_margin:.3f}, "
                f"depth {best_depth_ratio:.2f}"
            ),
        )
        return (
            Grounding2D(
                source="class_agnostic_clip",
                phrase=target_phrase,
                bbox_xyxy=l3_boxes[best_idx].astype(int),
                raw_score=float(best_score),
                rank_score=float(best_margin),
                conf=float(result.boxes.conf[best_idx].cpu().numpy()),
            ),
            l3_boxes,
            l3_best_idx,
        )

    def _build_visual_approach_result(
        self,
        scene,
        tsdf_planner,
        working_memory,
        candidate: TargetCandidate,
        img_path: str,
        target_phrase: str,
        cam_pose,
        view_yaw: float,
        l3_boxes,
        l3_best_idx,
        step_index: int,
    ) -> CandidateControllerResult:
        if l3_boxes is not None and l3_best_idx is not None:
            try:
                from src.utils import build_visual_approach_pose

                approach = build_visual_approach_pose(
                    l3_boxes[l3_best_idx],
                    scene.all_depths[img_path],
                    scene.intrinsics[:3, :3],
                    cam_pose,
                    tsdf_planner,
                )
                if approach is not None:
                    candidate.nav_target_kind = NavTargetKind.VISUAL_APPROACH_POSE
                    candidate.nav_goal_xyz = approach.xyz
                    candidate.nav_goal_yaw = approach.yaw
                    candidate.nav_status = NavStatus.PLANNED
                    if working_memory is not None:
                        working_memory.set_last_nav_candidate(candidate.candidate_id)
                    intent = VisualApproachIntent(
                        mode=NavigationMode.VISUAL_APPROACH,
                        candidate_id=candidate.candidate_id,
                        image_path=img_path,
                        target_phrase=target_phrase,
                        approach_xyz=approach.xyz,
                        approach_yaw=approach.yaw,
                        reason_code="VISUAL_RAY_APPROACH",
                        reason=(
                            "target likely visible but not grounded; "
                            "approach along visual ray"
                        ),
                    )
                    return CandidateControllerResult(
                        intent=intent,
                        candidate=candidate,
                        events=[
                            self._event(
                                EventType.VISUAL_APPROACH_READY,
                                step_index,
                                candidate.candidate_id,
                                {"image_path": img_path},
                            )
                        ],
                        reason=intent.reason,
                    )
            except Exception as exc:
                logging.info(f"[CandidateController] visual approach exception: {exc}")
        candidate.record_attempt(4, False, "visual approach unavailable")
        return self._reject_grounding_result(
            working_memory,
            candidate,
            img_path,
            step_index,
            f"VLM saw '{target_phrase}' but grounding failed",
            feedback_type=FB_AVU_VISUAL_ONLY,
        )

    def _reject_grounding_result(
        self,
        working_memory,
        candidate: TargetCandidate,
        img_path: str,
        step_index: int,
        reason: str,
        feedback_type: str = FB_AVU_FAIL,
    ) -> CandidateControllerResult:
        candidate.status = S_REJECTED
        candidate.pinned = False
        candidate.last_failure_reason = reason
        candidate.nav_target_kind = None
        candidate.nav_goal_xyz = None
        candidate.nav_goal_yaw = None
        candidate.nav_status = NavStatus.FAILED
        if working_memory is not None:
            working_memory.reject_candidate(candidate.candidate_id, reason)
            working_memory.add_feedback(
                step=step_index,
                type_=feedback_type,
                reason=reason,
                suggested_fix=working_memory.suggest_fix_for(
                    feedback_type, reason
                ),
                target_candidate_id=candidate.candidate_id,
            )
        return CandidateControllerResult(
            candidate=candidate,
            events=[
                self._event(
                    EventType.GROUNDING_FAILED,
                    step_index,
                    candidate.candidate_id,
                    {"image_path": img_path, "reason": reason},
                ),
                self._event(
                    EventType.CANDIDATE_REJECTED,
                    step_index,
                    candidate.candidate_id,
                    {"image_path": img_path, "reason": reason},
                ),
            ],
            reason=reason,
        )

    def _build_target_viewpoint_result(
        self,
        scene,
        tsdf_planner,
        working_memory,
        candidate: TargetCandidate,
        img_path: str,
        target_phrase: str,
        target_image,
        cam_pose,
        ground2d: Grounding2D,
        pts,
        step_index: int,
    ) -> CandidateControllerResult:
        (
            get_bounding_box,
            init_process_pcd,
            detections_to_obj_pcd_and_bbox,
        ) = _conceptgraph_slam_utils()
        xyxy_tensor = torch.as_tensor(
            ground2d.bbox_xyxy,
            dtype=torch.float32,
            device=scene.device,
        ).reshape(1, 4)
        sam_out = scene.sam_predictor.predict(
            target_image, bboxes=xyxy_tensor, verbose=False
        )
        masks_np = sam_out[0].masks.data.cpu().numpy()
        obj_pcds_and_bboxes = detections_to_obj_pcd_and_bbox(
            depth_array=scene.all_depths[img_path],
            masks=masks_np,
            cam_K=scene.intrinsics[:3, :3],
            image_rgb=target_image,
            trans_pose=cam_pose,
            min_points_threshold=scene.cfg_cg.min_points_threshold,
            spatial_sim_type=scene.cfg_cg.spatial_sim_type,
            obj_pcd_max_points=scene.cfg_cg.obj_pcd_max_points,
            device=scene.device,
        )
        for obj in obj_pcds_and_bboxes:
            if not obj:
                continue
            obj["pcd"] = init_process_pcd(
                pcd=obj["pcd"],
                downsample_voxel_size=scene.cfg_cg["downsample_voxel_size"],
                dbscan_remove_noise=scene.cfg_cg["dbscan_remove_noise"],
                dbscan_eps=scene.cfg_cg["dbscan_eps"],
                dbscan_min_points=scene.cfg_cg["dbscan_min_points"],
            )
            obj["bbox"] = get_bounding_box(
                spatial_sim_type=scene.cfg_cg["spatial_sim_type"],
                pcd=obj["pcd"],
            )
        valid_objs = [obj for obj in obj_pcds_and_bboxes if obj is not None]
        if not valid_objs:
            candidate.record_attempt(3, False, "SAM/pcd invalid")
            return self._reject_grounding_result(
                working_memory,
                candidate,
                img_path,
                step_index,
                "SAM or point cloud invalid after 2D grounding",
                feedback_type=FB_AVU_FAIL,
            )

        target_obj = valid_objs[0]
        if working_memory is not None:
            working_memory.grounded_candidate(candidate.candidate_id)
        else:
            candidate.status = S_GROUNDED_3D

        scene_points = []
        for obj in scene.objects.values():
            try:
                scene_points.append(np.asarray(obj["pcd"].points))
            except Exception:
                continue
        target_points = np.asarray(target_obj["pcd"].points)
        if scene_points:
            all_scene_points = np.concatenate(scene_points, axis=0)
        else:
            all_scene_points = target_points
        from src.utils import Visibility_based_Viewpoint_Decision

        obj_pos = Visibility_based_Viewpoint_Decision(
            target_points,
            all_scene_points,
            pts,
            tsdf_planner,
            self.cfg.dicision_radius,
        )
        if obj_pos is None:
            obj_pos = _select_navigation_corner(target_obj["bbox"], pts)
        candidate.nav_target_kind = NavTargetKind.VIEWPOINT_POSE
        candidate.nav_goal_xyz = obj_pos
        candidate.nav_status = NavStatus.PLANNED
        if working_memory is not None:
            working_memory.set_last_nav_candidate(candidate.candidate_id)

        intent = TargetViewpointIntent(
            mode=NavigationMode.TARGET_APPROACH,
            candidate_id=candidate.candidate_id,
            image_path=img_path,
            target_phrase=target_phrase,
            target_xyz=obj_pos,
            target_yaw=None,
            reason_code="GROUNDED_3D_VVD",
            reason="candidate grounded in 3D; navigate to target viewpoint",
        )
        return CandidateControllerResult(
            intent=intent,
            candidate=candidate,
            events=[
                self._event(
                    EventType.CANDIDATE_GROUNDED_3D,
                    step_index,
                    candidate.candidate_id,
                    {"image_path": img_path, "source": ground2d.source},
                ),
                self._event(
                    EventType.TARGET_VIEWPOINT_READY,
                    step_index,
                    candidate.candidate_id,
                    {"target_xyz": np.asarray(obj_pos).tolist()},
                ),
            ],
            reason=intent.reason,
        )

    def _get_or_create_candidate(
        self,
        working_memory,
        img_path,
        target_phrase,
        step_index,
        cam_pose,
        view_yaw,
    ) -> TargetCandidate:
        if working_memory is not None:
            return working_memory.get_or_create_candidate(
                image_path=img_path,
                target_phrase=target_phrase,
                source_step=step_index,
                camera_pose=cam_pose,
                view_yaw=view_yaw,
            )
        return TargetCandidate(
            candidate_id="C_TMP",
            subtask_id="",
            image_path=img_path,
            source_step=step_index,
            camera_pose=cam_pose,
            view_yaw=view_yaw,
            target_phrase=target_phrase,
            aliases=get_aliases(target_phrase),
        )

    @staticmethod
    def _event(
        type_: EventType,
        step: int,
        entity_id: Optional[str],
        payload: dict,
    ) -> TypedEvent:
        return TypedEvent(
            event_id=f"{type_.value}:{entity_id or 'candidate'}:{step}",
            type=type_,
            step=step,
            entity_id=entity_id,
            payload=payload,
        )

    @staticmethod
    def _store_events(
        working_memory: Optional[SubtaskWorkingMemory],
        events: List[TypedEvent],
    ) -> None:
        if working_memory is None:
            return
        for event in events:
            working_memory.add_typed_event(event)


def _cam_pose_to_yaw(cam_pose: np.ndarray) -> float:
    try:
        rot = np.asarray(cam_pose)[:3, :3]
        fwd = -rot[:3, 2]
        return float(np.arctan2(fwd[0], fwd[2]))
    except Exception:
        return 0.0


def _yolo_detect(scene, classes: List[str], image_rgb, conf: float):
    scene.detection_model.set_classes(classes)
    try:
        results = scene.detection_model.predict(image_rgb, conf=conf, verbose=False)
    finally:
        scene.detection_model.set_classes(scene.obj_classes.get_classes_arr())
    if len(results) == 0 or len(results[0].boxes) == 0:
        return None
    return results[0]


def _grounding_from_result(result, phrase: str, source: str) -> Grounding2D:
    conf = result.boxes.conf.cpu().numpy()
    best_idx = int(conf.argmax())
    return Grounding2D(
        source=source,
        phrase=phrase,
        bbox_xyxy=result.boxes.xyxy[best_idx].cpu().numpy().astype(int),
        conf=float(conf[best_idx]),
        raw_score=float(conf[best_idx]),
    )


def _conceptgraph_slam_utils():
    from src.conceptgraph.slam.utils import (
        get_bounding_box,
        init_process_pcd,
        detections_to_obj_pcd_and_bbox,
    )

    return get_bounding_box, init_process_pcd, detections_to_obj_pcd_and_bbox


def _clip_rerank_bbox(
    scene,
    image_rgb,
    xyxy_np: np.ndarray,
    target_phrase: str,
    aliases: List[str],
) -> Tuple[int, float, float]:
    try:
        from src.conceptgraph.utils.model_utils import clip_recognition
    except Exception:
        clip_recognition = None
    if clip_recognition is None or scene.clip_model is None:
        return 0, 0.0, 0.0
    prompts = [target_phrase] + [
        alias for alias in aliases if alias.lower() != target_phrase.lower()
    ]
    best_idx, best_score = 0, -1.0
    box_scores: List[float] = []
    for idx in range(len(xyxy_np)):
        x1, y1, x2, y2 = xyxy_np[idx].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_rgb.shape[1], x2), min(image_rgb.shape[0], y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            box_scores.append(0.0)
            continue
        crop = image_rgb[y1:y2, x1:x2]
        best_for_box = 0.0
        for prompt in prompts:
            try:
                probs = clip_recognition(
                    scene.clip_model,
                    scene.clip_tokenizer,
                    scene.clip_preprocess,
                    crop,
                    prompt,
                )
                score = float(probs[0]) if hasattr(probs, "__len__") else float(probs)
                best_for_box = max(best_for_box, score)
            except Exception:
                continue
        if best_for_box > best_score:
            best_score = best_for_box
            best_idx = idx
        box_scores.append(best_for_box)
    sorted_scores = sorted(box_scores, reverse=True)
    if len(sorted_scores) >= 2:
        margin = sorted_scores[0] - sorted_scores[1]
    elif sorted_scores:
        margin = sorted_scores[0]
    else:
        margin = 0.0
    return best_idx, best_score, float(margin)


def _bbox_valid_depth_ratio(depth_array, bbox_xyxy) -> float:
    if depth_array is None:
        return 1.0
    try:
        depth = np.asarray(depth_array)
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy[:4]]
    except Exception:
        return 0.0
    h, w = depth.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = depth[y1:y2, x1:x2]
    return float(np.count_nonzero(crop > 0) / max(crop.size, 1))


def _select_navigation_corner(aabb, robot_position) -> np.ndarray:
    try:
        corners = np.asarray(aabb.get_box_points())
    except Exception:
        try:
            center = np.asarray(aabb.get_center())
            return center
        except Exception:
            return np.asarray(robot_position)
    robot = np.asarray(robot_position)
    if corners.size == 0:
        return robot
    distances = np.linalg.norm(corners[:, [0, 2]] - robot[[0, 2]], axis=1)
    return corners[int(np.argmin(distances))]
