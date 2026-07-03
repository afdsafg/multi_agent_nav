import logging
from typing import Tuple, Optional, Union, List, Dict, Any
import random
import numpy as np

from src.explore_utils import (
    task_check,
    explore_two_step,
    Key_Subgraph_Selection,
    encode_tensor2base64,
)
from src.explore_multi_agent import explore_multi_agent
from src.utils import Visibility_based_Viewpoint_Decision
from src.tsdf_planner import TSDFPlanner, Frontier
from src.multimodal_3d_scene_graph import Scene
from src.conceptgraph.slam.utils import (
    get_bounding_box,
    init_process_pcd,
    detections_to_obj_pcd_and_bbox,
)
from src.memory_structures import (
    TargetCandidate, SubtaskWorkingMemory,
    S_GROUNDED_3D, S_VISUAL_ONLY, S_NEED_CLOSER_VIEW, S_REJECTED,
    FB_AVU_FAIL, FB_AVU_VISUAL_ONLY,
    get_aliases,
)

# Phase C: generic class proposals for class-agnostic grounding (Level 3).
_GENERIC_CLASSES = ["object", "item", "thing", "furniture", "appliance"]
# CLIP rerank threshold for accepting a class-agnostic proposal.
_CLIP_RERANK_THRESH = 0.20


def _cam_pose_to_yaw(cam_pose: np.ndarray) -> float:
    """Extract camera yaw (radians) from a 4x4 world->cam / cam->world pose.

    Habitat convention: camera forward is -z. We use the rotation submatrix
    to recover the heading. Returns 0.0 if pose is malformed.
    """
    try:
        R = cam_pose[:3, :3]
        # forward vector in world = R @ [0,0,-1] = -R[:,2]
        fwd = -R[:3, 2]
        return float(np.arctan2(fwd[0], fwd[2]))
    except Exception:
        return 0.0


def _yolo_detect(scene: Scene, classes: List[str], image_rgb, conf: float):
    """Run YOLOWorld with a temporary class list, then restore scene classes."""
    scene.detection_model.set_classes(classes)
    try:
        results = scene.detection_model.predict(image_rgb, conf=conf, verbose=False)
    finally:
        scene.detection_model.set_classes(scene.obj_classes.get_classes_arr())
    if len(results) == 0 or len(results[0].boxes) == 0:
        return None
    return results[0]


def _clip_rerank_bbox(
    scene: Scene, image_rgb, xyxy_np: np.ndarray, target_phrase: str,
    aliases: List[str],
) -> Tuple[int, float]:
    """Score each bbox crop against target_phrase + aliases via CLIP.

    Returns (best_idx, best_score).
    """
    from PIL import Image
    try:
        from src.conceptgraph.utils.model_utils import clip_recognition
    except Exception:
        clip_recognition = None
    if clip_recognition is None or scene.clip_model is None:
        return 0, 0.0
    prompts = [target_phrase] + [a for a in aliases if a.lower() != target_phrase.lower()]
    best_idx, best_score = 0, -1.0
    for i in range(len(xyxy_np)):
        x1, y1, x2, y2 = xyxy_np[i].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_rgb.shape[1], x2), min(image_rgb.shape[0], y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        crop = image_rgb[y1:y2, x1:x2]
        best_for_this = 0.0
        for p in prompts:
            try:
                probs = clip_recognition(
                    scene.clip_model, scene.clip_tokenizer,
                    scene.clip_preprocess, crop, p,
                )
                # probs is softmax over [prompt]; take the prompt's prob
                score = float(probs[0]) if hasattr(probs, '__len__') else float(probs)
                best_for_this = max(best_for_this, score)
            except Exception:
                continue
        if best_for_this > best_score:
            best_score = best_for_this
            best_idx = i
    return best_idx, best_score


def random_frontier_choice(tsdf_planner: TSDFPlanner, n_filtered_snapshots):
    """
    Choose a random frontier from the TSDF planner.
    """
    if len(tsdf_planner.frontiers) == 0:
        logging.error("No frontiers available, returning None.")
        return None
    idx = random.randint(0, len(tsdf_planner.frontiers)-1)
    random_frontier = tsdf_planner.frontiers[idx]
    logging.info(f"Randomly chosen frontier at {random_frontier.position}")
    return "frontier", random_frontier, n_filtered_snapshots, idx


def get_aabb_corner_points(aabb):
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()
    
    return np.array([
        [min_bound[0], min_bound[1], (max_bound[2] + min_bound[2]) / 2],
        [max_bound[0], min_bound[1], (max_bound[2] + min_bound[2]) / 2],
        [(max_bound[0] + min_bound[0]) / 2, min_bound[1], min_bound[2]], 
        [(max_bound[0] + min_bound[0]) / 2, min_bound[1], max_bound[2]], 
    ])

def select_navigation_corner(aabb, selection_strategy="closest_to_robot", robot_position=None):
    """
    Select a bounding box corner point as the navigation target.
    
    Parameters:
    aabb: Open3D Axis-Aligned Bounding Box object
    selection_strategy: selection strategy ("closest_to_robot", "lowest", "front_center")
    robot_position: (3,) array representing the robot's current position [x, y, z]
                    (only required if the strategy depends on robot position)
    
    Returns:
    target: (3,) array containing the coordinates of the selected corner point
    """
    # Retrieve all corner points
    corners = get_aabb_corner_points(aabb)
    
    if selection_strategy == "closest_to_robot" and robot_position is not None:
        # Select the corner point closest to the robot (XY-plane distance)
        distances = np.linalg.norm(corners[:, [0, 2]] - robot_position[[0, 2]], axis=1)
        return corners[np.argmin(distances)]
    
    elif selection_strategy == "lowest":
        # Select the corner point with the lowest height (assuming the object is grounded)
        return corners[np.argmin(corners[:, 1] - robot_position[1])]
    
    elif selection_strategy == "front_center" and robot_position is not None:
        # Select the middle point of the face oriented towards the robot
        
        # 1. Determine the face facing the robot
        center = aabb.get_center()
        to_robot = robot_position - center
        to_robot[2] = 0  # Ignore height difference
        to_robot /= np.linalg.norm(to_robot)
        
        # 2. Compute center points of each face
        face_centers = [
            np.mean(corners[[0, 1, 2, 3]], axis=0),  # front face (min x)
            np.mean(corners[[4, 5, 6, 7]], axis=0),  # back face (max x)
            np.mean(corners[[0, 1, 4, 5]], axis=0),  # left face (min y)
            np.mean(corners[[2, 3, 6, 7]], axis=0)   # right face (max y)
        ]
        
        # 3. Specify directions for each face
        face_directions = [
            np.array([-1, 0, 0]),  # front face facing +X
            np.array([1, 0, 0]),   # back face facing −X
            np.array([0, -1, 0]),  # left face facing +Y
            np.array([0, 1, 0])    # right face facing −Y
        ]
        
        # Determine the face most oriented towards the robot by dot product
        dot_products = [np.dot(dir, to_robot) for dir in face_directions]
        selected_face = np.argmax(dot_products)
        
        # 4. Return the lowest corner point of the selected face (assuming floor-level navigation)
        face_corners = {
            0: [0, 1, 2, 3],  # front face
            1: [4, 5, 6, 7],  # back face
            2: [0, 1, 4, 5],  # left face
            3: [2, 3, 6, 7]   # right face
        }[selected_face]
        
        face_points = corners[face_corners]
        return face_points[np.argmin(face_points[:, 2])]
    
    else:
        # Default: select the lowest-height corner point
        return corners[np.argmin(corners[:, 2])]

def query_vlm_for_response(
    subtask_metadata: dict,
    scene: Scene,
    tsdf_planner: TSDFPlanner,
    rgb_egocentric_views: list,
    cfg,
    pts = None,
    verbose: bool = False,
):
    # prepare input for vlm
    step_dict = {}

    # prepare object and image context
    object_id_to_name = {
        obj_id: obj["class_name"] for obj_id, obj in scene.objects.items()
    }
    object_id_to_room = {
        obj_id: [obj["room_label"],obj["room_conf"]] for obj_id, obj in scene.objects.items()
    }
    step_dict["obj_map"] = object_id_to_name

    step_dict["objects"] = scene.objects
    step_dict["all_imgs"] = scene.all_observations
    step_dict["edges"] = scene.edges
    step_dict["prompt_h"] = cfg.prompt_h
    step_dict["prompt_w"] = cfg.prompt_w
    step_dict["use_full_obj_list"] = cfg.use_full_obj_list

    # prepare frontier
    step_dict["frontier_imgs"] = [
        encode_tensor2base64(frontier.feature) for frontier in tsdf_planner.frontiers
    ]

    # prepare egocentric views
    if cfg.egocentric_views:
        step_dict["egocentric_views"] = rgb_egocentric_views
        step_dict["use_egocentric_views"] = True

    # prepare other metadata
    step_dict["question"] = subtask_metadata["question"]
    step_dict["task_type"] = subtask_metadata["task_type"]
    step_dict["class"] = subtask_metadata["class"]
    step_dict["image"] = subtask_metadata["image"]
    step_dict["CLR"] = subtask_metadata['CLR']
    step_dict["object_id_to_room"] = object_id_to_room
    step_dict['image_to_edges'] = scene.img_to_edge
    # query vlm
    (
        outputs,
        image_map_reverse,
        reason,
        n_filtered_snapshots,
    ) = explore_two_step(step_dict, cfg, verbose=verbose)
    if outputs is None:
        logging.error(f"explore_step failed and returned None, Choose a random frontier instead!")
        return random_frontier_choice(tsdf_planner, n_filtered_snapshots)
    logging.info(f"Response: [{outputs}]\nReason: [{reason}]")
    
    # parse returned results
    try:
        target_type, target_index = outputs.split(",")[0].strip().split(" ")
        logging.info(f"Prediction: {target_type}, {target_index}")
    except:
        logging.info(f"Wrong output format, Choose a random frontier instead!")
        return random_frontier_choice(tsdf_planner, n_filtered_snapshots)

    if target_type not in ["image", "frontier", "object"]:
        logging.info(f"Wrong target type: {target_type}, Choose a random frontier instead!")
        return random_frontier_choice(tsdf_planner, n_filtered_snapshots)


    if target_type == "image":
        #Implementation of AVU.
        #We update only the words we consider to be the target, 
        #so we perform re-perception directly and select it as the answer.
        if int(target_index) >= 0 and int(target_index) < len(image_map_reverse):
            target_index = image_map_reverse[int(target_index)]
        else:
            view_idx = int(target_index) - len(image_map_reverse)
            global_step = scene.global_step_cnt
            target_index = f"{global_step}-view_{view_idx}.png"
        target_image = scene.all_observations[target_index]
        try:
            object_class = outputs.split(",")[1].strip()
        except:
            logging.info(f"Wrong output format, Choose a random frontier instead!")
            return random_frontier_choice(tsdf_planner, n_filtered_snapshots)
        scene.detection_model.set_classes([object_class])
        results = scene.detection_model.predict(
            target_image, conf=cfg.AVU_conf_threshold, verbose=False
        )
        
        scene.detection_model.set_classes(scene.obj_classes.get_classes_arr())
        if len(results) == 0 or len(results[0].boxes) == 0:
            logging.info(
                f"No objects detected in the predicted image: {target_index}, Choose a random frontier instead!"
            )
            return random_frontier_choice(tsdf_planner, n_filtered_snapshots)
        
        confidences = results[0].boxes.conf.cpu().numpy()
        max_idx = confidences.argmax()
        max_confidence = confidences[max_idx: max_idx+1]
        detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        max_detection_class_ids = detection_class_ids[max_idx: max_idx+1]
        xyxy_tensor = results[0].boxes.xyxy[max_idx: max_idx+1, ...]
        sam_out = scene.sam_predictor.predict(
            target_image, bboxes=xyxy_tensor, verbose=False
        )
        masks_tensor = sam_out[0].masks.data
        masks_np = masks_tensor.cpu().numpy()
        obj_pcds_and_bboxes = detections_to_obj_pcd_and_bbox(
            depth_array=scene.all_depths[target_index],
            masks=masks_np,
            cam_K=scene.intrinsics[:3, :3],  # Camera intrinsics
            image_rgb=target_image,
            trans_pose=scene.all_cam_poses[target_index],
            min_points_threshold=scene.cfg_cg.min_points_threshold,
            spatial_sim_type=scene.cfg_cg.spatial_sim_type,
            obj_pcd_max_points=scene.cfg_cg.obj_pcd_max_points,
            device=scene.device,
        )
        
        for obj in obj_pcds_and_bboxes:
            if obj:
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
        try:
            a = []
            for idx in scene.objects.keys():
                a.append(scene.objects[idx]["pcd"].points)
            obj_pos = Visibility_based_Viewpoint_Decision(
                np.array(obj["pcd"].points),
                np.concatenate(a, axis=0),
                pts,
                tsdf_planner,
                cfg.dicision_radius,
            )
            if obj_pos is None:
                obj_pos = select_navigation_corner(aabb = obj["bbox"], robot_position = pts)
                logging.info(f"The index of target Image {target_index} : {obj_pos} (Closed Box Center, Confidence : {max_confidence})")
            else:
                logging.info(f"The index of target Image {target_index} : {obj_pos} (Visible Center, Confidence : {max_confidence})")
        except:
            logging.info(
                f"No Object Point Cloud in the predicted image: {target_index}, Choose a random frontier instead!"
            )
            return random_frontier_choice(tsdf_planner, n_filtered_snapshots)
        # logging.info(f"The index of target Image {target_index} : {obj_pos} (Mask Center, Confidence : {max_confidence})")
        
        
        # a = []
        # for idx in scene.objects.keys():
        #     if idx != target_index:
        #         a.append(scene.objects[idx]["pcd"].points)
        # np.save(f'/hsun/0625/3D-Mem/vis/other_obj_{pts[1]}.npy', np.concatenate(a, axis=0))
        # np.save(f'/hsun/0625/3D-Mem/vis/target_obj_{pts[1]}.npy', np.array(obj["pcd"].points))
        

        return target_type, obj_pos, n_filtered_snapshots, target_index
    elif target_type == "object":
        target_index = int(target_index)
        if target_index not in list(scene.objects.keys()):
            logging.info(
                f"Predicted object index not in list: {target_index}, Choose a random frontier instead!"
            )
            return random_frontier_choice(tsdf_planner, n_filtered_snapshots)
        # a = []
        # for idx in scene.objects.keys():
        #     if idx != target_index:
        #         a.append(scene.objects[idx]["pcd"].points)
        # np.save(f'/hsun/0625/3D-Mem/vis/other_obj_{pts[1]}.npy', np.concatenate(a, axis=0))
        # np.save(f'/hsun/0625/3D-Mem/vis/target_obj_{pts[1]}.npy', np.array(scene.objects[target_index]["pcd"].points))
        
        
        
        
        a = []
        for idx in scene.objects.keys():
            a.append(scene.objects[idx]["pcd"].points)
        target_point = Visibility_based_Viewpoint_Decision(
            np.array(scene.objects[target_index]["pcd"].points),
            np.concatenate(a, axis=0),
            pts,
            tsdf_planner,
            cfg.dicision_radius,
        )
        if target_point is None:
            target_point = select_navigation_corner(aabb = scene.objects[target_index]["bbox"], robot_position = pts)
            logging.info(f"Next choice: Object {target_point} (Closed Box Center)")
        else:
            logging.info(f"Next choice: Object {target_point} (Visible Center)")

        return target_type, target_point, n_filtered_snapshots, target_index
    else:  # target_type == "frontier"
        target_index = int(target_index)
        if target_index < 0 or target_index >= len(tsdf_planner.frontiers):
            logging.info(
                f"Predicted frontier target index out of range: {target_index}, Choose a random frontier instead!"
            )
            return random_frontier_choice(tsdf_planner, n_filtered_snapshots)
        target_point = tsdf_planner.frontiers[target_index].position
        logging.info(f"Next choice: Frontier at {target_point}")
        pred_target_frontier = tsdf_planner.frontiers[target_index]

        return target_type, pred_target_frontier, n_filtered_snapshots, target_index


def query_vlm_multi_agent(
    subtask_metadata: dict,
    scene: Scene,
    tsdf_planner: TSDFPlanner,
    rgb_egocentric_views: list,
    cfg,
    pts=None,
    verbose: bool = False,
):
    """Multi-agent variant of query_vlm_for_response.

    Uses explore_multi_agent (5-agent chain) instead of explore_two_step.
    Reuses the AVU+VVD logic from query_vlm_for_response's 'image' branch,
    but img_path now comes directly from explore_multi_agent (Answerer)
    rather than image_map_reverse mapping.
    """
    # prepare step_dict (mirrors query_vlm_for_response)
    step_dict = {}

    object_id_to_name = {
        obj_id: obj["class_name"] for obj_id, obj in scene.objects.items()
    }
    object_id_to_room = {
        obj_id: [obj["room_label"], obj["room_conf"]] for obj_id, obj in scene.objects.items()
    }
    step_dict["obj_map"] = object_id_to_name
    step_dict["objects"] = scene.objects
    step_dict["all_imgs"] = scene.all_observations
    step_dict["edges"] = scene.edges
    step_dict["prompt_h"] = cfg.prompt_h
    step_dict["prompt_w"] = cfg.prompt_w
    step_dict["use_full_obj_list"] = cfg.use_full_obj_list

    step_dict["frontier_imgs"] = [
        frontier.feature for frontier in tsdf_planner.frontiers
    ]
    if cfg.egocentric_views:
        step_dict["egocentric_views"] = rgb_egocentric_views
        step_dict["use_egocentric_views"] = True

    step_dict["question"] = subtask_metadata["question"]
    step_dict["task_type"] = subtask_metadata["task_type"]
    step_dict["class"] = subtask_metadata["class"]
    step_dict["image"] = subtask_metadata["image"]
    step_dict["CLR"] = subtask_metadata["CLR"]
    step_dict["object_id_to_room"] = object_id_to_room
    step_dict["image_to_edges"] = scene.img_to_edge
    step_dict["scene"] = scene
    step_dict["tsdf_planner"] = tsdf_planner
    step_dict["egocentric_views"] = rgb_egocentric_views

    # multi-agent specific metadata
    step_dict["is_new_subtask"] = subtask_metadata.get("is_new_subtask", False)
    step_dict["high_level_plan"] = subtask_metadata.get("high_level_plan", None)
    step_dict["step_index"] = subtask_metadata.get("current_step", 0)
    step_dict["current_step"] = subtask_metadata.get("current_step", 0)
    step_dict["current_position"] = pts
    # M4: pass episode_memory so High-Level Planner can retrieve step summaries
    step_dict["episode_memory"] = subtask_metadata.get("episode_memory", None)
    # Phase H: pass working_memory so agents can read candidates/feedback
    step_dict["working_memory"] = subtask_metadata.get("working_memory", None)

    # Bug2 fix: explore_multi_agent reads 'processed_images' and
    # 'image_map_reverse' (built by KSS) but they were never computed here,
    # so the snapshot pool was always empty. Run Key_Subgraph_Selection the
    # same way explore_two_step does (see explore_utils.py L801-818) and
    # populate both keys.
    # 设计: KSS 仅在 subtask 的 step0 (is_new_subtask=True) 运行并注入图像边,
    # 其他步骤跳过 KSS (processed_images 为空), explore_multi_agent 只追加 egocentric。
    step_dict["use_prefiltering"] = cfg.prefiltering
    step_dict["top_k_categories"] = cfg.top_k_categories
    step_dict["use_AVU"] = getattr(cfg, "use_AVU", step_dict.get("use_AVU", True))
    use_room_filter = cfg.use_room_filter
    is_new_subtask = subtask_metadata.get("is_new_subtask", False)
    if is_new_subtask:
        (
            _question_kss,
            _image_goal_kss,
            _egocentric_imgs_kss,
            _selected_objs,
            _selected_edges,
            processed_images,
            _frontier_imgs_kss,
        ) = Key_Subgraph_Selection(
            step_dict, verbose, cfg.use_ollama, use_room_filter
        )
        # Key_Subgraph_Selection returns processed_images but not
        # image_map_reverse; build the reverse index consistent with
        # Prompt_with_AVU_and_CLR (explore_utils.py L286-291).
        image_map_reverse = {
            idx: img_key for idx, img_key in enumerate(processed_images.keys())
        }
        step_dict["processed_images"] = processed_images
        step_dict["image_map_reverse"] = image_map_reverse
        # KSS returns b64-encoded egocentric_imgs and frontier_imgs; overwrite
        # the raw tensors in step_dict so explore_multi_agent consumes b64 directly.
        step_dict["egocentric_imgs"] = _egocentric_imgs_kss
        step_dict["frontier_imgs"] = _frontier_imgs_kss
        if verbose:
            logging.info(f"[KSS] step0 injected {len(processed_images)} edge images, "
                         f"{len(_egocentric_imgs_kss)} egocentric, "
                         f"{len(_frontier_imgs_kss)} frontiers")
    else:
        # non-step0: skip KSS, only encode egocentric + frontier for multi-agent
        step_dict["processed_images"] = {}
        step_dict["image_map_reverse"] = {}
        step_dict["egocentric_imgs"] = [
            encode_tensor2base64(v) for v in rgb_egocentric_views
        ] if cfg.egocentric_views else []
        step_dict["frontier_imgs"] = [
            encode_tensor2base64(f.feature) for f in tsdf_planner.frontiers
        ]
        if verbose:
            logging.info(f"[KSS] skipped (non-step0), "
                         f"{len(step_dict['egocentric_imgs'])} egocentric, "
                         f"{len(step_dict['frontier_imgs'])} frontiers")

    # query multi-agent vlm
    try:
        # F5: pass scene.image_pool into step_dict so explore_multi_agent
        # can carry it across steps instead of rebuilding from None each step
        step_dict["image_pool"] = scene.image_pool
        (
            target_type,
            target_index,
            reason,
            n_filtered_snapshots,
            class_name_if_image,
        ) = explore_multi_agent(step_dict, cfg, verbose=verbose)
        # F3: explore_multi_agent updates step_dict['high_level_plan'] locally;
        # write it back to subtask_metadata so the main loop reads the latest plan.
        subtask_metadata['high_level_plan'] = step_dict.get('high_level_plan')
        # F5: write image_pool back to scene for next step
        scene.image_pool = step_dict.get('image_pool')
    except Exception as e:
        logging.error(f"explore_multi_agent failed: {e}, stop (no random frontier)")
        return None

    if target_type is None:
        logging.error("explore_multi_agent returned None, stop (no random frontier)")
        return None

    logging.info(
        f"[multi_agent] target_type={target_type}, target_index={target_index}, "
        f"reason=[{reason}]"
    )

    # parse by target_type
    if target_type == "image":
        # Phase C: four-level grounding. img_path is target_index (Answerer).
        img_path = target_index
        object_class = class_name_if_image
        if object_class is None:
            logging.info(
                f"image target but no class_name, stop (no random frontier)"
            )
            return None
        if img_path not in scene.all_observations:
            logging.info(
                f"img_path {img_path} not in all_observations, stop (no random frontier)"
            )
            return None
        target_image = scene.all_observations[img_path]
        cam_pose = scene.all_cam_poses[img_path]
        view_yaw = _cam_pose_to_yaw(cam_pose)
        aliases = get_aliases(object_class)
        # working_memory may be None (Phase H wires it in). Create candidate if present.
        working_memory = subtask_metadata.get("working_memory", None)
        step_index = subtask_metadata.get("current_step", 0)
        candidate = None
        if working_memory is not None:
            candidate = working_memory.get_or_create_candidate(
                image_path=img_path,
                target_phrase=object_class,
                source_step=step_index,
                camera_pose=cam_pose,
                view_yaw=view_yaw,
            )
            logging.info(
                f"[AVU] candidate {candidate.candidate_id} phrase='{object_class}' "
                f"aliases={aliases} status={candidate.status}"
            )

        try:
            # ---- Level 1: YOLO(target_phrase) ----
            result = _yolo_detect(scene, [object_class], target_image, cfg.AVU_conf_threshold)
            if result is not None:
                logging.info(f"[AVU] L1 YOLO('{object_class}') detected")
                if candidate is not None:
                    candidate.record_attempt(1, True, "YOLO target phrase")
            else:
                logging.info(f"[AVU] L1 YOLO('{object_class}') no detection, trying aliases")
                if candidate is not None:
                    candidate.record_attempt(1, False, "no YOLO box for target phrase")
                # ---- Level 2: YOLO(aliases) ----
                for alias in aliases:
                    if alias.lower() == object_class.lower():
                        continue
                    result = _yolo_detect(scene, [alias], target_image, cfg.AVU_conf_threshold)
                    if result is not None:
                        logging.info(f"[AVU] L2 YOLO alias '{alias}' detected")
                        if candidate is not None:
                            candidate.record_attempt(2, True, f"alias '{alias}'")
                        break
                if result is None:
                    if candidate is not None:
                        candidate.record_attempt(2, False, "no alias detected")
                    # ---- Level 3: generic classes + CLIP rerank ----
                    result = _yolo_detect(scene, _GENERIC_CLASSES, target_image, cfg.AVU_conf_threshold)
                    if result is not None:
                        xyxy_np_all = result.boxes.xyxy.cpu().numpy()
                        best_idx, best_score = _clip_rerank_bbox(
                            scene, target_image, xyxy_np_all, object_class, aliases
                        )
                        if best_score >= _CLIP_RERANK_THRESH:
                            logging.info(
                                f"[AVU] L3 class-agnostic + CLIP rerank accepted "
                                f"(score={best_score:.3f})"
                            )
                            if candidate is not None:
                                candidate.record_attempt(3, True, f"CLIP score {best_score:.3f}")
                            # keep only the best bbox
                            keep = result.boxes.xyxy[best_idx:best_idx+1, ...]
                            result.boxes.xyxy = keep
                            result.boxes.conf = result.boxes.conf[best_idx:best_idx+1]
                            result.boxes.cls = result.boxes.cls[best_idx:best_idx+1]
                        else:
                            logging.info(
                                f"[AVU] L3 CLIP rerank rejected (score={best_score:.3f} "
                                f"< {_CLIP_RERANK_THRESH})"
                            )
                            if candidate is not None:
                                candidate.record_attempt(3, False, f"CLIP score {best_score:.3f}")
                            result = None
                    else:
                        if candidate is not None:
                            candidate.record_attempt(3, False, "no generic class detection")

            # ---- Level 4: evidence-pose navigation (no detection) ----
            if result is None:
                logging.info(
                    f"[AVU] L4 all detection failed, navigate to evidence-pose "
                    f"(img {img_path}, yaw={view_yaw:.2f})"
                )
                if candidate is not None:
                    candidate.record_attempt(4, False, "evidence-pose navigation")
                if working_memory is not None:
                    _reason_l4a = f"VLM saw '{object_class}' in {img_path} but YOLO/CLIP grounded nothing; navigating to evidence-pose"
                    _fix_l4a = working_memory.suggest_fix_for(FB_AVU_VISUAL_ONLY, _reason_l4a) if working_memory is not None else "re-observe from closer view; try aliases"
                    working_memory.add_feedback(
                        step=step_index,
                        type_=FB_AVU_VISUAL_ONLY,
                        reason=_reason_l4a,
                        suggested_fix=_fix_l4a,
                        target_candidate_id=candidate.candidate_id if candidate else None,
                    )
                # Return the camera pose that captured the evidence so the
                # agent navigates there and re-observes from a closer view.
                # cam_pose is a 4x4 world->cam; target_point is cam position
                # in habitat coords = cam_pose[:3, 3] (if cam->world) or the
                # inverse. scene stores cam_poses consistent with tsdf usage.
                # We return it as the navigation target; set_next_navigation_point
                # treats 'image' choice as a habitat position.
                try:
                    cam_pos_habitat = cam_pose[:3, 3]
                except Exception:
                    cam_pos_habitat = None
                if cam_pos_habitat is None:
                    logging.info(
                        f"[AVU] L4 evidence-pose unavailable, stop (no random frontier)"
                    )
                    if working_memory is not None:
                        _reason_nopose = f"VLM saw '{object_class}' in {img_path} but YOLO/CLIP grounded nothing; evidence-pose unavailable"
                        _fix_nopose = working_memory.suggest_fix_for(FB_AVU_VISUAL_ONLY, _reason_nopose) if working_memory is not None else "re-observe from closer view; try aliases"
                        working_memory.add_feedback(
                            step=step_index,
                            type_=FB_AVU_VISUAL_ONLY,
                            reason=_reason_nopose,
                            suggested_fix=_fix_nopose,
                            target_candidate_id=candidate.candidate_id if candidate else None,
                        )
                    return None
                return target_type, cam_pos_habitat, n_filtered_snapshots, target_index

            # ---- Grounded (L1/L2/L3): run SAM + point cloud + VVD ----
            confidences = result.boxes.conf.cpu().numpy()
            max_idx = int(confidences.argmax())
            max_confidence = confidences[max_idx: max_idx + 1]
            xyxy_tensor = result.boxes.xyxy[max_idx: max_idx + 1, ...]
            sam_out = scene.sam_predictor.predict(
                target_image, bboxes=xyxy_tensor, verbose=False
            )
            masks_tensor = sam_out[0].masks.data
            masks_np = masks_tensor.cpu().numpy()
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
                if obj:
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
            valid_objs = [o for o in obj_pcds_and_bboxes if o is not None]
            if not valid_objs:
                logging.info(
                    f"All detections invalid for {img_path}, navigate to evidence-pose"
                )
                if candidate is not None:
                    candidate.record_attempt(3, False, "SAM/pcd invalid")
                if working_memory is not None:
                    _reason_sam = f"VLM saw '{object_class}' in {img_path} but YOLO/CLIP grounded nothing; navigating to evidence-pose"
                    _fix_sam = working_memory.suggest_fix_for(FB_AVU_VISUAL_ONLY, _reason_sam) if working_memory is not None else "re-observe from closer view; try aliases"
                    working_memory.add_feedback(
                        step=step_index,
                        type_=FB_AVU_VISUAL_ONLY,
                        reason=_reason_sam,
                        suggested_fix=_fix_sam,
                        target_candidate_id=candidate.candidate_id if candidate else None,
                    )
                try:
                    cam_pos_habitat = cam_pose[:3, 3]
                except Exception:
                    cam_pos_habitat = None
                if cam_pos_habitat is None:
                    return None
                return target_type, cam_pos_habitat, n_filtered_snapshots, target_index
            target_obj = valid_objs[0]
            if candidate is not None:
                working_memory.grounded_candidate(candidate.candidate_id)
            a = []
            for idx in scene.objects.keys():
                a.append(scene.objects[idx]["pcd"].points)
            obj_pos = Visibility_based_Viewpoint_Decision(
                np.array(target_obj["pcd"].points),
                np.concatenate(a, axis=0),
                pts,
                tsdf_planner,
                cfg.dicision_radius,
            )
            if obj_pos is None:
                obj_pos = select_navigation_corner(
                    aabb=target_obj["bbox"], robot_position=pts
                )
                logging.info(
                    f"multi_agent target Image {img_path}: {obj_pos} "
                    f"(Closed Box Center, conf={max_confidence})"
                )
            else:
                logging.info(
                    f"multi_agent target Image {img_path}: {obj_pos} "
                    f"(Visible Center, conf={max_confidence})"
                )
            return target_type, obj_pos, n_filtered_snapshots, target_index
        except Exception as e:
            logging.info(
                f"AVU/VVD failed for {img_path}: {e}, navigate to evidence-pose or stop"
            )
            if candidate is not None:
                candidate.record_attempt(4, False, f"exception: {e}")
            if working_memory is not None:
                _reason_exc = f"AVU exception: {e}"
                _fix_exc = working_memory.suggest_fix_for(FB_AVU_FAIL, _reason_exc) if working_memory is not None else "retry grounding from evidence-pose; consider aliases"
                working_memory.add_feedback(
                    step=step_index,
                    type_=FB_AVU_FAIL,
                    reason=_reason_exc,
                    suggested_fix=_fix_exc,
                    target_candidate_id=candidate.candidate_id if candidate else None,
                )
            try:
                cam_pos_habitat = cam_pose[:3, 3]
            except Exception:
                cam_pos_habitat = None
            if cam_pos_habitat is None:
                return None
            return target_type, cam_pos_habitat, n_filtered_snapshots, target_index

    elif target_type == "frontier":
        target_index = int(target_index)
        if target_index < 0 or target_index >= len(tsdf_planner.frontiers):
            logging.info(
                f"Predicted frontier index out of range: {target_index}, "
                f"stop (no random frontier)"
            )
            return None
        pred_target_frontier = tsdf_planner.frontiers[target_index]
        logging.info(
            f"multi_agent next choice: Frontier at {pred_target_frontier.position}"
        )
        return target_type, pred_target_frontier, n_filtered_snapshots, target_index

    else:  # target_type == 'stop'
        logging.info("multi_agent Stop Exploration, returning None")
        return None


def query_vlm_for_response_end(
    subtask_metadata: dict,
    rgb_egocentric_views: dict,
    cfg,
    verbose: bool = False,
):
    # prepare input for vlm
    step_dict = {}
    # prepare egocentric views
    step_dict["egocentric_views"] = rgb_egocentric_views
    step_dict["use_egocentric_views"] = True

    # prepare other metadata
    step_dict["question"] = subtask_metadata["question"]
    step_dict["task_type"] = subtask_metadata["task_type"]
    step_dict["class"] = subtask_metadata["class"]
    step_dict["image"] = subtask_metadata["image"]
    # query vlm
    (
        outputs,
        reason,
    ) = task_check(step_dict, verbose=verbose)

    logging.info(f"Response: [{outputs}]\nReason: [{reason}]")

    # Phase F: return reason too so the main loop can build FeedbackEvent.
    return outputs, reason