"""Coupled multi-agent workflow for MSGNav GOAT-Bench evaluation.

Ported from Pred-EQA's 5-agent chain (Image Manager, Frontier Manager,
Answerer, High-Level Planner, Low-Level Executor) and adapted to MSGNav's
M3DSG scene graph, KSS image pool, and call_openai_api VLM interface.

Key adaptations vs Pred-EQA:
- image_pool entries carry M3DSG-connected objects (id, class_name) from
  scene.img_to_edge reverse lookup.
- Answerer outputs "Image i, <class>" (i = pool index) for all three GOAT-Bench
  task types (object / description / image), or "Continue Exploration".
- High-Level Planner plan is passed via step['high_level_plan'] and persists
  across subtasks (maintained by the main loop); this module only updates it.
- No Forced Answerer; step exhaustion => subtask failure.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple

from src.explore_utils import call_openai_api, encode_tensor2base64, resize_image, format_question
from src.plan_extraction_utils import extract_todo_list_from_text


# ---------------------------------------------------------------------------
# image_pool data structure
# ---------------------------------------------------------------------------
# image_pool: list of dict {
#     "img_path": str,                      # key into scene.all_observations
#     "img_b64": str,                       # resized RGB base64
#     "connected_objects": List[Tuple[int, str]],  # (obj_id, class_name)
#     "source": str,                        # "kss_edge" | "egocentric"
#     "step": int,                          # step at which it was added
# }


def build_image_pool_from_kss(
    step: int,
    scene,
    processed_images: Dict[str, str],
    image_map_reverse: Dict[int, str],
) -> List[Dict[str, Any]]:
    """Build image pool from KSS processed_images.

    For each image key in processed_images, reverse-lookup the connected
    object pairs via scene.img_to_edge and record (obj_id, class_name) for
    each unique object id appearing in those edges.

    Args:
        step: current step index.
        scene: MSGNav Scene instance.
        processed_images: {img_key: img_b64} from KSS edge_pruning_KSS.
        image_map_reverse: {idx: img_key} from KSS (kept for compatibility).

    Returns:
        image_pool list.
    """
    pool: List[Dict[str, Any]] = []
    for img_key, img_b64 in processed_images.items():
        connected_objects: List[Tuple[int, str]] = []
        edge_pairs = scene.img_to_edge.get(img_key, [])
        seen_ids = set()
        for pair in edge_pairs:
            for obj_id in pair:
                if obj_id in seen_ids:
                    continue
                seen_ids.add(obj_id)
                obj = scene.objects.get(obj_id)
                if obj is not None:
                    connected_objects.append((obj_id, obj["class_name"]))
        pool.append({
            "img_path": img_key,
            "img_b64": img_b64,
            "connected_objects": connected_objects,
            "source": "kss_edge",
            "step": step,
        })
    return pool


def add_egocentric_to_pool(
    pool: List[Dict[str, Any]],
    egocentric_imgs: List[str],
    step: int,
    scene=None,
    egocentric_img_paths: Optional[List[str]] = None,
) -> None:
    """Append current-step egocentric views to the image pool in-place.

    Args:
        pool: image pool list (mutated).
        egocentric_imgs: list of base64-encoded egocentric RGB images.
        step: current step index.
        scene: optional Scene, used to resolve img_path if egocentric_img_paths given.
        egocentric_img_paths: optional list of img_path keys parallel to
            egocentric_imgs. If None, a synthetic key f"{step}-view_{i}.png"
            is generated (matches MSGNav naming convention).
    """
    for i, img_b64 in enumerate(egocentric_imgs):
        if egocentric_img_paths is not None and i < len(egocentric_img_paths):
            img_path = egocentric_img_paths[i]
        else:
            img_path = f"{step}-view_{i}.png"
        pool.append({
            "img_path": img_path,
            "img_b64": img_b64,
            "connected_objects": [],
            "source": "egocentric",
            "step": step,
        })


# ---------------------------------------------------------------------------
# Prompt formatters
# ---------------------------------------------------------------------------

def _format_connected_objects(connected_objects: List[Tuple[int, str]]) -> str:
    if not connected_objects:
        return "none"
    return ", ".join(f"{name}({oid})" for oid, name in connected_objects)


def format_image_manager_prompt(
    question: str,
    pool: List[Dict[str, Any]],
    high_level_plan: Optional[str],
    task_type: str,
    image_goal: Optional[str],
) -> Tuple[str, list]:
    """Image Manager prompt.

    Ported from Pred-EQA format_manage_prompt. Displays each image as
    'Image i: <img> connected objects: chair(5), table(8)'. Outputs
    'Retain Images: {i,...}'.
    """
    sys_prompt = (
        "Task: You are an indoor MEMORY MANAGEMENT AGENT responsible for "
        "CURATING and PRESERVING visual images and spatial information "
        "collected by the embodied agent during its navigation, working in "
        "tandem with your existing TEXTUAL MEMORY and high-level plan.\n\n"
        "Instructions:\n"
        "1. CAREFULLY analyze the information needed to answer the question, "
        "paying special attention to location details, objectives, object "
        "relationships, and any mentioned or implied attributes.\n"
        "2. Review all available images thoroughly and cross-reference "
        "them with your TEXTUAL MEMORY. When deciding whether to retain an "
        "image, adopt a conservative approach - if there is ANY potential "
        "visual relevance to the current question or its context, it should "
        "be preserved. Specifically, retain images that include:\n"
        "   - Any room types or spaces that may be related to the question's "
        "context, even indirectly.\n"
        "   - Adjacent or connected areas that could provide spatial clues or "
        "lead to relevant locations.\n"
        "   - Partial views or incomplete perspectives of objects, appliances, "
        "or features that might be useful in reasoning.\n"
        "   - Environmental or contextual cues (e.g., lighting, layout, "
        "orientation) that help establish spatial understanding or support "
        "inference.\n"
        "   - Objects or categories explicitly mentioned in the question, as "
        "well as those that are semantically or functionally associated.\n"
        "   - Any image that provides visual background or situational "
        "information not fully captured by text, which could aid in answering "
        "the question or reconstructing the environment.\n"
        "3. MEMORY COMPACTION (Textual Redundancy Filter): To prevent critical "
        "visual clues from being overwhelmed by redundant trajectory images, "
        "you may DISCARD an image ONLY IF it meets BOTH of the following "
        "conditions:\n"
        "   - It is completely irrelevant to the primary question or objective "
        "(contains no target objects or contextual clues).\n"
        "   - Its environmental content, spatial relationships, or navigational "
        "cues are already adequately and comprehensively described in your "
        "existing textual memory.\n"
        "4. When in doubt-especially if you are unsure whether the textual "
        "memory fully captures the visual nuances of the scene-err on the "
        "side of retention. Even seemingly minor or indirect visual clues can "
        "become valuable during later stages of reasoning or path "
        "reconstruction.\n"
    )
    content = []

    # Question (with optional image goal for image-type tasks)
    text = f"Question: {question}\n"
    if image_goal is not None:
        content.append((text, image_goal))
        content.append(("\n",))
    else:
        content.append((text + "\n",))

    # High-level plan / textual memory
    if high_level_plan:
        content.append((f"Current High-Level Plan:\n{high_level_plan}\n",))
    else:
        content.append(("No high-level plan yet.\n",))

    # Images display
    content.append(("Available Images:\n",))
    if not pool:
        content.append(("No images available\n",))
    else:
        for i, snap in enumerate(pool):
            content.append((f"Image {i}: ", snap["img_b64"]))
            objs_str = _format_connected_objects(snap["connected_objects"])
            content.append((f" connected objects: {objs_str}\n",))

    # Output format
    text = (
        "Output Format:\n"
        "1. First, think step by step and explain your reasoning clearly.\n"
        '2. Then, provide your final answer in the exact format: '
        '"Retain Images: {i, ...}".'
    )
    content.append((text,))
    return sys_prompt, content


def format_frontier_manager_prompt(
    question: str,
    frontier_imgs: List[str],
    pool: List[Dict[str, Any]],
    high_level_plan: Optional[str],
    task_type: str,
    frontier_states: Optional[List[Any]] = None,
) -> Tuple[str, list]:
    """Frontier Manager prompt.

    Ported from Pred-EQA format_plan_manager_prompt. Outputs
    'Retain Frontiers: {F_XXX, ...}' using stable frontier IDs.
    """
    sys_prompt = (
        "Task: You are an EXPLORATION DIRECTION MANAGEMENT AGENT responsible "
        "for STRATEGICALLY SELECTING and PRUNING potential frontiers based on "
        "observed visual images. Your goal is to eliminate directions that "
        "have BOTH OBVIOUSLY BEEN EXPLORED AND ARE IRRELEVANT to answering "
        "the question.\n\n"
        "Instructions:\n"
        "1. CAREFULLY analyze the provided visual images to identify areas "
        "that have already been explored.\n"
        "2. Determine which frontiers (exploration directions) can be safely "
        "removed because they MEET BOTH CRITERIA:\n"
        "- They lead to areas ALREADY CONFIRMED AS VISITED with high "
        "certainty.\n"
        "- The area or objects within them are CLEARLY UNRELATED TO THE "
        "QUESTION or its context.\n"
        "3. ONLY remove such frontiers if BOTH conditions above are MET. If "
        "ANY DOUBT exists about either exploration status or relevance, KEEP "
        "THE FRONTIER.\n"
        "4. Retain all other frontiers, including those where there is ANY "
        "UNCERTAINTY regarding their exploration status or their relevance to "
        "the question.\n"
        "5. Maintain spatial awareness: even partially visible rooms or "
        "ambiguous paths should be preserved unless you are ABSOLUTELY "
        "CERTAIN about their irrelevance.\n"
        "6. REMEMBER, the key is to avoid deleting potentially useful "
        "information. When in doubt, err on the side of caution and retain "
        "the frontier.\n"
        "7. Each frontier has a STABLE ID like F_023. Use this ID (not the "
        "display index) in your output. Frontiers marked DO NOT SELECT or "
        "status=EXPLORED must be pruned.\n"
    )
    content = []

    content.append((f"Target Question: {question}\n",))

    if high_level_plan:
        content.append((f"Current High-Level Plan:\n{high_level_plan}\n",))
    else:
        content.append(("No high-level plan yet.\n",))

    # Previously observed clues (images)
    content.append(("Previously Observed Clues:\n",))
    if not pool:
        content.append(("No images available\n",))
    else:
        for snap in pool:
            content.append(("\n", snap["img_b64"]))
        content.append(("\n",))

    # Frontiers
    content.append(("\nAvailable Exploration Directions:\n",))
    if not frontier_imgs:
        content.append(("No frontiers available\n",))
    else:
        for i, img in enumerate(frontier_imgs):
            fs = frontier_states[i] if frontier_states and i < len(frontier_states) else None
            if fs is not None:
                header = f"Frontier {fs.to_prompt_str(local_index=i)}: "
            else:
                header = f"Frontier {i}: "
            content.append((header, img))
            content.append(("\n",))
        ids = [fs.frontier_id for fs in frontier_states] if frontier_states else list(range(len(frontier_imgs)))
        id_strs = ", ".join(f"F_{i:03d}" for i in ids)
        content.append((f"Available Frontier IDs: {id_strs}\n",))

    text = (
        "Output Format:\n"
        "1. First, think step by step and explain your reasoning clearly.\n"
        '2. Then, provide your final answer in the exact format: '
        '"Retain Frontiers: {F_XXX, F_YYY}" (retain at least 1 frontier).'
    )
    content.append((text,))
    return sys_prompt, content


def _format_clr_block(history_decision: Optional[Dict[str, Any]]) -> str:
    """Build a 'History Decisions (avoid repeating)' text block from CLR data.

    Mirrors the format of explore_utils.py Prompt_with_AVU_and_CLR L322-345.
    F2: inject ALL decisions with object_judge=='no', including frontier
    choices (previously only image/object were injected, leaving the Executor
    without frontier anti-repeat protection).
    F1: for image-type decisions, max_point_choice is an img_path (str), not
    a numeric index — display the path short name so the VLM can associate it
    with the pool entry rather than showing an opaque raw path.
    """
    if not history_decision:
        return ""
    cnt_step = history_decision.get("cnt_step", "?")
    max_step = history_decision.get("max_step", "?")
    lines = [
        f"History Decisions (avoid repeating): "
        f"(now step is {cnt_step}/{max_step}). "
        "Choosing those incorrect objects or images again is prohibited:"
    ]
    have_decision = False
    for s_key, decision in history_decision.items():
        if not isinstance(s_key, int):
            continue
        if not isinstance(decision, dict):
            continue
        if "target_type" not in decision:
            continue
        target_type = decision["target_type"]
        choice = decision.get("max_point_choice", "?")
        # F2: frontier decisions never get object_judge (no task_check for
        # frontier), so inject them unconditionally as "already explored" to
        # give the Executor anti-repeat context. image/object still require
        # object_judge=='no' (confirmed wrong by task_check).
        if target_type == "frontier":
            have_decision = True
            fid = decision.get("frontier_id")
            if fid is not None:
                fid_str = f"F_{int(fid):03d}"
            else:
                fid_str = str(choice)
            lines.append(
                f"    step {s_key}: Choosing Frontier {fid_str} to explore, "
                "already explored."
            )
            continue
        if decision.get("object_judge") != "no":
            continue
        have_decision = True
        if target_type == "image":
            # choice is img_path; show short name for VLM readability
            choice_str = str(choice)
            short = choice_str.split("/")[-1] if "/" in choice_str else choice_str
            lines.append(
                f"    step {s_key}: Choosing Image (path={short}) as answer, "
                "but not correct."
            )
        elif target_type == "object":
            lines.append(
                f"    step {s_key}: Choosing Object {choice} as answer, "
                "but not correct."
            )
        elif target_type == "frontier":
            lines.append(
                f"    step {s_key}: Choosing Frontier {choice} to explore, "
                "but not correct."
            )
        else:
            lines.append(
                f"    step {s_key}: Choosing {target_type} {choice} as answer, "
                "but not correct."
            )
    if not have_decision:
        return ""
    return "\n".join(lines) + "\n"


def format_answerer_prompt(
    question: str,
    pool: List[Dict[str, Any]],
    task_type: str,
    image_goal: Optional[str],
    high_level_plan: Optional[str],
    history_decision: Optional[Dict[str, Any]] = None,
    candidates_block: str = "",
    feedback_block: str = "",
) -> Tuple[str, list]:
    """Answerer prompt (Phase E tri-state for GOAT-Bench).

    Output format: structured tri-state decision.
    - NOT_FOUND: target not visible, continue exploration.
    - CANDIDATE_VISIBLE: target likely visible but not confirmed (small /
      occluded / edge of frame). Creates a TargetCandidate for grounding.
    - TARGET_CONFIRMED: target clearly visible and identifiable. Triggers AVU.

    For GOAT-Bench three task types:
    - object: find the object of the specified category.
    - description: find the object matching the natural-language description.
    - image: find the same object shown in the reference image.
    """
    sys_prompt = (
        "Task: You are an indoor agent that needs to determine if the current "
        "collected information is sufficient to answer the question.\n\n"
        "Instructions:\n"
        "1. CAREFULLY analyze the information needed to answer the question, "
        "especially location, objectives, relationships, and attributes.\n"
        "2. CAREFULLY analyze ALL available images (total observed clues). "
        "Each image shows the view and lists the connected objects with "
        "their IDs and class names.\n"
        "3. Judge based on the task type:\n"
    )
    if task_type == "object":
        sys_prompt += (
            "   - This is an OBJECT task: find the object of the specified "
            "category in the environment.\n"
        )
    elif task_type == "description":
        sys_prompt += (
            "   - This is a DESCRIPTION task: find the object exactly "
            "matching the natural-language description.\n"
        )
    elif task_type == "image":
        sys_prompt += (
            "   - This is an IMAGE task: find the same object shown in the "
            "reference image. Compare the reference image with each image "
            "in the pool.\n"
        )
    else:
        sys_prompt += (
            "   - Unknown task type: use the question to infer the target "
            "and check the images.\n"
        )
    sys_prompt += (
        "4. Output ONE of three decisions:\n"
        "   - NOT_FOUND: no image shows the target or any likely candidate.\n"
        "   - CANDIDATE_VISIBLE: an image shows a LIKELY candidate but it is "
        "small, partially occluded, at the edge of frame, or you are not "
        "fully certain. This will trigger closer-view grounding.\n"
        "   - TARGET_CONFIRMED: an image clearly and centrally shows the "
        "target object — directly visible, identifiable, not just inferred "
        "from common sense.\n"
        "5. TARGET_CONFIRMED is ONLY allowed when ALL of these hold:\n"
        "   - the target's main body is directly visible (not merely "
        "'probably in the room' or 'just outside frame');\n"
        "   - identification is from visual evidence, not common-sense "
        "guessing;\n"
        "   - for small objects, you can point to a specific region of the "
        "image;\n"
        "   - for attribute questions, the attribute itself is visible.\n"
        "   Otherwise use CANDIDATE_VISIBLE (if something plausible is "
        "visible) or NOT_FOUND.\n"
    )

    content = []

    # Question
    text = f"Question: {question}\n"
    if image_goal is not None:
        content.append((text, image_goal))
        content.append(("\n",))
    else:
        content.append((text + "\n",))

    # High-level plan
    if high_level_plan:
        content.append((f"Current High-Level Plan:\n{high_level_plan}\n",))
    else:
        content.append(("No high-level plan yet.\n",))

    # Phase E: active candidates + feedback
    if candidates_block:
        content.append((candidates_block,))
    if feedback_block:
        content.append((feedback_block,))

    # Images
    content.append(("Available Images:\n",))
    if not pool:
        content.append(("No images available\n",))
    else:
        for i, snap in enumerate(pool):
            content.append((f"Image {i}: ", snap["img_b64"]))
            objs_str = _format_connected_objects(snap["connected_objects"])
            content.append((f" connected objects: {objs_str}\n",))

    # F7: CLR - inject history of wrong decisions so Answerer avoids them
    clr_text = _format_clr_block(history_decision)
    if clr_text:
        content.append((clr_text,))

    # Output format
    text = (
        "Output Format:\n"
        "1. First, think step by step and explain your reasoning clearly.\n"
        "2. Then output a structured decision block in EXACTLY this format:\n"
        "Decision: NOT_FOUND | CANDIDATE_VISIBLE | TARGET_CONFIRMED\n"
        "Image: <i>          (image index; omit if NOT_FOUND)\n"
        "Target phrase: <class>   (target category; omit if NOT_FOUND)\n"
        "Visibility:\n"
        "  directly_visible: yes | no\n"
        "  central_enough: yes | no\n"
        "  partially_occluded: yes | no\n"
        "  approximate_location: <short text, e.g. 'right side near lamp'>\n"
        "  confidence: <0.0-1.0>\n"
        "Need action: move closer | rotate | ground with AVU | none\n"
        "\n"
        "Examples:\n"
        "Decision: TARGET_CONFIRMED\n"
        "Image: 3\n"
        "Target phrase: espresso machine\n"
        "...\n"
        "Decision: NOT_FOUND\n"
        "(no Image / Target phrase lines)\n"
    )
    content.append((text,))
    return sys_prompt, content


def format_high_level_planner_prompt(
    question: str,
    task_type: str,
    pool: List[Dict[str, Any]],
    frontier_imgs: List[str],
    high_level_plan_prev: Optional[str],
    is_new_subtask: bool,
    image_goal: Optional[str] = None,
    memory: Optional[Any] = None,
    feedback_block: str = "",
) -> Tuple[str, list]:
    """High-Level Planner prompt.

    Ported from Pred-EQA format_high_level_plan_prompt. Outputs XML
    <update_todo_list>. If is_new_subtask=True, prepends a NEW SUBTASK block
    instructing the planner to discard stale directives and plan fresh.

    Args:
        image_goal: base64 reference image for image-type subtasks.
        memory: TextLongTermMemory instance for retrieving step summaries.
    """
    sys_prompt = (
        "Task: You are a HIGH-LEVEL EXPLORATION PLANNER AGENT responsible for "
        "devising a long-term navigation and search plan to answer the user's "
        "question. Based on the question, you must break down the goal into a "
        "sequence of high-level tasks (e.g., go to a room, find an object, "
        "observe an attribute) and output them as an ordered to-do list. "
        "This plan will guide the low-level agents in subsequent steps.\n\n"
        "Instructions:\n"
        "1. Analyze the user's question and identify its target.\n"
    )
    # M2: GOAT-Bench three-class task_type adaptation
    if task_type == "object":
        sys_prompt += (
            "   - Target: find object of category "
            f"\"{question}\" in the environment.\n"
        )
    elif task_type == "description":
        sys_prompt += (
            "   - Target: find object matching natural language "
            f"description: \"{question}\".\n"
        )
    elif task_type == "image":
        sys_prompt += (
            "   - Target: find object matching the reference image provided.\n"
        )
    else:
        sys_prompt += (
            f"   - Target: find the object described by: \"{question}\".\n"
        )
    sys_prompt += (
        "2. Decompose the question into subgoals. For example:\n"
        "   - Determine which object to find and where it is likely "
        "located.\n"
        "   - Decide which locations or objects need exploration to "
        "understand their spatial arrangement.\n"
        "   - Use typical associations (e.g., kitchen contains a fridge) "
        "to infer where to search.\n"
        "3. For each subgoal, create a clear task (e.g., \"Go to the "
        "kitchen\", \"Find the refrigerator\", \"Check the microwave's door "
        "status\").\n"
        "4. Create Parallel Prediction-Based Branches for the immediate next "
        "step. For the most immediate unresolved navigation or search task, "
        "generate multiple parallel prediction-based exploration branches as "
        "testable hypotheses based on current observations and world "
        "knowledge.\n"
        "Example: instead of [ ] Find the kitchen, create:\n"
        "[ ] Explore the frontier leading to the hallway since it may lead "
        "to the kitchen\n"
        "[ ] Explore the frontier leading to the living area since it may "
        "also lead to the kitchen\n"
        "5. Combine these immediate predictive branches and the remaining "
        "downstream high-level tasks into a single, cohesive, ordered to-do "
        "list. Place the parallel predictive branches at the very top as the "
        "active starting point, followed by the subsequent tasks.\n"
        "6. Use the updateable checklist format for output. Mark tasks as "
        "[ ] pending, [-] in progress, or [x] completed based on what has "
        "been done so far. When agents investigate and eliminate predictive "
        "branches, mark the incorrect or dead-end branches as completed [x] "
        "with a brief inline explanation. Add new tasks immediately when they "
        "become apparent. Do not remove unfinished tasks unless they are "
        "truly irrelevant to the goal.\n"
        "Core Principles:\n"
        "- Before updating, always confirm which todos have been completed or "
        "invalidated since the last update.\n"
        "- You may update multiple statuses in a single update.\n"
        "- Dynamic Replanning: Because the environment is partially "
        "observable, new observations may completely invalidate your previous "
        "assumptions or downstream plans. If this happens, you MUST actively "
        "overhaul the plan.\n"
        "- When a prediction-based branch proves incorrect (a dead-end), OR "
        "when a downstream task becomes obsolete due to a plan overhaul, mark "
        "it as [x] AND append a brief inline comment explaining why it failed "
        "or was discarded.\n"
        "- Once ONE predictive branch successfully locates the target, "
        "immediately mark all other parallel predictive branches for that "
        "same goal as [x] with an explanation.\n"
        "- When a completely new actionable path is discovered that pivots the "
        "entire strategy, add the new tasks immediately and mark the old, "
        "now-irrelevant tasks as [x] with a brief explanation of the pivot.\n"
        "- For regular tasks that remain relevant to the current valid "
        "strategy, only mark them as completed [x] when fully accomplished "
        "successfully.\n"
        "Content Constraints:\n"
        "STRICTLY NEVER mention ANY image or frontier identifiers (e.g., "
        "\"Image 2\", \"Frontier 0\") - these labels are step-specific and "
        "will cause confusion in later steps when the current image is no "
        "longer available.\n"
        "AVOID relative directional references tied to transient views. "
        "Instead, describe spatial relationships using observable objects.\n"
    )
    content = []

    # C1+C2(3): NEW SUBTASK block FIRST (highest priority context)
    if is_new_subtask:
        new_subtask_text = (
            "--- NEW SUBTASK ---\n"
            "Previous subtask completed/failed. New subtask:\n"
            f"Task type: {task_type} (object|description|image)\n"
            f"Question: {question}\n\n"
            "IMPORTANT: The previous plan is from a DIFFERENT subtask with a "
            "DIFFERENT target. You MUST:\n"
            "1. Discard all stale spatial directives from the previous plan "
            "(e.g. \"Go to kitchen\" is irrelevant if new target is not in "
            "kitchen).\n"
            "2. Preserve useful spatial knowledge (rooms visited, object "
            "locations, layout connections) as context.\n"
            "3. Generate a FRESH TODO list for the new target. Mark old "
            "unrelated branches [x] with reason \"irrelevant to new "
            "subtask\".\n"
        )
        content.append((new_subtask_text,))

    content.append((f"Target Question: {question}\n",))
    content.append((f"Task type: {task_type}\n",))

    # M3: inject reference image for image-type subtasks
    if image_goal is not None and task_type == "image":
        content.append(("Reference Image:\n", image_goal))
        content.append(("\n",))

    # C1+C2(1): label previous plan as completed reference when new subtask
    if high_level_plan_prev:
        if is_new_subtask:
            content.append((
                "Previous Subtask Plan (COMPLETED, for reference only):\n"
                f"{high_level_plan_prev}\n",
            ))
        else:
            content.append((
                f"Previous High-Level Plan:\n{high_level_plan_prev}\n",
            ))
    else:
        content.append(("No previous high-level plan.\n",))

    # M4: inject recent step summaries from long-term memory
    if memory is not None:
        try:
            summaries = memory.retrieve_by_type(
                'step_summary_output', top_k=3
            )
            if summaries:
                summary_lines = ["Previous Steps Summary:\n"]
                for s in summaries:
                    summary_lines.append(f"- {s.content}\n")
                content.append(("".join(summary_lines),))
        except Exception:
            pass  # memory may not support retrieve_by_type

    # Images (clues) - images only, no ids per content constraint
    content.append(("Currently observed visual clues:\n",))
    if not pool:
        content.append(("No images available\n",))
    else:
        for snap in pool:
            content.append(("\n", snap["img_b64"]))
        content.append(("\n",))

    # Frontiers
    content.append(("\nAvailable Exploration Directions:\n",))
    if not frontier_imgs:
        content.append(("No frontiers available\n",))
    else:
        for i, img in enumerate(frontier_imgs):
            content.append((f"Frontier {i}: ", img))
            content.append(("\n",))

    # Output format
    text = (
        "Output Format:\n"
        "1. First, think step by step and explain your reasoning clearly.\n"
        "2. Always output your tasks in the following XML checklist format:\n"
        "<update_todo_list>\n"
        "<todos>\n"
        "[ ] Pending task description\n"
        "[-] In progress task description <!-- status; rationale -->\n"
        "[x] Completed or pruned task description <!-- status; rationale -->\n"
        "</todos>\n"
        "</update_todo_list>\n"
    )
    content.append((text,))

    # Phase F: recent failure feedback
    if feedback_block:
        content.append((feedback_block,))

    return sys_prompt, content


def format_executor_prompt(
    question: str,
    frontier_imgs: List[str],
    pool: List[Dict[str, Any]],
    high_level_plan: Optional[str],
    task_type: str,
    history_decision: Optional[Dict[str, Any]] = None,
    frontier_states: Optional[List[Any]] = None,
    feedback_block: str = "",
) -> Tuple[str, list]:
    """Low-Level Executor prompt.

    Ported from Pred-EQA format_explore_prompt. Outputs 'Frontier i' or
    'Stop Exploration'.
    """
    sys_prompt = (
        "Task: You are an indoor agent that needs to PHYSICALLY NAVIGATE "
        "through sequential frontier selections to finally find information "
        "needed for answering the question.\n\n"
        "Instructions:\n"
        "1. Analyze the question's information requirements, especially "
        "locations, objectives, relationships, and attributes. Identify "
        "target objects and their typical locations based on common sense.\n"
        "2. Assess the previously observed clues to determine already "
        "explored areas and objects.\n"
        "3. Given question needs and current exploration progress, choose a "
        "frontier based on the following Core Principles and constraints:\n"
        "principle 1: Use common room-object relationships to infer possible "
        "locations of the target object (e.g., \"refrigerator\" in kitchen, "
        "\"bed\" in bedroom). Use typical room connections to prioritize "
        "exploration directions.\n"
        "principle 2: If you are in an unrelated area, choose the frontier "
        "leading to a potentially relevant area. If previously observed clues "
        "do not suggest that the relevant area has already been explored, "
        "continue exploring without stopping until you reach the relevant "
        "area.\n"
        "principle 3: Balance proximity with strategic long-range exploration "
        "when clues suggest distant frontiers.\n"
        "constraint 1: If you find that you are still in an irrelevant area, "
        "you can only choose a frontier and continue walking in order to "
        "reach the relevant area.\n"
        "constraint 2: You can only access unvisited areas by selecting a "
        "frontier step-by-step.\n"
        "constraint 3: Keep selecting a frontier for moving until you find "
        "conclusive evidence enough to answer the question. Note that the "
        "objects mentioned in all questions are definitely available.\n"
    )
    content = []

    content.append((f"Target Question: {question}\n",))
    content.append((f"Task type: {task_type}\n",))

    if high_level_plan:
        content.append((f"Current High-Level Plan:\n{high_level_plan}\n",))
    else:
        content.append(("No high-level plan yet.\n",))

    # Previously observed clues
    content.append(("Previously Observed Clues:\n",))
    if not pool:
        content.append(("No images available\n",))
    else:
        for snap in pool:
            content.append(("\n", snap["img_b64"]))
        content.append(("\n",))

    # Frontiers
    content.append(("\nAvailable Exploration Directions:\n",))
    if not frontier_imgs:
        content.append(("No frontiers available\n",))
    else:
        for i, img in enumerate(frontier_imgs):
            fs = frontier_states[i] if frontier_states and i < len(frontier_states) else None
            if fs is not None:
                header = f"Frontier {fs.to_prompt_str(local_index=i)}: "
            else:
                header = f"Frontier {i}: "
            content.append((header, img))
            content.append(("\n",))
        ids = [fs.frontier_id for fs in frontier_states] if frontier_states else list(range(len(frontier_imgs)))
        id_strs = ", ".join(f"F_{i:03d}" for i in ids)
        content.append((f"Available Frontier IDs: {id_strs}\n",))

    # F7: CLR - inject history of wrong decisions so Executor avoids them
    clr_text = _format_clr_block(history_decision)
    if clr_text:
        content.append((clr_text,))

    # Phase F: recent failure feedback
    if feedback_block:
        content.append((feedback_block,))

    text = (
        "Output Format:\n"
        "1. First, think step by step and explain your reasoning clearly.\n"
        "2. Then, provide your final answer in the exact format: "
        '"Next Step: Frontier F_XXX" or "Stop Exploration", where F_XXX is '
        "the stable ID of the frontier you choose. Do NOT select frontiers "
        "marked DO NOT SELECT or status=EXPLORED."
    )
    content.append((text,))
    return sys_prompt, content


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def parse_retain_response(
    response: Optional[str],
    keyword: str,
) -> List[int]:
    """Parse 'Retain Images: {i,...}' or 'Retain Frontiers: {i,...}'.

    Args:
        response: raw VLM response.
        keyword: "Images" or "Frontiers".

    Returns:
        Sorted list of retained indices. Empty list on failure.
    """
    if response is None:
        return []
    # Robust to VLM output variants:
    #   "Retain Images: {0, 2}" | "Retain Images: 0, 2" | "Retain Images: {0,2}"
    #   trailing dot/semicolon; missing braces; extra whitespace.
    # Require ':' or '{' after keyword to skip descriptive "retain images that..."
    # in reasoning text. Group only digits/commas/spaces.
    pattern = (
        rf"Retain\s+{re.escape(keyword)}\s*[:：]\s*\{{?\s*"
        r"([\d,\s]+)"
        r"\s*\}?"
    )
    match = re.search(pattern, response, re.IGNORECASE)
    if not match:
        return []
    indices = []
    for tok in match.group(1).split(","):
        tok = tok.strip()
        if tok.isdigit():
            indices.append(int(tok))
    return sorted(set(indices))


def parse_answerer_response(
    response: Optional[str],
) -> Optional[Tuple[str, Optional[int], Optional[str]]]:
    """Parse Answerer tri-state response (Phase E).

    Returns:
        (decision, idx_or_None, class_name_or_None) where decision is one of
        'NOT_FOUND' | 'CANDIDATE_VISIBLE' | 'TARGET_CONFIRMED'.
        Returns ('NOT_FOUND', None, None) on parse failure / Continue.
    """
    if response is None:
        return ("NOT_FOUND", None, None)
    text = response.strip()
    lower = text.lower()

    # Detect decision keyword (case-insensitive). Prefer the last occurrence.
    decision = None
    for m in re.finditer(r"Decision\s*[:：]\s*([A-Z_]+)", text, re.IGNORECASE):
        decision = m.group(1).strip().upper()
    if decision is None:
        # Legacy / freeform: 'continue exploration' → NOT_FOUND
        if "continue exploration" in lower:
            return ("NOT_FOUND", None, None)
        # Legacy 'Image i, <class>' → treat as TARGET_CONFIRMED for back-compat
        pattern = r"(?:Image|Snapshot)\s+(\d+)\s*,\s*(.+?)(?:\n|$)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            idx_str, class_name = matches[-1]
            try:
                idx = int(idx_str)
            except ValueError:
                return ("NOT_FOUND", None, None)
            class_name = re.sub(r"\s*\(.*$", "", class_name.strip().rstrip(".")).strip()
            if class_name:
                return ("TARGET_CONFIRMED", idx, class_name)
        return ("NOT_FOUND", None, None)

    if decision == "NOT_FOUND":
        return ("NOT_FOUND", None, None)

    # CANDIDATE_VISIBLE or TARGET_CONFIRMED: extract Image + Target phrase
    idx = None
    class_name = None
    img_m = re.search(r"Image\s*[:：]\s*(\d+)", text, re.IGNORECASE)
    if img_m:
        try:
            idx = int(img_m.group(1))
        except ValueError:
            idx = None
    phrase_m = re.search(r"Target\s+phrase\s*[:：]\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if phrase_m:
        class_name = re.sub(r"\s*\(.*$", "", phrase_m.group(1).strip().rstrip(".")).strip()
        if not class_name:
            class_name = None
    if idx is None or class_name is None:
        # malformed → treat as NOT_FOUND to avoid bad AVU
        return ("NOT_FOUND", None, None)
    return (decision, idx, class_name)


def parse_executor_response(response: Optional[str]) -> int:
    """Parse Executor response.

    Returns:
        frontier index for 'Frontier i', or -1 for 'Stop Exploration',
        or -1 on parse failure.
    """
    if response is None:
        return -1
    text = response.strip()
    lower = text.lower()
    if "stop exploration" in lower:
        return -1
    # Match 'Next Step: Frontier i' or just 'Frontier i'
    pattern = r"Frontier\s+(\d+)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    if not matches:
        return -1
    try:
        return int(matches[-1])
    except ValueError:
        return -1


def parse_executor_frontier_id(response: Optional[str]) -> int:
    """Parse Executor response for a stable frontier ID.

    Returns:
        frontier_id (int) for 'Frontier F_XXX', or -1 for 'Stop Exploration'
        / parse failure.
    """
    if response is None:
        return -1
    text = response.strip()
    if "stop exploration" in text.lower():
        return -1
    # Match 'F_023' (stable ID). Fall back to bare integer for robustness.
    pattern = r"F_?(\d+)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        try:
            return int(matches[-1])
        except ValueError:
            return -1
    # Legacy fallback: 'Frontier i' with plain index
    pattern = r"Frontier\s+(\d+)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        try:
            return int(matches[-1])
        except ValueError:
            return -1
    return -1


def parse_retain_frontier_ids(
    response: Optional[str],
    valid_ids: Optional[List[int]] = None,
) -> List[int]:
    """Parse 'Retain Frontiers: {F_XXX, F_YYY}' into a list of frontier IDs.

    Args:
        valid_ids: if provided, filter parsed IDs to this set (defensive).
    """
    if response is None:
        return []
    pattern = (
        rf"Retain\s+{re.escape('Frontiers')}\s*[:：]\s*\{{?\s*"
        r"(F_?\d+(?:\s*,\s*F_?\d+|\s*,\s*\d+)*)"
        r"\s*\}?"
    )
    match = re.search(pattern, response, re.IGNORECASE)
    ids: List[int] = []
    if match:
        for tok in re.findall(r"F_?(\d+)|\b(\d+)\b", match.group(1)):
            val = tok[0] or tok[1]
            if val.isdigit():
                ids.append(int(val))
    if not ids:
        # legacy: plain indices 'Retain Frontiers: 0, 2'
        legacy = parse_retain_response(response, "Frontiers")
        ids = legacy
    if valid_ids is not None:
        valid = set(valid_ids)
        ids = [i for i in ids if i in valid]
    return sorted(set(ids))
    """Extract the <update_todo_list>...</update_todo_list> block (or
    <todos>...</todos>) from the planner response. Returns the raw XML
    block string, or None."""
    if response is None:
        return None
    pattern = r"<update_todo_list>([\s\S]*?)</update_todo_list>"
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return f"<update_todo_list>{match.group(1)}</update_todo_list>"
    # Fallback: <todos> block
    pattern = r"<todos>([\s\S]*?)</todos>"
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return f"<todos>{match.group(1)}</todos>"
    # m1: final fallback - extract todo items from plain text and wrap
    todo_list = extract_todo_list_from_text(response)
    if todo_list:
        lines = []
        for item in todo_list:
            status_char = {'pending': '[ ]', 'in_progress': '[-]',
                           'completed': '[x]'}.get(item.get('status', ''), '[ ]')
            lines.append(f"{status_char} {item.get('task', '')}")
        wrapped = "<update_todo_list>\n<todos>\n" + \
                  "\n".join(lines) + "\n</todos>\n</update_todo_list>"
        return wrapped
    return None


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def explore_multi_agent(
    step: Dict[str, Any],
    cfg,
    verbose: bool = False,
) -> Tuple[str, Any, Optional[str], int, Optional[str]]:
    """Multi-agent exploration workflow main entry.

    Runs the 5-agent chain: Image Manager -> Frontier Manager -> Answerer
    -> High-Level Planner -> Executor. Returns the decision for this step.

    Args:
        step: step dict. Expected keys:
            - question, task_type, image (path for image goal), CLR
            - egocentric_imgs (list of b64), frontier_imgs (list of b64)
            - processed_images ({img_key: b64} from KSS)
            - image_map_reverse ({idx: img_key} from KSS)
            - scene (Scene instance)
            - is_new_subtask (bool)
            - high_level_plan (str | None, maintained by main loop)
            - image_pool (list | None, maintained by main loop)
            - step_index (int)
            - prompt_h, prompt_w (int)
        cfg: config namespace.
        verbose: enable verbose logging.

    Returns:
        (target_type, target_index_or_choice, reason, n_filtered, class_name)
        - target_type: 'image' | 'frontier' | 'stop'
        - image: target_index = pool[i].img_path (for query_vlm depth/cam_pose)
        - frontier: target_index = frontier index
        - stop: target_index = None
        - reason: VLM reasoning string (may be "")
        - n_filtered: number of images filtered by Image Manager
        - class_name_if_image: target class name for image answers, else None
    """
    logging.info("[explore_multi_agent] start")

    # a. extract step fields
    question, image_goal = format_question(step)
    task_type = step.get("task_type", "object")
    egocentric_imgs = step.get("egocentric_imgs", [])
    frontier_imgs = step.get("frontier_imgs", [])
    processed_images = step.get("processed_images", {})
    image_map_reverse = step.get("image_map_reverse", {})
    scene = step["scene"]
    is_new_subtask = step.get("is_new_subtask", False)
    step_index = step.get("step_index", 0)
    high_level_plan = step.get("high_level_plan", None)
    history_decision = step.get("CLR", {})
    tsdf_planner = step.get("tsdf_planner", None)
    # Phase B: build frontier_states list aligned with frontier_imgs.
    # frontier_imgs are encoded from tsdf_planner.frontiers in order.
    frontier_states = []
    if tsdf_planner is not None:
        for f in tsdf_planner.frontiers:
            fs = tsdf_planner.frontier_registry.get(f.frontier_id)
            frontier_states.append(fs)
    # Phase B: working memory (SubtaskWorkingMemory) if provided by main loop
    working_memory = step.get("working_memory", None)

    # b. image pool maintenance
    pool = step.get("image_pool", None)
    if is_new_subtask or pool is None:
        # (Re)build pool from KSS processed_images
        pool = build_image_pool_from_kss(
            step_index, scene, processed_images, image_map_reverse
        )
        logging.info(
            f"[explore_multi_agent] built pool from KSS: {len(pool)} images"
        )
    else:
        # Append current-step egocentric views
        add_egocentric_to_pool(pool, egocentric_imgs, step_index)
        logging.info(
            f"[explore_multi_agent] added {len(egocentric_imgs)} egocentric, "
            f"pool size now {len(pool)}"
        )

    # Write pool back so the main loop can carry it across steps
    step["image_pool"] = pool

    n_filtered = 0

    # Phase D: pinned evidence (from active candidates) is forced-kept.
    pinned_paths = set()
    if working_memory is not None:
        pinned_paths = set(working_memory.pinned_ids)
    max_pool = getattr(cfg, "max_pool_size", 6)

    # c. Image Manager — only filters NON-pinned images; pinned are forced-kept.
    # Manager only runs when non-pinned count > 3 (pinned don't need filtering).
    nonpinned_idx = [i for i, s in enumerate(pool) if s.get("img_path") not in pinned_paths]
    if len(nonpinned_idx) > 3:
        sys_p, content = format_image_manager_prompt(
            question, pool, high_level_plan, task_type, image_goal
        )
        if verbose:
            logging.info("[Image Manager] calling VLM")
        raw = call_openai_api(sys_p, content)
        retain_idx = parse_retain_response(raw, "Images")
        if retain_idx:
            # retained set = pinned ∪ (retain_idx ∩ nonpinned)
            retained_nonpinned = set(i for i in retain_idx if i in nonpinned_idx and 0 <= i < len(pool))
            pinned_idx = [i for i in range(len(pool)) if pool[i].get("img_path") in pinned_paths]
            keep_idx = sorted(set(pinned_idx) | retained_nonpinned)
            # cap: if still over max_pool, drop oldest non-pinned first
            if len(keep_idx) > max_pool:
                pinned_set = set(pinned_idx)
                nonpinned_keep = [i for i in keep_idx if i not in pinned_set]
                drop_n = len(keep_idx) - max_pool
                nonpinned_keep = nonpinned_keep[drop_n:]  # drop oldest non-pinned
                keep_idx = sorted(set(pinned_set) | set(nonpinned_keep))
            new_pool = [pool[i] for i in keep_idx]
            n_filtered = len(pool) - len(new_pool)
            if n_filtered > 0 or len(new_pool) != len(pool):
                logging.info(
                    f"[Image Manager] filtered {n_filtered} images, "
                    f"{len(new_pool)} retained (pinned={len(pinned_idx)})"
                )
                pool = new_pool
                step["image_pool"] = pool
        else:
            # Fallback: keep pinned + latest (max_pool - pinned) non-pinned
            pinned_idx = [i for i in range(len(pool)) if pool[i].get("img_path") in pinned_paths]
            nonpinned_idx_cur = [i for i in range(len(pool)) if pool[i].get("img_path") not in pinned_paths]
            keep_nonpinned = nonpinned_idx_cur[-(max_pool - len(pinned_idx)):] if max_pool > len(pinned_idx) else []
            keep_idx = sorted(set(pinned_idx) | set(keep_nonpinned))
            if len(keep_idx) < len(pool):
                logging.info(
                    f"[Image Manager] no valid retain response, "
                    f"fallback keep {len(keep_idx)} (pinned={len(pinned_idx)})"
                )
                pool = [pool[i] for i in keep_idx]
                step["image_pool"] = pool
            else:
                logging.info(
                    "[Image Manager] no valid retain response, keeping pool"
                )
    else:
        logging.info(
            f"[Image Manager] skipped (non-pinned pool size "
            f"{len(nonpinned_idx)} <= 3, pinned={len(pinned_paths)})"
        )

    # d. Frontier Manager (only when frontier count > 1)
    if len(frontier_imgs) > 1:
        sys_p, content = format_frontier_manager_prompt(
            question, frontier_imgs, pool, high_level_plan, task_type,
            frontier_states=frontier_states,
        )
        if verbose:
            logging.info("[Frontier Manager] calling VLM")
        raw = call_openai_api(sys_p, content)
        valid_ids = [fs.frontier_id for fs in frontier_states if fs is not None]
        retain_ids = parse_retain_frontier_ids(raw, valid_ids=valid_ids)
        if retain_ids:
            # map retained frontier_ids back to positional indices
            id_to_pos = {
                fs.frontier_id: i for i, fs in enumerate(frontier_states) if fs is not None
            }
            new_frontier_imgs = [
                frontier_imgs[id_to_pos[fid]]
                for fid in retain_ids
                if fid in id_to_pos and 0 <= id_to_pos[fid] < len(frontier_imgs)
            ]
            if new_frontier_imgs:
                logging.info(
                    f"[Frontier Manager] filtered to "
                    f"{len(new_frontier_imgs)} frontiers (ids={retain_ids})"
                )
                frontier_imgs = new_frontier_imgs
                # keep frontier_states in sync (filter to retained ids)
                retained_set = set(retain_ids)
                frontier_states = [
                    fs for fs in frontier_states
                    if fs is not None and fs.frontier_id in retained_set
                ]
        else:
            logging.info(
                "[Frontier Manager] no valid retain response, keeping all"
            )
    else:
        logging.info(
            f"[Frontier Manager] skipped (frontier count "
            f"{len(frontier_imgs)} <= 1)"
        )

    # e. Answerer (Phase E: tri-state)
    candidates_block = ""
    feedback_block = ""
    if working_memory is not None:
        candidates_block = working_memory.candidates_prompt_block()
        feedback_block = working_memory.feedback_prompt_block()
    sys_p, content = format_answerer_prompt(
        question, pool, task_type, image_goal, high_level_plan,
        history_decision=history_decision,
        candidates_block=candidates_block,
        feedback_block=feedback_block,
    )
    if verbose:
        logging.info("[Answerer] calling VLM")
    raw = call_openai_api(sys_p, content)
    decision, idx, class_name = parse_answerer_response(raw)
    reason = _extract_reason(raw)
    if decision in ("CANDIDATE_VISIBLE", "TARGET_CONFIRMED") and idx is not None and 0 <= idx < len(pool):
        img_path = pool[idx]["img_path"]
        _record_step_summary(
            step,
            f"Answerer {decision} Image {idx} ({img_path}), class={class_name}"
        )
        logging.info(
            f"[Answerer] {decision} Image {idx} ({img_path}), class={class_name}"
        )
        return ("image", img_path, reason, n_filtered, class_name)
    elif decision in ("CANDIDATE_VISIBLE", "TARGET_CONFIRMED"):
        logging.info(
            f"[Answerer] {decision} but index {idx} out of pool range "
            f"{len(pool)}, falling through to exploration"
        )
    else:
        logging.info("[Answerer] NOT_FOUND, continue exploration")

    # f. High-Level Planner
    sys_p, content = format_high_level_planner_prompt(
        question,
        task_type,
        pool,
        frontier_imgs,
        high_level_plan,
        is_new_subtask,
        image_goal=image_goal,
        memory=step.get("episode_memory"),
        feedback_block=feedback_block,
    )
    if verbose:
        logging.info("[High-Level Planner] calling VLM")
    raw = call_openai_api(sys_p, content)
    new_plan = _parse_high_level_plan_response(raw)
    if new_plan:
        high_level_plan = new_plan
        step["high_level_plan"] = high_level_plan
        logging.info("[High-Level Planner] plan updated")
    else:
        logging.info("[High-Level Planner] no valid plan block, keeping old")

    # g. Executor
    sys_p, content = format_executor_prompt(
        question, frontier_imgs, pool, high_level_plan, task_type,
        history_decision=history_decision,
        frontier_states=frontier_states,
        feedback_block=feedback_block,
    )
    if verbose:
        logging.info("[Executor] calling VLM")
    raw = call_openai_api(sys_p, content)
    frontier_id = parse_executor_frontier_id(raw)
    reason = _extract_reason(raw)
    # map frontier_id -> positional index in the (possibly filtered) list
    id_to_pos = {
        fs.frontier_id: i for i, fs in enumerate(frontier_states) if fs is not None
    }
    if frontier_id >= 0 and frontier_id in id_to_pos:
        pos = id_to_pos[frontier_id]
        logging.info(f"[Executor] Frontier F_{frontier_id:03d} (pos {pos})")
        _record_step_summary(step, f"Executor chose Frontier F_{frontier_id:03d}")
        # Phase B: mark selected in registry + working memory
        if tsdf_planner is not None:
            tsdf_planner.mark_frontier_selected(frontier_id)
        if working_memory is not None:
            working_memory.mark_frontier_selected(frontier_id)
        return ("frontier", pos, reason, n_filtered, None)
    else:
        # 'Stop Exploration' or parse failure
        logging.info("[Executor] Stop Exploration")
        _record_step_summary(step, "Executor Stop Exploration")
        return ("stop", None, reason, n_filtered, None)


def _extract_reason(response: Optional[str]) -> str:
    """Best-effort extraction of reasoning text after the answer line.

    Returns everything after the first line as a single string, or "" if
    no reasoning is present.
    """
    if response is None:
        return ""
    lines = [l for l in response.strip().split("\n") if l.strip()]
    if len(lines) <= 1:
        return ""
    return " ".join(lines[1:])


def _record_step_summary(step: Dict[str, Any], summary: str) -> None:
    """Record a step summary into episode_memory (best-effort, never raises).

    F4: step_summary_output entries were retrieved by the High-Level Planner
    (M4 injection) but never written, making the injection dead code. This
    helper closes the loop by writing one summary per step.
    """
    memory = step.get("episode_memory")
    if memory is None or not hasattr(memory, "add_entry"):
        return
    try:
        step_index = step.get("step_index", 0)
        memory.add_entry(
            content=summary,
            importance=0.5,
            entry_type="step_summary_output",
            step=step_index,
        )
    except Exception:
        pass
