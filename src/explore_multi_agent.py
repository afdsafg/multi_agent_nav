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
) -> Tuple[str, list]:
    """Frontier Manager prompt.

    Ported from Pred-EQA format_plan_manager_prompt. Outputs
    'Retain Frontiers: {i,...}'.
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
            content.append((f"Frontier {i}: ", img))
            content.append(("\n",))
        if len(frontier_imgs) == 1:
            content.append(("Available Frontier indices: 0\n",))
        else:
            content.append((
                f"Available Frontier indices: 0-{len(frontier_imgs) - 1}\n",
            ))

    text = (
        "Output Format:\n"
        "1. First, think step by step and explain your reasoning clearly.\n"
        '2. Then, provide your final answer in the exact format: '
        '"Retain Frontiers: {i, ...}" (retain at least 1 frontier).'
    )
    content.append((text,))
    return sys_prompt, content


def format_answerer_prompt(
    question: str,
    pool: List[Dict[str, Any]],
    task_type: str,
    image_goal: Optional[str],
    high_level_plan: Optional[str],
) -> Tuple[str, list]:
    """Answerer prompt (rewritten for GOAT-Bench).

    Output format: 'Image i, <class>' (i = pool index, <class> = target
    category name) or 'Continue Exploration'.

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
            "category in the environment. If any image clearly contains "
            "the target object, report it.\n"
        )
    elif task_type == "description":
        sys_prompt += (
            "   - This is a DESCRIPTION task: find the object exactly "
            "matching the natural-language description. If any image "
            "contains an object matching the description, report it.\n"
        )
    elif task_type == "image":
        sys_prompt += (
            "   - This is an IMAGE task: find the same object shown in the "
            "reference image. Compare the reference image with each image "
            "in the pool and report the matching one.\n"
        )
    else:
        sys_prompt += (
            "   - Unknown task type: use the question to infer the target "
            "and check the images.\n"
        )
    sys_prompt += (
        "4. If ANY image contains information sufficient to identify the "
        "target object, output the image index and the target category "
        "name.\n"
        "5. If NO image provides sufficient information, output Continue "
        "Exploration.\n"
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

    # Images
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
        "2. If answerable, provide your final answer in the EXACT format: "
        '"Image i, <class>" where i is the image index and <class> is '
        "the target object category name. Example: Image 3, espresso "
        "machine\n"
        'If not answerable, use format: "Continue Exploration"'
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
) -> Tuple[str, list]:
    """High-Level Planner prompt.

    Ported from Pred-EQA format_high_level_plan_prompt. Outputs XML
    <update_todo_list>. If is_new_subtask=True, appends a NEW SUBTASK block
    instructing the planner to reorganize memory and plan.
    """
    sys_prompt = (
        "Task: You are a HIGH-LEVEL EXPLORATION PLANNER AGENT responsible for "
        "devising a long-term navigation and search plan to answer the user's "
        "question. Based on the question, you must break down the goal into a "
        "sequence of high-level tasks (e.g., go to a room, find an object, "
        "observe an attribute) and output them as an ordered to-do list. "
        "This plan will guide the low-level agents in subsequent steps.\n\n"
        "Instructions:\n"
        "1. Analyze the user's question and identify its type (object "
        "recognition, attribute recognition, spatial relationship, object "
        "state, functional reasoning, world knowledge, or object "
        "localization).\n"
        "2. Decompose the question into subgoals. For example:\n"
        "   - Object recognition: Determine which object to find and where "
        "it is likely located.\n"
        "   - Attribute recognition: Identify the object and which attribute "
        "to check.\n"
        "   - Spatial understanding: Decide which locations or objects need "
        "exploration to understand their spatial arrangement.\n"
        "   - Object state recognition: Determine which object's state to "
        "verify and how to observe it.\n"
        "   - Functional reasoning: Identify relevant objects that "
        "demonstrate the function in question.\n"
        "   - World knowledge: Use typical associations (e.g., kitchen "
        "contains a fridge) to infer where to search.\n"
        "   - Object localization: Plan a search sequence for locating the "
        "object in different rooms.\n"
        "3. For each subgoal, create a clear task (e.g., \"Go to the "
        "kitchen\", \"Find the refrigerator\", \"Check the microwave's door "
        "status\").\n"
        "4. Create Parallel Prediction-Based Branches for the immediate next "
        "step. For the most immediate unresolved navigation or search task, "
        "generate multiple parallel prediction-based exploration branches as "
        "testable hypotheses based on current observations and world "
        "knowledge.\n"
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

    content.append((f"Target Question: {question}\n",))
    content.append((f"Task type: {task_type}\n",))

    # Previous plan
    if high_level_plan_prev:
        content.append((
            f"Previous High-Level Plan:\n{high_level_plan_prev}\n",
        ))
    else:
        content.append(("No previous high-level plan.\n",))

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

    # New subtask injection
    if is_new_subtask:
        new_subtask_text = (
            "--- NEW SUBTASK ---\n"
            "Previous subtask completed/failed. "
            f"New subtask: task_type={task_type}, question={question}. "
            "Please review and reorganize your memory and plan for this new "
            "target."
        )
        content.append((new_subtask_text,))

    return sys_prompt, content


def format_executor_prompt(
    question: str,
    frontier_imgs: List[str],
    pool: List[Dict[str, Any]],
    high_level_plan: Optional[str],
    task_type: str,
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
            content.append((f"Frontier {i}: ", img))
            content.append(("\n",))
        if len(frontier_imgs) == 1:
            content.append(("Available Frontier indices: 0\n",))
        else:
            content.append((
                f"Available Frontier indices: 0-{len(frontier_imgs) - 1}\n",
            ))

    text = (
        "Output Format:\n"
        "1. First, think step by step and explain your reasoning clearly.\n"
        "2. Then, provide your final answer in the exact format: "
        '"Next Step: Frontier i" or "Stop Exploration", where i is the index '
        "of the frontier you choose."
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
    pattern = rf"Retain\s+{re.escape(keyword)}\s*:\s*\{{([^}}]*)\}}"
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
) -> Optional[Tuple[int, str]]:
    """Parse Answerer response.

    Returns:
        (pool_index, class_name) for 'Image i, <class>', or None for
        'Continue Exploration'. Returns None on parse failure too.
    """
    if response is None:
        return None
    text = response.strip()
    # Take the last non-empty line that looks like an answer (after reasoning)
    lower = text.lower()
    if "continue exploration" in lower:
        return None
    # Match 'Image i, <class>' or 'Snapshot i, <class>' (case-insensitive)
    pattern = r"(?:Image|Snapshot)\s+(\d+)\s*,\s*(.+?)(?:\n|$)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    if not matches:
        return None
    idx_str, class_name = matches[-1]
    try:
        idx = int(idx_str)
    except ValueError:
        return None
    class_name = class_name.strip().rstrip(".")
    # Strip trailing reasoning in parentheses
    class_name = re.sub(r"\s*\(.*$", "", class_name).strip()
    if not class_name:
        return None
    return idx, class_name


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


def _parse_high_level_plan_response(response: Optional[str]) -> Optional[str]:
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

    # c. Image Manager (only when pool length > 3)
    if len(pool) > 3:
        sys_p, content = format_image_manager_prompt(
            question, pool, high_level_plan, task_type, image_goal
        )
        if verbose:
            logging.info("[Image Manager] calling VLM")
        raw = call_openai_api(sys_p, content)
        retain_idx = parse_retain_response(raw, "Images")
        if retain_idx:
            new_pool = [pool[i] for i in retain_idx if 0 <= i < len(pool)]
            n_filtered = len(pool) - len(new_pool)
            if n_filtered > 0:
                logging.info(
                    f"[Image Manager] filtered {n_filtered} images, "
                    f"{len(new_pool)} retained"
                )
                pool = new_pool
                step["image_pool"] = pool
        else:
            logging.info(
                "[Image Manager] no valid retain response, keeping pool"
            )
    else:
        logging.info(
            f"[Image Manager] skipped (pool size {len(pool)} <= 3)"
        )

    # d. Frontier Manager (only when frontier count > 1)
    if len(frontier_imgs) > 1:
        sys_p, content = format_frontier_manager_prompt(
            question, frontier_imgs, pool, high_level_plan, task_type
        )
        if verbose:
            logging.info("[Frontier Manager] calling VLM")
        raw = call_openai_api(sys_p, content)
        retain_frontier_idx = parse_retain_response(raw, "Frontiers")
        if retain_frontier_idx:
            new_frontier_imgs = [
                frontier_imgs[i]
                for i in retain_frontier_idx
                if 0 <= i < len(frontier_imgs)
            ]
            if new_frontier_imgs:
                logging.info(
                    f"[Frontier Manager] filtered to "
                    f"{len(new_frontier_imgs)} frontiers"
                )
                frontier_imgs = new_frontier_imgs
        else:
            logging.info(
                "[Frontier Manager] no valid retain response, keeping all"
            )
    else:
        logging.info(
            f"[Frontier Manager] skipped (frontier count "
            f"{len(frontier_imgs)} <= 1)"
        )

    # e. Answerer
    sys_p, content = format_answerer_prompt(
        question, pool, task_type, image_goal, high_level_plan
    )
    if verbose:
        logging.info("[Answerer] calling VLM")
    raw = call_openai_api(sys_p, content)
    answer = parse_answerer_response(raw)
    if answer is not None:
        idx, class_name = answer
        if 0 <= idx < len(pool):
            img_path = pool[idx]["img_path"]
            reason = _extract_reason(raw)
            logging.info(
                f"[Answerer] Image {idx} ({img_path}), class={class_name}"
            )
            return ("image", img_path, reason, n_filtered, class_name)
        else:
            logging.info(
                f"[Answerer] index {idx} out of pool range {len(pool)}, "
                "falling through to exploration"
            )
    else:
        logging.info("[Answerer] Continue Exploration")

    # f. High-Level Planner
    sys_p, content = format_high_level_planner_prompt(
        question,
        task_type,
        pool,
        frontier_imgs,
        high_level_plan,
        is_new_subtask,
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
        question, frontier_imgs, pool, high_level_plan, task_type
    )
    if verbose:
        logging.info("[Executor] calling VLM")
    raw = call_openai_api(sys_p, content)
    frontier_idx = parse_executor_response(raw)
    reason = _extract_reason(raw)
    if frontier_idx >= 0 and frontier_idx < len(frontier_imgs):
        logging.info(f"[Executor] Frontier {frontier_idx}")
        return ("frontier", frontier_idx, reason, n_filtered, None)
    else:
        # 'Stop Exploration' or parse failure
        logging.info("[Executor] Stop Exploration")
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
