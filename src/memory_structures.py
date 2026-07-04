"""Graph-grounded Predictive Navigation memory structures.

Implements the three memory layers from 诊断.md:
- TargetCandidate: VLM-confirmed target with multi-level grounding state
- FrontierState: stable frontier identity with exploration status
- FeedbackEvent: failure reason that closes the loop back to agents
- SubtaskWorkingMemory: per-subtask scratch space (reset each subtask,
  long-term scene graph / all_observations live elsewhere and are NOT
  cleared here)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Navigation state enums (deep-research-report §TargetCandidateMemory)
# ---------------------------------------------------------------------------
class NavTargetKind(Enum):
    FRONTIER = auto()
    EVIDENCE_POSE = auto()
    VISUAL_APPROACH_POSE = auto()
    VIEWPOINT_POSE = auto()


class NavStatus(Enum):
    NONE = auto()
    PLANNED = auto()
    IN_PROGRESS = auto()
    REACHED = auto()
    FAILED = auto()
    STALE = auto()


class CandidateState(Enum):
    """Extended candidate state lifecycle (report §TargetCandidateMemory).
    Not wired into existing str-based status yet — kept for reference/TTL."""
    VISUAL_ONLY = auto()
    GROUNDED_2D = auto()
    GROUNDED_3D = auto()
    NEED_CLOSER_VIEW = auto()
    VIEWPOINT_READY = auto()
    VERIFY_PENDING = auto()
    SUCCEEDED = auto()
    REJECTED = auto()
    EXPIRED = auto()


# ---------------------------------------------------------------------------
# Target candidate
# ---------------------------------------------------------------------------
TARGET_PHRASE_ALIASES: Dict[str, List[str]] = {
    "carpet": ["rug", "floor mat", "mat", "floor covering"],
    "rug": ["carpet", "floor mat", "mat", "floor covering"],
    "ficus tree": ["ficus", "potted plant", "plant", "indoor plant", "green plant"],
    "ficus": ["ficus tree", "potted plant", "plant", "indoor plant"],
    "potted plant": ["plant", "indoor plant", "ficus", "ficus tree", "green plant"],
    "plant": ["potted plant", "indoor plant", "ficus", "ficus tree", "green plant"],
    "sofa": ["couch", "settee", "loveseat"],
    "couch": ["sofa", "settee", "loveseat"],
    "tv": ["television", "monitor", "screen", "tv monitor"],
    "television": ["tv", "monitor", "screen"],
    "fridge": ["refrigerator", "cooler"],
    "refrigerator": ["fridge", "cooler"],
    "stove": ["range", "oven", "cooktop"],
    "oven": ["stove", "range", "cooktop"],
    "desk": ["table", "workstation"],
    "table": ["desk"],
    "bed": ["mattress", "cot"],
    "chair": ["seat", "stool"],
    "stool": ["chair", "seat"],
    "cabinet": ["cupboard", "locker"],
    "cupboard": ["cabinet", "locker"],
    "bathtub": ["tub", "bath"],
    "tub": ["bathtub", "bath"],
    "sink": ["basin", "washbasin"],
    "basin": ["sink", "washbasin"],
    "toilet": ["commode", "wc"],
    "door": ["doorway", "entrance"],
    "shelf": ["bookshelf", "rack", "shelving"],
    "bookshelf": ["shelf", "bookcase", "rack"],
    "lamp": ["light", "light fixture", "light source"],
    "light": ["lamp", "light fixture", "light source"],
    "picture": ["painting", "frame", "artwork"],
    "painting": ["picture", "frame", "artwork"],
    "clock": ["wall clock", "timepiece"],
    "curtain": ["drape", "drapes", "curtains"],
    "counter": ["countertop", "worktop"],
    "countertop": ["counter", "worktop"],
    "fireplace": ["hearth"],
    "dishwasher": ["dish washer"],
    "microwave": ["microwave oven"],
    "washer": ["washing machine", "washingmachine"],
    "washing machine": ["washer"],
    "dryer": ["drying machine", "tumble dryer"],
    "air conditioner": ["ac", "ac unit", "air conditioning"],
    "heater": ["radiator", "heating"],
    "coffee maker": ["coffee machine", "coffeemaker"],
    "coffee machine": ["coffee maker", "coffeemaker"],
    "toaster": ["toaster oven"],
    "blender": ["mixer", "liquidiser"],
    "vase": ["urn", "jar"],
    "jar": ["vase", "urn"],
    "bin": ["trash can", "garbage can", "wastebin", "dustbin"],
    "trash can": ["bin", "garbage can", "wastebin", "dustbin"],
    "garbage can": ["bin", "trash can", "wastebin", "dustbin"],
    "pillow": ["cushion"],
    "cushion": ["pillow"],
    "blanket": ["throw", "quilt"],
    "towel": ["rag"],
    "mirror": ["looking glass"],
    "whiteboard": ["white board", "dry erase board"],
    "blackboard": ["chalkboard", "black board"],
    "keyboard": ["keypad"],
    "mouse": ["computer mouse"],
    "remote": ["remote control", "clicker"],
    "fan": ["ceiling fan", "exhaust fan"],
    "printer": ["scanner", "multifunction printer"],
    "speaker": ["loudspeaker", "stereo speaker"],
    "guitar": ["acoustic guitar", "electric guitar"],
    "piano": ["keyboard instrument", "upright piano", "grand piano"],
    "bicycle": ["bike"],
    "bike": ["bicycle"],
    "machine": ["device", "appliance"],
    "appliance": ["machine", "device"],
    "instrument": ["tool", "device"],
}


def get_aliases(target_phrase: str) -> List[str]:
    """Return alias list for a target phrase (lowercased, deduped)."""
    key = target_phrase.strip().lower()
    aliases = list(TARGET_PHRASE_ALIASES.get(key, []))
    # also include the original phrase so Level 2 tries it too
    if key not in aliases:
        aliases = [key] + aliases
    return aliases


# ---------------------------------------------------------------------------
# Grounding2D — Ultralytics-free intermediate (report §Visual Candidate Approach)
# ---------------------------------------------------------------------------
@dataclass
class Grounding2D:
    source: str               # "yolo_target" | "yolo_alias" | "class_agnostic_clip"
    phrase: str
    bbox_xyxy: Any            # np.ndarray shape (4,) int
    mask: Optional[Any] = None
    raw_score: float = 0.0
    rank_score: Optional[float] = None
    conf: float = 0.0


@dataclass
class ApproachPose:
    xyz: Any
    yaw: float
    source: str
    valid_depth_ratio: float


# ---------------------------------------------------------------------------
# Target candidate
# ---------------------------------------------------------------------------
# Status values
S_GROUNDED_3D = "GROUNDED_3D"            # bbox/mask/point cloud obtained
S_VISUAL_ONLY = "VISUAL_ONLY"            # VLM saw it, detector failed
S_NEED_CLOSER_VIEW = "NEED_CLOSER_VIEW"  # visible but small/occluded
S_REJECTED = "REJECTED"                  # VLM re-confirmed: not the target


@dataclass
class GroundingAttempt:
    level: int            # 1=YOLO target, 2=YOLO alias, 3=class-agnostic+CLIP, 4=evidence-pose
    success: bool
    reason: str = ""


def should_release(c: "TargetCandidate", step: int, ttl_steps: int = 12,
                   max_closer_views: int = 3) -> bool:
    """Report §TargetCandidateMemory release gate. Only reject/expire/cap
    release; success release is handled by release_after_navigation."""
    if c.status == S_REJECTED:
        return True
    if c.closer_view_attempts >= max_closer_views:
        return True
    if step - c.source_step > ttl_steps and c.status == S_VISUAL_ONLY:
        return True
    return False


@dataclass
class TargetCandidate:
    candidate_id: str
    subtask_id: str
    image_path: str
    source_step: int
    camera_pose: Any               # 4x4 numpy array (world->cam) — from scene.all_cam_poses
    view_yaw: float                # camera yaw when the evidence was captured
    target_phrase: str
    aliases: List[str] = field(default_factory=list)
    status: str = S_VISUAL_ONLY    # newly created = VLM saw, not yet grounded
    grounding_attempts: List[GroundingAttempt] = field(default_factory=list)
    pinned: bool = True
    last_failure_reason: Optional[str] = None
    closer_view_attempts: int = 0  # how many times we navigated closer and re-tried
    # Navigation target tracking (report §AVU→VVD→task_check)
    nav_target_kind: Optional[NavTargetKind] = None
    nav_status: NavStatus = NavStatus.NONE
    nav_goal_xyz: Optional[Any] = None       # habitat (x,y,z)
    nav_goal_yaw: Optional[float] = None

    def record_attempt(self, level: int, success: bool, reason: str = "") -> None:
        self.grounding_attempts.append(
            GroundingAttempt(level=level, success=success, reason=reason)
        )
        if success:
            self.status = S_GROUNDED_3D
        elif level == 4 and not success:
            self.status = S_NEED_CLOSER_VIEW
            self.closer_view_attempts += 1
        else:
            self.last_failure_reason = reason

    def release(self) -> None:
        """Mark as no longer pinned (candidate resolved or definitively rejected)."""
        self.pinned = False

    def to_prompt_str(self) -> str:
        return (
            f"  - candidate {self.candidate_id} (image={self.image_path}, "
            f"phrase='{self.target_phrase}', status={self.status}, "
            f"closer_views={self.closer_view_attempts}, "
            f"last_failure={self.last_failure_reason or 'none'})"
        )


# ---------------------------------------------------------------------------
# Frontier state
# ---------------------------------------------------------------------------
# Status values
F_ACTIVE = "ACTIVE"
F_EXPLORED = "EXPLORED"
F_BLOCKED = "BLOCKED"
F_STALE = "STALE"

# last_result values
FR_NO_NEW_INFO = "NO_NEW_INFO"
FR_BLOCKED = "BLOCKED"
FR_LED_TO_ROOM = "LED_TO_ROOM"
FR_FOUND_CANDIDATE = "FOUND_CANDIDATE"


@dataclass
class FrontierState:
    frontier_id: int               # matches Frontier.frontier_id in tsdf_planner
    centroid: np.ndarray           # (x, y, z) world
    area: float
    view_yaw: float
    first_seen_step: int
    last_seen_step: int
    selected_count: int = 0
    reached_count: int = 0
    status: str = F_ACTIVE
    last_result: Optional[str] = None

    def to_prompt_str(self, local_index: Optional[int] = None) -> str:
        prefix = f"F_{self.frontier_id:03d}"
        if local_index is not None:
            prefix += f" (display {local_index})"
        parts = [
            prefix,
            f"status={self.status}",
            f"selected={self.selected_count}",
            f"reached={self.reached_count}",
        ]
        if self.last_result:
            parts.append(f"last={self.last_result}")
        if self.status == F_EXPLORED or self.selected_count >= 2:
            parts.append("DO NOT SELECT")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Feedback event
# ---------------------------------------------------------------------------
FB_TASK_CHECK_FAIL = "TASK_CHECK_FAIL"
FB_TASK_CHECK_PASS = "TASK_CHECK_PASS"
FB_AVU_FAIL = "AVU_FAIL"
FB_AVU_VISUAL_ONLY = "AVU_VISUAL_ONLY"
FB_FRONTIER_NO_INFO = "FRONTIER_NO_INFO"
FB_PLANNER_STALE = "PLANNER_STALE"
FB_WRONG_INSTANCE = "WRONG_INSTANCE"


@dataclass
class FeedbackEvent:
    step: int
    type: str
    reason: str
    target_candidate_id: Optional[str] = None
    frontier_id: Optional[int] = None
    suggested_fix: str = ""
    ttl_steps: int = 4
    created_step: int = 0

    def to_prompt_str(self) -> str:
        s = f"- Step {self.step} [{self.type}]: {self.reason}"
        if self.suggested_fix:
            s += f" → {self.suggested_fix}"
        return s


# ---------------------------------------------------------------------------
# Subtask working memory
# ---------------------------------------------------------------------------
class SubtaskWorkingMemory:
    """Per-subtask scratch space.

    Reset at the start of every subtask. Long-term memory
    (scene.all_observations / scene.objects / scene.edges / M3DSG)
    lives on the scene object and is NOT touched here.
    """

    def __init__(self) -> None:
        self.subtask_id: str = ""
        self.question: str = ""
        self.pinned_ids: set = set()             # image paths that cannot be dropped
        self.target_candidates: Dict[str, TargetCandidate] = {}
        self.frontier_registry: Dict[int, FrontierState] = {}
        self.feedback: List[FeedbackEvent] = []
        self.plan_stale_count: int = 0
        self.high_level_plan: Optional[str] = None
        self.last_plan_normalized: Optional[str] = None
        self.recent_frontier_ids: List[int] = []  # ordered, for recent_window
        self._last_nav_candidate_id: Optional[str] = None

    # ---- subtask lifecycle -------------------------------------------------
    def reset_for_new_subtask(
        self,
        subtask_id: str,
        question: str,
    ) -> None:
        """Clear working memory; keep frontier_registry geometry but mark STALE.

        Frontier geometry is still valid (walls don't move between subtasks in
        the same episode), but their exploration status resets because the
        target changed — a frontier that gave no info for object A may still
        be useful for object B. We mark STALE so they can be re-considered but
        are not preferred.
        """
        self.subtask_id = subtask_id
        self.question = question
        self.pinned_ids = set()
        self.target_candidates = {}
        self.feedback = []
        self.plan_stale_count = 0
        self.high_level_plan = None
        self.last_plan_normalized = None
        self.recent_frontier_ids = []
        self._last_nav_candidate_id = None
        for fs in self.frontier_registry.values():
            fs.status = F_STALE
            fs.selected_count = 0
            fs.reached_count = 0
            fs.last_result = None
            # keep centroid/area/view_yaw/first_seen_step

    # ---- candidates --------------------------------------------------------
    def get_or_create_candidate(
        self,
        image_path: str,
        target_phrase: str,
        source_step: int,
        camera_pose: Any,
        view_yaw: float,
    ) -> TargetCandidate:
        """Reuse an existing VISUAL_ONLY candidate for the same image+phrase,
        otherwise create a new one. New candidates are pinned by default."""
        for c in self.target_candidates.values():
            if (
                c.image_path == image_path
                and c.target_phrase.lower() == target_phrase.lower()
                and c.status not in (S_REJECTED, S_GROUNDED_3D)
            ):
                return c
        cid = f"C{len(self.target_candidates):03d}"
        cand = TargetCandidate(
            candidate_id=cid,
            subtask_id=self.subtask_id,
            image_path=image_path,
            source_step=source_step,
            camera_pose=camera_pose,
            view_yaw=view_yaw,
            target_phrase=target_phrase,
            aliases=get_aliases(target_phrase),
        )
        self.target_candidates[cid] = cand
        self.pinned_ids.add(image_path)
        return cand

    def reject_candidate(self, candidate_id: str, reason: str) -> None:
        c = self.target_candidates.get(candidate_id)
        if c is None:
            return
        c.status = S_REJECTED
        c.pinned = False
        c.last_failure_reason = reason
        self.pinned_ids.discard(c.image_path)

    def grounded_candidate(self, candidate_id: str) -> None:
        c = self.target_candidates.get(candidate_id)
        if c is None:
            return
        c.status = S_GROUNDED_3D
        # keep pinned until navigation succeeds — VVD still needs the image

    def release_candidate(self, candidate_id: str) -> None:
        c = self.target_candidates.get(candidate_id)
        if c is None:
            return
        c.release()
        self.pinned_ids.discard(c.image_path)

    def release_after_navigation(self, candidate_id: str) -> None:
        """§2 release condition 1: GROUNDED_3D + navigation succeeded."""
        self.release_candidate(candidate_id)

    def reject_candidate_by_vlm(self, image_path: str, reason: str) -> None:
        """§2 release condition 2: VLM re-confirms target NOT visible.
        Find any active (non-rejected, non-released) candidate for the image
        and reject it."""
        for c in self.target_candidates.values():
            if c.image_path == image_path and c.status not in (S_REJECTED,) and c.pinned:
                self.reject_candidate(c.candidate_id, reason)
                return

    def check_closer_view_limit(self, candidate_id: str, max_attempts: int) -> bool:
        """§2 release condition 3: N closer-view attempts failed → release.
        Returns True if limit reached (and candidate released)."""
        c = self.target_candidates.get(candidate_id)
        if c is None:
            return False
        if c.closer_view_attempts >= max_attempts:
            self.release_candidate(candidate_id)
            return True
        return False

    def active_candidates(self) -> List[TargetCandidate]:
        return [
            c for c in self.target_candidates.values()
            if c.status in (S_VISUAL_ONLY, S_NEED_CLOSER_VIEW)
        ]

    # ---- nav candidate tracking (report §AVU→VVD→task_check) --------------
    def set_last_nav_candidate(self, candidate_id: str) -> None:
        self._last_nav_candidate_id = candidate_id

    def get_last_nav_candidate(self) -> Optional[TargetCandidate]:
        cid = self._last_nav_candidate_id
        return self.target_candidates.get(cid) if cid else None

    # ---- frontiers ---------------------------------------------------------
    def upsert_frontier(
        self,
        frontier_id: int,
        centroid: np.ndarray,
        area: float,
        view_yaw: float,
        step: int,
    ) -> FrontierState:
        fs = self.frontier_registry.get(frontier_id)
        if fs is None:
            fs = FrontierState(
                frontier_id=frontier_id,
                centroid=centroid.copy(),
                area=area,
                view_yaw=view_yaw,
                first_seen_step=step,
                last_seen_step=step,
            )
            self.frontier_registry[frontier_id] = fs
        else:
            fs.last_seen_step = step
            fs.centroid = centroid.copy()
            fs.area = area
            fs.view_yaw = view_yaw
            if fs.status == F_STALE:
                fs.status = F_ACTIVE
        return fs

    def mark_frontier_selected(self, frontier_id: int) -> None:
        fs = self.frontier_registry.get(frontier_id)
        if fs is not None:
            fs.selected_count += 1
            if frontier_id not in self.recent_frontier_ids:
                self.recent_frontier_ids.append(frontier_id)

    def mark_frontier_reached(
        self,
        frontier_id: int,
        result: str = FR_NO_NEW_INFO,
    ) -> None:
        fs = self.frontier_registry.get(frontier_id)
        if fs is not None:
            fs.reached_count += 1
            fs.last_result = result
            if result == FR_NO_NEW_INFO:
                fs.status = F_EXPLORED
            elif result == FR_FOUND_CANDIDATE:
                fs.status = F_ACTIVE  # keep it available
            # LED_TO_ROOM / BLOCKED: leave status as-is for now

    def get_valid_frontiers(
        self,
        all_frontier_ids: List[int],
        max_reselect: int = 2,
        recent_window: int = 3,
    ) -> List[int]:
        """Return frontier_ids that are selectable right now."""
        recent = set(self.recent_frontier_ids[-recent_window:]) if recent_window > 0 else set()
        out = []
        for fid in all_frontier_ids:
            fs = self.frontier_registry.get(fid)
            if fs is None:
                out.append(fid)  # unknown → treat as fresh
                continue
            if fs.status == F_EXPLORED:
                continue
            if fs.status == F_BLOCKED:
                continue
            if fs.selected_count >= max_reselect:
                continue
            if fid in recent:
                continue
            out.append(fid)
        return out

    def prune_recent(self, window: int = 3) -> None:
        if window > 0 and len(self.recent_frontier_ids) > window * 2:
            self.recent_frontier_ids = self.recent_frontier_ids[-window * 2:]

    # ---- feedback ----------------------------------------------------------
    def add_feedback(
        self,
        step: int,
        type_: str,
        reason: str,
        suggested_fix: str = "",
        target_candidate_id: Optional[str] = None,
        frontier_id: Optional[int] = None,
        created_step: Optional[int] = None,
    ) -> None:
        self.feedback.append(
            FeedbackEvent(
                step=step,
                type=type_,
                reason=reason,
                target_candidate_id=target_candidate_id,
                frontier_id=frontier_id,
                suggested_fix=suggested_fix,
                created_step=step if created_step is None else created_step,
            )
        )
        # keep only the most recent N to bound prompt size
        if len(self.feedback) > 8:
            self.feedback = self.feedback[-8:]

    def recent_feedback(self, n: int = 4, current_step: int = 0) -> List[FeedbackEvent]:
        alive = [e for e in self.feedback if current_step - e.created_step < e.ttl_steps]
        return alive[-n:] if alive else []

    def summarize_feedback(self, agent_name: str, current_step: int = 0) -> List[FeedbackEvent]:
        """Report §FeedbackMemory: agent-specific event filtering."""
        alive = [e for e in self.feedback if current_step - e.created_step < e.ttl_steps]
        _MAP = {
            "ImageManager": {FB_AVU_FAIL, FB_AVU_VISUAL_ONLY},
            "Answerer": {FB_TASK_CHECK_FAIL, FB_WRONG_INSTANCE},
            "Planner": {FB_TASK_CHECK_FAIL, FB_AVU_FAIL, FB_FRONTIER_NO_INFO, FB_PLANNER_STALE},
            "Executor": {FB_FRONTIER_NO_INFO},
        }
        types = _MAP.get(agent_name, set())
        return [e for e in alive if e.type in types]

    def suggest_fix_for(self, type_: str, reason: str = "") -> str:
        """§7 failure-type → suggested_fix mapping. reason used to disambiguate
        TASK_CHECK_FAIL (facing vs instance)."""
        t = type_
        r = (reason or "").lower()
        if t == FB_TASK_CHECK_FAIL:
            if "facing" in r or "view" in r or "orient" in r:
                return "rotate toward target / use VVD viewpoint"
            return "mark candidate rejected, find other instances"
        if t in (FB_AVU_FAIL, FB_AVU_VISUAL_ONLY):
            return "pin image, try aliases / class-agnostic grounding / navigate closer"
        if t == FB_FRONTIER_NO_INFO:
            return "select a different frontier"
        if t == FB_PLANNER_STALE:
            return "revise plan: mark branch completed/failed or add new branch"
        if t == FB_WRONG_INSTANCE:
            return "reject candidate, search for other instances"
        return ""

    def feedback_prompt_block(
        self,
        n: int = 4,
        agent_name: Optional[str] = None,
        current_step: int = 0,
    ) -> str:
        if agent_name is not None:
            events = self.summarize_feedback(agent_name, current_step)
        else:
            events = self.recent_feedback(n, current_step)
        if not events:
            return ""
        lines = ["Recent Failure Feedback:"]
        for ev in events:
            lines.append(ev.to_prompt_str())
        lines.append(
            "Consider this feedback when deciding the next action. Do not "
            "repeat actions that led to the same failure."
        )
        return "\n".join(lines) + "\n"

    # ---- plan staleness ----------------------------------------------------
    @staticmethod
    def _normalize_plan(plan: Optional[str]) -> str:
        if plan is None:
            return ""
        return " ".join(plan.split()).lower()

    def update_plan(self, new_plan: Optional[str]) -> None:
        norm = self._normalize_plan(new_plan)
        if norm and norm == self.last_plan_normalized:
            self.plan_stale_count += 1
        else:
            self.plan_stale_count = 0
        self.high_level_plan = new_plan
        self.last_plan_normalized = norm

    def plan_is_stale(self, threshold: int = 2) -> bool:
        return self.plan_stale_count >= threshold

    # ---- prompt helpers ----------------------------------------------------
    def candidates_prompt_block(self) -> str:
        active = self.active_candidates()
        if not active:
            return ""
        lines = ["Active Target Candidates (VLM saw but not yet grounded):"]
        for c in active:
            lines.append(c.to_prompt_str())
        return "\n".join(lines) + "\n"

    def progress_signals_block(
        self,
        current_pose: Any = None,
        distance_moved: Optional[float] = None,
        new_objects: Optional[List[str]] = None,
        new_rooms: Optional[List[str]] = None,
        last_frontier_id: Optional[int] = None,
        last_frontier_result: Optional[str] = None,
        stale_plan_count: Optional[int] = None,
    ) -> str:
        """§8 Progress Signals block for the Planner prompt each step. Omits
        lines whose argument is None. Uses active_candidates / recent_feedback /
        plan_stale_count for the derived lines. (stale_plan_count arg is
        accepted for caller compatibility but the instance attribute wins.)"""
        lines: List[str] = ["Progress Signals:"]
        if current_pose is not None:
            lines.append(f"- current pose: {current_pose}")
        if distance_moved is not None:
            lines.append(f"- distance moved since last step: {distance_moved}m")
        if new_objects is not None:
            lines.append(f"- newly observed objects: {new_objects}")
        if new_rooms is not None:
            lines.append(f"- newly observed room cues: {new_rooms}")
        if last_frontier_id is not None:
            res = last_frontier_result if last_frontier_result is not None else "none"
            lines.append(f"- last frontier: F_{last_frontier_id:03d}, result={res}")
        # candidate grounding status (always present, may be empty)
        active = self.active_candidates()
        if active:
            status_str = "; ".join(
                f"{c.candidate_id}={c.status}" for c in active
            )
        else:
            status_str = "none"
        lines.append(f"- candidate grounding status: {status_str}")
        # recent feedback (last 2)
        events = self.recent_feedback(2)
        if events:
            fb_str = " | ".join(
                f"[{e.type}] {e.reason}" for e in events
            )
        else:
            fb_str = "none"
        lines.append(f"- recent feedback: {fb_str}")
        lines.append(f"- stale_plan_count: {self.plan_stale_count}")
        return "\n".join(lines) + "\n"
