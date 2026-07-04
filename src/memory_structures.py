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

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

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


class NavigationMode(Enum):
    EXPLORE = "EXPLORE"
    VISUAL_APPROACH = "VISUAL_APPROACH"
    TARGET_APPROACH = "TARGET_APPROACH"
    VERIFY = "VERIFY"
    STOP = "STOP"


class BranchTaskStatus(Enum):
    NEW = "NEW"
    ADVANCING = "ADVANCING"
    STALLED = "STALLED"
    REVISITING = "REVISITING"
    CLOSED = "CLOSED"


class HypothesisStatus(Enum):
    ACTIVE = "ACTIVE"
    SUPPORTED = "SUPPORTED"
    REJECTED = "REJECTED"
    CONFIRMED = "CONFIRMED"


class EventType(Enum):
    SPATIAL_BRANCH_CREATED = "SPATIAL_BRANCH_CREATED"
    SPATIAL_BRANCH_ADVANCED = "SPATIAL_BRANCH_ADVANCED"
    SPATIAL_BRANCH_STALLED = "SPATIAL_BRANCH_STALLED"
    SPATIAL_BRANCH_REVISITING = "SPATIAL_BRANCH_REVISITING"
    NEW_FRONTIER = "NEW_FRONTIER"
    FRONTIER_REACHED = "FRONTIER_REACHED"
    NO_VALID_FRONTIER = "NO_VALID_FRONTIER"
    SEMANTIC_EVIDENCE = "SEMANTIC_EVIDENCE"
    EVIDENCE_CONFLICT = "EVIDENCE_CONFLICT"
    CANDIDATE_VISIBLE = "CANDIDATE_VISIBLE"
    CANDIDATE_GROUNDED_3D = "CANDIDATE_GROUNDED_3D"
    VISUAL_APPROACH_READY = "VISUAL_APPROACH_READY"
    TARGET_VIEWPOINT_READY = "TARGET_VIEWPOINT_READY"
    TARGET_VIEWPOINT_REACHED = "TARGET_VIEWPOINT_REACHED"
    VERIFY_SUCCESS = "VERIFY_SUCCESS"
    VERIFY_FAILED = "VERIFY_FAILED"
    WRONG_INSTANCE = "WRONG_INSTANCE"
    POOR_VIEW = "POOR_VIEW"
    TARGET_NOT_VISIBLE = "TARGET_NOT_VISIBLE"
    WORKING_MEMORY_OVER_BUDGET = "WORKING_MEMORY_OVER_BUDGET"
    HYPOTHESIS_UPDATE_REQUIRED = "HYPOTHESIS_UPDATE_REQUIRED"
    MEMORY_UPDATE_REQUIRED = "MEMORY_UPDATE_REQUIRED"
    NAVIGATION_FAILED = "NAVIGATION_FAILED"


class AnswererDecision(Enum):
    NOT_FOUND = "NOT_FOUND"
    CANDIDATE_VISIBLE = "CANDIDATE_VISIBLE"
    TARGET_CONFIRMED = "TARGET_CONFIRMED"
    ANSWER_READY = "ANSWER_READY"


class ExecutorActionMode(Enum):
    CONTINUE_SPATIAL_BRANCH = "CONTINUE_SPATIAL_BRANCH"
    SWITCH_SPATIAL_BRANCH = "SWITCH_SPATIAL_BRANCH"
    REVISIT_SPATIAL_BRANCH = "REVISIT_SPATIAL_BRANCH"
    STOP = "STOP"


class VerifyStatus(Enum):
    SUCCESS = "SUCCESS"
    WRONG_INSTANCE = "WRONG_INSTANCE"
    POOR_VIEW = "POOR_VIEW"
    TARGET_NOT_VISIBLE = "TARGET_NOT_VISIBLE"


T = TypeVar("T", bound="JsonDataclassMixin")


def _enum_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return {f.name: _jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return value
    return value


def _coerce_enum(enum_cls: Type[Enum], value: Any) -> Enum:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError:
            return enum_cls[value]
    return enum_cls(value)


class JsonDataclassMixin:
    """Small JSON bridge for typed rebuild records.

    This intentionally avoids a dependency on pydantic. Runtime code can pass
    numpy arrays, enum instances, or primitive dicts; persisted state stays JSON
    compatible.
    """

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: _jsonable(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        if data is None:
            raise ValueError(f"cannot build {cls.__name__} from None")
        return cls(**dict(data))


@dataclass
class FrontierAnchor(JsonDataclassMixin):
    frontier_id: int
    spatial_branch_id: Optional[str] = None


@dataclass
class SpatialBranchAnchor(JsonDataclassMixin):
    spatial_branch_id: str


@dataclass
class ImageAnchor(JsonDataclassMixin):
    image_path: str
    candidate_id: Optional[str] = None


@dataclass
class ObjectAnchor(JsonDataclassMixin):
    object_id: Union[int, str]
    class_name: Optional[str] = None


@dataclass
class CandidateAnchor(JsonDataclassMixin):
    candidate_id: str
    image_path: Optional[str] = None
    target_phrase: Optional[str] = None


Anchor = Union[
    FrontierAnchor,
    SpatialBranchAnchor,
    ImageAnchor,
    ObjectAnchor,
    CandidateAnchor,
]


def anchor_from_dict(data: Dict[str, Any]) -> Anchor:
    kind = (data or {}).get("kind") or (data or {}).get("type")
    if kind:
        kind = str(kind).lower()
    if kind == "frontier" or "frontier_id" in data:
        return FrontierAnchor(
            frontier_id=int(data["frontier_id"]),
            spatial_branch_id=data.get("spatial_branch_id"),
        )
    if kind == "spatial_branch" or "spatial_branch_id" in data:
        return SpatialBranchAnchor(spatial_branch_id=str(data["spatial_branch_id"]))
    if kind == "object" or "object_id" in data:
        return ObjectAnchor(
            object_id=data["object_id"],
            class_name=data.get("class_name"),
        )
    if kind == "candidate" or "candidate_id" in data:
        return CandidateAnchor(
            candidate_id=str(data["candidate_id"]),
            image_path=data.get("image_path"),
            target_phrase=data.get("target_phrase"),
        )
    if kind == "image" or "image_path" in data:
        return ImageAnchor(
            image_path=str(data["image_path"]),
            candidate_id=data.get("candidate_id"),
        )
    raise ValueError(f"unknown anchor payload: {data}")


def _anchor_to_dict(anchor: Anchor) -> Dict[str, Any]:
    payload = anchor.to_dict() if hasattr(anchor, "to_dict") else _jsonable(anchor)
    if isinstance(anchor, FrontierAnchor):
        payload["kind"] = "frontier"
    elif isinstance(anchor, SpatialBranchAnchor):
        payload["kind"] = "spatial_branch"
    elif isinstance(anchor, ImageAnchor):
        payload["kind"] = "image"
    elif isinstance(anchor, ObjectAnchor):
        payload["kind"] = "object"
    elif isinstance(anchor, CandidateAnchor):
        payload["kind"] = "candidate"
    return payload


@dataclass
class FrontierInstance(JsonDataclassMixin):
    frontier_id: int
    position: Any
    orientation: Any = None
    area: float = 0.0
    local_index: Optional[int] = None
    spatial_branch_id: Optional[str] = None
    image: Optional[str] = None
    is_selectable: bool = True
    created_step: int = 0
    updated_step: int = 0

    def __post_init__(self) -> None:
        self.frontier_id = int(self.frontier_id)
        self.position = _jsonable(self.position)
        self.orientation = _jsonable(self.orientation)

    def to_prompt_str(self) -> str:
        branch = self.spatial_branch_id or "unassigned"
        return (
            f"F_{self.frontier_id:03d}/{branch}: "
            f"area={self.area:.1f}, selectable={self.is_selectable}"
        )


@dataclass
class SpatialBranchRecord(JsonDataclassMixin):
    spatial_branch_id: str
    frontier_ids: List[int] = field(default_factory=list)
    spine: List[Any] = field(default_factory=list)
    active_tip_frontier_id: Optional[int] = None
    aliases: List[str] = field(default_factory=list)
    merged_into: Optional[str] = None
    created_step: int = 0
    updated_step: int = 0

    def __post_init__(self) -> None:
        self.frontier_ids = [int(fid) for fid in self.frontier_ids]
        if self.active_tip_frontier_id is not None:
            self.active_tip_frontier_id = int(self.active_tip_frontier_id)
        self.spine = [_jsonable(p) for p in self.spine]

    def to_prompt_str(self) -> str:
        tip = (
            f"F_{self.active_tip_frontier_id:03d}"
            if self.active_tip_frontier_id is not None
            else "none"
        )
        merged = f", merged_into={self.merged_into}" if self.merged_into else ""
        return (
            f"{self.spatial_branch_id}: tip={tip}, "
            f"frontiers={self.frontier_ids}{merged}"
        )


@dataclass
class BranchTaskState(JsonDataclassMixin):
    spatial_branch_id: str
    status: BranchTaskStatus = BranchTaskStatus.NEW
    progress_score: float = 0.0
    steps_without_progress: int = 0
    selected_count: int = 0
    reached_count: int = 0
    active_hypothesis_id: Optional[str] = None
    last_frontier_id: Optional[int] = None
    closed_reason: Optional[str] = None
    updated_step: int = 0

    def __post_init__(self) -> None:
        self.status = _coerce_enum(BranchTaskStatus, self.status)
        if self.last_frontier_id is not None:
            self.last_frontier_id = int(self.last_frontier_id)

    def to_prompt_str(self) -> str:
        fid = (
            f"F_{self.last_frontier_id:03d}"
            if self.last_frontier_id is not None
            else "none"
        )
        return (
            f"{self.spatial_branch_id}: status={self.status.value}, "
            f"progress={self.progress_score:.2f}, last={fid}"
        )


@dataclass
class HypothesisBranch(JsonDataclassMixin):
    hypothesis_id: str
    description: str
    anchors: List[Anchor] = field(default_factory=list)
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    created_step: int = 0
    updated_step: int = 0
    writer: str = "HypothesisManager"

    def __post_init__(self) -> None:
        self.status = _coerce_enum(HypothesisStatus, self.status)
        normalized = []
        for anchor in self.anchors:
            if isinstance(anchor, dict):
                normalized.append(anchor_from_dict(anchor))
            else:
                normalized.append(anchor)
        self.anchors = normalized

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["anchors"] = [_anchor_to_dict(a) for a in self.anchors]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HypothesisBranch":
        payload = dict(data)
        payload["anchors"] = [
            anchor_from_dict(a) if isinstance(a, dict) else a
            for a in payload.get("anchors", [])
        ]
        return cls(**payload)

    def to_prompt_str(self) -> str:
        anchor_str = ", ".join(
            _anchor_to_dict(anchor).get("kind", "anchor")
            for anchor in self.anchors
        ) or "none"
        return (
            f"{self.hypothesis_id}: {self.description} "
            f"(status={self.status.value}, conf={self.confidence:.2f}, "
            f"anchors={anchor_str})"
        )


@dataclass
class TypedEvent(JsonDataclassMixin):
    event_id: str
    type: EventType
    step: int
    entity_id: Optional[str] = None
    active_hypothesis_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"
    ttl_steps: int = 4
    created_step: int = 0

    def __post_init__(self) -> None:
        self.type = _coerce_enum(EventType, self.type)
        if self.created_step == 0:
            self.created_step = self.step

    def debounce_key(self) -> tuple:
        return (self.type.value, self.entity_id, self.active_hypothesis_id)

    def to_prompt_str(self) -> str:
        entity = f" entity={self.entity_id}" if self.entity_id else ""
        return f"- step {self.step} [{self.type.value}]{entity}: {self.payload}"


@dataclass
class ExploreIntent(JsonDataclassMixin):
    mode: NavigationMode = NavigationMode.EXPLORE
    frontier_id: Optional[int] = None
    spatial_branch_id: Optional[str] = None
    hypothesis_id: Optional[str] = None
    action_mode: ExecutorActionMode = ExecutorActionMode.CONTINUE_SPATIAL_BRANCH
    reason_code: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        self.mode = _coerce_enum(NavigationMode, self.mode)
        self.action_mode = _coerce_enum(ExecutorActionMode, self.action_mode)
        if self.frontier_id is not None:
            self.frontier_id = int(self.frontier_id)


@dataclass
class VisualApproachIntent(JsonDataclassMixin):
    mode: NavigationMode = NavigationMode.VISUAL_APPROACH
    candidate_id: str = ""
    image_path: str = ""
    target_phrase: str = ""
    approach_xyz: Any = None
    approach_yaw: Optional[float] = None
    reason_code: str = "VISUAL_APPROACH"
    reason: str = ""

    def __post_init__(self) -> None:
        self.mode = _coerce_enum(NavigationMode, self.mode)
        self.approach_xyz = _jsonable(self.approach_xyz)


@dataclass
class TargetViewpointIntent(JsonDataclassMixin):
    mode: NavigationMode = NavigationMode.TARGET_APPROACH
    candidate_id: str = ""
    image_path: str = ""
    target_phrase: str = ""
    target_xyz: Any = None
    target_yaw: Optional[float] = None
    viewpoint_id: Optional[str] = None
    reason_code: str = "TARGET_VIEWPOINT"
    reason: str = ""

    def __post_init__(self) -> None:
        self.mode = _coerce_enum(NavigationMode, self.mode)
        self.target_xyz = _jsonable(self.target_xyz)


NavigationIntent = Union[ExploreIntent, VisualApproachIntent, TargetViewpointIntent]


def navigation_intent_from_dict(data: Dict[str, Any]) -> NavigationIntent:
    mode = _coerce_enum(NavigationMode, data.get("mode", NavigationMode.EXPLORE))
    if mode == NavigationMode.VISUAL_APPROACH:
        return VisualApproachIntent.from_dict(data)
    if mode in (NavigationMode.TARGET_APPROACH, NavigationMode.VERIFY):
        return TargetViewpointIntent.from_dict(data)
    return ExploreIntent.from_dict(data)


@dataclass
class NavigationResult(JsonDataclassMixin):
    mode: NavigationMode
    success: bool
    target_arrived: bool = False
    intent: Optional[NavigationIntent] = None
    failure_reason: Optional[str] = None
    verify_status: Optional[VerifyStatus] = None
    frontier_id: Optional[int] = None
    candidate_id: Optional[str] = None
    step: int = 0

    def __post_init__(self) -> None:
        self.mode = _coerce_enum(NavigationMode, self.mode)
        if self.verify_status is not None:
            self.verify_status = _coerce_enum(VerifyStatus, self.verify_status)
        if isinstance(self.intent, dict):
            self.intent = navigation_intent_from_dict(self.intent)
        if self.frontier_id is not None:
            self.frontier_id = int(self.frontier_id)


@dataclass
class StepOutcome(JsonDataclassMixin):
    step: int
    mode: NavigationMode
    intent: Optional[NavigationIntent] = None
    navigation_result: Optional[NavigationResult] = None
    events: List[TypedEvent] = field(default_factory=list)
    answerer_decision: AnswererDecision = AnswererDecision.NOT_FOUND
    verification_result: Optional[VerifyStatus] = None
    done: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mode = _coerce_enum(NavigationMode, self.mode)
        self.answerer_decision = _coerce_enum(
            AnswererDecision, self.answerer_decision
        )
        if self.verification_result is not None:
            self.verification_result = _coerce_enum(
                VerifyStatus, self.verification_result
            )
        if isinstance(self.intent, dict):
            self.intent = navigation_intent_from_dict(self.intent)
        if isinstance(self.navigation_result, dict):
            self.navigation_result = NavigationResult.from_dict(
                self.navigation_result
            )
        self.events = [
            TypedEvent.from_dict(e) if isinstance(e, dict) else e
            for e in self.events
        ]


def format_prompt_summary(items: Any, max_items: int = 8) -> str:
    """Render typed rebuild state for VLM prompts without leaking raw dicts."""
    if items is None:
        return ""
    if not isinstance(items, (list, tuple)):
        items = [items]
    lines = []
    for item in list(items)[:max_items]:
        if hasattr(item, "to_prompt_str"):
            lines.append(item.to_prompt_str())
        elif isinstance(item, dict):
            lines.append(str(_jsonable(item)))
        else:
            lines.append(str(item))
    if len(items) > max_items:
        lines.append(f"... {len(items) - max_items} more")
    return "\n".join(lines)


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
S_SUCCEEDED = "SUCCEEDED"                # navigation + task_check passed
S_EXPIRED = "EXPIRED"                    # TTL or closer-view cap reached


@dataclass
class GroundingAttempt:
    level: int            # 1=YOLO target, 2=YOLO alias, 3=class-agnostic+CLIP, 4=evidence-pose
    success: bool
    reason: str = ""


def should_release(c: "TargetCandidate", step: int, ttl_steps: int = 12,
                   max_closer_views: int = 3) -> bool:
    """Report §TargetCandidateMemory release gate. Only reject/expire/cap
    release; success release is handled by release_after_navigation."""
    if c.status in (S_REJECTED, S_SUCCEEDED, S_EXPIRED):
        return True
    if c.closer_view_attempts >= max_closer_views:
        c.status = S_EXPIRED
        return True
    _ref_step = c.updated_step if c.updated_step else c.source_step
    if step - _ref_step > ttl_steps and c.status == S_VISUAL_ONLY:
        c.status = S_EXPIRED
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
    updated_step: int = 0                    # last status-change step (0 → use source_step)

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
FB_BLOCK_FRONTIER = "BLOCK_FRONTIER"
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
    event_id: str = ""
    subtask_id: str = ""
    reason_code: str = ""
    message: str = ""
    blocked_actions: list = field(default_factory=list)
    severity: str = "info"

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
        self.spatial_branches: Dict[str, SpatialBranchRecord] = {}
        self.frontier_to_branch: Dict[int, str] = {}
        self.branch_task_states: Dict[str, BranchTaskState] = {}
        self.hypotheses: Dict[str, HypothesisBranch] = {}
        self.typed_events: List[TypedEvent] = []
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
        self.branch_task_states = {}
        self.hypotheses = {}
        self.typed_events = []
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

    # ---- typed rebuild state ----------------------------------------------
    def upsert_spatial_branch(
        self,
        branch: SpatialBranchRecord,
    ) -> SpatialBranchRecord:
        self.spatial_branches[branch.spatial_branch_id] = branch
        for fid in branch.frontier_ids:
            self.frontier_to_branch[int(fid)] = branch.spatial_branch_id
        if branch.active_tip_frontier_id is not None:
            self.frontier_to_branch[
                int(branch.active_tip_frontier_id)
            ] = branch.spatial_branch_id
        return branch

    def get_branch_for_frontier(self, frontier_id: int) -> Optional[SpatialBranchRecord]:
        bid = self.frontier_to_branch.get(int(frontier_id))
        return self.spatial_branches.get(bid) if bid else None

    def upsert_branch_task_state(
        self,
        state: BranchTaskState,
    ) -> BranchTaskState:
        self.branch_task_states[state.spatial_branch_id] = state
        return state

    def eligible_spatial_branches(self) -> List[SpatialBranchRecord]:
        out = []
        for bid, branch in self.spatial_branches.items():
            if branch.merged_into:
                continue
            state = self.branch_task_states.get(bid)
            if state is not None and state.status == BranchTaskStatus.CLOSED:
                continue
            out.append(branch)
        return out

    def set_hypotheses_from_manager(
        self,
        hypotheses: List[HypothesisBranch],
        writer: str = "HypothesisManager",
    ) -> None:
        """Single-writer gate for HypothesisBranch updates."""
        for hyp in hypotheses:
            if hyp.writer != writer:
                raise ValueError(
                    "HypothesisBranch updates must come from HypothesisManager"
                )
            self.hypotheses[hyp.hypothesis_id] = hyp

    def add_typed_event(self, event: TypedEvent) -> None:
        self.typed_events.append(event)
        if len(self.typed_events) > 32:
            self.typed_events = self.typed_events[-32:]

    def clear_task_scope_events(self) -> None:
        self.typed_events = []

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
            "Executor": {FB_FRONTIER_NO_INFO, FB_BLOCK_FRONTIER},
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
