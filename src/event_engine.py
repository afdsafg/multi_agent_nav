"""Typed event detection and trigger routing for the rebuild flow."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from src.memory_structures import (
    AnswererDecision,
    BranchTaskStatus,
    EventType,
    HypothesisStatus,
    NavigationMode,
    NavigationResult,
    StepOutcome,
    SubtaskWorkingMemory,
    TypedEvent,
    VerifyStatus,
)


@dataclass
class EventEngineConfig:
    debounce_steps: int = 2
    memory_pool_budget: int = 6
    spatial_stalled_steps_threshold: int = 2
    spatial_revisit_reversal_threshold: int = 2
    spatial_revisit_overlap_threshold: float = 0.7


@dataclass
class EventRoutingDecision:
    call_memory_manager: bool = False
    call_hypothesis_manager: bool = False
    reasons: List[str] = field(default_factory=list)


class EventEngine:
    """Creates task-scoped typed events and debounces trigger routing."""

    HYPOTHESIS_TRIGGERS = {
        EventType.NO_ACTIVE_HYPOTHESIS,
        EventType.HYPOTHESIS_SUPPORTED,
        EventType.HYPOTHESIS_CONTRADICTED,
        EventType.HYPOTHESIS_TEST_COMPLETED,
        EventType.TRAJECTORY_LOOP,
        EventType.CANDIDATE_REJECTED,
        EventType.SPATIAL_BRANCH_STALLED,
        EventType.SPATIAL_BRANCH_REVISITING,
        EventType.NO_ELIGIBLE_FRONTIER,
        EventType.NO_VALID_FRONTIER,
        EventType.WRONG_INSTANCE,
        EventType.ANSWERER_EVIDENCE_CONFLICT,
        EventType.HYPOTHESIS_REVISED,
        EventType.HYPOTHESIS_UPDATE_REQUIRED,
    }
    MEMORY_TRIGGERS = {
        EventType.WORKING_MEMORY_OVER_BUDGET,
        EventType.HYPOTHESIS_REVISED,
        EventType.ANSWERER_EVIDENCE_CONFLICT,
        EventType.MEMORY_UPDATE_REQUIRED,
    }

    def __init__(self, config: Optional[EventEngineConfig] = None) -> None:
        self.config = config or EventEngineConfig()
        self._last_emitted: Dict[
            Tuple[str, Optional[str], Optional[str]], TypedEvent
        ] = {}

    def reset_task_scope(self) -> None:
        self._last_emitted = {}

    def emit(
        self,
        type_: EventType,
        step: int,
        entity_id: Optional[str] = None,
        active_hypothesis_id: Optional[str] = None,
        payload: Optional[Dict] = None,
        severity: str = "info",
    ) -> Optional[TypedEvent]:
        event = TypedEvent(
            event_id=f"{type_.value}:{entity_id or 'global'}:{step}",
            type=type_,
            step=step,
            entity_id=entity_id,
            active_hypothesis_id=active_hypothesis_id,
            payload=payload or {},
            severity=severity,
        )
        key = event.debounce_key()
        last_event = self._last_emitted.get(key)
        if self._should_suppress(event, last_event):
            return None
        self._last_emitted[key] = event
        return event

    def debounce_events(self, events: Iterable[TypedEvent]) -> List[TypedEvent]:
        """Apply EventEngine debounce to events created by code modules."""
        debounced = []
        for event in events:
            key = event.debounce_key()
            last_event = self._last_emitted.get(key)
            if self._should_suppress(event, last_event):
                continue
            self._last_emitted[key] = event
            debounced.append(event)
        return debounced

    def _should_suppress(
        self,
        event: TypedEvent,
        last_event: Optional[TypedEvent],
    ) -> bool:
        if last_event is None:
            return False
        if event.step - last_event.step >= self.config.debounce_steps:
            return False
        if self._severity_rank(event.severity) > self._severity_rank(last_event.severity):
            return False
        return self._material_signature(event) == self._material_signature(last_event)

    @staticmethod
    def _severity_rank(severity: str) -> int:
        return {
            "debug": 0,
            "info": 1,
            "warning": 2,
            "error": 3,
            "critical": 4,
        }.get(str(severity or "info").lower(), 1)

    @staticmethod
    def _material_signature(event: TypedEvent) -> str:
        try:
            return json.dumps(event.payload or {}, sort_keys=True, default=str)
        except TypeError:
            return repr(event.payload)

    def detect_memory_events(
        self,
        pool_size: int,
        step: int,
        working_memory: Optional[SubtaskWorkingMemory] = None,
    ) -> List[TypedEvent]:
        if pool_size <= self.config.memory_pool_budget:
            return []
        event = self.emit(
            EventType.WORKING_MEMORY_OVER_BUDGET,
            step=step,
            entity_id="image_pool",
            payload={
                "pool_size": pool_size,
                "budget": self.config.memory_pool_budget,
            },
            severity="warning",
        )
        if event is None:
            return []
        if working_memory is not None:
            working_memory.add_typed_event(event)
        return [event]

    def detect_answerer_events(
        self,
        decision: AnswererDecision,
        step: int,
        candidate_id: Optional[str] = None,
        image_path: Optional[str] = None,
        evidence_updates: Optional[List] = None,
        evidence_conflict: bool = False,
        working_memory: Optional[SubtaskWorkingMemory] = None,
    ) -> List[TypedEvent]:
        events: List[TypedEvent] = []
        for idx, update in enumerate(evidence_updates or []):
            if not isinstance(update, dict):
                payload = {"raw": update}
                hypothesis_id = None
                result = ""
            else:
                payload = dict(update)
                hypothesis_id = (
                    payload.get("hypothesis_id")
                    or payload.get("id")
                    or payload.get("hypothesis")
                )
                result = str(payload.get("result") or payload.get("status") or "").upper()
            event_type = EventType.SEMANTIC_EVIDENCE
            if result in {"SUPPORT", "SUPPORTED", "STRENGTHEN"}:
                event_type = EventType.HYPOTHESIS_SUPPORTED
            elif result in {"WEAKEN", "CONTRADICT", "CONTRADICTED", "REJECT"}:
                event_type = EventType.HYPOTHESIS_CONTRADICTED
            elif result in {"TEST_COMPLETED", "COMPLETED", "DONE"}:
                event_type = EventType.HYPOTHESIS_TEST_COMPLETED
            event = self.emit(
                event_type,
                step=step,
                entity_id=str(hypothesis_id or f"evidence_{idx}"),
                active_hypothesis_id=str(hypothesis_id) if hypothesis_id else None,
                payload=payload,
                severity="warning"
                if event_type == EventType.HYPOTHESIS_CONTRADICTED
                else "info",
            )
            if event is not None:
                events.append(event)
        if decision in (
            AnswererDecision.CANDIDATE_VISIBLE,
            AnswererDecision.TARGET_CONFIRMED,
            AnswererDecision.ANSWER_READY,
        ):
            event = self.emit(
                EventType.CANDIDATE_VISIBLE,
                step=step,
                entity_id=candidate_id or image_path,
                payload={
                    "decision": decision.value,
                    "candidate_id": candidate_id,
                    "image_path": image_path,
                },
            )
            if event is not None:
                events.append(event)
        if evidence_conflict:
            event = self.emit(
                EventType.ANSWERER_EVIDENCE_CONFLICT,
                step=step,
                entity_id=candidate_id or image_path,
                payload={"candidate_id": candidate_id, "image_path": image_path},
                severity="warning",
            )
            if event is not None:
                events.append(event)
        if working_memory is not None:
            for event in events:
                working_memory.add_typed_event(event)
        return events

    def detect_navigation_events(
        self,
        result: NavigationResult,
        working_memory: Optional[SubtaskWorkingMemory] = None,
    ) -> List[TypedEvent]:
        events: List[TypedEvent] = []
        if result.mode == NavigationMode.EXPLORE and result.target_arrived:
            event = self.emit(
                EventType.FRONTIER_REACHED,
                step=result.step,
                entity_id=(
                    f"F_{result.frontier_id:03d}"
                    if result.frontier_id is not None
                    else None
                ),
                payload={"frontier_id": result.frontier_id},
            )
            if event is not None:
                events.append(event)
        elif result.mode == NavigationMode.TARGET_APPROACH and result.target_arrived:
            event = self.emit(
                EventType.TARGET_VIEWPOINT_REACHED,
                step=result.step,
                entity_id=result.candidate_id,
                payload={"candidate_id": result.candidate_id},
            )
            if event is not None:
                events.append(event)

        if not result.success:
            event = self.emit(
                EventType.NAVIGATION_FAILED,
                step=result.step,
                entity_id=result.candidate_id
                or (
                    f"F_{result.frontier_id:03d}"
                    if result.frontier_id is not None
                    else None
                ),
                payload={"failure_reason": result.failure_reason},
                severity="warning",
            )
            if event is not None:
                events.append(event)

        if result.verify_status is not None:
            verify_type = {
                VerifyStatus.SUCCESS: EventType.VERIFY_SUCCESS,
                VerifyStatus.WRONG_INSTANCE: EventType.WRONG_INSTANCE,
                VerifyStatus.POOR_VIEW: EventType.POOR_VIEW,
                VerifyStatus.TARGET_NOT_VISIBLE: EventType.TARGET_NOT_VISIBLE,
            }[result.verify_status]
            event = self.emit(
                verify_type,
                step=result.step,
                entity_id=result.candidate_id,
                payload={"verify_status": result.verify_status.value},
                severity="info"
                if result.verify_status == VerifyStatus.SUCCESS
                else "warning",
            )
            if event is not None:
                events.append(event)

        if working_memory is not None:
            for event in events:
                working_memory.add_typed_event(event)
        return events

    def route(
        self,
        events: Iterable[TypedEvent],
        working_memory: Optional[SubtaskWorkingMemory] = None,
    ) -> EventRoutingDecision:
        routing = EventRoutingDecision()
        for event in events:
            if event.type in self.MEMORY_TRIGGERS:
                routing.call_memory_manager = True
                routing.reasons.append(event.type.value)
            if event.type in self.HYPOTHESIS_TRIGGERS and self._should_route_hypothesis_event(
                event, working_memory
            ):
                routing.call_hypothesis_manager = True
                routing.reasons.append(event.type.value)
        return routing

    def _should_route_hypothesis_event(
        self,
        event: TypedEvent,
        working_memory: Optional[SubtaskWorkingMemory],
    ) -> bool:
        if event.type not in (
            EventType.SPATIAL_BRANCH_STALLED,
            EventType.SPATIAL_BRANCH_REVISITING,
        ):
            return True
        if working_memory is None or not event.entity_id:
            return False
        if not self._branch_linked_to_active_hypothesis(
            working_memory, event.entity_id
        ):
            return False
        state = working_memory.branch_task_states.get(event.entity_id)
        if state is None:
            return False
        if event.type == EventType.SPATIAL_BRANCH_STALLED:
            return (
                state.status == BranchTaskStatus.STALLED
                and state.steps_without_progress
                >= self.config.spatial_stalled_steps_threshold
            )
        return (
            state.status == BranchTaskStatus.REVISITING
            and (
                state.reversal_count
                >= self.config.spatial_revisit_reversal_threshold
                or state.recent_region_overlap
                >= self.config.spatial_revisit_overlap_threshold
            )
        )

    @staticmethod
    def _branch_linked_to_active_hypothesis(
        working_memory: SubtaskWorkingMemory,
        branch_id: str,
    ) -> bool:
        active_statuses = {
            HypothesisStatus.PENDING,
            HypothesisStatus.ACTIVE,
            HypothesisStatus.SUPPORTED,
            HypothesisStatus.CONFIRMED,
        }
        for hyp in working_memory.hypotheses.values():
            if hyp.status not in active_statuses:
                continue
            if branch_id in hyp.linked_spatial_branches:
                return True
        return False

    def build_step_outcome(
        self,
        step: int,
        mode: NavigationMode,
        events: Iterable[TypedEvent],
        decision: AnswererDecision = AnswererDecision.NOT_FOUND,
        navigation_result: Optional[NavigationResult] = None,
        done: bool = False,
    ) -> StepOutcome:
        return StepOutcome(
            step=step,
            mode=mode,
            events=list(events),
            answerer_decision=decision,
            navigation_result=navigation_result,
            done=done,
        )
