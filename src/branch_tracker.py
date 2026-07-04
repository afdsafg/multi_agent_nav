"""Spatial branch tracking for the rebuild navigation flow.

Frontier IDs are transient TSDF observations. SpatialBranchRecord is the
durable spatial memory used by Executor and Hypothesis Manager prompts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.memory_structures import (
    BranchTaskState,
    BranchTaskStatus,
    FrontierInstance,
    SpatialBranchRecord,
    SubtaskWorkingMemory,
    TypedEvent,
    EventType,
)


def _as_np_xy(pos) -> np.ndarray:
    arr = np.asarray(pos, dtype=float)
    if arr.ndim == 0:
        return np.zeros(2, dtype=float)
    if arr.shape[0] >= 3:
        return arr[[0, 2]].astype(float)
    if arr.shape[0] >= 2:
        return arr[:2].astype(float)
    return np.zeros(2, dtype=float)


@dataclass
class BranchTrackerConfig:
    match_distance_m: float = 1.0
    stalled_progress_epsilon: float = 0.05
    stalled_after_steps: int = 3
    revisit_distance_m: float = 0.6


class BranchTracker:
    """Maintains durable spatial branches from current TSDF frontiers."""

    def __init__(self, config: Optional[BranchTrackerConfig] = None) -> None:
        self.config = config or BranchTrackerConfig()
        self._counter = 0

    def reset_task_scope(self) -> None:
        """Do not clear branch records; only task-local state lives in memory."""
        pass

    def sync_frontiers(
        self,
        tsdf_planner,
        working_memory: SubtaskWorkingMemory,
        step: int,
        current_position=None,
    ) -> Tuple[List[FrontierInstance], List[TypedEvent]]:
        instances: List[FrontierInstance] = []
        events: List[TypedEvent] = []
        current_fids = set()

        for local_index, frontier in enumerate(getattr(tsdf_planner, "frontiers", [])):
            fid = int(getattr(frontier, "frontier_id"))
            current_fids.add(fid)
            position_habitat = self._frontier_position_habitat(tsdf_planner, frontier)
            area = float(np.sum(getattr(frontier, "region", np.zeros(1))))
            branch = self._assign_branch(
                working_memory=working_memory,
                frontier_id=fid,
                position=position_habitat,
                step=step,
            )
            instance = FrontierInstance(
                frontier_id=fid,
                position=position_habitat,
                orientation=getattr(frontier, "orientation", None),
                area=area,
                local_index=local_index,
                spatial_branch_id=branch.spatial_branch_id,
                image=getattr(frontier, "image", None),
                created_step=branch.created_step,
                updated_step=step,
            )
            instances.append(instance)

            state = working_memory.branch_task_states.get(branch.spatial_branch_id)
            if state is None:
                state = BranchTaskState(
                    spatial_branch_id=branch.spatial_branch_id,
                    status=BranchTaskStatus.NEW,
                    progress_score=0.0,
                    last_frontier_id=fid,
                    updated_step=step,
                )
                working_memory.upsert_branch_task_state(state)
                events.append(
                    self._event(
                        EventType.SPATIAL_BRANCH_CREATED,
                        step,
                        branch.spatial_branch_id,
                        {"frontier_id": fid},
                    )
                )
            else:
                self._update_branch_progress(
                    state, branch, position_habitat, current_position, step
                )
                events.append(
                    self._event(
                        EventType.SPATIAL_BRANCH_ADVANCED,
                        step,
                        branch.spatial_branch_id,
                        {
                            "frontier_id": fid,
                            "status": state.status.value,
                            "progress_score": state.progress_score,
                        },
                    )
                )
                if state.status == BranchTaskStatus.STALLED:
                    events.append(
                        self._event(
                            EventType.SPATIAL_BRANCH_STALLED,
                            step,
                            branch.spatial_branch_id,
                            {"frontier_id": fid},
                        )
                    )
                elif state.status == BranchTaskStatus.REVISITING:
                    events.append(
                        self._event(
                            EventType.SPATIAL_BRANCH_REVISITING,
                            step,
                            branch.spatial_branch_id,
                            {"frontier_id": fid},
                        )
                    )

            working_memory.frontier_to_branch[fid] = branch.spatial_branch_id

        self._close_missing_active_tips(working_memory, current_fids, step)
        for event in events:
            working_memory.add_typed_event(event)
        return instances, events

    def eligible_frontier_instances(
        self,
        instances: Iterable[FrontierInstance],
        working_memory: SubtaskWorkingMemory,
        recent_ids: Optional[List[int]] = None,
        recent_window: int = 3,
        max_reselect: int = 2,
    ) -> List[FrontierInstance]:
        recent = set(recent_ids[-recent_window:]) if recent_ids and recent_window > 0 else set()
        eligible = []
        for inst in instances:
            branch = working_memory.spatial_branches.get(inst.spatial_branch_id or "")
            state = working_memory.branch_task_states.get(inst.spatial_branch_id or "")
            if branch is not None and branch.merged_into:
                continue
            if state is not None and state.status == BranchTaskStatus.CLOSED:
                continue
            if inst.frontier_id in recent:
                continue
            fs = working_memory.frontier_registry.get(inst.frontier_id)
            if fs is not None and getattr(fs, "selected_count", 0) >= max_reselect:
                continue
            inst.is_selectable = True
            eligible.append(inst)
        return eligible

    def mark_selected(
        self,
        frontier_id: int,
        working_memory: SubtaskWorkingMemory,
        step: int,
    ) -> None:
        branch = working_memory.get_branch_for_frontier(frontier_id)
        if branch is None:
            return
        state = working_memory.branch_task_states.get(branch.spatial_branch_id)
        if state is None:
            state = BranchTaskState(spatial_branch_id=branch.spatial_branch_id)
        state.selected_count += 1
        state.last_frontier_id = int(frontier_id)
        state.updated_step = step
        if state.status == BranchTaskStatus.NEW:
            state.status = BranchTaskStatus.ADVANCING
        working_memory.upsert_branch_task_state(state)

    def mark_reached(
        self,
        frontier_id: int,
        working_memory: SubtaskWorkingMemory,
        step: int,
        progress_delta: float = 0.0,
    ) -> Optional[TypedEvent]:
        branch = working_memory.get_branch_for_frontier(frontier_id)
        if branch is None:
            return None
        state = working_memory.branch_task_states.get(branch.spatial_branch_id)
        if state is None:
            state = BranchTaskState(spatial_branch_id=branch.spatial_branch_id)
        state.reached_count += 1
        state.last_frontier_id = int(frontier_id)
        state.progress_score = max(state.progress_score, state.progress_score + progress_delta)
        state.updated_step = step
        working_memory.upsert_branch_task_state(state)
        event = self._event(
            EventType.FRONTIER_REACHED,
            step,
            branch.spatial_branch_id,
            {"frontier_id": int(frontier_id), "progress_delta": progress_delta},
        )
        working_memory.add_typed_event(event)
        return event

    def _assign_branch(
        self,
        working_memory: SubtaskWorkingMemory,
        frontier_id: int,
        position,
        step: int,
    ) -> SpatialBranchRecord:
        existing = working_memory.get_branch_for_frontier(frontier_id)
        if existing is not None:
            self._update_branch(existing, frontier_id, position, step)
            return existing

        match = self._nearest_branch(working_memory.spatial_branches.values(), position)
        if match is not None:
            self._update_branch(match, frontier_id, position, step)
            working_memory.upsert_spatial_branch(match)
            return match

        bid = self._next_branch_id(working_memory.spatial_branches.keys())
        branch = SpatialBranchRecord(
            spatial_branch_id=bid,
            frontier_ids=[int(frontier_id)],
            spine=[position],
            active_tip_frontier_id=int(frontier_id),
            created_step=step,
            updated_step=step,
        )
        working_memory.upsert_spatial_branch(branch)
        return branch

    def _update_branch(
        self,
        branch: SpatialBranchRecord,
        frontier_id: int,
        position,
        step: int,
    ) -> None:
        fid = int(frontier_id)
        if fid not in branch.frontier_ids:
            branch.frontier_ids.append(fid)
        branch.active_tip_frontier_id = fid
        branch.spine.append(position)
        if len(branch.spine) > 24:
            branch.spine = branch.spine[-24:]
        branch.updated_step = step

    def _nearest_branch(
        self,
        branches: Iterable[SpatialBranchRecord],
        position,
    ) -> Optional[SpatialBranchRecord]:
        pos = _as_np_xy(position)
        best = None
        best_d = float("inf")
        for branch in branches:
            if branch.merged_into or not branch.spine:
                continue
            d = float(np.linalg.norm(pos - _as_np_xy(branch.spine[-1])))
            if d < best_d:
                best = branch
                best_d = d
        if best is not None and best_d <= self.config.match_distance_m:
            return best
        return None

    def _update_branch_progress(
        self,
        state: BranchTaskState,
        branch: SpatialBranchRecord,
        position,
        current_position,
        step: int,
    ) -> None:
        prev = _as_np_xy(branch.spine[-2]) if len(branch.spine) >= 2 else _as_np_xy(position)
        cur = _as_np_xy(position)
        progress = float(np.linalg.norm(cur - prev))
        state.progress_score = max(state.progress_score, progress)
        if progress <= self.config.stalled_progress_epsilon:
            state.steps_without_progress += 1
        else:
            state.steps_without_progress = 0
        if current_position is not None and len(branch.spine) > 2:
            d_to_old = min(
                float(np.linalg.norm(_as_np_xy(current_position) - _as_np_xy(p)))
                for p in branch.spine[:-1]
            )
            if d_to_old <= self.config.revisit_distance_m:
                state.status = BranchTaskStatus.REVISITING
            else:
                state.status = BranchTaskStatus.ADVANCING
        elif state.status == BranchTaskStatus.NEW:
            state.status = BranchTaskStatus.ADVANCING
        if state.steps_without_progress >= self.config.stalled_after_steps:
            state.status = BranchTaskStatus.STALLED
        state.updated_step = step

    def _close_missing_active_tips(
        self,
        working_memory: SubtaskWorkingMemory,
        current_fids: set,
        step: int,
    ) -> None:
        for branch in working_memory.spatial_branches.values():
            tip = branch.active_tip_frontier_id
            if tip is None or tip in current_fids:
                continue
            state = working_memory.branch_task_states.get(branch.spatial_branch_id)
            if state is None:
                continue
            if state.status == BranchTaskStatus.ADVANCING:
                state.status = BranchTaskStatus.STALLED
                state.updated_step = step

    def _frontier_position_habitat(self, tsdf_planner, frontier):
        try:
            return tsdf_planner.voxel2habitat(frontier.position)
        except Exception:
            return getattr(frontier, "position", [0.0, 0.0, 0.0])

    def _next_branch_id(self, existing_ids: Iterable[str]) -> str:
        used = set(existing_ids)
        while True:
            self._counter += 1
            bid = f"B{self._counter:03d}"
            if bid not in used:
                return bid

    @staticmethod
    def _event(
        type_: EventType,
        step: int,
        entity_id: str,
        payload: Dict,
    ) -> TypedEvent:
        return TypedEvent(
            event_id=f"{type_.value}:{entity_id}:{step}",
            type=type_,
            step=step,
            entity_id=entity_id,
            payload=payload,
        )
