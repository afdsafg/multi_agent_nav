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
    split_distance_m: float = 0.8
    split_tip_distance_m: float = 1.6
    split_after_steps: int = 2
    merge_distance_m: float = 0.5


class BranchTracker:
    """Maintains durable spatial branches from current TSDF frontiers."""

    def __init__(self, config: Optional[BranchTrackerConfig] = None) -> None:
        self.config = config or BranchTrackerConfig()
        self._counter = 0
        self._split_candidates: Dict[int, Tuple[str, int]] = {}

    def reset_task_scope(self) -> None:
        """Do not clear branch records; only task-local state lives in memory."""
        self._split_candidates = {}

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
            split_from = self._persistent_split_source(
                [existing], frontier_id, position
            )
            if split_from is not None:
                self._split_candidates.pop(int(frontier_id), None)
                branch = self._new_branch(
                    working_memory,
                    frontier_id=frontier_id,
                    position=position,
                    step=step,
                )
                branch.aliases.append(f"split_from:{split_from.spatial_branch_id}")
                working_memory.upsert_spatial_branch(branch)
                return branch
            if self._pending_split_source(working_memory, frontier_id) is not None:
                return existing
            self._update_branch(existing, frontier_id, position, step)
            working_memory.upsert_spatial_branch(existing)
            self._mark_close_branch_merges(working_memory, existing, position)
            return existing

        match = self._nearest_branch(working_memory.spatial_branches.values(), position)
        if match is not None:
            self._update_branch(match, frontier_id, position, step)
            working_memory.upsert_spatial_branch(match)
            self._mark_close_branch_merges(working_memory, match, position)
            return match

        split_from = self._persistent_split_source(
            working_memory.spatial_branches.values(), frontier_id, position
        )
        if split_from is not None:
            self._split_candidates.pop(int(frontier_id), None)
            branch = self._new_branch(
                working_memory,
                frontier_id=frontier_id,
                position=position,
                step=step,
            )
            branch.aliases.append(f"split_from:{split_from.spatial_branch_id}")
            working_memory.upsert_spatial_branch(branch)
            return branch
        pending_source = self._pending_split_source(working_memory, frontier_id)
        if pending_source is not None:
            return pending_source

        branch = self._new_branch(
            working_memory,
            frontier_id=frontier_id,
            position=position,
            step=step,
        )
        merged_into = self._merge_target(
            working_memory.spatial_branches.values(), branch, position
        )
        if merged_into is not None:
            branch.merged_into = merged_into.spatial_branch_id
            branch.aliases.append(f"merged_into:{merged_into.spatial_branch_id}")
        working_memory.upsert_spatial_branch(branch)
        return branch

    def _new_branch(
        self,
        working_memory: SubtaskWorkingMemory,
        frontier_id: int,
        position,
        step: int,
    ) -> SpatialBranchRecord:
        bid = self._next_branch_id(working_memory.spatial_branches.keys())
        branch = SpatialBranchRecord(
            spatial_branch_id=bid,
            frontier_ids=[int(frontier_id)],
            spine=[position],
            active_tip_frontier_id=int(frontier_id),
            created_step=step,
            updated_step=step,
        )
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

    def _persistent_split_source(
        self,
        branches: Iterable[SpatialBranchRecord],
        frontier_id: int,
        position,
    ) -> Optional[SpatialBranchRecord]:
        pos = _as_np_xy(position)
        source = None
        source_d = float("inf")
        for branch in branches:
            if branch.merged_into or len(branch.spine) < 2:
                continue
            tip_d = float(np.linalg.norm(pos - _as_np_xy(branch.spine[-1])))
            if tip_d <= self.config.split_tip_distance_m:
                continue
            spine_d = min(
                float(np.linalg.norm(pos - _as_np_xy(p)))
                for p in branch.spine[:-1]
            )
            if spine_d <= self.config.split_distance_m and spine_d < source_d:
                source = branch
                source_d = spine_d
        if source is None:
            self._split_candidates.pop(int(frontier_id), None)
            return None

        prev_bid, count = self._split_candidates.get(int(frontier_id), ("", 0))
        count = count + 1 if prev_bid == source.spatial_branch_id else 1
        self._split_candidates[int(frontier_id)] = (source.spatial_branch_id, count)
        return source if count >= self.config.split_after_steps else None

    def _pending_split_source(
        self,
        working_memory: SubtaskWorkingMemory,
        frontier_id: int,
    ) -> Optional[SpatialBranchRecord]:
        pending = self._split_candidates.get(int(frontier_id))
        if pending is None:
            return None
        bid, _count = pending
        return working_memory.spatial_branches.get(bid)

    def _merge_target(
        self,
        branches: Iterable[SpatialBranchRecord],
        branch: SpatialBranchRecord,
        position,
    ) -> Optional[SpatialBranchRecord]:
        pos = _as_np_xy(position)
        best = None
        best_d = float("inf")
        for other in branches:
            if other.spatial_branch_id == branch.spatial_branch_id:
                continue
            if other.merged_into or not other.spine:
                continue
            d = float(np.linalg.norm(pos - _as_np_xy(other.spine[-1])))
            if d <= self.config.merge_distance_m and d < best_d:
                best = other
                best_d = d
        return best

    def _mark_close_branch_merges(
        self,
        working_memory: SubtaskWorkingMemory,
        target_branch: SpatialBranchRecord,
        position,
    ) -> None:
        pos = _as_np_xy(position)
        for other in working_memory.spatial_branches.values():
            if other.spatial_branch_id == target_branch.spatial_branch_id:
                continue
            if other.merged_into or not other.spine:
                continue
            d = float(np.linalg.norm(pos - _as_np_xy(other.spine[-1])))
            if d <= self.config.merge_distance_m:
                other.merged_into = target_branch.spatial_branch_id
                alias = f"merged_into:{target_branch.spatial_branch_id}"
                if alias not in other.aliases:
                    other.aliases.append(alias)

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
