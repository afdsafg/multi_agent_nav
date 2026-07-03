import json
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import os
import re


@dataclass
class TextMemoryEntry:
    id: str
    content: str  # 文本描述内容
    timestamp: datetime = field(default_factory=datetime.now)  # 时间戳
    importance: float = 1.0  # 重要性评分
    entry_type: str = "general"  # 记忆类型
    step: int = 0  # 探索步骤
    position: Optional[np.ndarray] = None  # 位置信息
    raw_response: Optional[str] = None  # 保留原始VLM响应
    structured_decision: Optional[Dict[str, Any]] = None  # 结构化决策信息


class TextLongTermMemory:
    """优化的文本长期记忆系统"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.entries: List[TextMemoryEntry] = []
        self.id_counter = 0
        # 记忆索引，用于快速检索
        self.type_index: Dict[str, List[str]] = {}  # type -> entry_ids
        self.step_index: Dict[int, List[str]] = {}  # step -> entry_ids
        self.timestamp_index: List[str] = []  # 按时间戳排序的entry_ids

    def add_entry(self, content: str, importance: float = 1.0, entry_type: str = "general",
                  step: int = 0, position: Optional[np.ndarray] = None,
                  raw_response: Optional[str] = None,
                  structured_decision: Optional[Dict[str, Any]] = None) -> str:
        """添加文本记忆条目"""
        entry_id = f"mem_{self.id_counter}"
        self.id_counter += 1

        entry = TextMemoryEntry(
            id=entry_id,
            content=content,
            importance=importance,
            entry_type=entry_type,
            step=step,
            position=position,
            raw_response=raw_response,
            structured_decision=structured_decision
        )

        self.entries.append(entry)

        # 更新索引
        if entry_type not in self.type_index:
            self.type_index[entry_type] = []
        self.type_index[entry_type].append(entry_id)

        if step not in self.step_index:
            self.step_index[step] = []
        self.step_index[step].append(entry_id)

        # 重新排序时间戳索引
        self.timestamp_index = [e.id for e in sorted(self.entries, key=lambda x: x.timestamp)]

        return entry_id


    def record_structured_agent_output(self, step: int, agent_type: str, structured_output: Dict,
                                      raw_response: str, position: Optional[np.ndarray] = None) -> str:
        """记录agent的结构化输出信息"""
        # 检查是否已经记录过相同步骤和agent_type的输出，防止重复记录
        existing_entries = self.retrieve_by_step(step, top_k=10)
        for entry in existing_entries:
            if (entry.entry_type == f"{agent_type}_output" and
                entry.step == step and
                entry.content == f"Step {step} - {agent_type.upper()} Output: {structured_output.get('parsed_decision', 'Unknown')}"):
                print(f"跳过重复的{agent_type}记录，步骤{step}")
                return entry.id

        # 提取关键信息
        action = structured_output.get('structured_output', {}).get('action', 'unknown')
        parsed_decision = structured_output.get('parsed_decision', 'Unknown')

        content = f"Step {step} - {agent_type.upper()} Output: {parsed_decision}"

        # 构建结构化决策信息
        structured_decision = {
            "agent_type": agent_type,
            "action": action,
            "step": step,
            "position": position.tolist() if position is not None and isinstance(position, np.ndarray) else position,
            "structured_output": structured_output.get('structured_output', {}),
            # 优先使用raw_response_summary，如果没有则使用reasoning（兼容性）
            "raw_response_summary": structured_output.get('raw_response_summary') or structured_output.get('reasoning', ''),
        }

        # 根据agent类型添加特定信息
        if agent_type == "planner":
            structured_decision.update({
                "target_type": structured_output.get('structured_output', {}).get('target_type', 'unknown'),
                "target_id": structured_output.get('structured_output', {}).get('target_id', None)
            })
        elif agent_type == "answerer":
            answer_text = structured_output.get('structured_output', {}).get('answer_text', '')
            if len(answer_text) > 100:  # 限制答案长度
                answer_text = answer_text[:100] + "...[truncated]"
            structured_decision.update({
                "answer_text": answer_text,
                "evidence_snapshot": structured_output.get('structured_output', {}).get('evidence_snapshot', None)
            })
        elif agent_type == "snapshot_manager":
            structured_decision.update({
                "retained_snapshots": structured_output.get('structured_output', {}).get('snapshot_ids', [])
            })
        elif agent_type == "frontier_manager":
            structured_decision.update({
                "retained_frontiers": structured_output.get('structured_output', {}).get('frontier_ids', [])
            })
        elif agent_type == "high_level_planner":
            # 特殊处理high_level_planner的todo_list
            todo_list = structured_output.get('todo_list', structured_output.get('structured_output', {}).get('todo_list', []))
            structured_decision.update({
                "todo_list": todo_list,
                "num_tasks": len(todo_list)
            })

        return self.add_entry(
            content=content,
            importance=0.8,
            entry_type=f"{agent_type}_output",
            step=step,
            position=position,
            raw_response=raw_response,  # 保留完整的原始响应
            structured_decision=structured_decision
        )

    def retrieve_by_type(self, entry_type: str, top_k: int = 5) -> List[TextMemoryEntry]:
        """按类型检索记忆"""
        # 使用索引快速检索
        entry_ids = self.type_index.get(entry_type, [])
        matching_entries = [entry for entry in self.entries if entry.id in entry_ids]
        matching_entries.sort(key=lambda x: x.importance, reverse=True)
        return matching_entries[:top_k]

    def retrieve_by_step(self, step: int, top_k: int = 5) -> List[TextMemoryEntry]:
        """按步骤检索记忆"""
        entry_ids = self.step_index.get(step, [])
        matching_entries = [entry for entry in self.entries if entry.id in entry_ids]
        matching_entries.sort(key=lambda x: x.timestamp)
        return matching_entries[:top_k]

    def retrieve_by_step_and_type(self, step: int, entry_type: str, top_k: int = 5) -> List[TextMemoryEntry]:
        """按步骤和类型检索记忆"""
        # 先获取指定步骤的所有条目
        step_entry_ids = self.step_index.get(step, [])
        # 然后过滤出指定类型的条目
        matching_entries = [entry for entry in self.entries if entry.id in step_entry_ids and entry.entry_type == entry_type]
        matching_entries.sort(key=lambda x: x.timestamp)
        return matching_entries[:top_k]

    def retrieve_by_time(self, minutes_back: int = 60, top_k: int = 5) -> List[TextMemoryEntry]:
        """按时间检索记忆（最近的）"""
        time_threshold = datetime.now() - timedelta(minutes=minutes_back)
        recent_entries = [entry for entry in self.entries if entry.timestamp >= time_threshold]
        recent_entries.sort(key=lambda x: x.timestamp, reverse=True)
        return recent_entries[:top_k]

    def get_latest_high_level_plan(self) -> Optional[TextMemoryEntry]:
        """返回最近一条 type='high_level_planner_output' 的记录，无则 None。"""
        matching = [e for e in self.entries if e.entry_type == "high_level_planner_output"]
        if not matching:
            return None
        matching.sort(key=lambda x: x.timestamp)
        return matching[-1]

    def start_new_subtask(self, subtask_id: str) -> None:
        """记录新 subtask_id 到内部列表，不清空记忆。"""
        if not hasattr(self, "subtask_ids"):
            self.subtask_ids: List[str] = []
        self.subtask_ids.append(subtask_id)
