from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import SerializableMixin


@dataclass
class TurnRecord(SerializableMixin):
    case_id: str
    turn_id: int
    timestamp: str
    doctor_action: dict[str, Any] = field(default_factory=dict)
    observation: dict[str, Any] = field(default_factory=dict)
    case_state_before: dict[str, Any] = field(default_factory=dict)
    case_state_after: dict[str, Any] = field(default_factory=dict)
    retrieved_memory_ids: dict[str, list[str]] = field(default_factory=dict)
    selected_memory_ids: list[str] = field(default_factory=list)
    rejected_memory_ids: list[str] = field(default_factory=list)
    decision_rationale: str = ""
    reward_signal: float | None = None
    error_signal: list[str] | None = None
    memory_guidance: dict[str, Any] = field(default_factory=dict)
    source_field_refs: list[str] = field(default_factory=list)


@dataclass
class TurnFeedback(SerializableMixin):
    turn_id: int
    local_gain: dict[str, Any] = field(default_factory=dict)
    local_cost: dict[str, Any] = field(default_factory=dict)
    safety_signal: dict[str, Any] = field(default_factory=dict)
    source_field_refs: list[str] = field(default_factory=list)


@dataclass
class EpisodeFeedback(SerializableMixin):
    episode_id: str
    offline_supervision: dict[str, Any] = field(default_factory=dict)
    trajectory_metrics: dict[str, Any] = field(default_factory=dict)
    reward: dict[str, Any] = field(default_factory=dict)
    source_field_refs: list[str] = field(default_factory=list)


@dataclass
class DistilledEpisode(SerializableMixin):
    episode_id: str
    summary: dict[str, Any] = field(default_factory=dict)
    candidate_experience_items: list[dict[str, Any]] = field(default_factory=list)
    candidate_skill_items: list[dict[str, Any]] = field(default_factory=list)
    revision_signals: dict[str, Any] = field(default_factory=dict)
    source_field_refs: list[str] = field(default_factory=list)


@dataclass
class MemoryUpdateOperation(SerializableMixin):
    op_type: str
    target_memory: str
    target_item_id: str = ""
    source_item_ids: list[str] = field(default_factory=list)
    reason: str = ""
    source_field_refs: list[str] = field(default_factory=list)


@dataclass
class MemoryUpdatePlan(SerializableMixin):
    episode_id: str
    operations: list[MemoryUpdateOperation] = field(default_factory=list)
    backend_updates: dict[str, Any] = field(default_factory=dict)
    source_field_refs: list[str] = field(default_factory=list)
