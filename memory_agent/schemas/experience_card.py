from __future__ import annotations

from dataclasses import dataclass, field

from .common import SerializableMixin


@dataclass
class ExperienceCard(SerializableMixin):
    memory_id: str
    memory_type: str = "experience"
    description: str = "" # LLM 生成的一段自然语言，包含 situation + local goal + uncertainty
    action_sequence: list[dict] = field(default_factory=list)
    outcome_shift: str = ""
    outcome_type: str = "partial_success" # success | partial_success | failure | unsafe
    failure_mode: str | None = None
    boundary: str = ""
    modality_flags: list[str] = field(default_factory=list)
    retrieval_tags: list[str] = field(default_factory=list)
    risk_tags: list[str] = field(default_factory=list)
    confidence: float = 0.5
    support_count: int = 1
    conflict_group_id: str | None = None
    source_episode_ids: list[str] = field(default_factory=list)
    source_case_ids: list[str] = field(default_factory=list)
    source_turn_ids: list[int] = field(default_factory=list)
