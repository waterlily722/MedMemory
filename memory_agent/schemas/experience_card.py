from __future__ import annotations

from dataclasses import dataclass, field

from .common import SerializableMixin


@dataclass
class ExperienceCard(SerializableMixin):
    memory_id: str
    memory_type: str = "experience"

    situation_text: str = ""
    action_text: str = ""
    outcome_text: str = ""
    boundary_text: str = ""

    action_sequence: list[dict[str, str]] = field(default_factory=list)

    outcome_type: str = "partial_success"  # success | partial_success | failure | unsafe
    failure_mode: str = ""

    retrieval_tags: list[str] = field(default_factory=list)
    risk_tags: list[str] = field(default_factory=list)

    confidence: float = 0.5
    support_count: int = 1
    conflict_group_id: str = ""

    source_episode_ids: list[str] = field(default_factory=list)
    source_case_ids: list[str] = field(default_factory=list)
    source_turn_ids: list[int] = field(default_factory=list)