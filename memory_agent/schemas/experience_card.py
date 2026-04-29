from __future__ import annotations

from dataclasses import dataclass, field

from .common import SerializableMixin


@dataclass
class ExperienceCard(SerializableMixin):
    memory_id: str
    memory_type: str = "experience"
    situation_anchor: str = ""
    local_goal: str = ""
    uncertainty_state: str = ""
    key_evidence: list[str] = field(default_factory=list)
    missing_info: list[str] = field(default_factory=list)
    active_hypotheses: list[str] = field(default_factory=list)
    action_sequence: list[dict] = field(default_factory=list)
    outcome_shift: str = ""
    outcome_type: str = "partial_success"
    failure_mode: str | None = None
    boundary: str = ""
    applicability_conditions: list[str] = field(default_factory=list)
    non_applicability_conditions: list[str] = field(default_factory=list)
    modality_flags: list[str] = field(default_factory=list)
    retrieval_tags: list[str] = field(default_factory=list)
    risk_tags: list[str] = field(default_factory=list)
    confidence: float = 0.5
    support_count: int = 1
    conflict_group_id: str | None = None
    source_episode_ids: list[str] = field(default_factory=list)
    source_case_ids: list[str] = field(default_factory=list)
