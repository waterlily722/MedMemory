from __future__ import annotations

from dataclasses import dataclass, field

from .common import SerializableMixin


@dataclass
class ExperienceCard(SerializableMixin):
    item_id: str
    experience_id: str = ""
    situation_anchor: str = ""
    local_goal: str = ""
    action_sequence: list[dict[str, str]] = field(default_factory=list)
    outcome_shift: str = ""
    boundary: str = ""
    outcome_type: str = "partial_success"
    key_evidence: list[str] = field(default_factory=list)
    missing_info: list[str] = field(default_factory=list)
    applicability_conditions: list[str] = field(default_factory=list)
    non_applicability_conditions: list[str] = field(default_factory=list)
    retrieval_tags: list[str] = field(default_factory=list)
    confidence: float = 0.5
    uncertainty_state: str = ""
    success_signal: str = "partial"
    failure_mode: str = ""
    error_tag: list[str] = field(default_factory=list)
    support_count: int = 1
    conflict_group_id: str = ""
    hypotheses: list[str] = field(default_factory=list)
    source_turn_ids: list[str] = field(default_factory=list)
    source_episode_ids: list[str] = field(default_factory=list)
    source_case_ids: list[str] = field(default_factory=list)
    visual_signature: dict[str, str] = field(default_factory=dict)
    source_field_refs: list[str] = field(default_factory=list)
