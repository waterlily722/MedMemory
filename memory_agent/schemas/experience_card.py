from __future__ import annotations

from dataclasses import dataclass, field

from .common import SerializableMixin


@dataclass
class ExperienceCard(SerializableMixin):
    item_id: str
    situation_anchor: str
    local_goal: str
    action_sequence: list[dict[str, str]] = field(default_factory=list)
    outcome_shift: str = ""
    boundary: str = ""
    outcome_type: str = "partial_success"
    error_tag: list[str] = field(default_factory=list)
    support_count: int = 1
    source_episode_ids: list[str] = field(default_factory=list)
    source_case_ids: list[str] = field(default_factory=list)
    visual_signature: dict[str, str] = field(default_factory=dict)
    source_field_refs: list[str] = field(default_factory=list)
