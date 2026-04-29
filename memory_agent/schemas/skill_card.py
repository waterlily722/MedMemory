from __future__ import annotations

from dataclasses import dataclass, field

from .common import SerializableMixin


@dataclass
class SkillCard(SerializableMixin):
    memory_id: str
    memory_type: str = "skill"
    skill_name: str = ""
    clinical_situation: str = ""
    local_goal: str = ""
    trigger_conditions: list[str] = field(default_factory=list)
    procedure: list[dict] = field(default_factory=list)
    stop_conditions: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    contraindications: list[str] = field(default_factory=list)
    required_modalities: list[str] = field(default_factory=list)
    applicability_boundary: str = ""
    source_experience_ids: list[str] = field(default_factory=list)
    evidence_count: int = 0
    confidence: float = 0.5
    version: int = 1
    unique_case_count: int = 0
    success_rate: float = 0.0
    unsafe_rate: float = 0.0
