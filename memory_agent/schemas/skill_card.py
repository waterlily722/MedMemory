from __future__ import annotations

from dataclasses import dataclass, field

from .common import SerializableMixin


@dataclass
class SkillCard(SerializableMixin):
    memory_id: str
    memory_type: str = "skill"

    skill_name: str = ""

    situation_text: str = ""
    goal_text: str = ""
    procedure_text: str = ""
    boundary_text: str = ""

    procedure: list[dict[str, str]] = field(default_factory=list)
    contraindications: list[str] = field(default_factory=list)

    source_experience_ids: list[str] = field(default_factory=list)

    evidence_count: int = 0
    unique_case_count: int = 0
    success_rate: float = 0.0
    unsafe_rate: float = 0.0

    confidence: float = 0.5
    version: int = 1
