from __future__ import annotations

from dataclasses import dataclass, field

from .common import SerializableMixin


@dataclass
class SkillProcedureStep(SerializableMixin):
    step_id: int
    action_type: str
    action_label: str
    expected_observation: str = ""
    fallback_action: str = ""
    source_field_refs: list[str] = field(default_factory=list)


@dataclass
class SkillCard(SerializableMixin):
    skill_id: str
    skill_name: str
    skill_trigger: str
    clinical_goal: str
    preconditions: list[str] = field(default_factory=list)
    procedure_template: list[SkillProcedureStep] = field(default_factory=list)
    stop_condition: list[str] = field(default_factory=list)
    boundary: list[str] = field(default_factory=list)
    contraindications: list[str] = field(default_factory=list)
    source_experience_ids: list[str] = field(default_factory=list)
    support_count: int = 1
    success_rate: float = 1.0
    unsafe_rate: float = 0.0
    confidence: float = 0.5
    visual_trigger: dict[str, str] = field(default_factory=dict)
    source_case_ids: list[str] = field(default_factory=list)
    source_field_refs: list[str] = field(default_factory=list)
