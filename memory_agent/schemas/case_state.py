from __future__ import annotations

from dataclasses import dataclass, field

from .common import SerializableMixin


@dataclass
class CaseState(SerializableMixin):
    case_id: str
    turn_id: int = 0
    problem_summary: str = ""
    key_evidence: list[str] = field(default_factory=list)
    negative_evidence: list[str] = field(default_factory=list)
    missing_info: list[str] = field(default_factory=list)
    active_hypotheses: list[str] = field(default_factory=list)
    local_goal: str = ""
    uncertainty_summary: str = ""
    finalize_risk: str = "high" # low | medium | high
    modality_flags: list[str] = field(default_factory=list)
    reviewed_modalities: list[str] = field(default_factory=list)
    interaction_history_summary: str = ""