from __future__ import annotations

from dataclasses import dataclass, field

from .common import SerializableMixin


@dataclass
class MemoryQuery(SerializableMixin):
    query_text: str
    situation_anchor: str
    local_goal: str
    uncertainty_focus: str
    positive_evidence: list[str] = field(default_factory=list)
    negative_evidence: list[str] = field(default_factory=list)
    missing_info: list[str] = field(default_factory=list)
    active_hypotheses: list[str] = field(default_factory=list)
    modality_need: list[str] = field(default_factory=list)
    candidate_action_need: list[str] = field(default_factory=list)
    finalize_risk: str = "high"
    finalize_risk_reason: str = ""
    retrieval_intent: str = "mixed"
