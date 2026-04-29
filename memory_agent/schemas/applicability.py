from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .common import SerializableMixin


@dataclass
class MemoryApplicabilityAssessment(SerializableMixin):
    memory_id: str
    memory_type: str
    applicability: str = "medium"
    reason: str = ""
    matched_aspects: list[str] = field(default_factory=list)
    mismatched_aspects: list[str] = field(default_factory=list)
    boundary_violation: Optional[str] = None
    action_bias: dict[str, float] = field(default_factory=dict)
    blocked_actions: list[str] = field(default_factory=list)
    controller_decision: str = "ignore"


@dataclass
class ActionAssessment(SerializableMixin):
    action_type: str
    action_label: str
    score_delta: float = 0.0
    blocked: bool = False
    reason: str = ""
    supporting_memory_ids: list[str] = field(default_factory=list)
    warning_memory_ids: list[str] = field(default_factory=list)


@dataclass
class ApplicabilityResult(SerializableMixin):
    memory_assessments: list[MemoryApplicabilityAssessment] = field(default_factory=list)
    action_assessments: list[ActionAssessment] = field(default_factory=list)
    hard_blocked_actions: list[str] = field(default_factory=list)
    risk_warning: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, object] | None):
        if data is None:
            return cls()
        payload = dict(data)
        payload["memory_assessments"] = [MemoryApplicabilityAssessment.from_dict(item) for item in payload.get("memory_assessments", [])]
        payload["action_assessments"] = [ActionAssessment.from_dict(item) for item in payload.get("action_assessments", [])]
        return cls(**payload)
