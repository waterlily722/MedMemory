from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schemas import (
    ActionAssessment,
    ActionCandidate,
    ActionDecision,
    ApplicabilityResult,
    CandidateRanking,
    CaseState,
    DistilledEpisode,
    EpisodeFeedback,
    ExecutionResult,
    ExperienceCard,
    IntentPlan,
    KnowledgeItem,
    MedEnvCaseBundle,
    MemoryQuery,
    MemoryQueryStructured,
    MemoryRetrievalResult,
    MemoryUpdateOperation,
    MemoryUpdatePlan,
    RetrievalHit,
    SerializableMixin,
    SkillCard,
    TurnFeedback,
)


@dataclass
class RankedIntent(SerializableMixin):
    intent_type: str
    score: float = 0.0
    rationale: str = ""
    source_field_refs: list[str] = field(default_factory=list)


@dataclass
class CanonicalEvidence(SerializableMixin):
    evidence_id: str
    turn_id: str
    source_type: str
    raw_field_refs: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    raw_structured: dict[str, Any] = field(default_factory=dict)
    raw_image_refs: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    negated_facts: list[str] = field(default_factory=list)
    uncertainty_patterns: list[str] = field(default_factory=list)
    symptom_patterns: list[str] = field(default_factory=list)
    test_patterns: list[str] = field(default_factory=list)
    route_flags: dict[str, bool] = field(default_factory=dict)
    source_field_refs: list[str] = field(default_factory=list)


# Backward-compatible aliases used by existing wrapper/pipeline code.
CaseMemory = CaseState
ExperienceItem = ExperienceCard
SkillItem = SkillCard
GuardrailItem = KnowledgeItem

__all__ = [
    "SerializableMixin",
    "RankedIntent",
    "ActionCandidate",
    "RetrievalHit",
    "ActionAssessment",
    "CandidateRanking",
    "MemoryUpdateOperation",
    "MedEnvCaseBundle",
    "CanonicalEvidence",
    "CaseMemory",
    "IntentPlan",
    "ExperienceItem",
    "SkillItem",
    "GuardrailItem",
    "MemoryRetrievalResult",
    "ApplicabilityResult",
    "ActionDecision",
    "ExecutionResult",
    "TurnFeedback",
    "EpisodeFeedback",
    "DistilledEpisode",
    "MemoryUpdatePlan",
    "CaseState",
    "MemoryQuery",
    "MemoryQueryStructured",
    "ExperienceCard",
    "SkillCard",
    "KnowledgeItem",
]
