from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import SerializableMixin


@dataclass
class EvidenceItem(SerializableMixin):
    evidence_id: str
    content: str
    source: str
    modality: str
    polarity: str
    confidence: float = 0.5
    linked_hypotheses: list[str] = field(default_factory=list)
    turn_id: int = 0
    source_field_refs: list[str] = field(default_factory=list)
    signal_tags: list[str] = field(default_factory=list)


@dataclass
class HypothesisState(SerializableMixin):
    name: str
    probability_hint: str = "medium"
    supporting_evidence: list[str] = field(default_factory=list)
    conflicting_evidence: list[str] = field(default_factory=list)
    source_field_refs: list[str] = field(default_factory=list)


@dataclass
class CaseState(SerializableMixin):
    case_id: str
    turn_id: int = 0
    problem_summary: str = ""
    evidence_items: list[EvidenceItem] = field(default_factory=list)
    missing_info: list[str] = field(default_factory=list)
    active_hypotheses: list[HypothesisState] = field(default_factory=list)
    local_goal: str = ""
    uncertainty_summary: str = ""
    finalize_risk: str = "high"
    modality_flags: list[str] = field(default_factory=list)
    next_action_constraints: list[str] = field(default_factory=list)
    source_field_refs: list[str] = field(default_factory=list)


@dataclass
class MemoryQueryStructured(SerializableMixin):
    situation_anchor: str = ""
    local_goal: str = ""
    uncertainty_focus: str = ""
    active_hypotheses: list[str] = field(default_factory=list)
    key_positive_evidence: list[str] = field(default_factory=list)
    key_negative_evidence: list[str] = field(default_factory=list)
    missing_info: list[str] = field(default_factory=list)
    modality_flags: list[str] = field(default_factory=list)
    finalize_risk: str = "high"
    finalize_risk_reason: str = ""
    retrieval_intent: str = "mixed"
    current_action_candidates: list[str] = field(default_factory=list)
    source_field_refs: list[str] = field(default_factory=list)


@dataclass
class MemoryApplicabilityAssessment(SerializableMixin):
    memory_id: str
    memory_type: str = "experience"
    memory_content: dict[str, Any] = field(default_factory=dict)
    applicability: str = "medium"
    reason: str = ""
    matched_aspects: list[str] = field(default_factory=list)
    mismatched_aspects: list[str] = field(default_factory=list)
    boundary_violation: bool = False
    action_bias: dict[str, float] = field(default_factory=dict)
    blocked_actions: list[str] = field(default_factory=list)
    controller_decision: str = "hint"
    relevance_score: float = 0.0
    source_field_refs: list[str] = field(default_factory=list)


@dataclass
class MemoryQuery(SerializableMixin):
    query_text: str = ""
    structured: MemoryQueryStructured = field(default_factory=MemoryQueryStructured)
    source_field_refs: list[str] = field(default_factory=list)


@dataclass
class ActionCandidate(SerializableMixin):
    action_id: str
    action_type: str
    action_label: str
    action_content: str = ""
    planner_score: float = 0.0
    source_field_refs: list[str] = field(default_factory=list)


@dataclass
class IntentPlan(SerializableMixin):
    turn_id: int
    action_candidates: list[ActionCandidate] = field(default_factory=list)
    memory_query: MemoryQuery = field(default_factory=MemoryQuery)
    source_field_refs: list[str] = field(default_factory=list)


@dataclass
class RetrievalHit(SerializableMixin):
    item_id: str
    retrieval_score: float
    matched_fields: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    source_field_refs: list[str] = field(default_factory=list)


@dataclass
class MemoryRetrievalResult(SerializableMixin):
    turn_id: int
    experience_hits: list[RetrievalHit] = field(default_factory=list)
    negative_experience_hits: list[RetrievalHit] = field(default_factory=list)
    skill_hits: list[RetrievalHit] = field(default_factory=list)
    knowledge_hits: list[RetrievalHit] = field(default_factory=list)
    source_field_refs: list[str] = field(default_factory=list)


@dataclass
class ActionAssessment(SerializableMixin):
    action_id: str
    decision: str = "hint"
    rationale: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    supporting_experience_ids: list[str] = field(default_factory=list)
    supporting_skill_ids: list[str] = field(default_factory=list)
    supporting_knowledge_ids: list[str] = field(default_factory=list)
    source_field_refs: list[str] = field(default_factory=list)


@dataclass
class ApplicabilityResult(SerializableMixin):
    turn_id: int
    memory_assessments: list[MemoryApplicabilityAssessment] = field(default_factory=list)
    action_assessments: list[ActionAssessment] = field(default_factory=list)
    controller_summary: dict[str, Any] = field(default_factory=dict)
    hard_block_actions: list[str] = field(default_factory=list)
    source_field_refs: list[str] = field(default_factory=list)


@dataclass
class CandidateRanking(SerializableMixin):
    action_id: str
    planner_score: float = 0.0
    controller_adjustment: float = 0.0
    final_score: float = 0.0
    blocked: bool = False
    source_field_refs: list[str] = field(default_factory=list)


@dataclass
class ActionDecision(SerializableMixin):
    turn_id: int
    chosen_action: dict[str, Any] = field(default_factory=dict)
    candidate_rankings: list[CandidateRanking] = field(default_factory=list)
    final_rationale: str = ""
    source_field_refs: list[str] = field(default_factory=list)


@dataclass
class ExecutionResult(SerializableMixin):
    turn_id: int
    executed_action: dict[str, Any] = field(default_factory=dict)
    env_response: dict[str, Any] = field(default_factory=dict)
    execution_status: str = "success"
    source_field_refs: list[str] = field(default_factory=list)


@dataclass
class MedEnvCaseBundle(SerializableMixin):
    case_id: str = ""
    ehr: dict[str, Any] = field(default_factory=dict)
    source_field_refs: list[str] = field(default_factory=lambda: ["ehr"])
