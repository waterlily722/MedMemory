from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Union, get_args, get_origin


def _serialize(value: Any) -> Any:
    if isinstance(value, SerializableMixin):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


def _unwrap_optional(tp: Any) -> Any:
    origin = get_origin(tp)
    if origin not in (Union, Optional):
        return tp
    args = [arg for arg in get_args(tp) if arg is not type(None)]
    return args[0] if len(args) == 1 else tp


def _deserialize(tp: Any, value: Any) -> Any:
    if value is None:
        return None

    tp = _unwrap_optional(tp)
    origin = get_origin(tp)

    if origin is list:
        item_type = get_args(tp)[0] if get_args(tp) else Any
        return [_deserialize(item_type, item) for item in value]

    if origin is dict:
        key_type, val_type = get_args(tp) if get_args(tp) else (Any, Any)
        return {
            _deserialize(key_type, key): _deserialize(val_type, item)
            for key, item in value.items()
        }

    if isinstance(tp, type) and issubclass(tp, SerializableMixin):
        if isinstance(value, tp):
            return value
        return tp.from_dict(value)

    return value


class SerializableMixin:
    source_field_refs: List[str]

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for f in fields(self):
            data[f.name] = _serialize(getattr(self, f.name))
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        data = data or {}
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            kwargs[f.name] = _deserialize(f.type, data.get(f.name))
        return cls(**kwargs)


@dataclass
class RankedIntent(SerializableMixin):
    intent_type: str
    score: float = 0.0
    rationale: str = ""
    source_field_refs: List[str] = field(default_factory=list)


@dataclass
class ActionCandidate(SerializableMixin):
    action_id: str
    action_type: str
    action_text: str = ""
    action_args: Dict[str, Any] = field(default_factory=dict)
    planner_score: float = 0.0
    stop_conditions: List[str] = field(default_factory=list)
    source_field_refs: List[str] = field(default_factory=list)


@dataclass
class RetrievalHit(SerializableMixin):
    item_id: str
    retrieval_score: float = 0.0
    matched_fields: List[str] = field(default_factory=list)
    source_field_refs: List[str] = field(default_factory=list)


@dataclass
class ActionAssessment(SerializableMixin):
    action_id: str
    supporting_experience_ids: List[str] = field(default_factory=list)
    supporting_skill_ids: List[str] = field(default_factory=list)
    triggered_guardrail_ids: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    decision: str = "hint"
    rationale: str = ""
    source_field_refs: List[str] = field(default_factory=list)


@dataclass
class CandidateRanking(SerializableMixin):
    action_id: str
    planner_score: float = 0.0
    controller_adjustment: float = 0.0
    final_score: float = 0.0
    blocked: bool = False
    source_field_refs: List[str] = field(default_factory=list)


@dataclass
class MemoryUpdateOperation(SerializableMixin):
    op_type: str
    target_memory: str
    target_item_id: str = ""
    source_item_ids: List[str] = field(default_factory=list)
    reason: str = ""
    source_field_refs: List[str] = field(default_factory=list)


@dataclass
class MedEnvCaseBundle(SerializableMixin):
    case_id: str = ""
    ehr: Dict[str, Any] = field(default_factory=dict)
    source_field_refs: List[str] = field(default_factory=lambda: ["ehr"])


@dataclass
class CanonicalEvidence(SerializableMixin):
    evidence_id: str
    turn_id: str
    source_type: str
    raw_field_refs: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    raw_structured: Dict[str, Any] = field(default_factory=dict)
    raw_image_refs: List[str] = field(default_factory=list)
    facts: List[str] = field(default_factory=list)
    negated_facts: List[str] = field(default_factory=list)
    uncertainty_patterns: List[str] = field(default_factory=list)
    symptom_patterns: List[str] = field(default_factory=list)
    test_patterns: List[str] = field(default_factory=list)
    route_flags: Dict[str, bool] = field(default_factory=dict)
    source_field_refs: List[str] = field(default_factory=list)


@dataclass
class CaseMemory(SerializableMixin):
    meta: Dict[str, Any] = field(default_factory=dict)
    raw_snapshot: Dict[str, Any] = field(default_factory=dict)
    derived_state: Dict[str, Any] = field(default_factory=dict)
    source_field_refs: List[str] = field(default_factory=list)


@dataclass
class IntentPlan(SerializableMixin):
    turn_id: str
    ranked_intents: List[RankedIntent] = field(default_factory=list)
    action_candidates: List[ActionCandidate] = field(default_factory=list)
    query_signature: Dict[str, Any] = field(default_factory=dict)
    source_field_refs: List[str] = field(default_factory=list)


@dataclass
class ExperienceItem(SerializableMixin):
    item_id: str
    source_episode_ids: List[str] = field(default_factory=list)
    source_case_ids: List[str] = field(default_factory=list)
    state_signature: Dict[str, Any] = field(default_factory=dict)
    trigger_signature: Dict[str, Any] = field(default_factory=dict)
    action_sequence: List[Dict[str, Any]] = field(default_factory=list)
    effect_summary: Dict[str, Any] = field(default_factory=dict)
    utility_stats: Dict[str, Any] = field(default_factory=dict)
    applicability: Dict[str, Any] = field(default_factory=dict)
    failure_mechanism: Dict[str, Any] = field(default_factory=dict)
    safer_alternative: Dict[str, Any] = field(default_factory=dict)
    source_field_refs: List[str] = field(default_factory=list)


@dataclass
class SkillItem(SerializableMixin):
    item_id: str
    source_experience_ids: List[str] = field(default_factory=list)
    source_case_ids: List[str] = field(default_factory=list)
    skill_meta: Dict[str, Any] = field(default_factory=dict)
    trigger_signature: Dict[str, Any] = field(default_factory=dict)
    action_sequence: List[Dict[str, Any]] = field(default_factory=list)
    success_criteria: List[Dict[str, Any]] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)
    reliability: Dict[str, Any] = field(default_factory=dict)
    source_field_refs: List[str] = field(default_factory=list)


@dataclass
class GuardrailItem(SerializableMixin):
    item_id: str
    source_episode_ids: List[str] = field(default_factory=list)
    source_case_ids: List[str] = field(default_factory=list)
    trigger_signature: Dict[str, Any] = field(default_factory=dict)
    risky_action: Dict[str, Any] = field(default_factory=dict)
    failure_mechanism: Dict[str, Any] = field(default_factory=dict)
    safer_alternative: Dict[str, Any] = field(default_factory=dict)
    risk_stats: Dict[str, Any] = field(default_factory=dict)
    source_field_refs: List[str] = field(default_factory=list)


@dataclass
class MemoryRetrievalResult(SerializableMixin):
    turn_id: str
    query_signature: Dict[str, Any] = field(default_factory=dict)
    experience_hits: List[RetrievalHit] = field(default_factory=list)
    skill_hits: List[RetrievalHit] = field(default_factory=list)
    guardrail_hits: List[RetrievalHit] = field(default_factory=list)
    source_field_refs: List[str] = field(default_factory=list)


@dataclass
class ApplicabilityResult(SerializableMixin):
    turn_id: str
    action_assessments: List[ActionAssessment] = field(default_factory=list)
    source_field_refs: List[str] = field(default_factory=list)


@dataclass
class ActionDecision(SerializableMixin):
    turn_id: str
    chosen_action: Dict[str, Any] = field(default_factory=dict)
    candidate_rankings: List[CandidateRanking] = field(default_factory=list)
    final_rationale: str = ""
    source_field_refs: List[str] = field(default_factory=list)


@dataclass
class ExecutionResult(SerializableMixin):
    turn_id: str
    executed_action: Dict[str, Any] = field(default_factory=dict)
    env_response: Dict[str, Any] = field(default_factory=dict)
    execution_status: str = "success"
    source_field_refs: List[str] = field(default_factory=list)


@dataclass
class TurnFeedback(SerializableMixin):
    turn_id: str
    local_gain: Dict[str, Any] = field(default_factory=dict)
    local_cost: Dict[str, Any] = field(default_factory=dict)
    safety_signal: Dict[str, Any] = field(default_factory=dict)
    source_field_refs: List[str] = field(default_factory=list)


@dataclass
class EpisodeFeedback(SerializableMixin):
    episode_id: str
    offline_supervision: Dict[str, Any] = field(default_factory=dict)
    trajectory_metrics: Dict[str, Any] = field(default_factory=dict)
    reward: Dict[str, Any] = field(default_factory=dict)
    source_field_refs: List[str] = field(default_factory=list)


@dataclass
class DistilledEpisode(SerializableMixin):
    episode_id: str
    summary: Dict[str, Any] = field(default_factory=dict)
    candidate_experience_items: List[ExperienceItem] = field(default_factory=list)
    candidate_skill_items: List[SkillItem] = field(default_factory=list)
    candidate_guardrail_items: List[GuardrailItem] = field(default_factory=list)
    revision_signals: Dict[str, Any] = field(default_factory=dict)
    source_field_refs: List[str] = field(default_factory=list)


@dataclass
class MemoryUpdatePlan(SerializableMixin):
    episode_id: str
    operations: List[MemoryUpdateOperation] = field(default_factory=list)
    backend_updates: Dict[str, Any] = field(default_factory=dict)
    source_field_refs: List[str] = field(default_factory=list)
