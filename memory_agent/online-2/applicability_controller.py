from __future__ import annotations

from typing import Any

from ..llm import LLMClient, applicability_prompt, parse_validate_repair
from ..llm.schemas import APPLICABILITY_SCHEMA
from ..schemas import (
    ActionAssessment,
    ApplicabilityResult,
    CaseState,
    MemoryApplicabilityAssessment,
    MemoryQuery,
    MemoryRetrievalResult,
    RetrievalHit,
)

DEFAULT_ACTIONS = [
    "ASK",
    "REQUEST_LAB",
    "REVIEW_IMAGE",
    "UPDATE_HYPOTHESIS",
    "FINALIZE_DIAGNOSIS",
]

UNSAFE_BLOCK_MIN_SCORE = 0.35
FAILURE_BLOCK_MIN_SCORE = 0.50
SKILL_APPLY_MIN_SCORE = 0.45


def _all_hits(retrieval_result: MemoryRetrievalResult) -> list[RetrievalHit]:
    return (
        retrieval_result.positive_experience_hits
        + retrieval_result.negative_experience_hits
        + retrieval_result.skill_hits
        + retrieval_result.knowledge_hits
    )


def _outcome_type(hit: RetrievalHit) -> str:
    if hit.memory_type != "experience":
        return ""
    return str(hit.content.get("outcome_type") or "").lower()


def _is_unsafe_experience(hit: RetrievalHit) -> bool:
    return _outcome_type(hit) == "unsafe"


def _is_failure_experience(hit: RetrievalHit) -> bool:
    return _outcome_type(hit) == "failure"


def _first_action_from_steps(steps: Any) -> str:
    if isinstance(steps, list) and steps:
        first = steps[0]
        if isinstance(first, dict):
            return str(first.get("action_type") or "").upper()
    return ""


def _infer_action_from_memory(hit: RetrievalHit) -> str:
    if hit.memory_type == "experience":
        action = _first_action_from_steps(hit.content.get("action_sequence") or [])
        if action:
            return action
        action_text = str(hit.content.get("action_text") or "").upper()
        for action_type in DEFAULT_ACTIONS:
            if action_type in action_text:
                return action_type

    if hit.memory_type == "skill":
        action = _first_action_from_steps(hit.content.get("procedure") or [])
        if action:
            return action
        procedure_text = str(hit.content.get("procedure_text") or "").upper()
        for action_type in DEFAULT_ACTIONS:
            if action_type in procedure_text:
                return action_type

    return ""


def _rule_memory_assessment(hit: RetrievalHit) -> MemoryApplicabilityAssessment:
    action = _infer_action_from_memory(hit)

    if _is_unsafe_experience(hit):
        should_block = bool(action and hit.score >= UNSAFE_BLOCK_MIN_SCORE)
        return MemoryApplicabilityAssessment(
            memory_id=hit.memory_id,
            memory_type=hit.memory_type,
            decision="block" if should_block else "hint",
            reason=(
                "Retrieved unsafe experience. Block only when similarity is high enough; "
                "otherwise use it as a warning."
            ),
            action_bias={action: -0.8} if action else {},
            blocked_actions=[action] if should_block else [],
        )

    if _is_failure_experience(hit):
        # Failure is usually a discouraging signal, not an automatic block.
        should_block = bool(action and hit.score >= FAILURE_BLOCK_MIN_SCORE)
        return MemoryApplicabilityAssessment(
            memory_id=hit.memory_id,
            memory_type=hit.memory_type,
            decision="block" if should_block else "hint",
            reason=(
                "Retrieved failed experience. Discourage repeating this path; "
                "block only at very high similarity."
            ),
            action_bias={action: -0.45} if action else {},
            blocked_actions=[action] if should_block else [],
        )

    if hit.memory_type == "experience":
        return MemoryApplicabilityAssessment(
            memory_id=hit.memory_id,
            memory_type=hit.memory_type,
            decision="hint",
            reason="Retrieved positive experience may provide a useful local action hint.",
            action_bias={action: 0.35} if action else {},
            blocked_actions=[],
        )

    if hit.memory_type == "skill":
        # Retrieved skill is not automatically applicable. It is a candidate hint
        # unless LLM/hybrid mode later judges it applicable under its boundary.
        return MemoryApplicabilityAssessment(
            memory_id=hit.memory_id,
            memory_type=hit.memory_type,
            decision="hint",
            reason="Retrieved skill is a candidate procedure and still requires applicability checking.",
            action_bias={action: 0.30} if action else {},
            blocked_actions=[],
        )

    return MemoryApplicabilityAssessment(
        memory_id=hit.memory_id,
        memory_type=hit.memory_type,
        decision="hint",
        reason="Retrieved knowledge may provide relevant background guidance.",
        action_bias={},
        blocked_actions=[],
    )


def _postprocess_llm_assessment(
    hit: RetrievalHit,
    assessment: MemoryApplicabilityAssessment,
    fallback: MemoryApplicabilityAssessment,
) -> MemoryApplicabilityAssessment:
    if assessment.decision not in {"apply", "hint", "block", "ignore"}:
        return fallback

    assessment.memory_id = hit.memory_id
    assessment.memory_type = hit.memory_type

    cleaned_bias: dict[str, float] = {}
    for action, delta in (assessment.action_bias or {}).items():
        action_type = str(action).upper()
        if action_type not in DEFAULT_ACTIONS:
            continue
        try:
            cleaned_bias[action_type] = float(delta)
        except Exception:
            continue
    assessment.action_bias = cleaned_bias

    assessment.blocked_actions = [
        str(action).upper()
        for action in (assessment.blocked_actions or [])
        if str(action).upper() in DEFAULT_ACTIONS
    ]

    if hit.memory_type == "skill" and assessment.decision == "apply" and hit.score < SKILL_APPLY_MIN_SCORE:
        assessment.decision = "hint"
        assessment.reason = (
            assessment.reason
            or "Skill was retrieved but similarity is not high enough for automatic apply."
        )

    if _is_failure_experience(hit) and assessment.decision == "block" and hit.score < FAILURE_BLOCK_MIN_SCORE:
        assessment.decision = "hint"
        assessment.blocked_actions = []
        assessment.reason = assessment.reason or "Failure memory is not similar enough to block."

    if _is_unsafe_experience(hit) and assessment.decision == "block" and hit.score < UNSAFE_BLOCK_MIN_SCORE:
        assessment.decision = "hint"
        assessment.blocked_actions = []
        assessment.reason = assessment.reason or "Unsafe memory is not similar enough to block."

    return assessment


def _llm_memory_assessment(
    hit: RetrievalHit,
    case_state: CaseState,
    memory_query: MemoryQuery,
    llm_client: LLMClient,
) -> MemoryApplicabilityAssessment:
    fallback = _rule_memory_assessment(hit)
    if not llm_client.available():
        return fallback

    payload = {
        "case_state": case_state.to_dict(),
        "memory_query": memory_query.to_dict(),
        "retrieved_memory": {
            "memory_id": hit.memory_id,
            "memory_type": hit.memory_type,
            "score": hit.score,
            "content": hit.content,
        },
        "allowed_decisions": ["apply", "hint", "block", "ignore"],
        "allowed_actions": DEFAULT_ACTIONS,
        "rules": [
            "Do not apply a skill merely because it was retrieved; check boundary_text and current case_state.",
            "Failure experience usually discourages; unsafe experience can block if clearly applicable.",
            "Do not invent action types.",
        ],
    }
    parsed, _, _ = parse_validate_repair(
        llm_client.generate_json(applicability_prompt(payload), max_tokens=900),
        APPLICABILITY_SCHEMA,
        fallback.to_dict(),
    )
    try:
        assessment = MemoryApplicabilityAssessment.from_dict(parsed)
    except Exception:
        return fallback
    return _postprocess_llm_assessment(hit, assessment, fallback)


def _hard_block_actions(case_state: CaseState) -> tuple[list[str], str]:
    blocked: list[str] = []
    warnings: list[str] = []

    if case_state.finalize_risk == "high":
        blocked.append("FINALIZE_DIAGNOSIS")
        warnings.append("finalize_risk is high")

    if len(case_state.missing_info) >= 3:
        if "FINALIZE_DIAGNOSIS" not in blocked:
            blocked.append("FINALIZE_DIAGNOSIS")
        warnings.append("multiple critical missing information slots remain unresolved")

    if "image" in case_state.modality_flags and "image" not in case_state.reviewed_modalities:
        warnings.append("image modality is available but not yet reviewed")

    return list(dict.fromkeys(blocked)), "; ".join(warnings)


def _aggregate_action_assessments(
    memory_assessments: list[MemoryApplicabilityAssessment],
    hard_blocked_actions: list[str],
) -> list[ActionAssessment]:
    action_map: dict[str, ActionAssessment] = {}

    def ensure(action: str) -> ActionAssessment:
        action = str(action).upper()
        if action not in action_map:
            action_map[action] = ActionAssessment(action_type=action)
        return action_map[action]

    for action in hard_blocked_actions:
        item = ensure(action)
        item.blocked = True
        item.reason = "Hard safety rule blocked this action from current CaseState."

    for assessment in memory_assessments:
        for action, delta in assessment.action_bias.items():
            action = str(action).upper()
            if not action:
                continue
            item = ensure(action)
            try:
                item.score_delta += float(delta)
            except Exception:
                pass
            if assessment.decision in {"apply", "hint"} and item.score_delta >= 0:
                item.supporting_memory_ids.append(assessment.memory_id)
            if assessment.decision in {"hint", "block"} and item.score_delta < 0:
                item.warning_memory_ids.append(assessment.memory_id)

        for action in assessment.blocked_actions:
            action = str(action).upper()
            if not action:
                continue
            item = ensure(action)
            item.blocked = True
            item.warning_memory_ids.append(assessment.memory_id)
            item.reason = assessment.reason or "Blocked by retrieved memory."

    for item in action_map.values():
        item.supporting_memory_ids = list(dict.fromkeys(item.supporting_memory_ids))
        item.warning_memory_ids = list(dict.fromkeys(item.warning_memory_ids))
        if not item.reason:
            if item.score_delta > 0:
                item.reason = "Supported by retrieved memory."
            elif item.score_delta < 0:
                item.reason = "Discouraged by retrieved memory."
            else:
                item.reason = "Neutral."

    return sorted(action_map.values(), key=lambda item: item.action_type)


def apply_applicability_control(
    case_state: CaseState,
    memory_query: MemoryQuery,
    retrieval_result: MemoryRetrievalResult,
    mode: str = "rule",
    llm_client: LLMClient | None = None,
) -> ApplicabilityResult:
    memory_assessments: list[MemoryApplicabilityAssessment] = []

    for hit in _all_hits(retrieval_result):
        if mode in {"llm", "hybrid"} and llm_client is not None:
            assessment = _llm_memory_assessment(hit, case_state, memory_query, llm_client)
        else:
            assessment = _rule_memory_assessment(hit)
        memory_assessments.append(assessment)

    hard_blocked, risk_warning = _hard_block_actions(case_state)
    action_assessments = _aggregate_action_assessments(memory_assessments, hard_blocked)

    return ApplicabilityResult(
        memory_assessments=memory_assessments,
        action_assessments=action_assessments,
        hard_blocked_actions=hard_blocked,
        risk_warning=risk_warning,
    )
