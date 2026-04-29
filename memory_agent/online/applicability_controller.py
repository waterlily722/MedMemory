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


def _all_hits(retrieval_result: MemoryRetrievalResult) -> list[RetrievalHit]:
    return (
        retrieval_result.positive_experience_hits
        + retrieval_result.negative_experience_hits
        + retrieval_result.skill_hits
        + retrieval_result.knowledge_hits
    )


def _is_negative_experience(hit: RetrievalHit) -> bool:
    if hit.memory_type != "experience":
        return False
    outcome_type = str(hit.content.get("outcome_type") or "")
    return outcome_type in {"failure", "unsafe"}


def _infer_action_from_memory(hit: RetrievalHit) -> str:
    if hit.memory_type == "experience":
        sequence = hit.content.get("action_sequence") or []
        if isinstance(sequence, list) and sequence:
            first = sequence[0]
            if isinstance(first, dict):
                return str(first.get("action_type") or "").upper()
        action_text = str(hit.content.get("action_text") or "").upper()
        for action in DEFAULT_ACTIONS:
            if action in action_text:
                return action

    if hit.memory_type == "skill":
        procedure = hit.content.get("procedure") or []
        if isinstance(procedure, list) and procedure:
            first = procedure[0]
            if isinstance(first, dict):
                return str(first.get("action_type") or "").upper()
        procedure_text = str(hit.content.get("procedure_text") or "").upper()
        for action in DEFAULT_ACTIONS:
            if action in procedure_text:
                return action

    return ""


def _rule_memory_assessment(hit: RetrievalHit) -> MemoryApplicabilityAssessment:
    action = _infer_action_from_memory(hit)

    if _is_negative_experience(hit):
        blocked_actions = [action] if action else []
        return MemoryApplicabilityAssessment(
            memory_id=hit.memory_id,
            memory_type=hit.memory_type,
            decision="block" if blocked_actions else "hint",
            reason=(
                "Retrieved negative experience with similar situation/action. "
                "Use as warning against repeating a risky path."
            ),
            action_bias={action: -0.8} if action else {},
            blocked_actions=blocked_actions,
        )

    if hit.memory_type == "experience":
        return MemoryApplicabilityAssessment(
            memory_id=hit.memory_id,
            memory_type=hit.memory_type,
            decision="hint",
            reason="Retrieved positive experience may provide a useful local action hint.",
            action_bias={action: 0.4} if action else {},
            blocked_actions=[],
        )

    if hit.memory_type == "skill":
        return MemoryApplicabilityAssessment(
            memory_id=hit.memory_id,
            memory_type=hit.memory_type,
            decision="apply",
            reason="Retrieved skill provides a reusable procedure for this uncertainty state.",
            action_bias={action: 0.6} if action else {},
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
        "instruction": (
            "Judge whether this memory is applicable to the current case. "
            "Return only memory_id, memory_type, decision, reason, action_bias, blocked_actions. "
            "Do not invent new action types."
        ),
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

    if assessment.decision not in {"apply", "hint", "block", "ignore"}:
        return fallback

    assessment.memory_id = hit.memory_id
    assessment.memory_type = hit.memory_type
    return assessment


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

    return blocked, "; ".join(warnings)


def _aggregate_action_assessments(
    memory_assessments: list[MemoryApplicabilityAssessment],
    hard_blocked_actions: list[str],
) -> list[ActionAssessment]:
    action_map: dict[str, ActionAssessment] = {}

    def ensure(action: str) -> ActionAssessment:
        if action not in action_map:
            action_map[action] = ActionAssessment(action_type=action)
        return action_map[action]

    for action in hard_blocked_actions:
        item = ensure(action)
        item.blocked = True
        item.reason = "Hard safety rule blocked this action."

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

            if assessment.decision in {"apply", "hint"}:
                item.supporting_memory_ids.append(assessment.memory_id)
            if assessment.decision == "block":
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
        if mode == "hybrid" and llm_client is not None:
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