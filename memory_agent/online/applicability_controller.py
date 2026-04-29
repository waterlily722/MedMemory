from __future__ import annotations

from typing import Any

from ..llm import LLMClient, applicability_prompt, parse_validate_repair
from ..llm.schemas import APPLICABILITY_SCHEMA
from ..schemas import ActionAssessment, ApplicabilityResult, CaseState, MemoryApplicabilityAssessment, MemoryQuery, MemoryRetrievalResult
from ..utils.action_vocab import action_label
from ..utils.config import APPLICABILITY_CONFIG
from ..utils.scoring import cosine_similarity, overlap_score


def _memory_text(hit) -> str:
    return str(hit.content or {})


def _rule_memory_assessment(case_state: CaseState, query: MemoryQuery, hit, candidate_actions: list[str]) -> MemoryApplicabilityAssessment:
    content = hit.content if isinstance(hit.content, dict) else {}
    text = _memory_text(hit).lower()
    matched_aspects: list[str] = []
    mismatched_aspects: list[str] = []
    action_bias = {action: 0.0 for action in candidate_actions}
    blocked_actions: list[str] = []
    applicability = "medium"
    reason = "rule-based memory assessment"
    boundary_violation = None
    controller_decision = "ignore"

    if hit.memory_type == "experience":
        if query.local_goal and query.local_goal.lower() in text:
            matched_aspects.append("local_goal")
        if overlap_score(query.positive_evidence, content.get("key_evidence", []) if isinstance(content.get("key_evidence"), list) else []) > 0:
            matched_aspects.append("positive_evidence")
        if hit.score >= 0.75:
            applicability = "high"
            controller_decision = "apply"
        elif hit.score >= 0.35:
            applicability = "medium"
            controller_decision = "hint"
        else:
            applicability = "low"
            controller_decision = "ignore"
        if content.get("outcome_type") in {"failure", "unsafe"}:
            applicability = "low"
            controller_decision = "block"
            reason = "negative experience"
            for action in candidate_actions:
                if action.lower() in text:
                    blocked_actions.append(action)
                    action_bias[action] -= 0.4
        if content.get("outcome_type") in {"success", "partial_success"}:
            for action in candidate_actions:
                if action.lower() in text:
                    action_bias[action] += 0.2
                    matched_aspects.append(action)
    elif hit.memory_type == "negative_experience":
        applicability = "low"
        controller_decision = "block"
        reason = "negative experience"
        for action in candidate_actions:
            if action.lower() in text:
                blocked_actions.append(action)
                action_bias[action] -= 0.5
        boundary_violation = "negative experience warning"
    elif hit.memory_type == "skill":
        applicability = "high" if hit.score >= 0.55 else "medium"
        controller_decision = "apply" if applicability == "high" else "hint"
        for action in candidate_actions:
            if action.lower() in text:
                action_bias[action] += 0.3
                matched_aspects.append(action)
    else:
        applicability = "medium" if hit.score >= 0.35 else "low"
        controller_decision = "hint" if applicability == "medium" else "ignore"

    if not matched_aspects:
        mismatched_aspects.extend(candidate_actions)

    if query.finalize_risk == "high" and any(action == "FINALIZE_DIAGNOSIS" for action in candidate_actions):
        boundary_violation = boundary_violation or "high finalize risk"
        blocked_actions.append("FINALIZE_DIAGNOSIS")
        action_bias["FINALIZE_DIAGNOSIS"] = min(action_bias.get("FINALIZE_DIAGNOSIS", 0.0), -1.0)
        controller_decision = "block"
        applicability = "reject"
        reason = "hard finalize safety rule"

    return MemoryApplicabilityAssessment(
        memory_id=hit.memory_id,
        memory_type=hit.memory_type,
        applicability=applicability,
        reason=reason,
        matched_aspects=list(dict.fromkeys(matched_aspects)),
        mismatched_aspects=list(dict.fromkeys(mismatched_aspects)),
        boundary_violation=boundary_violation,
        action_bias=action_bias,
        blocked_actions=list(dict.fromkeys(blocked_actions)),
        controller_decision=controller_decision,
    )


def _llm_memory_assessment(case_state: CaseState, query: MemoryQuery, hit, candidate_actions: list[str], llm_client: LLMClient) -> MemoryApplicabilityAssessment:
    fallback = _rule_memory_assessment(case_state, query, hit, candidate_actions).to_dict()
    payload = {
        "case_state": case_state.to_dict(),
        "memory_query": query.to_dict(),
        "memory_item": hit.to_dict(),
        "candidate_actions": candidate_actions,
    }
    parsed, _, _ = parse_validate_repair(llm_client.generate_json(applicability_prompt(payload)), APPLICABILITY_SCHEMA, fallback)
    return MemoryApplicabilityAssessment.from_dict(parsed)


def _aggregate(case_state: CaseState, query: MemoryQuery, memory_assessments: list[MemoryApplicabilityAssessment], candidate_actions: list[str]) -> tuple[list[ActionAssessment], list[str], str]:
    hard_blocked: list[str] = []
    assessments: list[ActionAssessment] = []
    risk_warning = ""

    for action in candidate_actions:
        support_ids: list[str] = []
        warning_ids: list[str] = []
        score_delta = 0.0
        blocked = False
        reason_parts: list[str] = []
        for assessment in memory_assessments:
            score_delta += float(assessment.action_bias.get(action, 0.0))
            if action in assessment.blocked_actions:
                blocked = True
                warning_ids.append(assessment.memory_id)
            if action in assessment.matched_aspects:
                support_ids.append(assessment.memory_id)
            if assessment.boundary_violation:
                warning_ids.append(assessment.memory_id)
        if query.finalize_risk == "high" and action == "FINALIZE_DIAGNOSIS":
            blocked = True
            reason_parts.append("hard finalize risk")
        if action == "FINALIZE_DIAGNOSIS" and case_state.missing_info and query.finalize_risk != "low":
            blocked = True
            reason_parts.append("missing critical info")
        if action == "REVIEW_IMAGE" and "image" in case_state.modality_flags and "image" not in case_state.reviewed_modalities:
            pass
        if blocked:
            hard_blocked.append(action)
        reason = "; ".join(reason_parts) if reason_parts else ("blocked by memory" if blocked else "memory guidance")
        assessments.append(
            ActionAssessment(
                action_type=action,
                action_label=action_label(action),
                score_delta=round(score_delta, 4),
                blocked=blocked,
                reason=reason,
                supporting_memory_ids=list(dict.fromkeys(support_ids)),
                warning_memory_ids=list(dict.fromkeys(warning_ids)),
            )
        )

    if any(action == "FINALIZE_DIAGNOSIS" for action in hard_blocked):
        risk_warning = "Finalize diagnosis is blocked by hard safety rules."
    return assessments, list(dict.fromkeys(hard_blocked)), risk_warning


def apply_applicability_control(
    case_state: CaseState,
    query: MemoryQuery,
    retrieval: MemoryRetrievalResult,
    candidate_actions: list[str],
    mode: str = "rule",
    llm_client: LLMClient | None = None,
) -> ApplicabilityResult:
    memory_assessments: list[MemoryApplicabilityAssessment] = []
    for hit in retrieval.positive_experience_hits:
        memory_assessments.append(
            _llm_memory_assessment(case_state, query, hit, candidate_actions, llm_client) if mode == "llm" and llm_client and llm_client.available() else _rule_memory_assessment(case_state, query, hit, candidate_actions)
        )
    for hit in retrieval.negative_experience_hits:
        memory_assessments.append(
            _llm_memory_assessment(case_state, query, hit, candidate_actions, llm_client) if mode == "llm" and llm_client and llm_client.available() else _rule_memory_assessment(case_state, query, hit, candidate_actions)
        )
    for hit in retrieval.skill_hits:
        memory_assessments.append(
            _llm_memory_assessment(case_state, query, hit, candidate_actions, llm_client) if mode == "llm" and llm_client and llm_client.available() else _rule_memory_assessment(case_state, query, hit, candidate_actions)
        )
    for hit in retrieval.knowledge_hits:
        memory_assessments.append(
            _llm_memory_assessment(case_state, query, hit, candidate_actions, llm_client) if mode == "llm" and llm_client and llm_client.available() else _rule_memory_assessment(case_state, query, hit, candidate_actions)
        )

    action_assessments, hard_blocked, risk_warning = _aggregate(case_state, query, memory_assessments, candidate_actions)
    return ApplicabilityResult(
        memory_assessments=memory_assessments,
        action_assessments=action_assessments,
        hard_blocked_actions=hard_blocked,
        risk_warning=risk_warning,
    )
