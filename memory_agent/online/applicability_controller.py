from __future__ import annotations

from ..schemas import ActionAssessment, ApplicabilityResult, CaseState, IntentPlan, MemoryApplicabilityAssessment, MemoryRetrievalResult
from ..utils.config import APPLICABILITY_CONFIG
from .llm_applicability_judge import llm_judge_applicability


def _payload_content(hit) -> dict:
    payload = hit.payload if isinstance(hit.payload, dict) else {}
    content = payload.get("content")
    return content if isinstance(content, dict) else payload


def _matches_action(hit_text: str, action_type: str) -> bool:
    lowered = hit_text.lower()
    action = action_type.lower()
    return action in lowered or action.replace("_", " ") in lowered


def _assess_memory_item(
    case_state: CaseState,
    hit,
    candidate_actions: list[str],
    structured_query: dict,
    mode: str = "rule",
    llm_client=None,
) -> MemoryApplicabilityAssessment:
    memory_content = _payload_content(hit)
    memory_type = str((hit.payload or {}).get("memory_type", "experience"))
    memory_id = str(hit.item_id)
    hit_text = str(memory_content)

    matched_actions = [action for action in candidate_actions if _matches_action(hit_text, action)]
    applicability = "high" if hit.retrieval_score >= 0.75 else "medium" if hit.retrieval_score >= 0.45 else "low"
    reason = "rule-based applicability from retrieval score and action match"
    boundary_violation = False
    blocked_actions: list[str] = []
    action_bias = {action: 0.0 for action in candidate_actions}
    controller_decision = "hint"

    if memory_type == "negative_experience":
        applicability = "low"
        boundary_violation = True
        reason = "negative experience memory"
        blocked_actions = matched_actions[:]
        controller_decision = "block"
    elif memory_type == "skill":
        reason = "skill trigger and preconditions matched"
        controller_decision = "apply" if applicability == "high" else "hint"
        for action in matched_actions:
            action_bias[action] = 0.25
    elif memory_type == "knowledge":
        reason = "knowledge prior matches current query"
        controller_decision = "hint" if applicability != "low" else "escalate"
    else:
        controller_decision = "apply" if applicability == "high" else "hint"

    if mode in {"llm", "hybrid"} and llm_client is not None:
        judge, _, _ = llm_judge_applicability(
            case_state=case_state.to_dict(),
            structured_memory_query=structured_query,
            memory_item={
                "memory_id": memory_id,
                "memory_type": memory_type,
                "content": memory_content,
                "retrieval_score": hit.retrieval_score,
                "matched_fields": hit.matched_fields,
            },
            candidate_actions=candidate_actions,
            local_goal=case_state.local_goal,
            finalize_risk=case_state.finalize_risk,
            llm_client=llm_client,
        )
        llm_decision = str(judge.get("controller_decision", controller_decision))
        llm_reason = str(judge.get("reason", reason))
        if mode == "llm":
            controller_decision = llm_decision
            reason = llm_reason
        elif mode == "hybrid":
            if controller_decision == "block" or llm_decision == "block":
                controller_decision = "block"
                reason = llm_reason or reason
            elif llm_decision == "apply":
                controller_decision = "apply"
                reason = llm_reason or reason
            elif llm_decision == "hint" and controller_decision == "escalate":
                controller_decision = "hint"
                reason = llm_reason or reason

    return MemoryApplicabilityAssessment(
        memory_id=memory_id,
        memory_type=memory_type,
        memory_content=memory_content,
        applicability=applicability,
        reason=reason,
        matched_aspects=matched_actions,
        mismatched_aspects=[action for action in candidate_actions if action not in matched_actions],
        boundary_violation=boundary_violation,
        action_bias=action_bias,
        blocked_actions=blocked_actions,
        controller_decision=controller_decision,
        relevance_score=float(hit.retrieval_score),
        source_field_refs=hit.source_field_refs,
    )


def _hard_safety_block(case_state: CaseState, action_type: str, retrieval: MemoryRetrievalResult) -> tuple[bool, str]:
    if action_type != "FINALIZE_DIAGNOSIS":
        return False, ""
    if case_state.finalize_risk == "high":
        return True, "hard rule: finalize_risk is high"
    if case_state.missing_info:
        return True, "hard rule: missing critical info before finalize"
    if "image" in case_state.modality_flags:
        image_reviewed = any("review_image" in h.item_id.lower() or "image" in str(h.payload).lower() for h in retrieval.experience_hits)
        if not image_reviewed:
            return True, "hard rule: image modality required but image evidence not reviewed"
    return False, ""


def _aggregate_action_support(
    action_type: str,
    memory_assessments: list[MemoryApplicabilityAssessment],
) -> tuple[float, list[str], list[str], list[str]]:
    exp_ids: list[str] = []
    skill_ids: list[str] = []
    kn_ids: list[str] = []
    score = 0.0

    for assessment in memory_assessments:
        if action_type not in assessment.matched_aspects and action_type.lower() not in str(assessment.memory_content).lower():
            continue
        if assessment.memory_type in {"experience", "skill", "knowledge"} and assessment.controller_decision in {"apply", "hint"}:
            score += assessment.relevance_score
            if assessment.memory_type == "experience":
                exp_ids.append(assessment.memory_id)
            elif assessment.memory_type == "skill":
                skill_ids.append(assessment.memory_id)
            else:
                kn_ids.append(assessment.memory_id)
        elif assessment.memory_type == "negative_experience" or assessment.boundary_violation:
            score -= max(0.15, assessment.relevance_score * 0.5)

    return score, list(dict.fromkeys(exp_ids))[:3], list(dict.fromkeys(skill_ids))[:2], list(dict.fromkeys(kn_ids))[:2]


def apply_applicability_control(
    case_state: CaseState,
    plan: IntentPlan,
    retrieval: MemoryRetrievalResult,
    mode: str = "rule",
    llm_client=None,
) -> ApplicabilityResult:
    memory_assessments: list[MemoryApplicabilityAssessment] = []
    for hit in retrieval.experience_hits:
        memory_assessments.append(_assess_memory_item(case_state, hit, [c.action_type for c in plan.action_candidates], plan.memory_query.structured.to_dict(), mode=mode, llm_client=llm_client))
    for hit in retrieval.negative_experience_hits:
        memory_assessments.append(_assess_memory_item(case_state, hit, [c.action_type for c in plan.action_candidates], plan.memory_query.structured.to_dict(), mode=mode, llm_client=llm_client))
    for hit in retrieval.skill_hits:
        memory_assessments.append(_assess_memory_item(case_state, hit, [c.action_type for c in plan.action_candidates], plan.memory_query.structured.to_dict(), mode=mode, llm_client=llm_client))
    for hit in retrieval.knowledge_hits:
        memory_assessments.append(_assess_memory_item(case_state, hit, [c.action_type for c in plan.action_candidates], plan.memory_query.structured.to_dict(), mode=mode, llm_client=llm_client))

    assessments: list[ActionAssessment] = []
    hard_block_actions: list[str] = []
    for cand in plan.action_candidates:
        support, exp_ids, skill_ids, kn_ids = _aggregate_action_support(cand.action_type, memory_assessments)
        risk_penalty = 0.0
        boundary_conflict = False
        hard_block, hard_reason = _hard_safety_block(case_state, cand.action_type, retrieval)

        if cand.action_type == "FINALIZE_DIAGNOSIS" and case_state.finalize_risk in {"high", "medium"}:
            if APPLICABILITY_CONFIG["block_premature_finalize"] and case_state.finalize_risk == "high":
                boundary_conflict = True
            risk_penalty = 0.45 if case_state.finalize_risk == "high" else 0.2

        if APPLICABILITY_CONFIG["reject_on_missing_modality"] and cand.action_type in {"REVIEW_IMAGE", "REQUEST_IMAGING"}:
            if "image" not in case_state.modality_flags:
                boundary_conflict = True

        score = max(0.0, support - risk_penalty)
        if hard_block:
            decision = "block"
            rationale = hard_reason
            hard_block_actions.append(cand.action_id)
        elif boundary_conflict:
            decision = "block"
            rationale = "Rejected due to modality/boundary conflict or premature finalize risk."
            hard_block_actions.append(cand.action_id)
        elif score >= APPLICABILITY_CONFIG["accept_threshold"]:
            decision = "apply"
            rationale = "High applicability under current case state and memory evidence."
        elif score >= APPLICABILITY_CONFIG["weak_hint_threshold"]:
            decision = "hint"
            rationale = "Weakly applicable, treat as soft guidance."
        else:
            decision = "escalate"
            rationale = "Low applicability, prefer to request missing evidence first."

        if mode in {"llm", "hybrid"} and llm_client is not None:
            payload = {
                "memory_id": exp_ids[0] if exp_ids else (skill_ids[0] if skill_ids else (kn_ids[0] if kn_ids else "")),
                "item_id": exp_ids[0] if exp_ids else (skill_ids[0] if skill_ids else (kn_ids[0] if kn_ids else "")),
            }
            judge, _, _ = llm_judge_applicability(
                case_state=case_state.to_dict(),
                structured_memory_query=plan.memory_query.to_dict(),
                memory_item=payload,
                candidate_actions=[c.action_type for c in plan.action_candidates],
                local_goal=case_state.local_goal,
                finalize_risk=case_state.finalize_risk,
                llm_client=llm_client,
            )
            llm_decision = str(judge.get("controller_decision", "hint"))
            llm_reason = str(judge.get("reason", ""))
            if mode == "llm":
                decision = llm_decision
                rationale = llm_reason or rationale
            elif mode == "hybrid":
                if decision == "block":
                    pass
                elif llm_decision == "block":
                    decision = "block"
                    rationale = llm_reason or rationale
                elif llm_decision == "apply" and decision in {"hint", "escalate"}:
                    decision = "hint"
                    rationale = llm_reason or rationale

        assessments.append(
            ActionAssessment(
                action_id=cand.action_id,
                decision=decision,
                rationale=rationale,
                scores={"support": round(support, 4), "risk_penalty": round(risk_penalty, 4), "final": round(score, 4)},
                supporting_experience_ids=exp_ids,
                supporting_skill_ids=skill_ids,
                supporting_knowledge_ids=kn_ids,
                source_field_refs=cand.source_field_refs,
            )
        )

    return ApplicabilityResult(
        turn_id=plan.turn_id,
        memory_assessments=memory_assessments,
        action_assessments=assessments,
        controller_summary={"hard_block_actions": hard_block_actions, "mode": mode},
        hard_block_actions=hard_block_actions,
        source_field_refs=plan.source_field_refs,
    )
