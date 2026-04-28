from __future__ import annotations

from ..schemas import ActionAssessment, ApplicabilityResult, CaseState, IntentPlan, MemoryRetrievalResult
from ..utils.config import APPLICABILITY_CONFIG


def _best_support(action_type: str, retrieval: MemoryRetrievalResult) -> tuple[float, list[str], list[str], list[str]]:
    exp_ids = [h.item_id for h in retrieval.experience_hits if action_type.lower() in str(h.payload).lower()][:3]
    skill_ids = [h.item_id for h in retrieval.skill_hits if action_type.lower() in str(h.payload).lower()][:2]
    kn_ids = [h.item_id for h in retrieval.knowledge_hits][:2]
    exp_score = max([h.retrieval_score for h in retrieval.experience_hits if h.item_id in exp_ids] or [0.0])
    skill_score = max([h.retrieval_score for h in retrieval.skill_hits if h.item_id in skill_ids] or [0.0])
    know_score = max([h.retrieval_score for h in retrieval.knowledge_hits if h.item_id in kn_ids] or [0.0])
    return (0.45 * exp_score + 0.4 * skill_score + 0.15 * know_score, exp_ids, skill_ids, kn_ids)


def apply_applicability_control(case_state: CaseState, plan: IntentPlan, retrieval: MemoryRetrievalResult) -> ApplicabilityResult:
    assessments: list[ActionAssessment] = []
    for cand in plan.action_candidates:
        support, exp_ids, skill_ids, kn_ids = _best_support(cand.action_type, retrieval)
        risk_penalty = 0.0
        boundary_conflict = False

        if cand.action_type == "FINALIZE_DIAGNOSIS" and case_state.finalize_risk in {"high", "medium"}:
            if APPLICABILITY_CONFIG["block_premature_finalize"] and case_state.finalize_risk == "high":
                boundary_conflict = True
            risk_penalty = 0.45 if case_state.finalize_risk == "high" else 0.2

        if APPLICABILITY_CONFIG["reject_on_missing_modality"] and cand.action_type in {"REVIEW_IMAGE", "REQUEST_IMAGING"}:
            if "image" not in case_state.modality_flags:
                boundary_conflict = True

        score = max(0.0, support - risk_penalty)
        if boundary_conflict:
            decision = "block"
            rationale = "Rejected due to modality/boundary conflict or premature finalize risk."
        elif score >= APPLICABILITY_CONFIG["accept_threshold"]:
            decision = "apply"
            rationale = "High applicability under current case state and memory evidence."
        elif score >= APPLICABILITY_CONFIG["weak_hint_threshold"]:
            decision = "hint"
            rationale = "Weakly applicable, treat as soft guidance."
        else:
            decision = "escalate"
            rationale = "Low applicability, prefer to request missing evidence first."

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

    return ApplicabilityResult(turn_id=plan.turn_id, action_assessments=assessments, source_field_refs=plan.source_field_refs)
