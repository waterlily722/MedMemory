from __future__ import annotations

from typing import Dict, List

from .schemas import ActionAssessment, ApplicabilityResult, CaseMemory, IntentPlan, MemoryRetrievalResult


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _guardrail_risk_for_action(action_type: str, retrieval_result: MemoryRetrievalResult) -> tuple[float, list[str]]:
    ids: list[str] = []
    scores: list[float] = []
    for hit in retrieval_result.guardrail_hits:
        if "risky_action" in hit.matched_fields or not hit.matched_fields or action_type in hit.matched_fields:
            ids.append(hit.item_id)
            scores.append(hit.retrieval_score)
    return _mean(scores), ids


def apply_controller(
    case_memory: CaseMemory,
    intent_plan: IntentPlan,
    retrieval_result: MemoryRetrievalResult,
) -> ApplicabilityResult:
    assessments: List[ActionAssessment] = []
    safety_state = case_memory.derived_state.get("safety_state", {})
    modality_available = (case_memory.derived_state.get("modality_state", {}) or {}).get("available", {})

    for candidate in intent_plan.action_candidates:
        supporting_experience_ids = [hit.item_id for hit in retrieval_result.experience_hits if candidate.action_type in hit.matched_fields or not hit.matched_fields]
        supporting_skill_ids = [hit.item_id for hit in retrieval_result.skill_hits if candidate.action_type in hit.matched_fields or not hit.matched_fields]
        guardrail_risk, guardrail_ids = _guardrail_risk_for_action(candidate.action_type, retrieval_result)

        state_match = _mean([hit.retrieval_score for hit in retrieval_result.experience_hits[:3]])
        evidence_compatibility = 0.7
        modality_match = 1.0
        historical_reliability = min(1.0, 0.2 + 0.15 * len(supporting_experience_ids) + 0.15 * len(supporting_skill_ids))
        applicability_boundary = 0.7

        if candidate.action_type == "finalize":
            risk = safety_state.get("premature_finalize_risk", "mid")
            if risk == "high":
                evidence_compatibility = 0.15
                applicability_boundary = 0.1
                guardrail_risk = max(guardrail_risk, 0.95)
            elif risk == "mid":
                evidence_compatibility = 0.45
                applicability_boundary = 0.35
                guardrail_risk = max(guardrail_risk, 0.55)

        if candidate.action_type == "cxr" and not modality_available.get("cxr"):
            modality_match = 0.0
        if candidate.action_type == "cxr_grounding" and not modality_available.get("cxr_grounding"):
            modality_match = 0.0

        if candidate.action_type == "request_exam" and case_memory.derived_state.get("missing_critical_slots"):
            evidence_compatibility = max(evidence_compatibility, 0.82)

        if candidate.action_type == "ask":
            evidence_compatibility = max(0.78, evidence_compatibility)

        decision = "hint"
        rationale = "Moderately compatible with the current case state."
        if modality_match == 0.0:
            decision = "block"
            rationale = "Action blocked because the required modality is unavailable in this case."
        elif guardrail_risk >= 0.8:
            decision = "block"
            rationale = "Action blocked by a high-risk guardrail signal."
        elif candidate.action_type == "finalize" and safety_state.get("premature_finalize_risk") == "mid":
            decision = "escalate"
            rationale = "Final diagnosis should be deferred until more evidence is gathered."
        elif evidence_compatibility >= 0.75 and historical_reliability >= 0.45:
            decision = "apply"
            rationale = "Action is well supported by the current state and retrieved memory."

        assessments.append(
            ActionAssessment(
                action_id=candidate.action_id,
                supporting_experience_ids=supporting_experience_ids,
                supporting_skill_ids=supporting_skill_ids,
                triggered_guardrail_ids=guardrail_ids,
                scores={
                    "state_match": round(state_match, 4),
                    "evidence_compatibility": round(evidence_compatibility, 4),
                    "modality_match": round(modality_match, 4),
                    "historical_reliability": round(historical_reliability, 4),
                    "applicability_boundary": round(applicability_boundary, 4),
                    "guardrail_risk": round(guardrail_risk, 4),
                },
                decision=decision,
                rationale=rationale,
                source_field_refs=candidate.source_field_refs,
            )
        )

    return ApplicabilityResult(
        turn_id=intent_plan.turn_id,
        action_assessments=assessments,
        source_field_refs=intent_plan.source_field_refs,
    )
