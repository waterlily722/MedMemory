from __future__ import annotations

from typing import Dict

from .schemas import ActionDecision, ApplicabilityResult, CandidateRanking, IntentPlan


ADJUSTMENT_MAP = {
    "apply": 0.25,
    "hint": 0.05,
    "escalate": -0.2,
    "block": -1.0,
}


def decide_action(
    intent_plan: IntentPlan,
    applicability_result: ApplicabilityResult,
) -> ActionDecision:
    assessment_by_id = {assessment.action_id: assessment for assessment in applicability_result.action_assessments}
    candidate_rankings: list[CandidateRanking] = []
    best_candidate = None
    best_score = float("-inf")

    for candidate in intent_plan.action_candidates:
        assessment = assessment_by_id.get(candidate.action_id)
        decision = assessment.decision if assessment else "hint"
        adjustment = ADJUSTMENT_MAP.get(decision, 0.0)
        final_score = candidate.planner_score + adjustment
        blocked = decision == "block"
        candidate_rankings.append(
            CandidateRanking(
                action_id=candidate.action_id,
                planner_score=candidate.planner_score,
                controller_adjustment=adjustment,
                final_score=round(final_score, 4),
                blocked=blocked,
                source_field_refs=candidate.source_field_refs,
            )
        )
        if not blocked and final_score > best_score and candidate.action_type != "defer_finalize":
            best_candidate = candidate
            best_score = final_score

    candidate_rankings.sort(key=lambda item: item.final_score, reverse=True)

    if best_candidate is None and intent_plan.action_candidates:
        for candidate in intent_plan.action_candidates:
            if candidate.action_type != "finalize":
                best_candidate = candidate
                break
        best_candidate = best_candidate or intent_plan.action_candidates[0]

    chosen_action = {
        "action_id": best_candidate.action_id if best_candidate else "",
        "action_type": best_candidate.action_type if best_candidate else "ask",
        "action_text": best_candidate.action_text if best_candidate else "Can you tell me more about your symptoms?",
        "action_args": best_candidate.action_args if best_candidate else {"question": "Can you tell me more about your symptoms?"},
    }
    rationale = "Selected the highest-ranked non-blocked action after planner/controller fusion."
    if best_candidate:
        assessment = assessment_by_id.get(best_candidate.action_id)
        if assessment:
            rationale = assessment.rationale

    return ActionDecision(
        turn_id=intent_plan.turn_id,
        chosen_action=chosen_action,
        candidate_rankings=candidate_rankings,
        final_rationale=rationale,
        source_field_refs=intent_plan.source_field_refs,
    )
