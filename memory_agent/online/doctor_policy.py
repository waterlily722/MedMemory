from __future__ import annotations

from ..schemas import ActionDecision, ApplicabilityResult, CandidateRanking, CaseState, IntentPlan

ADJUSTMENT_MAP = {
    "apply": 0.25,
    "hint": 0.05,
    "escalate": -0.15,
    "block": -1.0,
}


def choose_next_action(case_state: CaseState, plan: IntentPlan, applicability: ApplicabilityResult) -> ActionDecision:
    by_id = {a.action_id: a for a in applicability.action_assessments}
    rankings: list[CandidateRanking] = []
    best = None
    best_score = float("-inf")

    for cand in plan.action_candidates:
        a = by_id.get(cand.action_id)
        decision = a.decision if a else "hint"
        adj = ADJUSTMENT_MAP.get(decision, 0.0)
        score = cand.planner_score + adj
        blocked = decision == "block"
        rankings.append(
            CandidateRanking(
                action_id=cand.action_id,
                planner_score=cand.planner_score,
                controller_adjustment=adj,
                final_score=round(score, 4),
                blocked=blocked,
                source_field_refs=cand.source_field_refs,
            )
        )
        if not blocked and score > best_score:
            best = cand
            best_score = score

    rankings.sort(key=lambda r: r.final_score, reverse=True)

    if best is None and plan.action_candidates:
        best = plan.action_candidates[0]

    if best is None:
        chosen = {
            "action_id": "fallback_ask",
            "action_type": "ASK",
            "action_label": "ask_onset",
            "action_content": "When did your symptoms start?",
        }
    else:
        chosen = {
            "action_id": best.action_id,
            "action_type": best.action_type,
            "action_label": best.action_label,
            "action_content": best.action_content,
        }

    rationale = by_id.get(chosen["action_id"]).rationale if by_id.get(chosen["action_id"]) else "Fallback policy."
    return ActionDecision(turn_id=plan.turn_id, chosen_action=chosen, candidate_rankings=rankings, final_rationale=rationale, source_field_refs=plan.source_field_refs)
