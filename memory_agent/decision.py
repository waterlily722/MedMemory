from __future__ import annotations

from .online.doctor_policy import choose_next_action
from .schemas import ActionDecision, ApplicabilityResult, CaseState, IntentPlan


def decide_action(
    intent_plan: IntentPlan,
    applicability_result: ApplicabilityResult,
    case_memory: CaseState | None = None,
) -> ActionDecision:
    memory = case_memory if case_memory is not None else CaseState(case_id="", turn_id=intent_plan.turn_id)
    return choose_next_action(memory, intent_plan, applicability_result)
