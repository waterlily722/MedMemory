from __future__ import annotations

from .online.applicability_controller import apply_applicability_control
from .schemas import ApplicabilityResult, CaseMemory, IntentPlan, MemoryRetrievalResult


def apply_controller(
    case_memory: CaseMemory,
    intent_plan: IntentPlan,
    retrieval_result: MemoryRetrievalResult,
) -> ApplicabilityResult:
    return apply_applicability_control(case_memory, intent_plan, retrieval_result)
