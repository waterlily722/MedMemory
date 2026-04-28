from __future__ import annotations

from .online.applicability_controller import apply_applicability_control
from .schemas import ApplicabilityResult, CaseMemory, IntentPlan, MemoryRetrievalResult


def apply_controller(
    case_memory: CaseMemory,
    intent_plan: IntentPlan,
    retrieval_result: MemoryRetrievalResult,
    mode: str = "rule",
    llm_client=None,
) -> ApplicabilityResult:
    return apply_applicability_control(case_memory, intent_plan, retrieval_result, mode=mode, llm_client=llm_client)
