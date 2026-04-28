from __future__ import annotations

from .online.case_updater import init_case_state, update_case_state
from .schemas import CaseMemory, CanonicalEvidence, MedEnvCaseBundle


def init_case_memory(bundle: MedEnvCaseBundle | dict, no_cxr: bool) -> CaseMemory:
    return init_case_state(bundle, no_cxr=no_cxr)


def update_case_memory(
    prev_case_memory: CaseMemory,
    evidence_list: list[CanonicalEvidence],
    executed_action=None,
    execution_result=None,
) -> CaseMemory:
    # executed_action/execution_result are accepted for backward compatibility.
    _ = executed_action
    _ = execution_result
    return update_case_state(prev_case_memory, evidence_list)
