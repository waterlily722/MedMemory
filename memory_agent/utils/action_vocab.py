from __future__ import annotations

from ..schemas import CaseState

ACTION_LABELS = {
    "ASK": "ask_patient",
    "REVIEW_HISTORY": "review_history",
    "REQUEST_EXAM": "request_exam",
    "REQUEST_LAB": "request_lab",
    "REVIEW_IMAGE": "review_image",
    "UPDATE_HYPOTHESIS": "update_hypothesis",
    "DEFER_FINALIZE": "defer_finalize",
    "FINALIZE_DIAGNOSIS": "finalize_diagnosis",
}

TOOL_NAMES = {
    "ASK": "ask_patient",
    "REVIEW_HISTORY": "retrieve",
    "REQUEST_EXAM": "request_exam",
    "REQUEST_LAB": "request_exam",
    "REVIEW_IMAGE": "cxr",
    "UPDATE_HYPOTHESIS": "retrieve",
    "DEFER_FINALIZE": "ask_patient",
    "FINALIZE_DIAGNOSIS": "diagnosis",
}


def action_label(action_type: str) -> str:
    return ACTION_LABELS.get(action_type, action_type.lower())


def tool_name(action_type: str) -> str:
    return TOOL_NAMES.get(action_type, "ask_patient")


def candidate_actions(case_state: CaseState) -> list[str]:
    actions = ["ASK", "REVIEW_HISTORY", "UPDATE_HYPOTHESIS", "DEFER_FINALIZE"]
    if "lab" in case_state.modality_flags:
        actions.append("REQUEST_LAB")
    if "image" in case_state.modality_flags:
        actions.append("REVIEW_IMAGE")
    if case_state.missing_info:
        actions.append("REQUEST_EXAM")
    if case_state.finalize_risk in {"low", "medium", "high"}:
        actions.append("FINALIZE_DIAGNOSIS")
    return list(dict.fromkeys(actions))
