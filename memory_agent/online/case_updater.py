from __future__ import annotations

from typing import Any

from ..schemas import CaseState
from ..utils.medenv_adapter import nested_get, unwrap_osce_examination


CRITICAL_SLOT_TEMPLATES = {
    "chest pain": ["onset", "radiation", "associated dyspnea", "exertional trigger", "cardiac history", "troponin or ECG"],
    "shortness of breath": ["onset", "progression", "fever or cough", "oxygen saturation", "cardiac history", "cxr or imaging"],
    "altered mental status": ["onset", "baseline mental status", "focal neurologic symptoms", "fever", "medication exposure", "trauma"],
}


def _critical_slots(problem_summary: str) -> list[str]:
    lowered = (problem_summary or "").lower()
    for key, slots in CRITICAL_SLOT_TEMPLATES.items():
        if key in lowered:
            return list(slots)
    return ["timeline", "associated symptoms", "targeted exam"]


def init_case_state(bundle: Any, no_cxr: bool = False) -> CaseState:
    if isinstance(bundle, dict):
        case_id = str(bundle.get("case_id", ""))
        ehr = dict(bundle.get("ehr") or {})
    else:
        case_id = str(getattr(bundle, "case_id", ""))
        ehr = dict(getattr(bundle, "ehr", {}) or {})
    osce = unwrap_osce_examination(ehr)
    history = nested_get(osce, ["Patient_Actor", "History"], {})
    symptoms = nested_get(osce, ["Patient_Actor", "Symptoms"], {})
    chief = str(symptoms.get("Chief_Complaint") or history.get("Chief_Complaint") or osce.get("Objective_for_Doctor", ""))
    modality_flags = ["text", "lab"]
    test_results = nested_get(osce, ["Test_Results"], {})
    if not no_cxr and isinstance(test_results, dict) and test_results.get("CXR"):
        modality_flags.append("image")
    return CaseState(
        case_id=case_id,
        turn_id=0,
        problem_summary=" | ".join(text for text in [chief, str(osce.get("Objective_for_Doctor", ""))] if text),
        key_evidence=[],
        negative_evidence=[],
        missing_info=_critical_slots(chief or str(osce.get("Objective_for_Doctor", ""))),
        active_hypotheses=[],
        local_goal="gather_high_value_evidence" if chief else "clarify_problem",
        uncertainty_summary="initial state",
        finalize_risk="high" if chief else "medium",
        modality_flags=modality_flags,
        reviewed_modalities=[],
        interaction_history_summary="",
        source_turn_ids=[],
    )


def _collect_texts(payload: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(payload, dict):
        for value in payload.values():
            texts.extend(_collect_texts(value))
    elif isinstance(payload, list):
        for value in payload:
            texts.extend(_collect_texts(value))
    elif payload is not None:
        text = str(payload).strip()
        if text:
            texts.append(text)
    return texts


def update_case_state(prev_case_state: CaseState, observation: Any) -> CaseState:
    state = CaseState.from_dict(prev_case_state.to_dict())
    state.turn_id += 1
    state.source_turn_ids.append(prev_case_state.turn_id)
    state.interaction_history_summary = (state.interaction_history_summary + " | " if state.interaction_history_summary else "") + f"turn_{prev_case_state.turn_id}"

    texts = _collect_texts(observation)
    if isinstance(observation, dict):
        if isinstance(observation.get("question"), str):
            texts.append(observation["question"])
        if isinstance(observation.get("tool_outputs"), dict):
            texts.extend(str(value) for value in observation["tool_outputs"].values())

    for text in texts:
        lowered = text.lower()
        if any(slot.lower() in lowered for slot in state.missing_info):
            state.key_evidence.append(text[:200])
        else:
            state.negative_evidence.append(text[:200])
        if "image" in lowered or "cxr" in lowered:
            if "image" not in state.reviewed_modalities:
                state.reviewed_modalities.append("image")
        if "lab" in lowered or "troponin" in lowered or "cbc" in lowered:
            if "lab" not in state.reviewed_modalities:
                state.reviewed_modalities.append("lab")

    state.key_evidence = list(dict.fromkeys(state.key_evidence))[:20]
    state.negative_evidence = list(dict.fromkeys(state.negative_evidence))[:20]

    resolved = [slot for slot in state.missing_info if slot.lower() in " ".join(texts).lower()]
    state.missing_info = [slot for slot in state.missing_info if slot not in resolved]
    if len(state.missing_info) >= 3:
        state.finalize_risk = "high"
        state.local_goal = "collect_missing_critical_info"
        state.uncertainty_summary = "multiple critical slots unresolved"
    elif len(state.missing_info) >= 1:
        state.finalize_risk = "medium"
        state.local_goal = "reduce_uncertainty"
        state.uncertainty_summary = "some critical slots unresolved"
    else:
        state.finalize_risk = "low"
        state.local_goal = "verify_before_finalize"
        state.uncertainty_summary = "core evidence mostly available"
    return state
