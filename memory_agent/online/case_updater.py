from __future__ import annotations

from typing import Any

from ..schemas import CaseState


CRITICAL_SLOT_TEMPLATES = {
    "chest pain": [
        "onset",
        "radiation",
        "associated dyspnea",
        "exertional trigger",
        "cardiac history",
        "ECG or troponin",
    ],
    "shortness of breath": [
        "onset",
        "progression",
        "fever or cough",
        "oxygen saturation",
        "cardiac history",
        "CXR or imaging",
    ],
    "altered mental status": [
        "onset",
        "baseline mental status",
        "focal neurologic symptoms",
        "fever",
        "medication exposure",
        "trauma",
    ],
    "abdominal pain": [
        "onset",
        "location",
        "duration",
        "fever",
        "vomiting",
        "stool or urinary symptoms",
    ],
    "bleeding": [
        "amount",
        "duration",
        "hemodynamic symptoms",
        "medication exposure",
        "prior bleeding history",
    ],
}


def _nested_get(payload: dict[str, Any], path: list[str], default: Any = None) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return current if current is not None else default


def _unwrap_osce_examination(ehr: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ehr, dict):
        return {}
    if isinstance(ehr.get("OSCE_Examination"), dict):
        return ehr["OSCE_Examination"]
    if isinstance(ehr.get("osce_examination"), dict):
        return ehr["osce_examination"]
    return ehr


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


def _critical_slots(problem_summary: str) -> list[str]:
    lowered = (problem_summary or "").lower()

    for key, slots in CRITICAL_SLOT_TEMPLATES.items():
        if key in lowered:
            return list(slots)

    return [
        "timeline",
        "associated symptoms",
        "severity",
        "risk factors",
        "targeted exam",
    ]


def _case_id_from_bundle(bundle: Any) -> str:
    if isinstance(bundle, dict):
        return str(
            bundle.get("case_id")
            or bundle.get("id")
            or bundle.get("uid")
            or ""
        )
    return str(
        getattr(bundle, "case_id", "")
        or getattr(bundle, "id", "")
        or getattr(bundle, "uid", "")
        or ""
    )


def _ehr_from_bundle(bundle: Any) -> dict[str, Any]:
    if isinstance(bundle, dict):
        ehr = bundle.get("ehr") or bundle.get("EHR") or bundle
    else:
        ehr = getattr(bundle, "ehr", None) or getattr(bundle, "EHR", None) or {}

    return ehr if isinstance(ehr, dict) else {}


def _problem_summary_from_ehr(ehr: dict[str, Any]) -> str:
    osce = _unwrap_osce_examination(ehr)

    history = _nested_get(osce, ["Patient_Actor", "History"], {})
    symptoms = _nested_get(osce, ["Patient_Actor", "Symptoms"], {})

    chief = ""
    if isinstance(symptoms, dict):
        chief = str(symptoms.get("Chief_Complaint") or "")
    if not chief and isinstance(history, dict):
        chief = str(history.get("Chief_Complaint") or "")

    objective = str(osce.get("Objective_for_Doctor") or osce.get("objective") or "")

    parts = [part.strip() for part in [chief, objective] if part and part.strip()]
    return " | ".join(parts)


def _modality_flags_from_ehr(ehr: dict[str, Any], no_cxr: bool = False) -> list[str]:
    flags = ["text"]

    text_blob = " ".join(_collect_texts(ehr)).lower()

    if any(token in text_blob for token in ["lab", "cbc", "troponin", "bmp", "cmp"]):
        flags.append("lab")

    if not no_cxr and any(token in text_blob for token in ["cxr", "x-ray", "xray", "image", "imaging"]):
        flags.append("image")

    return list(dict.fromkeys(flags))


def init_case_state(bundle: Any, no_cxr: bool = False) -> CaseState:
    ehr = _ehr_from_bundle(bundle)
    case_id = _case_id_from_bundle(bundle)

    problem_summary = _problem_summary_from_ehr(ehr)
    if not problem_summary:
        problem_summary = "initial clinical problem unclear"

    return CaseState(
        case_id=case_id,
        turn_id=0,
        problem_summary=problem_summary,
        key_evidence=[],
        negative_evidence=[],
        missing_info=_critical_slots(problem_summary),
        active_hypotheses=[],
        local_goal="collect_missing_critical_info",
        uncertainty_summary="initial state with unresolved diagnostic uncertainty",
        finalize_risk="high",
        modality_flags=_modality_flags_from_ehr(ehr, no_cxr=no_cxr),
        reviewed_modalities=[],
        interaction_history_summary="",
    )


def _is_negative_text(text: str) -> bool:
    lowered = text.lower()
    negation_markers = [
        "no ",
        "denies",
        "without",
        "negative for",
        "not present",
        "absent",
        "normal",
        "unremarkable",
    ]
    return any(marker in lowered for marker in negation_markers)


def _update_reviewed_modalities(state: CaseState, text: str) -> None:
    lowered = text.lower()

    if any(token in lowered for token in ["cxr", "x-ray", "xray", "image", "imaging", "radiology"]):
        if "image" not in state.reviewed_modalities:
            state.reviewed_modalities.append("image")

    if any(token in lowered for token in ["lab", "cbc", "troponin", "bmp", "cmp", "test result"]):
        if "lab" not in state.reviewed_modalities:
            state.reviewed_modalities.append("lab")


def _resolve_missing_info(state: CaseState, text_blob: str) -> None:
    lowered = text_blob.lower()
    state.missing_info = [
        slot for slot in state.missing_info
        if slot.lower() not in lowered
    ]


def _update_risk_and_goal(state: CaseState) -> None:
    if len(state.missing_info) >= 3:
        state.finalize_risk = "high"
        state.local_goal = "collect_missing_critical_info"
        state.uncertainty_summary = "multiple critical slots remain unresolved"
    elif len(state.missing_info) >= 1:
        state.finalize_risk = "medium"
        state.local_goal = "reduce_remaining_uncertainty"
        state.uncertainty_summary = "some important clinical uncertainty remains"
    else:
        state.finalize_risk = "low"
        state.local_goal = "verify_before_finalize"
        state.uncertainty_summary = "core missing information appears mostly resolved"


def update_case_state(prev_case_state: CaseState, observation: Any) -> CaseState:
    state = CaseState.from_dict(prev_case_state.to_dict())
    state.turn_id += 1

    texts = _collect_texts(observation)
    text_blob = " ".join(texts)

    if text_blob:
        prior_summary = state.interaction_history_summary
        turn_note = f"turn_{state.turn_id}: {text_blob[:240]}"
        state.interaction_history_summary = (
            f"{prior_summary} | {turn_note}" if prior_summary else turn_note
        )

    for text in texts:
        clipped = text[:300].strip()
        if not clipped:
            continue

        if _is_negative_text(clipped):
            state.negative_evidence.append(clipped)
        else:
            state.key_evidence.append(clipped)

        _update_reviewed_modalities(state, clipped)

    _resolve_missing_info(state, text_blob)
    _update_risk_and_goal(state)

    state.key_evidence = list(dict.fromkeys(state.key_evidence))[:30]
    state.negative_evidence = list(dict.fromkeys(state.negative_evidence))[:30]
    state.reviewed_modalities = list(dict.fromkeys(state.reviewed_modalities))

    return state