from __future__ import annotations

import json
from typing import Any

from ..llm import LLMClient, parse_validate_repair
from ..schemas import CaseState


CASE_STATE_FIELDS = [
    "case_id",
    "turn_id",
    "problem_summary",
    "key_evidence",
    "negative_evidence",
    "missing_info",
    "active_hypotheses",
    "local_goal",
    "uncertainty_summary",
    "finalize_risk",
    "modality_flags",
    "reviewed_modalities",
    "interaction_history_summary",
]

CASE_STATE_UPDATE_SCHEMA = {
    "required": CASE_STATE_FIELDS,
    "list_fields": [
        "key_evidence",
        "negative_evidence",
        "missing_info",
        "active_hypotheses",
        "modality_flags",
        "reviewed_modalities",
    ],
    "dict_fields": [],
    "enum_fields": {"finalize_risk": ["low", "medium", "high"]},
}

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


def _dedupe_keep_order(values: list[Any], limit: int | None = None) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
        if limit is not None and len(output) >= limit:
            break
    return output


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
        return str(bundle.get("case_id") or bundle.get("id") or bundle.get("uid") or "")
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
    if not no_cxr and any(
        token in text_blob for token in ["cxr", "x-ray", "xray", "image", "imaging"]
    ):
        flags.append("image")
    return _dedupe_keep_order(flags)


def init_case_state(bundle: Any, no_cxr: bool = False) -> CaseState:
    ehr = _ehr_from_bundle(bundle)
    case_id = _case_id_from_bundle(bundle)
    problem_summary = _problem_summary_from_ehr(ehr) or "initial clinical problem unclear"
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
        state.reviewed_modalities.append("image")
    if any(token in lowered for token in ["lab", "cbc", "troponin", "bmp", "cmp", "test result"]):
        state.reviewed_modalities.append("lab")
    state.reviewed_modalities = _dedupe_keep_order(state.reviewed_modalities)


def update_case_state_rule(prev_case_state: CaseState, observation: Any) -> CaseState:
    """Low-risk bookkeeping fallback.

    This function intentionally does not infer missing_info, local_goal,
    active_hypotheses, uncertainty_summary, or finalize_risk. Those are clinical
    abstraction fields and should be updated by update_case_state_llm when an LLM
    is available.
    """

    state = CaseState.from_dict(prev_case_state.to_dict())
    state.turn_id += 1

    texts = _collect_texts(observation)
    text_blob = " ".join(texts).strip()
    if text_blob:
        turn_note = f"turn_{state.turn_id}: {text_blob[:240]}"
        state.interaction_history_summary = (
            f"{state.interaction_history_summary} | {turn_note}"
            if state.interaction_history_summary
            else turn_note
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

    state.key_evidence = _dedupe_keep_order(state.key_evidence, limit=30)
    state.negative_evidence = _dedupe_keep_order(state.negative_evidence, limit=30)
    state.missing_info = _dedupe_keep_order(state.missing_info, limit=20)
    state.active_hypotheses = _dedupe_keep_order(state.active_hypotheses, limit=12)
    state.modality_flags = _dedupe_keep_order(state.modality_flags, limit=8)
    state.reviewed_modalities = _dedupe_keep_order(state.reviewed_modalities, limit=8)
    return state


def _case_state_update_prompt(payload: dict[str, Any]) -> str:
    return f"""
You update the CaseState for a medical memory system.
Return one JSON object only. Do not use markdown.

Use exactly these CaseState fields and do not add extra keys:
{json.dumps(CASE_STATE_FIELDS, ensure_ascii=False)}

Field meanings:
- problem_summary: concise current clinical problem.
- key_evidence: important positive or supporting evidence already known.
- negative_evidence: important absent/denied/normal findings already known.
- missing_info: unresolved information that would change next action or finalization.
- active_hypotheses: current differential or working hypotheses.
- local_goal: immediate next reasoning goal, not a final diagnosis.
- uncertainty_summary: why the case is still uncertain.
- finalize_risk: low, medium, or high risk of finalizing now.
- modality_flags: modalities available for this case, such as text, lab, image.
- reviewed_modalities: modalities already explicitly reviewed.
- interaction_history_summary: compact summary of recent interaction.

Important rules:
- Base your update only on previous_case_state, rule_baseline_state, and new_observation.
- Do not delete unresolved missing_info unless the new observation actually answers it.
- If key evidence is insufficient or important modalities are unreviewed, finalize_risk should remain medium or high.
- Keep lists concise and non-duplicated.
- Return valid JSON only.

Input:
{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}
""".strip()


def _sanitize_case_state_payload(parsed: dict[str, Any], fallback: CaseState) -> dict[str, Any]:
    fallback_dict = fallback.to_dict()
    data = {key: parsed.get(key, fallback_dict.get(key)) for key in CASE_STATE_FIELDS}

    data["case_id"] = str(data.get("case_id") or fallback.case_id)
    try:
        data["turn_id"] = int(data.get("turn_id", fallback.turn_id))
    except Exception:
        data["turn_id"] = fallback.turn_id

    for field in CASE_STATE_UPDATE_SCHEMA["list_fields"]:
        data[field] = _dedupe_keep_order(data.get(field) or [])

    for field in [
        "problem_summary",
        "local_goal",
        "uncertainty_summary",
        "finalize_risk",
        "interaction_history_summary",
    ]:
        data[field] = str(data.get(field) or "").strip()

    if data["finalize_risk"] not in {"low", "medium", "high"}:
        data["finalize_risk"] = fallback.finalize_risk or "high"

    return data


def update_case_state_llm(
    prev_case_state: CaseState,
    observation: Any,
    llm_client: LLMClient,
) -> CaseState:
    rule_state = update_case_state_rule(prev_case_state, observation)
    if not llm_client.available():
        return rule_state

    payload = {
        "previous_case_state": prev_case_state.to_dict(),
        "rule_baseline_state": rule_state.to_dict(),
        "new_observation": observation,
    }
    fallback = rule_state.to_dict()
    raw = llm_client.generate_json(_case_state_update_prompt(payload), max_tokens=1500)
    parsed, _, _ = parse_validate_repair(raw, CASE_STATE_UPDATE_SCHEMA, fallback)
    sanitized = _sanitize_case_state_payload(parsed, rule_state)

    try:
        return CaseState.from_dict(sanitized)
    except Exception:
        return rule_state


def update_case_state(
    prev_case_state: CaseState,
    observation: Any,
    mode: str = "llm",
    llm_client: LLMClient | None = None,
) -> CaseState:
    """Update CaseState while respecting the existing CaseState schema.

    mode="rule": bookkeeping only.
    mode="llm" or "hybrid": rule baseline plus LLM clinical abstraction.
    """

    if mode == "rule" or llm_client is None:
        return update_case_state_rule(prev_case_state, observation)
    return update_case_state_llm(prev_case_state, observation, llm_client)
