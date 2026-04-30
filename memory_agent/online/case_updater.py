from __future__ import annotations

import json
import logging
from typing import Any

from ..llm import LLMClient, parse_validate_repair
from ..schemas import CaseState

logger = logging.getLogger(__name__)

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
    return list(dict.fromkeys(flags))


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
        if "image" not in state.reviewed_modalities:
            state.reviewed_modalities.append("image")
    if any(token in lowered for token in ["lab", "cbc", "troponin", "bmp", "cmp", "test result"]):
        if "lab" not in state.reviewed_modalities:
            state.reviewed_modalities.append("lab")


def _dedupe_strs(values: list[Any], limit: int) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def update_case_state_rule(prev_case_state: CaseState, observation: Any) -> CaseState:
    """
    Safe deterministic updater.

    It only performs low-risk bookkeeping: turn count, interaction summary,
    raw evidence append, and reviewed modalities. Clinical abstraction fields
    such as missing_info, active_hypotheses, local_goal, uncertainty_summary,
    and finalize_risk are preserved for the LLM updater.
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

    state.key_evidence = _dedupe_strs(state.key_evidence, 30)
    state.negative_evidence = _dedupe_strs(state.negative_evidence, 30)
    state.missing_info = _dedupe_strs(state.missing_info, 20)
    state.active_hypotheses = _dedupe_strs(state.active_hypotheses, 12)
    state.modality_flags = _dedupe_strs(state.modality_flags, 8)
    state.reviewed_modalities = _dedupe_strs(state.reviewed_modalities, 8)
    return state


def _truncate_payload(value: Any, max_chars: int = 5000) -> Any:
    if isinstance(value, dict):
        return {str(key): _truncate_payload(item, max_chars=max_chars) for key, item in value.items()}
    if isinstance(value, list):
        return [_truncate_payload(item, max_chars=max_chars) for item in value[:50]]
    text = str(value)
    if len(text) > max_chars:
        return text[:max_chars] + "...[truncated]"
    return value


def _case_state_update_prompt(
    previous_case_state: CaseState,
    rule_updated_case_state: CaseState,
    observation: Any,
) -> str:
    payload = {
        "previous_case_state": previous_case_state.to_dict(),
        "rule_updated_case_state": rule_updated_case_state.to_dict(),
        "new_observation": _truncate_payload(observation),
        "allowed_finalize_risk": ["low", "medium", "high"],
        "schema_fields": CASE_STATE_FIELDS,
    }
    return f"""
You update CaseState for a medical memory system.
Return one JSON object only. Do not use markdown. Do not add extra keys.

Use exactly these keys:
{json.dumps(CASE_STATE_FIELDS, ensure_ascii=False)}

Rules:
- Preserve case_id and turn_id from rule_updated_case_state.
- Use only evidence present in previous_case_state, rule_updated_case_state, and new_observation.
- Update missing_info, active_hypotheses, local_goal, uncertainty_summary, and finalize_risk by clinical reasoning.
- finalize_risk must be one of: low, medium, high.
- Keep lists concise and non-duplicated.
- Do not invent tests, diagnoses, or imaging findings not supported by input.

Input:
{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}
""".strip()


def _sanitize_case_state_dict(parsed: dict[str, Any], fallback: CaseState) -> dict[str, Any]:
    fallback_dict = fallback.to_dict()
    cleaned: dict[str, Any] = {}
    for field in CASE_STATE_FIELDS:
        cleaned[field] = parsed.get(field, fallback_dict.get(field))

    cleaned["case_id"] = fallback.case_id
    cleaned["turn_id"] = fallback.turn_id

    for field in CASE_STATE_UPDATE_SCHEMA["list_fields"]:
        cleaned[field] = _dedupe_strs(list(cleaned.get(field) or []), 30)

    cleaned["problem_summary"] = str(cleaned.get("problem_summary") or fallback.problem_summary)
    cleaned["local_goal"] = str(cleaned.get("local_goal") or fallback.local_goal)
    cleaned["uncertainty_summary"] = str(
        cleaned.get("uncertainty_summary") or fallback.uncertainty_summary
    )
    cleaned["interaction_history_summary"] = str(
        cleaned.get("interaction_history_summary") or fallback.interaction_history_summary
    )

    risk = str(cleaned.get("finalize_risk") or fallback.finalize_risk).lower()
    if risk not in {"low", "medium", "high"}:
        risk = fallback.finalize_risk if fallback.finalize_risk in {"low", "medium", "high"} else "high"
    cleaned["finalize_risk"] = risk
    return cleaned


def update_case_state_llm(
    prev_case_state: CaseState,
    observation: Any,
    llm_client: LLMClient,
) -> CaseState:
    fallback = update_case_state_rule(prev_case_state, observation)
    if not llm_client.available():
        return fallback

    parsed, ok, errors = parse_validate_repair(
        llm_client.generate_json(
            _case_state_update_prompt(prev_case_state, fallback, observation),
            max_tokens=1400,
        ),
        CASE_STATE_UPDATE_SCHEMA,
        fallback.to_dict(),
    )
    if not ok:
        logger.warning(
            "CaseState LLM update validation errors for turn %d: %s",
            prev_case_state.turn_id + 1, errors,
        )
    try:
        return CaseState.from_dict(_sanitize_case_state_dict(parsed, fallback))
    except Exception as exc:
        logger.warning(
            "CaseState LLM update failed for turn %d: %s; using rule fallback",
            prev_case_state.turn_id + 1, exc,
        )
        return fallback


def update_case_state(
    prev_case_state: CaseState,
    observation: Any,
    mode: str = "llm",
    llm_client: LLMClient | None = None,
) -> CaseState:
    if mode == "llm" and llm_client is not None:
        return update_case_state_llm(prev_case_state, observation, llm_client)
    return update_case_state_rule(prev_case_state, observation)
