from __future__ import annotations

from typing import Any

from ..llm import LLMClient
from ..schemas import CaseState

CASE_STATE_FIELDS = [
    "case_id",
    "turn_id",
    "chief_complaint",
    "acquired_information",
]

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


def _collect_information_records(payload: Any, path: str = "observation") -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            records.extend(_collect_information_records(value, f"{path}.{key}"))
    elif isinstance(payload, list):
        for idx, value in enumerate(payload):
            records.extend(_collect_information_records(value, f"{path}[{idx}]"))
    elif payload is not None:
        text = str(payload).strip()
        if text:
            records.append({"source_path": path, "content": text})
    return records


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
        ehr = (
            bundle.get("ehr")
            or bundle.get("EHR")
            or _nested_get(bundle, ["medenv_case_bundle", "ehr"])
            or _nested_get(bundle, ["case_bundle", "ehr"])
            or bundle
        )
    else:
        ehr = getattr(bundle, "ehr", None) or getattr(bundle, "EHR", None) or {}
    if isinstance(ehr, dict) and isinstance(ehr.get("ehr"), dict):
        ehr = ehr["ehr"]
    return ehr if isinstance(ehr, dict) else {}


def _chief_complaint_from_ehr(ehr: dict[str, Any]) -> str:
    osce = _unwrap_osce_examination(ehr)
    history = (
        _nested_get(osce, ["Patient_Actor", "History"], {})
        or osce.get("History")
        or ehr.get("History")
        or {}
    )
    symptoms = (
        _nested_get(osce, ["Patient_Actor", "Symptoms"], {})
        or osce.get("Symptoms")
        or ehr.get("Symptoms")
        or {}
    )

    chief = ""
    if isinstance(symptoms, dict):
        chief = str(symptoms.get("Chief_Complaint") or "")
    if not chief and isinstance(history, dict):
        chief = str(history.get("Chief_Complaint") or "")

    return chief.strip()


def init_case_state(bundle: Any, no_cxr: bool = False) -> CaseState:
    ehr = _ehr_from_bundle(bundle)
    case_id = _case_id_from_bundle(bundle)
    chief_complaint = _chief_complaint_from_ehr(ehr)

    return CaseState(
        case_id=case_id,
        turn_id=0,
        chief_complaint=chief_complaint,
        acquired_information=[],
    )


def _dedupe_records(values: list[Any], limit: int) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for value in values or []:
        if not isinstance(value, dict):
            continue
        source_path = str(value.get("source_path") or "").strip()
        content = str(value.get("content") or "").strip()
        turn_id = str(value.get("turn_id") or "").strip()
        if not content:
            continue
        key = (turn_id, source_path, content)
        if key in seen:
            continue
        seen.add(key)
        item = dict(value)
        item["source_path"] = source_path
        item["content"] = content
        cleaned.append(item)
        if len(cleaned) >= limit:
            break
    return cleaned


def update_case_state_rule(prev_case_state: CaseState, observation: Any) -> CaseState:
    """
    Deterministic observation ledger.

    CaseState records the complete information already exposed to the doctor
    agent by the environment or tools. It does not summarize, truncate, or infer.
    """
    state = CaseState.from_dict(prev_case_state.to_dict())
    state.turn_id += 1

    records = _collect_information_records(observation)
    for record in records:
        text = record["content"]
        content = text.strip()
        if not content:
            continue
        state.acquired_information.append(
            {
                "turn_id": state.turn_id,
                "source_path": record.get("source_path", ""),
                "content": content,
            }
        )

    state.acquired_information = _dedupe_records(state.acquired_information, 200)
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


def update_case_state_llm(
    prev_case_state: CaseState,
    observation: Any,
    llm_client: LLMClient,
    debug: dict[str, Any] | None = None,
    strict: bool = True,
) -> CaseState:
    result = update_case_state_rule(prev_case_state, observation)
    if debug is not None:
        debug["mode"] = "observed"
        debug["llm_skipped"] = True
        debug["skip_reason"] = "CaseState is an observed-information ledger; no LLM extraction is used"
        debug["previous_case_state"] = prev_case_state.to_dict()
        debug["observation"] = _truncate_payload(observation)
        debug["final_case_state"] = result.to_dict()
    return result


def update_case_state(
    prev_case_state: CaseState,
    observation: Any,
    mode: str = "observed",
    llm_client: LLMClient | None = None,
    debug: dict[str, Any] | None = None,
    strict: bool = True,
) -> CaseState:
    result = update_case_state_rule(prev_case_state, observation)
    if debug is not None:
        debug["mode"] = "observed"
        debug["requested_mode"] = mode
        debug["llm_skipped"] = True
        debug["previous_case_state"] = prev_case_state.to_dict()
        debug["observation"] = _truncate_payload(observation)
        debug["final_case_state"] = result.to_dict()
    return result
