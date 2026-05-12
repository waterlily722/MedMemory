from __future__ import annotations

from typing import Any

from ..llm import LLMClient, case_memory_prompt, parse_validate_repair, query_builder_prompt
from ..llm.schemas import CASE_MEMORY_SCHEMA, QUERY_BUILDER_SCHEMA
from ..schemas import CaseMemory, CaseState, MemoryQuery


def _join(values: list[Any], limit: int = 6) -> str:
    cleaned = [str(value).strip() for value in values[:limit] if str(value).strip()]
    return "; ".join(cleaned)


def _record_to_text(item: dict[str, Any]) -> str:
    turn_id = item.get("turn_id")
    source_path = str(item.get("source_path") or "").strip()
    content = str(item.get("content") or "").strip()
    if not content:
        return ""
    prefix = f"turn_{turn_id}" if turn_id else ""
    if source_path:
        prefix = f"{prefix} {source_path}".strip()
    return f"{prefix}: {content}" if prefix else content


def _turn_information(case_state: CaseState, turn_id: int | None = None) -> list[str]:
    target_turn = case_state.turn_id if turn_id is None else turn_id
    out: list[str] = []
    for item in case_state.acquired_information or []:
        if not isinstance(item, dict):
            continue
        if item.get("turn_id") == target_turn:
            text = _record_to_text(item)
            if text:
                out.append(text)
    return out


def _prior_information_text(case_state: CaseState) -> str:
    lines: list[str] = []
    for item in case_state.acquired_information or []:
        if not isinstance(item, dict):
            continue
        if item.get("turn_id") == case_state.turn_id:
            continue
        text = _record_to_text(item)
        if text:
            lines.append(text)
    return "\n".join(lines)


def build_case_memory_rule(case_state: CaseState) -> CaseMemory:
    return CaseMemory(
        case_id=case_state.case_id,
        turn_id=case_state.turn_id,
        chief_complaint=case_state.chief_complaint,
        current_turn_information=_turn_information(case_state),
        prior_information_summary=_prior_information_text(case_state),
    )


def build_case_memory_llm(
    case_state: CaseState,
    llm_client: LLMClient,
    debug: dict[str, Any] | None = None,
    strict: bool = True,
) -> CaseMemory:
    rule_memory = build_case_memory_rule(case_state)
    payload = {
        "case_state": case_state.to_dict(),
        "instruction": (
            "Extract CaseMemory from CaseState. CaseState is the full observed ledger. "
            "Use latest turn records as current_turn_information and summarize earlier "
            "records as prior_information_summary."
        ),
    }
    prompt = case_memory_prompt(payload)
    if debug is not None:
        debug["case_memory_mode"] = "llm"
        debug["case_state"] = case_state.to_dict()
        debug["rule_case_memory"] = rule_memory.to_dict()
        debug["case_memory_llm_available"] = llm_client.available()
        debug["case_memory_prompt"] = prompt
        debug["case_memory_payload"] = payload
    if not llm_client.available():
        message = "CaseMemory LLM mode requested but memory LLM is unavailable"
        if strict:
            raise RuntimeError(message)
        if debug is not None:
            debug["case_memory_used_fallback"] = True
            debug["case_memory_fallback_reason"] = "llm_unavailable"
            debug["final_case_memory"] = rule_memory.to_dict()
        return rule_memory

    raw_output = llm_client.generate_json(prompt, max_tokens=1200)
    raw_empty = not str(raw_output or "").strip() or str(raw_output or "").strip() == "{}"
    parsed, ok, errors = parse_validate_repair(
        raw_output,
        CASE_MEMORY_SCHEMA,
        rule_memory.to_dict(),
    )
    if raw_empty or not ok:
        message = (
            f"CaseMemory LLM output invalid for case_id={case_state.case_id!r} "
            f"turn_id={case_state.turn_id}: errors={errors}, raw_output={raw_output!r}"
        )
        if strict and not raw_empty:
            raise RuntimeError(message)
        parsed = rule_memory.to_dict()
    try:
        result = CaseMemory.from_dict(parsed)
    except Exception as exc:
        if strict:
            raise RuntimeError(f"CaseMemory parse failed: {exc}") from exc
        result = rule_memory
    if debug is not None:
        debug["case_memory_raw_output"] = raw_output
        debug["case_memory_parsed_output"] = parsed
        debug["case_memory_validation_ok"] = ok
        debug["case_memory_validation_errors"] = errors
        debug["case_memory_used_fallback"] = result.to_dict() == rule_memory.to_dict() and (raw_empty or not ok)
        debug["final_case_memory"] = result.to_dict()
    return result


def build_case_memory(
    case_state: CaseState,
    mode: str = "rule",
    llm_client: LLMClient | None = None,
    debug: dict[str, Any] | None = None,
    strict: bool = True,
) -> CaseMemory:
    if mode == "llm" and llm_client is not None:
        return build_case_memory_llm(case_state, llm_client, debug=debug, strict=strict)
    if mode == "llm" and strict:
        raise RuntimeError("CaseMemory LLM mode requested but llm_client is None")
    result = build_case_memory_rule(case_state)
    if debug is not None:
        debug["case_memory_mode"] = "rule"
        debug["case_state"] = case_state.to_dict()
        debug["final_case_memory"] = result.to_dict()
    return result


def _case_memory_to_dict(case_memory: CaseMemory | dict[str, Any]) -> dict[str, Any]:
    if isinstance(case_memory, CaseMemory):
        return case_memory.to_dict()
    return dict(case_memory or {})


def _build_memory_query_rule_from_case_memory(
    case_memory: CaseMemory | dict[str, Any],
    candidate_actions: list[Any] | None = None,
) -> MemoryQuery:
    cm = _case_memory_to_dict(case_memory)
    sections: list[str] = []
    for field in ["chief_complaint", "current_turn_information", "prior_information_summary"]:
        value = cm.get(field)
        if isinstance(value, list):
            text = _join(value, limit=12)
        else:
            text = str(value or "").strip()
        if text:
            sections.append(f"{field}: {text}")

    if candidate_actions:
        actions = [_action_to_text(action) for action in candidate_actions]
        actions_text = _join(actions, limit=12)
        if actions_text:
            sections.append(f"candidate_actions: {actions_text}")

    query_text = "\n".join(sections).strip()
    if not query_text:
        raise RuntimeError(
            f"Cannot build memory query for case_id={cm.get('case_id')!r} "
            f"turn_id={cm.get('turn_id')}: CaseMemory contains no queryable information"
        )

    return MemoryQuery(
        case_id=str(cm.get("case_id") or ""),
        turn_id=int(cm.get("turn_id") or 0),
        query_text=query_text,
    )


def _action_to_text(action: Any) -> str:
    if isinstance(action, dict):
        action_type = str(action.get("action_type") or action.get("tool") or "").strip()
        action_label = str(action.get("action_label") or action.get("label") or "").strip()
        if action_type and action_label:
            return f"{action_type}: {action_label}"
        return action_type or action_label
    return str(action).strip()


def build_memory_query_rule(
    case_state: CaseState,
    candidate_actions: list[Any] | None = None,
) -> MemoryQuery:
    """
    Build a natural-language retrieval query from existing CaseState fields only.
    The MemoryQuery schema stays minimal: case_id, turn_id, query_text.
    """
    return _build_memory_query_rule_from_case_memory(
        build_case_memory_rule(case_state),
        candidate_actions,
    )


def build_memory_query_llm(
    case_state: CaseState,
    candidate_actions: list[Any] | None,
    llm_client: LLMClient,
    debug: dict[str, Any] | None = None,
    strict: bool = True,
) -> MemoryQuery:
    case_memory = build_case_memory(
        case_state,
        mode="llm",
        llm_client=llm_client,
        debug=debug,
        strict=strict,
    )
    rule_query = _build_memory_query_rule_from_case_memory(case_memory, candidate_actions)
    actions = [_action_to_text(action) for action in candidate_actions or []]
    payload = {
        "case_memory": case_memory.to_dict(),
        "candidate_actions": actions,
        "instruction": (
            "Create one concise retrieval query for memory search. "
            "Use only case_memory and candidate_actions from the input. "
            "Focus on the current turn plus information already exposed to the doctor agent. "
            "Do not restate the full historical dialogue. "
            "Do not infer new diagnoses, missing information, uncertainty, or risk labels. "
            "The query_text should naturally include the clinical situation, newly exposed facts, "
            "available/reviewed modalities, and useful next-action needs when present. "
            "Return JSON with only query_text."
        ),
    }
    prompt = query_builder_prompt(payload)
    if debug is not None:
        debug["mode"] = "llm"
        debug["case_memory"] = case_memory.to_dict()
        debug["candidate_actions"] = actions
        debug["rule_query"] = rule_query.to_dict()
        debug["llm_available"] = llm_client.available()
        debug["payload"] = payload
        debug["prompt"] = prompt
    if not llm_client.available():
        message = "Memory query LLM mode requested but memory LLM is unavailable"
        if strict:
            raise RuntimeError(message)
        if debug is not None:
            debug["used_fallback"] = True
            debug["fallback_reason"] = "llm_unavailable"
            debug["final_query"] = rule_query.to_dict()
        return rule_query

    raw_output = llm_client.generate_json(prompt, max_tokens=800)
    raw_empty = not str(raw_output or "").strip() or str(raw_output or "").strip() == "{}"
    parsed, ok, errors = parse_validate_repair(
        raw_output,
        QUERY_BUILDER_SCHEMA,
        {"query_text": rule_query.query_text},
    )
    query_text = str(parsed.get("query_text") or "").strip()
    if raw_empty or not ok or not query_text:
        message = (
            f"Memory query LLM output invalid for case_id={case_state.case_id!r} "
            f"turn_id={case_state.turn_id}: errors={errors}, raw_output={raw_output!r}"
        )
        if strict and not raw_empty:
            raise RuntimeError(message)
        query_text = rule_query.query_text
    result = MemoryQuery(
        case_id=case_state.case_id,
        turn_id=case_state.turn_id,
        query_text=query_text,
    )
    if debug is not None:
        debug["raw_output"] = raw_output
        debug["parsed_output"] = parsed
        debug["validation_ok"] = ok
        debug["validation_errors"] = errors
        debug["used_fallback"] = query_text == rule_query.query_text and (raw_empty or not ok)
        debug["final_query"] = result.to_dict()
    return result


def build_memory_query(
    case_state: CaseState,
    candidate_actions: list[Any] | None = None,
    mode: str = "rule",
    llm_client: LLMClient | None = None,
    debug: dict[str, Any] | None = None,
    strict: bool = True,
) -> MemoryQuery:
    if mode == "llm" and llm_client is not None:
        return build_memory_query_llm(case_state, candidate_actions, llm_client, debug=debug, strict=strict)
    if mode == "llm" and strict:
        raise RuntimeError("Memory query LLM mode requested but llm_client is None")
    result = build_memory_query_rule(case_state, candidate_actions)
    if debug is not None:
        debug["mode"] = "rule"
        debug["case_memory"] = build_case_memory_rule(case_state).to_dict()
        debug["candidate_actions"] = [_action_to_text(action) for action in candidate_actions or []]
        debug["final_query"] = result.to_dict()
    return result
