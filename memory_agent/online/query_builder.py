from __future__ import annotations

from typing import Any

from ..llm import LLMClient, parse_validate_repair, query_builder_prompt
from ..llm.schemas import QUERY_BUILDER_SCHEMA
from ..schemas import CaseState, MemoryQuery


def _join(values: list[Any], limit: int = 6) -> str:
    cleaned = [str(value).strip() for value in values[:limit] if str(value).strip()]
    return "; ".join(cleaned)


def _latest_turn_text(history: str) -> str:
    parts = [part.strip() for part in str(history or "").split(" | ") if part.strip()]
    return parts[-1] if parts else ""


def _exclude_current_turn(values: list[Any], current_turn: str, limit: int = 5) -> list[str]:
    """Keep recent evidence that is not just the current turn repeated."""
    current = str(current_turn or "")
    cleaned: list[str] = []
    for value in values or []:
        text = str(value).strip()
        if not text:
            continue
        if text[:80] and text[:80] in current:
            continue
        cleaned.append(text)
    return cleaned[-limit:]


def case_memory_for_query(case_state: CaseState) -> dict[str, Any]:
    """Compact current-turn case memory used to build retrieval queries.

    This intentionally avoids the full cumulative interaction history so each
    turn query reflects the current state without repeating every prior turn.
    """
    current_turn = _latest_turn_text(case_state.interaction_history_summary)
    return {
        "case_id": case_state.case_id,
        "turn_id": case_state.turn_id,
        "problem_summary": case_state.problem_summary,
        "current_turn": current_turn,
        "recent_key_evidence": _exclude_current_turn(case_state.key_evidence, current_turn, 5),
        "recent_negative_evidence": _exclude_current_turn(case_state.negative_evidence, current_turn, 5),
        "missing_info": list(case_state.missing_info or [])[:10],
        "active_hypotheses": list(case_state.active_hypotheses or [])[:8],
        "local_goal": case_state.local_goal,
        "uncertainty_summary": case_state.uncertainty_summary,
        "finalize_risk": case_state.finalize_risk,
        "modality_flags": list(case_state.modality_flags or [])[:8],
        "reviewed_modalities": list(case_state.reviewed_modalities or [])[:8],
    }


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
    sections: list[str] = []

    case_memory = case_memory_for_query(case_state)

    scalar_fields = [
        "problem_summary",
        "current_turn",
        "local_goal",
        "uncertainty_summary",
        "finalize_risk",
    ]
    for field in scalar_fields:
        value = str(case_memory.get(field) or "").strip()
        if value:
            sections.append(f"{field}: {value}")

    list_fields = [
        ("recent_key_evidence", 5),
        ("recent_negative_evidence", 5),
        ("missing_info", 10),
        ("active_hypotheses", 8),
        ("modality_flags", 8),
        ("reviewed_modalities", 8),
    ]
    for field, limit in list_fields:
        value = _join(list(case_memory.get(field) or []), limit=limit)
        if value:
            sections.append(f"{field}: {value}")

    if candidate_actions:
        actions = [_action_to_text(action) for action in candidate_actions]
        actions_text = _join(actions, limit=12)
        if actions_text:
            sections.append(f"candidate_actions: {actions_text}")

    query_text = "\n".join(sections).strip()
    if not query_text:
        raise RuntimeError(
            f"Cannot build memory query for case_id={case_state.case_id!r} "
            f"turn_id={case_state.turn_id}: CaseState contains no queryable information"
        )

    return MemoryQuery(
        case_id=case_state.case_id,
        turn_id=case_state.turn_id,
        query_text=query_text,
    )


def build_memory_query_llm(
    case_state: CaseState,
    candidate_actions: list[Any] | None,
    llm_client: LLMClient,
    debug: dict[str, Any] | None = None,
    strict: bool = True,
) -> MemoryQuery:
    case_memory = case_memory_for_query(case_state)
    rule_query = build_memory_query_rule(case_state, candidate_actions)
    actions = [_action_to_text(action) for action in candidate_actions or []]
    payload = {
        "case_memory": case_memory,
        "candidate_actions": actions,
        "instruction": (
            "Create one concise retrieval query for memory search. "
            "Use only case_memory and candidate_actions from the input. "
            "Focus on the current turn plus the compact current case memory. "
            "Do not restate the full historical dialogue. "
            "Mention clinical situation, new information, uncertainty, missing information, "
            "finalize risk, modalities, and useful next-action needs. "
            "Return JSON with only query_text."
        ),
    }
    prompt = query_builder_prompt(payload)
    if debug is not None:
        debug["mode"] = "llm"
        debug["case_memory"] = case_memory
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
        if strict:
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
        debug["used_fallback"] = query_text == rule_query.query_text and not ok
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
        debug["case_memory"] = case_memory_for_query(case_state)
        debug["candidate_actions"] = [_action_to_text(action) for action in candidate_actions or []]
        debug["final_query"] = result.to_dict()
    return result
