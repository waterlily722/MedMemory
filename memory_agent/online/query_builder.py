from __future__ import annotations

from typing import Any

from ..llm import LLMClient, parse_validate_repair, query_builder_prompt
from ..llm.schemas import QUERY_BUILDER_SCHEMA
from ..schemas import CaseState, MemoryQuery


def _join(values: list[Any], limit: int = 6) -> str:
    cleaned = [str(value).strip() for value in values[:limit] if str(value).strip()]
    return "; ".join(cleaned)


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

    scalar_fields = [
        "problem_summary",
        "local_goal",
        "uncertainty_summary",
        "finalize_risk",
        "interaction_history_summary",
    ]
    for field in scalar_fields:
        value = str(getattr(case_state, field, "") or "").strip()
        if value:
            sections.append(f"{field}: {value}")

    list_fields = [
        ("key_evidence", 8),
        ("negative_evidence", 8),
        ("missing_info", 10),
        ("active_hypotheses", 8),
        ("modality_flags", 8),
        ("reviewed_modalities", 8),
    ]
    for field, limit in list_fields:
        value = _join(list(getattr(case_state, field, []) or []), limit=limit)
        if value:
            sections.append(f"{field}: {value}")

    if candidate_actions:
        actions = [_action_to_text(action) for action in candidate_actions]
        actions_text = _join(actions, limit=12)
        if actions_text:
            sections.append(f"candidate_actions: {actions_text}")

    query_text = "\n".join(sections).strip()
    if not query_text:
        query_text = "clinical case with unresolved diagnostic uncertainty"

    return MemoryQuery(
        case_id=case_state.case_id,
        turn_id=case_state.turn_id,
        query_text=query_text,
    )


def build_memory_query_llm(
    case_state: CaseState,
    candidate_actions: list[Any] | None,
    llm_client: LLMClient,
) -> MemoryQuery:
    fallback = build_memory_query_rule(case_state, candidate_actions)
    if not llm_client.available():
        return fallback

    payload = {
        "case_state": case_state.to_dict(),
        "candidate_actions": [_action_to_text(action) for action in candidate_actions or []],
        "instruction": (
            "Create one concise retrieval query for memory search. "
            "Use only CaseState fields and candidate_actions from the input. "
            "Mention clinical situation, uncertainty, missing information, finalize risk, "
            "modalities, and useful next-action needs. Return JSON with only query_text."
        ),
    }
    parsed, _, _ = parse_validate_repair(
        llm_client.generate_json(query_builder_prompt(payload), max_tokens=800),
        QUERY_BUILDER_SCHEMA,
        {"query_text": fallback.query_text},
    )
    query_text = str(parsed.get("query_text") or fallback.query_text).strip()
    return MemoryQuery(
        case_id=case_state.case_id,
        turn_id=case_state.turn_id,
        query_text=query_text or fallback.query_text,
    )


def build_memory_query(
    case_state: CaseState,
    candidate_actions: list[Any] | None = None,
    mode: str = "rule",
    llm_client: LLMClient | None = None,
) -> MemoryQuery:
    if mode == "llm" and llm_client is not None:
        return build_memory_query_llm(case_state, candidate_actions, llm_client)
    return build_memory_query_rule(case_state, candidate_actions)
