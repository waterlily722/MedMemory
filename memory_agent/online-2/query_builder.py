from __future__ import annotations

from typing import Any

from ..llm import LLMClient, parse_validate_repair, query_builder_prompt
from ..llm.schemas import QUERY_BUILDER_SCHEMA
from ..schemas import CaseState, MemoryQuery


def _join(values: list[Any], limit: int = 6) -> str:
    output: list[str] = []
    for value in values or []:
        text = str(value).strip()
        if text:
            output.append(text)
        if len(output) >= limit:
            break
    return "; ".join(output)


def _format_candidate_action(action: Any) -> str:
    if isinstance(action, dict):
        action_type = str(action.get("action_type") or action.get("tool") or action.get("name") or "").strip()
        action_label = str(action.get("action_label") or action.get("argument") or action.get("content") or "").strip()
        if action_type and action_label:
            return f"{action_type}: {action_label}"
        return action_type or action_label
    return str(action).strip()


def _format_candidate_actions(candidate_actions: list[Any] | None) -> list[str]:
    return [text for text in (_format_candidate_action(action) for action in candidate_actions or []) if text]


def build_memory_query_rule(
    case_state: CaseState,
    candidate_actions: list[Any] | None = None,
) -> MemoryQuery:
    """Build retrieval query using only existing CaseState fields.

    Candidate actions are runtime policy/tool options, not CaseState fields.
    The MemoryQuery schema remains {case_id, turn_id, query_text}.
    """

    sections: list[str] = []

    if case_state.problem_summary:
        sections.append(f"problem_summary: {case_state.problem_summary}")
    if case_state.local_goal:
        sections.append(f"local_goal: {case_state.local_goal}")
    if case_state.uncertainty_summary:
        sections.append(f"uncertainty_summary: {case_state.uncertainty_summary}")
    if case_state.finalize_risk:
        sections.append(f"finalize_risk: {case_state.finalize_risk}")
    if case_state.key_evidence:
        sections.append(f"key_evidence: {_join(case_state.key_evidence[-8:], limit=8)}")
    if case_state.negative_evidence:
        sections.append(f"negative_evidence: {_join(case_state.negative_evidence[-8:], limit=8)}")
    if case_state.missing_info:
        sections.append(f"missing_info: {_join(case_state.missing_info, limit=10)}")
    if case_state.active_hypotheses:
        sections.append(f"active_hypotheses: {_join(case_state.active_hypotheses, limit=8)}")
    if case_state.modality_flags:
        sections.append(f"modality_flags: {_join(case_state.modality_flags, limit=8)}")
    if case_state.reviewed_modalities:
        sections.append(f"reviewed_modalities: {_join(case_state.reviewed_modalities, limit=8)}")
    if case_state.interaction_history_summary:
        sections.append(f"interaction_history_summary: {case_state.interaction_history_summary}")

    formatted_actions = _format_candidate_actions(candidate_actions)
    if formatted_actions:
        sections.append(f"candidate_actions: {_join(formatted_actions, limit=12)}")

    query_text = "\n".join(sections).strip()
    if not query_text:
        query_text = "clinical uncertainty; collect missing information before final diagnosis"

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
        "candidate_actions": _format_candidate_actions(candidate_actions),
        "instruction": (
            "Create one concise retrieval query for memory search. "
            "Use only the provided CaseState fields and candidate_actions. "
            "Return JSON with only query_text."
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
