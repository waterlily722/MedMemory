from __future__ import annotations

from ..llm import LLMClient, parse_validate_repair, query_builder_prompt
from ..llm.schemas import QUERY_BUILDER_SCHEMA
from ..schemas import CaseState, MemoryQuery


def _join(values: list[str], limit: int = 6) -> str:
    return ", ".join(value for value in values[:limit] if value)


def build_memory_query_rule(
    case_state: CaseState,
    candidate_actions: list[str],
) -> MemoryQuery:
    parts = [
        f"Problem: {case_state.problem_summary}",
        f"Current goal: {case_state.local_goal}",
        f"Uncertainty: {case_state.uncertainty_summary}",
        f"Positive evidence: {_join(case_state.key_evidence)}",
        f"Negative evidence: {_join(case_state.negative_evidence)}",
        f"Missing information: {_join(case_state.missing_info)}",
        f"Active hypotheses: {_join(case_state.active_hypotheses)}",
        f"Finalize risk: {case_state.finalize_risk}",
        f"Available modalities: {_join(case_state.modality_flags)}",
        f"Reviewed modalities: {_join(case_state.reviewed_modalities)}",
        f"Candidate actions: {_join(candidate_actions, limit=10)}",
        f"Interaction summary: {case_state.interaction_history_summary}",
    ]

    query_text = "\n".join(
        part for part in parts
        if part and not part.endswith(": ")
    )

    return MemoryQuery(
        case_id=case_state.case_id,
        turn_id=case_state.turn_id,
        query_text=query_text,
    )


def build_memory_query_llm(
    case_state: CaseState,
    candidate_actions: list[str],
    llm_client: LLMClient,
) -> MemoryQuery:
    fallback = build_memory_query_rule(case_state, candidate_actions)

    if not llm_client.available():
        return fallback

    payload = {
        "case_state": case_state.to_dict(),
        "candidate_actions": candidate_actions,
        "instruction": (
            "Create one concise retrieval query for memory search. "
            "The query should mention the clinical situation, uncertainty, "
            "missing information, finalize risk, and useful next-action needs. "
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
    candidate_actions: list[str],
    mode: str = "rule",
    llm_client: LLMClient | None = None,
) -> MemoryQuery:
    if mode == "llm" and llm_client is not None:
        return build_memory_query_llm(case_state, candidate_actions, llm_client)

    return build_memory_query_rule(case_state, candidate_actions)