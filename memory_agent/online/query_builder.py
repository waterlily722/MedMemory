from __future__ import annotations

from typing import Any

from ..llm import LLMClient, parse_validate_repair, query_builder_prompt
from ..llm.schemas import QUERY_BUILDER_SCHEMA
from ..schemas import CaseState, MemoryQuery


def _base_query(case_state: CaseState, candidate_actions: list[str]) -> MemoryQuery:
    positive = list(case_state.key_evidence[:8])
    negative = list(case_state.negative_evidence[:8])
    missing = list(case_state.missing_info[:8])
    hypotheses = list(case_state.active_hypotheses[:8])
    modality_need = []
    if "lab" in case_state.modality_flags:
        modality_need.append("lab")
    if "image" in case_state.modality_flags:
        modality_need.append("image")
    if "text" in case_state.modality_flags:
        modality_need.append("text")

    risk_reason = "missing_critical_info" if missing else "image_needed" if "image" in modality_need and "image" not in case_state.reviewed_modalities else "other"
    retrieval_intent = "mixed"
    lower_actions = [action.lower() for action in candidate_actions]
    if any("finalize" in action for action in lower_actions):
        retrieval_intent = "mixed"
    elif any("image" in action for action in lower_actions):
        retrieval_intent = "experience"
    elif any("lab" in action or "exam" in action for action in lower_actions):
        retrieval_intent = "skill"

    query_text = " | ".join(
        [
            case_state.problem_summary,
            case_state.local_goal,
            case_state.uncertainty_summary,
            " ".join(positive[:4]),
            "missing:" + ",".join(missing[:4]),
            "risk:" + case_state.finalize_risk,
            case_state.interaction_history_summary,
        ]
    )
    return MemoryQuery(
        query_text=query_text,
        situation_anchor=case_state.problem_summary,
        local_goal=case_state.local_goal,
        uncertainty_focus=case_state.uncertainty_summary,
        positive_evidence=positive,
        negative_evidence=negative,
        missing_info=missing,
        active_hypotheses=hypotheses,
        modality_need=modality_need,
        candidate_action_need=list(candidate_actions),
        finalize_risk=case_state.finalize_risk,
        finalize_risk_reason=risk_reason,
        retrieval_intent=retrieval_intent,
    )


def build_memory_query_rule(case_state: CaseState, candidate_actions: list[str]) -> MemoryQuery:
    return _base_query(case_state, candidate_actions)


def build_memory_query_llm(case_state: CaseState, candidate_actions: list[str], llm_client: LLMClient) -> MemoryQuery:
    rule_query = _base_query(case_state, candidate_actions)
    if not llm_client.available():
        return rule_query
    payload = {
        "case_state": case_state.to_dict(),
        "candidate_actions": candidate_actions,
        "structured_query": rule_query.to_dict(),
    }
    fallback = rule_query.to_dict()
    parsed, _, _ = parse_validate_repair(llm_client.generate_json(query_builder_prompt(payload)), QUERY_BUILDER_SCHEMA, fallback)
    return MemoryQuery.from_dict(parsed)


def build_memory_query(case_state: CaseState, candidate_actions: list[str], mode: str = "rule", llm_client: LLMClient | None = None) -> MemoryQuery:
    if mode == "llm" and llm_client is not None:
        return build_memory_query_llm(case_state, candidate_actions, llm_client)
    return build_memory_query_rule(case_state, candidate_actions)
