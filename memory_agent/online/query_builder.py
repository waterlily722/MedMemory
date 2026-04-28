from __future__ import annotations

from ..schemas import CaseState, MemoryQuery, MemoryQueryStructured
from .llm_query_builder import llm_build_query_payload


def build_memory_query(case_state: CaseState, action_candidates: list[str]) -> MemoryQuery:
    pos = [item.content for item in case_state.evidence_items if item.polarity == "positive"][:8]
    neg = [item.content for item in case_state.evidence_items if item.polarity == "negative"][:6]
    hypotheses = [h.name for h in case_state.active_hypotheses][:6]
    finalize_risk_reason = "missing critical info" if case_state.missing_info else "high uncertainty" if case_state.finalize_risk == "high" else "bounded"
    retrieval_intent = "mixed"
    if any("review" in action.lower() for action in action_candidates):
        retrieval_intent = "experience"
    elif any("request" in action.lower() for action in action_candidates):
        retrieval_intent = "skill"

    structured = MemoryQueryStructured(
        situation_anchor=case_state.problem_summary,
        local_goal=case_state.local_goal,
        uncertainty_focus=case_state.uncertainty_summary,
        active_hypotheses=hypotheses,
        key_positive_evidence=pos,
        key_negative_evidence=neg,
        missing_info=case_state.missing_info[:8],
        modality_flags=case_state.modality_flags,
        finalize_risk=case_state.finalize_risk,
        finalize_risk_reason=finalize_risk_reason,
        retrieval_intent=retrieval_intent,
        current_action_candidates=action_candidates,
        source_field_refs=case_state.source_field_refs,
    )
    query_text = " | ".join(
        [
            case_state.problem_summary,
            case_state.uncertainty_summary,
            case_state.local_goal,
            " ".join(hypotheses),
            " ".join(pos[:4]),
            "missing:" + ",".join(case_state.missing_info[:4]),
            "risk:" + case_state.finalize_risk,
        ]
    )
    return MemoryQuery(query_text=query_text, structured=structured, source_field_refs=case_state.source_field_refs)


def build_memory_query_with_mode(
    case_state: CaseState,
    action_candidates: list[str],
    mode: str = "rule",
    llm_client=None,
    observation: dict | None = None,
    interaction_history_summary: str = "",
) -> MemoryQuery:
    rule_query = build_memory_query(case_state, action_candidates)
    if mode != "llm" or llm_client is None:
        return rule_query

    payload, _, _ = llm_build_query_payload(
        case_state=case_state.to_dict(),
        observation=observation,
        interaction_history_summary=interaction_history_summary,
        candidate_actions=action_candidates,
        local_goal=case_state.local_goal,
        uncertainty=case_state.uncertainty_summary,
        llm_client=llm_client,
    )

    structured = MemoryQueryStructured(
        situation_anchor=str(payload.get("situation_anchor", rule_query.structured.situation_anchor)),
        local_goal=str(payload.get("local_goal", case_state.local_goal)),
        uncertainty_focus=str(payload.get("uncertainty_focus", case_state.uncertainty_summary)),
        active_hypotheses=[str(x) for x in payload.get("active_hypotheses", [])],
        key_positive_evidence=[str(x) for x in payload.get("positive_evidence", [])],
        key_negative_evidence=[str(x) for x in payload.get("negative_evidence", [])],
        missing_info=[str(x) for x in payload.get("missing_info", [])],
        modality_flags=[str(x) for x in payload.get("modality_need", [])],
        finalize_risk=str(payload.get("finalize_risk", case_state.finalize_risk)),
        finalize_risk_reason=str(payload.get("finalize_risk_reason", rule_query.structured.finalize_risk_reason)),
        retrieval_intent=str(payload.get("retrieval_intent", rule_query.structured.retrieval_intent)),
        current_action_candidates=[str(x) for x in payload.get("candidate_action_need", action_candidates)],
        source_field_refs=case_state.source_field_refs,
    )
    return MemoryQuery(
        query_text=str(payload.get("query_text", rule_query.query_text)),
        structured=structured,
        source_field_refs=case_state.source_field_refs,
    )
