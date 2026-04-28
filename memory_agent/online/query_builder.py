from __future__ import annotations

from ..schemas import CaseState, MemoryQuery, MemoryQueryStructured


def build_memory_query(case_state: CaseState, action_candidates: list[str]) -> MemoryQuery:
    pos = [item.content for item in case_state.evidence_items if item.polarity == "positive"][:8]
    neg = [item.content for item in case_state.evidence_items if item.polarity == "negative"][:6]
    hypotheses = [h.name for h in case_state.active_hypotheses][:6]

    structured = MemoryQueryStructured(
        local_goal=case_state.local_goal,
        active_hypotheses=hypotheses,
        key_positive_evidence=pos,
        key_negative_evidence=neg,
        missing_info=case_state.missing_info[:8],
        modality_flags=case_state.modality_flags,
        finalize_risk=case_state.finalize_risk,
        current_action_candidates=action_candidates,
        source_field_refs=case_state.source_field_refs,
    )
    query_text = " | ".join(
        [
            case_state.problem_summary,
            case_state.local_goal,
            " ".join(hypotheses),
            " ".join(pos[:4]),
            "missing:" + ",".join(case_state.missing_info[:4]),
            "risk:" + case_state.finalize_risk,
        ]
    )
    return MemoryQuery(query_text=query_text, structured=structured, source_field_refs=case_state.source_field_refs)
