from __future__ import annotations

from typing import Any

from ..llm import LLMClient, QUERY_BUILDER_SCHEMA, parse_validate_repair, query_builder_prompt


def llm_build_query_payload(
    *,
    case_state: dict[str, Any],
    observation: dict[str, Any] | None,
    interaction_history_summary: str,
    candidate_actions: list[str],
    local_goal: str,
    uncertainty: str,
    llm_client: LLMClient,
) -> tuple[dict[str, Any], bool, str]:
    fallback = {
        "situation_anchor": case_state.get("problem_summary", ""),
        "local_goal": local_goal,
        "uncertainty_focus": uncertainty,
        "positive_evidence": [],
        "negative_evidence": [],
        "missing_info": case_state.get("missing_info", []),
        "active_hypotheses": [h.get("name", "") for h in case_state.get("active_hypotheses", [])],
        "modality_need": case_state.get("modality_flags", []),
        "candidate_action_need": candidate_actions,
        "finalize_risk_reason": f"finalize_risk={case_state.get('finalize_risk', 'high')}",
        "retrieval_intent": "mixed",
        "query_text": f"{case_state.get('problem_summary', '')} {local_goal} {uncertainty}",
    }

    prompt = query_builder_prompt(
        {
            "case_state": case_state,
            "observation": observation or {},
            "interaction_history_summary": interaction_history_summary,
            "candidate_actions": candidate_actions,
            "local_goal": local_goal,
            "uncertainty": uncertainty,
        }
    )
    raw = llm_client.generate_json(prompt)
    return parse_validate_repair(raw, QUERY_BUILDER_SCHEMA, fallback)
