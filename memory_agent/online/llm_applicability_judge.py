from __future__ import annotations

from typing import Any

from ..llm import APPLICABILITY_JUDGE_SCHEMA, LLMClient, applicability_prompt, parse_validate_repair


def llm_judge_applicability(
    *,
    case_state: dict[str, Any],
    structured_memory_query: dict[str, Any],
    memory_item: dict[str, Any],
    candidate_actions: list[str],
    local_goal: str,
    finalize_risk: str,
    llm_client: LLMClient,
) -> tuple[dict[str, Any], bool, str]:
    memory_id = str(memory_item.get("memory_id") or memory_item.get("item_id") or "")
    memory_type = str(memory_item.get("memory_type") or "experience")
    memory_content = memory_item.get("content") if isinstance(memory_item.get("content"), dict) else memory_item
    fallback = {
        "memory_id": memory_id,
        "memory_type": memory_type,
        "memory_content": memory_content if isinstance(memory_content, dict) else {},
        "applicability": "medium",
        "reason": "fallback rule: moderate applicability",
        "matched_aspects": [],
        "mismatched_aspects": [],
        "boundary_violation": None,
        "action_bias": {a: 0.0 for a in candidate_actions},
        "blocked_actions": [],
        "controller_decision": "hint",
    }

    prompt = applicability_prompt(
        {
            "case_state": case_state,
            "structured_memory_query": structured_memory_query,
            "memory_item": memory_item,
            "candidate_actions": candidate_actions,
            "local_goal": local_goal,
            "finalize_risk": finalize_risk,
        }
    )
    raw = llm_client.generate_json(prompt)
    return parse_validate_repair(raw, APPLICABILITY_JUDGE_SCHEMA, fallback)
