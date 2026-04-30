from __future__ import annotations

import logging
import uuid
from typing import Any

from ..llm import LLMClient, experience_merge_prompt, parse_validate_repair

logger = logging.getLogger(__name__)
from ..llm.schemas import EXPERIENCE_MERGE_SCHEMA
from ..schemas import ExperienceCard, OutcomeType
from ..utils.config import MERGE_CONFIG
from ..utils.scoring import cosine_similarity


def _threshold(name: str, default: float) -> float:
    try:
        return float(MERGE_CONFIG.get(name, default))
    except Exception:
        return default


def _same_direction(left: ExperienceCard, right: ExperienceCard) -> bool:
    left_positive = left.outcome_type in {
        OutcomeType.SUCCESS.value, OutcomeType.PARTIAL_SUCCESS.value
    }
    right_positive = right.outcome_type in {
        OutcomeType.SUCCESS.value, OutcomeType.PARTIAL_SUCCESS.value
    }
    return left_positive == right_positive


def _same_trigger(left: ExperienceCard, right: ExperienceCard) -> bool:
    situation_threshold = _threshold("semantic_threshold", 0.80)
    action_threshold = _threshold("action_threshold", 0.75)

    return (
        cosine_similarity(left.situation_text, right.situation_text) >= situation_threshold
        and cosine_similarity(left.action_text, right.action_text) >= action_threshold
    )


def _boundary_compatible(left: ExperienceCard, right: ExperienceCard) -> bool:
    if not left.boundary_text or not right.boundary_text:
        return True
    boundary_threshold = _threshold("boundary_threshold", 0.50)
    return cosine_similarity(left.boundary_text, right.boundary_text) >= boundary_threshold


def _can_merge(left: ExperienceCard, right: ExperienceCard) -> bool:
    return (
        _same_trigger(left, right)
        and _same_direction(left, right)
        and _boundary_compatible(left, right)
    )


def _unique(values: list[Any]) -> list[Any]:
    seen = set()
    output = []
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _choose_longer(left: str, right: str) -> str:
    left = left or ""
    right = right or ""
    return left if len(left) >= len(right) else right


def merge_experience(base: ExperienceCard, incoming: ExperienceCard) -> ExperienceCard:
    """
    Merge two experiences with same trigger and same outcome direction.
    Keep base memory_id stable.
    """
    merged = ExperienceCard.from_dict(base.to_dict())

    merged.support_count = max(1, base.support_count) + max(1, incoming.support_count)

    merged.situation_text = _choose_longer(base.situation_text, incoming.situation_text)
    merged.action_text = _choose_longer(base.action_text, incoming.action_text)
    merged.outcome_text = _choose_longer(base.outcome_text, incoming.outcome_text)
    merged.boundary_text = _choose_longer(base.boundary_text, incoming.boundary_text)

    merged.action_sequence = (
        base.action_sequence
        if len(base.action_sequence) >= len(incoming.action_sequence)
        else incoming.action_sequence
    )

    merged.retrieval_tags = _unique(base.retrieval_tags + incoming.retrieval_tags)
    merged.risk_tags = _unique(base.risk_tags + incoming.risk_tags)

    merged.confidence = max(base.confidence, incoming.confidence)

    merged.source_episode_ids = _unique(base.source_episode_ids + incoming.source_episode_ids)
    merged.source_case_ids = _unique(base.source_case_ids + incoming.source_case_ids)
    merged.source_turn_ids = _unique(base.source_turn_ids + incoming.source_turn_ids)

    return merged


def decide_merge_rule(
    new_experience: ExperienceCard,
    similar_existing: list[ExperienceCard],
) -> dict[str, Any]:
    for existing in similar_existing:
        if _same_trigger(existing, new_experience) and not _same_direction(existing, new_experience):
            conflict_group_id = existing.conflict_group_id or f"conflict_{uuid.uuid4().hex[:10]}"
            return {
                "merge_decision": "conflict",
                "target_memory_ids": [existing.memory_id, new_experience.memory_id],
                "reason": "same situation/action but opposite outcome",
                "merged_experience": {},
                "conflict_group_id": conflict_group_id,
            }

        if _can_merge(existing, new_experience):
            merged = merge_experience(existing, new_experience)
            return {
                "merge_decision": "merge",
                "target_memory_ids": [existing.memory_id, new_experience.memory_id],
                "reason": "same situation/action/outcome direction",
                "merged_experience": merged.to_dict(),
                "conflict_group_id": "",
            }

    return {
        "merge_decision": "insert_new",
        "target_memory_ids": [],
        "reason": "no compatible existing memory",
        "merged_experience": new_experience.to_dict(),
        "conflict_group_id": "",
    }


def decide_merge_llm(
    new_experience: ExperienceCard,
    similar_existing: list[ExperienceCard],
    llm_client: LLMClient,
) -> dict[str, Any]:
    fallback = decide_merge_rule(new_experience, similar_existing)

    if not llm_client.available():
        return fallback

    payload = {
        "new_experience": new_experience.to_dict(),
        "similar_existing": [item.to_dict() for item in similar_existing],
        "rule_decision": fallback,
        "instruction": (
            "Decide whether to insert, merge, discard, or mark conflict. "
            "Never merge opposite outcomes. "
            "If same situation/action but opposite outcome, use conflict."
        ),
    }

    parsed, _, _ = parse_validate_repair(
        llm_client.generate_json(experience_merge_prompt(payload), max_tokens=1200),
        EXPERIENCE_MERGE_SCHEMA,
        fallback,
    )

    decision = str(parsed.get("merge_decision") or "insert_new")
    if decision not in {"insert_new", "merge", "discard", "conflict"}:
        logger.warning(
            "LLM merge returned invalid decision=%r; falling back to rule", decision
        )
        return fallback

    if decision == "merge" and not isinstance(parsed.get("merged_experience"), dict):
        logger.warning("LLM merge decided 'merge' but merged_experience is not dict; fallback")
        return fallback

    return parsed