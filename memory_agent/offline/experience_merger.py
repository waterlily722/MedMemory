from __future__ import annotations

import uuid
from typing import Any

from ..llm import LLMClient, experience_merge_prompt, parse_validate_repair
from ..llm.schemas import EXPERIENCE_MERGE_SCHEMA
from ..schemas import ExperienceCard
from ..utils.config import MERGE_CONFIG
from ..utils.scoring import cosine_similarity, overlap_score


def _same_direction(left: ExperienceCard, right: ExperienceCard) -> bool:
    left_pos = left.outcome_type in {"success", "partial_success"}
    right_pos = right.outcome_type in {"success", "partial_success"}
    return left_pos == right_pos


def _same_trigger(left: ExperienceCard, right: ExperienceCard) -> bool:
    return (
        cosine_similarity(left.situation_text, right.situation_text) >= MERGE_CONFIG["semantic_threshold"]
        and cosine_similarity(left.action_text, right.action_text) >= MERGE_CONFIG["action_threshold"]
    )


def _boundary_compatible(left: ExperienceCard, right: ExperienceCard) -> bool:
    if not left.boundary_text or not right.boundary_text:
        return True
    return cosine_similarity(left.boundary_text, right.boundary_text) >= 0.5


def _can_merge(left: ExperienceCard, right: ExperienceCard) -> bool:
    return (
        _same_trigger(left, right)
        and _same_direction(left, right)
        and _boundary_compatible(left, right)
    )


def merge_experience(base: ExperienceCard, incoming: ExperienceCard) -> ExperienceCard:
    merged = ExperienceCard.from_dict(base.to_dict())

    merged.support_count = base.support_count + incoming.support_count

    merged.situation_text = base.situation_text or incoming.situation_text
    merged.action_text = base.action_text or incoming.action_text
    merged.outcome_text = base.outcome_text or incoming.outcome_text
    merged.boundary_text = base.boundary_text or incoming.boundary_text

    merged.action_sequence = (
        base.action_sequence
        if len(base.action_sequence) >= len(incoming.action_sequence)
        else incoming.action_sequence
    )

    merged.retrieval_tags = list(dict.fromkeys(base.retrieval_tags + incoming.retrieval_tags))
    merged.risk_tags = list(dict.fromkeys(base.risk_tags + incoming.risk_tags))

    merged.confidence = max(base.confidence, incoming.confidence)

    merged.source_episode_ids = list(dict.fromkeys(base.source_episode_ids + incoming.source_episode_ids))
    merged.source_case_ids = list(dict.fromkeys(base.source_case_ids + incoming.source_case_ids))
    merged.source_turn_ids = list(dict.fromkeys(base.source_turn_ids + incoming.source_turn_ids))

    return merged


def decide_merge_rule(new_experience: ExperienceCard, similar_existing: list[ExperienceCard]) -> dict[str, Any]:
    for existing in similar_existing:
        same_anchor = cosine_similarity(existing.situation_anchor, new_experience.situation_anchor) >= MERGE_CONFIG["semantic_threshold"]
        opposite_outcome = _same_direction(existing, new_experience) is False
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
                "reason": "rule merge",
                "merged_experience": merged.to_dict(),
                "conflict_group_id": None,
            }
    return {
        "merge_decision": "insert_new",
        "target_memory_ids": [],
        "reason": "new experience",
        "merged_experience": new_experience.to_dict(),
        "conflict_group_id": None,
    }


def decide_merge_llm(new_experience: ExperienceCard, similar_existing: list[ExperienceCard], llm_client: LLMClient) -> dict[str, Any]:
    fallback = decide_merge_rule(new_experience, similar_existing)
    if not llm_client.available():
        return fallback
    payload = {"new_experience": new_experience.to_dict(), "similar_existing": [item.to_dict() for item in similar_existing]}
    parsed, _, _ = parse_validate_repair(llm_client.generate_json(experience_merge_prompt(payload)), EXPERIENCE_MERGE_SCHEMA, fallback)
    return parsed
