from __future__ import annotations

import uuid
from typing import Any

from ..llm import LLMClient, experience_merge_prompt, parse_validate_repair
from ..llm.schemas import EXPERIENCE_MERGE_SCHEMA
from ..schemas import ExperienceCard
from ..utils.config import MERGE_CONFIG
from ..utils.scoring import cosine_similarity, overlap_score


def _action_labels(experience: ExperienceCard) -> list[str]:
    return [str(step.get("action_label", step.get("action_type", ""))) for step in experience.action_sequence if isinstance(step, dict)]


def _same_boundary(left: ExperienceCard, right: ExperienceCard) -> bool:
    if not left.boundary or not right.boundary:
        return True
    return left.boundary == right.boundary or cosine_similarity(left.boundary, right.boundary) >= MERGE_CONFIG["semantic_threshold"]


def _same_direction(left: ExperienceCard, right: ExperienceCard) -> bool:
    success_left = left.outcome_type in {"success", "partial_success"}
    success_right = right.outcome_type in {"success", "partial_success"}
    return success_left == success_right


def _can_merge(left: ExperienceCard, right: ExperienceCard) -> bool:
    return (
        cosine_similarity(left.situation_anchor, right.situation_anchor) >= MERGE_CONFIG["semantic_threshold"]
        and cosine_similarity(left.local_goal, right.local_goal) >= MERGE_CONFIG["goal_threshold"]
        and cosine_similarity(" ".join(_action_labels(left)), " ".join(_action_labels(right))) >= MERGE_CONFIG["action_threshold"]
        and overlap_score(left.active_hypotheses, right.active_hypotheses) >= 0.4
        and _same_boundary(left, right)
        and _same_direction(left, right)
    )


def merge_experience(base: ExperienceCard, incoming: ExperienceCard) -> ExperienceCard:
    merged = ExperienceCard.from_dict(base.to_dict())
    merged.support_count = base.support_count + incoming.support_count
    merged.key_evidence = list(dict.fromkeys(base.key_evidence + incoming.key_evidence))
    merged.missing_info = list(dict.fromkeys(base.missing_info + incoming.missing_info))
    merged.active_hypotheses = list(dict.fromkeys(base.active_hypotheses + incoming.active_hypotheses))
    merged.action_sequence = base.action_sequence if len(base.action_sequence) >= len(incoming.action_sequence) else incoming.action_sequence
    merged.outcome_shift = base.outcome_shift or incoming.outcome_shift
    merged.boundary = base.boundary or incoming.boundary
    merged.applicability_conditions = list(dict.fromkeys(base.applicability_conditions + incoming.applicability_conditions))
    merged.non_applicability_conditions = list(dict.fromkeys(base.non_applicability_conditions + incoming.non_applicability_conditions))
    merged.modality_flags = list(dict.fromkeys(base.modality_flags + incoming.modality_flags))
    merged.retrieval_tags = list(dict.fromkeys(base.retrieval_tags + incoming.retrieval_tags))
    merged.risk_tags = list(dict.fromkeys(base.risk_tags + incoming.risk_tags))
    merged.confidence = max(base.confidence, incoming.confidence)
    merged.source_episode_ids = list(dict.fromkeys(base.source_episode_ids + incoming.source_episode_ids))
    merged.source_case_ids = list(dict.fromkeys(base.source_case_ids + incoming.source_case_ids))
    return merged


def decide_merge_rule(new_experience: ExperienceCard, similar_existing: list[ExperienceCard]) -> dict[str, Any]:
    for existing in similar_existing:
        same_anchor = cosine_similarity(existing.situation_anchor, new_experience.situation_anchor) >= MERGE_CONFIG["semantic_threshold"]
        opposite_outcome = _same_direction(existing, new_experience) is False
        if same_anchor and opposite_outcome:
            conflict_group_id = existing.conflict_group_id or f"conflict_{uuid.uuid4().hex[:10]}"
            return {
                "merge_decision": "conflict",
                "target_memory_ids": [existing.memory_id, new_experience.memory_id],
                "reason": "same trigger but opposite outcome",
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
