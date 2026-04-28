from __future__ import annotations

import uuid

from ..llm import EXPERIENCE_MERGE_SCHEMA, LLMClient, experience_merge_prompt, parse_validate_repair
from ..schemas import ExperienceCard
from ..utils.config import MERGE_CONFIG
from ..utils.scoring import cosine_similarity, overlap_score


def _action_distance(a: list[dict[str, str]], b: list[dict[str, str]]) -> int:
    labels_a = [x.get("action_label", "") for x in a]
    labels_b = [x.get("action_label", "") for x in b]
    if labels_a == labels_b:
        return 0
    if abs(len(labels_a) - len(labels_b)) > 1:
        return 2
    mismatch = sum(1 for x, y in zip(labels_a, labels_b) if x != y)
    mismatch += abs(len(labels_a) - len(labels_b))
    return mismatch


def _same_boundary(base: ExperienceCard, candidate: ExperienceCard) -> bool:
    if not base.boundary or not candidate.boundary:
        return True
    return base.boundary == candidate.boundary or cosine_similarity(base.boundary, candidate.boundary) > 0.75


def can_merge(base: ExperienceCard, candidate: ExperienceCard) -> bool:
    semantic = cosine_similarity(base.situation_anchor, candidate.situation_anchor)
    local_goal_match = base.local_goal == candidate.local_goal or cosine_similarity(base.local_goal, candidate.local_goal) > 0.85
    action_ok = _action_distance(base.action_sequence, candidate.action_sequence) <= MERGE_CONFIG["action_edit_distance"]
    hypo_overlap = overlap_score(base.hypotheses, candidate.hypotheses)
    boundary_ok = _same_boundary(base, candidate)
    return (
        semantic >= MERGE_CONFIG["semantic_threshold"]
        and local_goal_match
        and action_ok
        and hypo_overlap >= MERGE_CONFIG["min_hypothesis_overlap"]
        and boundary_ok
    )


def merge_experience(base: ExperienceCard, candidate: ExperienceCard) -> ExperienceCard:
    merged = ExperienceCard.from_dict(base.to_dict())
    merged.experience_id = merged.experience_id or merged.item_id
    merged.source_episode_ids = list(dict.fromkeys(base.source_episode_ids + candidate.source_episode_ids))
    merged.source_case_ids = list(dict.fromkeys(base.source_case_ids + candidate.source_case_ids))
    merged.source_turn_ids = list(dict.fromkeys(base.source_turn_ids + candidate.source_turn_ids))
    merged.source_field_refs = list(dict.fromkeys(base.source_field_refs + candidate.source_field_refs))
    merged.support_count = int(base.support_count) + int(candidate.support_count)
    merged.key_evidence = list(dict.fromkeys(base.key_evidence + candidate.key_evidence))
    merged.missing_info = list(dict.fromkeys(base.missing_info + candidate.missing_info))
    merged.applicability_conditions = list(dict.fromkeys(base.applicability_conditions + candidate.applicability_conditions))
    merged.non_applicability_conditions = list(dict.fromkeys(base.non_applicability_conditions + candidate.non_applicability_conditions))
    merged.retrieval_tags = list(dict.fromkeys(base.retrieval_tags + candidate.retrieval_tags))
    merged.hypotheses = list(dict.fromkeys(base.hypotheses + candidate.hypotheses))
    merged.confidence = max(base.confidence, candidate.confidence)
    if base.outcome_type != candidate.outcome_type:
        merged.outcome_type = "partial_success"
        merged.error_tag = list(dict.fromkeys(base.error_tag + candidate.error_tag + ["conflict_group"]))
    return merged


def decide_merge_rule(new_experience: ExperienceCard, similar_existing: list[ExperienceCard]) -> tuple[str, list[str], ExperienceCard | None, str | None]:
    for existing in similar_existing:
        if existing.outcome_type != new_experience.outcome_type and cosine_similarity(existing.situation_anchor, new_experience.situation_anchor) > 0.85:
            conflict_group_id = existing.conflict_group_id or f"conflict_{uuid.uuid4().hex[:10]}"
            return "conflict", [existing.item_id, new_experience.item_id], None, conflict_group_id
        if can_merge(existing, new_experience):
            return "merge_with_existing", [existing.item_id], merge_experience(existing, new_experience), None
    return "insert_new", [], new_experience, None


def decide_merge_llm(
    new_experience: ExperienceCard,
    similar_existing: list[ExperienceCard],
    llm_client: LLMClient,
) -> dict:
    fallback = {
        "merge_decision": "insert_new",
        "target_memory_ids": [],
        "reason": "fallback insert_new",
        "merged_experience": new_experience.to_dict(),
        "discard_reason": None,
    }
    prompt = experience_merge_prompt(
        {
            "new_experience": new_experience.to_dict(),
            "similar_existing": [x.to_dict() for x in similar_existing],
        }
    )
    parsed, _, _ = parse_validate_repair(llm_client.generate_json(prompt), EXPERIENCE_MERGE_SCHEMA, fallback)
    if parsed.get("merge_decision") == "conflict" and not parsed.get("discard_reason"):
        parsed["discard_reason"] = f"conflict_group:{parsed.get('conflict_group_id', '')}"
    return parsed
