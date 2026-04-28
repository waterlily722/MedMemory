from __future__ import annotations

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


def can_merge(base: ExperienceCard, candidate: ExperienceCard) -> bool:
    semantic = cosine_similarity(base.situation_anchor, candidate.situation_anchor)
    local_goal_match = base.local_goal == candidate.local_goal or cosine_similarity(base.local_goal, candidate.local_goal) > 0.85
    action_ok = _action_distance(base.action_sequence, candidate.action_sequence) <= MERGE_CONFIG["action_edit_distance"]
    hypo_overlap = overlap_score(base.error_tag, candidate.error_tag)
    boundary_ok = not (base.boundary and candidate.boundary and base.boundary != candidate.boundary and "conflict" in base.boundary + candidate.boundary)
    return (
        semantic >= MERGE_CONFIG["semantic_threshold"]
        and local_goal_match
        and action_ok
        and hypo_overlap >= MERGE_CONFIG["min_hypothesis_overlap"]
        and boundary_ok
    )


def merge_experience(base: ExperienceCard, candidate: ExperienceCard) -> ExperienceCard:
    merged = ExperienceCard.from_dict(base.to_dict())
    merged.source_episode_ids = list(dict.fromkeys(base.source_episode_ids + candidate.source_episode_ids))
    merged.source_case_ids = list(dict.fromkeys(base.source_case_ids + candidate.source_case_ids))
    merged.source_field_refs = list(dict.fromkeys(base.source_field_refs + candidate.source_field_refs))
    merged.support_count = int(base.support_count) + int(candidate.support_count)
    if base.outcome_type != candidate.outcome_type:
        merged.outcome_type = "partial_success"
        merged.error_tag = list(dict.fromkeys(base.error_tag + candidate.error_tag + ["conflict_group"]))
    return merged
