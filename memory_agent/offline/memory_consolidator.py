from __future__ import annotations

from ..schemas import ExperienceCard, SkillCard
from .experience_merger import decide_merge_llm, decide_merge_rule
from .skill_miner import mine_skill_llm, mine_skill_rule


def consolidate_experience(
    *,
    new_experience: ExperienceCard,
    similar_existing: list[ExperienceCard],
    mode: str = "rule",
    llm_client=None,
) -> dict:
    if mode == "llm" and llm_client is not None:
        return decide_merge_llm(new_experience, similar_existing, llm_client)

    decision, ids, merged, discard_reason = decide_merge_rule(new_experience, similar_existing)
    return {
        "merge_decision": decision,
        "target_memory_ids": ids,
        "reason": "rule merge",
        "merged_experience": merged.to_dict() if merged is not None else {},
        "discard_reason": discard_reason,
    }


def consolidate_skill(
    *,
    clustered_success_experiences: list[ExperienceCard],
    mode: str = "rule",
    llm_client=None,
) -> SkillCard | None:
    if mode == "llm" and llm_client is not None:
        return mine_skill_llm(clustered_success_experiences, llm_client)
    return mine_skill_rule(clustered_success_experiences)
