from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from ..schemas import DistilledEpisode, EpisodeFeedback, ExperienceCard, SkillCard, SkillProcedureStep, TurnFeedback
from ..utils.config import EXPERIENCE_WRITE_CONFIG
from .experience_extractor import extract_experiences_llm, extract_experiences_rule
from .skill_miner import mine_skill_llm, mine_skill_rule


def _as_turn_feedbacks(turn_feedback_list: list[Any]) -> list[TurnFeedback]:
    output: list[TurnFeedback] = []
    for item in turn_feedback_list or []:
        output.append(item if isinstance(item, TurnFeedback) else TurnFeedback.from_dict(item))
    return output


def distill_from_trajectory(
    trajectory: Any,
    turn_feedback_list: list[Any],
    episode_feedback: EpisodeFeedback | dict[str, Any],
    experience_extraction_mode: str = "rule",
    skill_mining_mode: str = "rule",
    llm_client=None,
) -> DistilledEpisode:
    epi = episode_feedback if isinstance(episode_feedback, EpisodeFeedback) else EpisodeFeedback.from_dict(episode_feedback)
    turn_feedbacks = _as_turn_feedbacks(turn_feedback_list)

    if experience_extraction_mode == "llm" and llm_client is not None:
        experiences = extract_experiences_llm(trajectory, turn_feedbacks, epi, llm_client)
    else:
        experiences = extract_experiences_rule(trajectory, turn_feedbacks, epi)

    skills: list[SkillCard] = []
    if len(experiences) >= 2 and float((epi.reward or {}).get("total_score", 0.0)) >= 0.7:
        if skill_mining_mode == "llm" and llm_client is not None:
            mined = mine_skill_llm(experiences, llm_client)
        else:
            mined = mine_skill_rule(experiences)
        if mined is not None:
            skills.append(mined)

    return DistilledEpisode(
        episode_id=epi.episode_id,
        summary={
            "timestamp": datetime.utcnow().isoformat(),
            "experience_count": len(experiences),
            "skill_count": len(skills),
            "reward": epi.reward,
        },
        candidate_experience_items=[x.to_dict() for x in experiences],
        candidate_skill_items=[x.to_dict() for x in skills],
        revision_signals={"item_ids_to_revise": [], "item_ids_to_deprecate": []},
        source_field_refs=epi.source_field_refs,
    )
