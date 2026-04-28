from __future__ import annotations

from ..schemas import DistilledEpisode


def summarize_distilled_episode(distilled: DistilledEpisode) -> dict[str, float]:
    exp_count = len(distilled.candidate_experience_items)
    skill_count = len(distilled.candidate_skill_items)
    reward = float((distilled.summary or {}).get("reward", {}).get("total_score", 0.0))
    return {
        "experience_count": float(exp_count),
        "skill_count": float(skill_count),
        "reward_total_score": reward,
    }
