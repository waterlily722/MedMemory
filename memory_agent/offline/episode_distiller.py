from __future__ import annotations

from datetime import datetime
from typing import Any

from ..schemas import DistilledEpisode, EpisodeFeedback


def _turn_records_from_trajectory(trajectory: Any) -> list[dict[str, Any]]:
    memory_info = ((getattr(trajectory, "info", {}) or {}).get("memory_agent", {}) or {})
    records = memory_info.get("turn_records", [])
    return [record for record in records if isinstance(record, dict)]


def distill_from_trajectory(trajectory: Any, episode_feedback: EpisodeFeedback | dict[str, Any]) -> DistilledEpisode:
    feedback = episode_feedback if isinstance(episode_feedback, EpisodeFeedback) else EpisodeFeedback.from_dict(episode_feedback)
    turn_records = _turn_records_from_trajectory(trajectory)
    return DistilledEpisode(
        episode_id=feedback.episode_id,
        case_id=feedback.case_id,
        turn_records=turn_records,
        episode_feedback=feedback.to_dict(),
        candidate_experience_items=[],
        summary={
            "timestamp": datetime.utcnow().isoformat(),
            "turn_count": len(turn_records),
            "total_reward": feedback.total_reward,
            "success": feedback.success,
        },
    )
