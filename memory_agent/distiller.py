from __future__ import annotations

from .offline.episode_distiller import distill_from_trajectory
from .schemas import DistilledEpisode, EpisodeFeedback, TurnFeedback


def distill_episode(
    trajectory,
    turn_feedback_list,
    episode_feedback,
) -> DistilledEpisode:
    feedback = episode_feedback if isinstance(episode_feedback, EpisodeFeedback) else EpisodeFeedback.from_dict(episode_feedback)
    turns = [item if isinstance(item, TurnFeedback) else TurnFeedback.from_dict(item) for item in (turn_feedback_list or [])]
    return distill_from_trajectory(trajectory, turns, feedback)
