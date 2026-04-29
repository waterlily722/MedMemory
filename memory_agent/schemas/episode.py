from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import SerializableMixin


@dataclass
class EpisodeFeedback(SerializableMixin):
    episode_id: str
    case_id: str = ""
    total_reward: float = 0.0
    success: bool = False
    final_diagnosis: str = ""
    gold_diagnosis: str = ""
    trajectory_metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class DistilledEpisode(SerializableMixin):
    episode_id: str
    case_id: str = ""
    turn_records: list[dict[str, Any]] = field(default_factory=list)
    episode_feedback: dict[str, Any] = field(default_factory=dict)
    candidate_experience_items: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
