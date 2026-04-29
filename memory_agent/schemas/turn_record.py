from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import SerializableMixin


@dataclass
class TurnRecord(SerializableMixin):
    episode_id: str = ""
    case_id: str = ""
    turn_id: int = 0

    case_state: dict = field(default_factory=dict)
    memory_query: dict = field(default_factory=dict)
    retrieval_result: dict = field(default_factory=dict)
    applicability_result: dict = field(default_factory=dict)
    memory_guidance: dict = field(default_factory=dict)

    selected_action: dict = field(default_factory=dict)
    env_observation: dict = field(default_factory=dict)
    env_info: dict = field(default_factory=dict)

    reward: float = 0.0
    done: bool = False