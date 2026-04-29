from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import SerializableMixin


@dataclass
class TurnRecord(SerializableMixin):
    episode_id: str = ""
    case_id: str = ""
    turn_id: int = 0

    case_state: dict[str, Any] = field(default_factory=dict)
    memory_query: dict[str, Any] = field(default_factory=dict)
    retrieval_result: dict[str, Any] = field(default_factory=dict)
    applicability_result: dict[str, Any] = field(default_factory=dict)
    memory_guidance: dict[str, Any] = field(default_factory=dict)

    selected_action: dict[str, Any] = field(default_factory=dict)

    env_observation: dict[str, Any] | str = field(default_factory=dict)
    env_info: dict[str, Any] = field(default_factory=dict)

    reward: float = 0.0
    done: bool = False