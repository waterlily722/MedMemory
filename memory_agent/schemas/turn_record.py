from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import SerializableMixin


@dataclass
class TurnRecord(SerializableMixin):
    turn_id: int
    case_state_before: dict[str, Any] = field(default_factory=dict)
    memory_query: dict[str, Any] = field(default_factory=dict)
    retrieval_result: dict[str, Any] = field(default_factory=dict)
    applicability_result: dict[str, Any] = field(default_factory=dict)
    memory_guidance: dict[str, Any] = field(default_factory=dict)
    selected_action: dict[str, Any] = field(default_factory=dict)
    env_observation: dict[str, Any] = field(default_factory=dict)
    env_info: dict[str, Any] = field(default_factory=dict)
    case_state_after: dict[str, Any] = field(default_factory=dict)
    blocked_actions: list[str] = field(default_factory=list)
    selected_action_blocked: bool = False
    reward: float = 0.0
    done: bool = False
