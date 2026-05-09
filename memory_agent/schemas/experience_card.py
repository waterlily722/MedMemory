from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Any

from .common import OutcomeType, SerializableMixin


@dataclass
class ExperienceCard(SerializableMixin):
    memory_id: str
    memory_type: str = "experience"

    # retrieval + applicability core
    situation_text: str = ""      # 什么时候适用：clinical trigger + uncertainty state
    action_text: str = ""         # 做了什么/应该避免什么
    outcome_text: str = ""        # 带来了什么结果/教训
    boundary_text: str = ""       # 什么时候不要用

    # action-level structure
    action_sequence: list[dict[str, str]] = field(default_factory=list)

    # safety / retrieval
    outcome_type: str = OutcomeType.PARTIAL_SUCCESS.value
    tags: list[str] = field(default_factory=list)

    # lightweight confidence / provenance
    confidence: float = 0.5
    support_count: int = 1
    source: dict[str, list[str]] = field(default_factory=dict)

    source_episode_ids: InitVar[list[str] | None] = None
    source_case_ids: InitVar[list[str] | None] = None
    source_turn_ids: InitVar[list[str] | None] = None

    def __post_init__(
        self,
        source_episode_ids: list[str] | None,
        source_case_ids: list[str] | None,
        source_turn_ids: list[str] | None,
    ) -> None:
        legacy_sources = {
            "episode_ids": source_episode_ids or [],
            "case_ids": source_case_ids or [],
            "turn_ids": source_turn_ids or [],
        }
        if not isinstance(self.source, dict):
            self.source = {}
        for key, values in legacy_sources.items():
            if self.source.get(key):
                continue
            normalized = [str(item) for item in values if str(item)]
            if normalized:
                self.source[key] = normalized

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        if data is None:
            raise ValueError(f"{cls.__name__}.from_dict received None")
        if not isinstance(data, dict):
            raise TypeError(f"{cls.__name__}.from_dict expected dict, got {type(data)}")

        values = dict(data)
        if not values.get("tags"):
            values["tags"] = list(values.get("retrieval_tags") or [])

        raw_source = values.get("source")
        source = dict(raw_source) if isinstance(raw_source, dict) else {}
        legacy_sources = {
            "episode_ids": values.get("source_episode_ids") or [],
            "case_ids": values.get("source_case_ids") or [],
            "turn_ids": values.get("source_turn_ids") or [],
        }
        for key, legacy_values in legacy_sources.items():
            if source.get(key):
                continue
            source[key] = [str(item) for item in legacy_values if str(item)]
        values["source"] = source

        return super().from_dict(values)
