from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import SerializableMixin


@dataclass
class SkillCard(SerializableMixin):
    memory_id: str
    memory_type: str = "skill"

    # skill identity
    skill_name: str = ""
    situation_text: str = ""      # trigger / when to use
    goal_text: str = ""           # what this skill tries to achieve

    # executable workflow
    procedure_text: str = ""
    procedure: list[dict[str, str]] = field(default_factory=list)

    # safety boundary
    boundary_text: str = ""
    tags: list[str] = field(default_factory=list)

    # lightweight evidence
    confidence: float = 0.5
    support_count: int = 1
    source: dict[str, list[str]] = field(default_factory=dict)

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
        if not source.get("experience_ids"):
            source["experience_ids"] = list(values.get("source_experience_ids") or [])
        values["source"] = source

        if not values.get("support_count"):
            values["support_count"] = int(values.get("evidence_count") or 1)

        return super().from_dict(values)
