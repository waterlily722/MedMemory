from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import SerializableMixin


@dataclass
class RetrievalHit(SerializableMixin):
    memory_id: str
    memory_type: str
    content: dict[str, Any]
    score: float
 

@dataclass
class MemoryRetrievalResult(SerializableMixin):
    positive_experience_hits: list[RetrievalHit] = field(default_factory=list)
    negative_experience_hits: list[RetrievalHit] = field(default_factory=list)
    skill_hits: list[RetrievalHit] = field(default_factory=list)
    knowledge_hits: list[RetrievalHit] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        if data is None:
            return cls()
        payload = dict(data)
        payload["positive_experience_hits"] = [RetrievalHit.from_dict(item) for item in payload.get("positive_experience_hits", [])]
        payload["negative_experience_hits"] = [RetrievalHit.from_dict(item) for item in payload.get("negative_experience_hits", [])]
        payload["skill_hits"] = [RetrievalHit.from_dict(item) for item in payload.get("skill_hits", [])]
        payload["knowledge_hits"] = [RetrievalHit.from_dict(item) for item in payload.get("knowledge_hits", [])]
        return cls(**payload)
