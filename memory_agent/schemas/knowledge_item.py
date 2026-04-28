from __future__ import annotations

from dataclasses import dataclass, field

from .common import SerializableMixin


@dataclass
class KnowledgeItem(SerializableMixin):
    item_id: str
    title: str
    content: str
    disease_tags: list[str] = field(default_factory=list)
    symptom_tags: list[str] = field(default_factory=list)
    modality_tags: list[str] = field(default_factory=list)
    risk_tags: list[str] = field(default_factory=list)
    source: str = "guideline"
    confidence: float = 0.8
    source_field_refs: list[str] = field(default_factory=list)
