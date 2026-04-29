from __future__ import annotations

from dataclasses import dataclass, field

from .common import SerializableMixin


@dataclass
class MemoryGuidance(SerializableMixin):
    recommended_actions: list[str] = field(default_factory=list)
    discouraged_actions: list[str] = field(default_factory=list)
    blocked_actions: list[str] = field(default_factory=list)
    used_memory_ids: list[str] = field(default_factory=list)
    warning_memory_ids: list[str] = field(default_factory=list)
    rationale: str = ""
    risk_warning: str = ""
    why_not_finalize: str = ""
