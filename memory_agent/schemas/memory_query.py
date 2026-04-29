from __future__ import annotations

from dataclasses import dataclass, field

from .common import SerializableMixin


@dataclass
class MemoryQuery(SerializableMixin):
    case_id: str
    turn_id: int = 0
    query_text: str = ""