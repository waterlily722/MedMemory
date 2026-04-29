from __future__ import annotations

from dataclasses import dataclass, field

from .common import SerializableMixin


@dataclass
class MemoryQuery(SerializableMixin):
    case_id: str
    turn_id: int = 0
    query_text: str = ""
    # 当前病例概况、local goal、uncertainty、positive / negative evidence
    # 关键 missing info、当前不应该 finalize 的原因、可能有价值的下一步 action