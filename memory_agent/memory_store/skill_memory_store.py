from __future__ import annotations

from pathlib import Path

from ..schemas import SkillCard
from .base_store import JsonMemoryStore


class SkillMemoryStore(JsonMemoryStore):
    def __init__(self, root_dir: str | Path):
        super().__init__(root_dir=root_dir, filename="skill_memory_store.json", item_cls=SkillCard)
