from __future__ import annotations

from pathlib import Path

from ..schemas import SkillItem
from .base_store import FlatMemoryStore


class SkillStore(FlatMemoryStore):
    def __init__(self, root_dir: str | Path):
        super().__init__(root_dir=root_dir, filename="skill_store.json", item_cls=SkillItem)
