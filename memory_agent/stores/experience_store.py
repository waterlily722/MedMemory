from __future__ import annotations

from pathlib import Path

from ..schemas import ExperienceItem
from .base_store import FlatMemoryStore


class ExperienceStore(FlatMemoryStore):
    def __init__(self, root_dir: str | Path):
        super().__init__(root_dir=root_dir, filename="experience_store.json", item_cls=ExperienceItem)
