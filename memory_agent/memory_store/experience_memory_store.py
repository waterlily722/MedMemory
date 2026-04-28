from __future__ import annotations

from pathlib import Path

from ..schemas import ExperienceCard
from .base_store import JsonMemoryStore


class ExperienceMemoryStore(JsonMemoryStore):
    def __init__(self, root_dir: str | Path):
        super().__init__(root_dir=root_dir, filename="experience_memory_store.json", item_cls=ExperienceCard)
