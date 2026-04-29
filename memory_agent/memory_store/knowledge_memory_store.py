from __future__ import annotations

from pathlib import Path

from ..schemas import KnowledgeItem
from .base_store import JsonMemoryStore


class KnowledgeMemoryStore(JsonMemoryStore):
    def __init__(self, root_dir: str | Path):
        super().__init__(root_dir=root_dir, filename="knowledge_memory.jsonl", item_cls=KnowledgeItem)
