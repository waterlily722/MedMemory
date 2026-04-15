from __future__ import annotations

from pathlib import Path

from ..schemas import GuardrailItem
from .base_store import FlatMemoryStore


class GuardrailStore(FlatMemoryStore):
    def __init__(self, root_dir: str | Path):
        super().__init__(root_dir=root_dir, filename="guardrail_store.json", item_cls=GuardrailItem)
