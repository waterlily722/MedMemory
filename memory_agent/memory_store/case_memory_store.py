from __future__ import annotations

from pathlib import Path

from ..schemas import CaseState
from .base_store import JsonMemoryStore


class CaseMemoryStore(JsonMemoryStore):
    def __init__(self, root_dir: str | Path):
        super().__init__(root_dir=root_dir, filename="case_memory_store.json", item_cls=CaseState)

    def upsert_case_state(self, state: CaseState) -> str:
        return self.upsert(state, id_field="case_id")

    def get_case_state(self, case_id: str) -> CaseState | None:
        item = self.get(case_id, id_field="case_id")
        return item if isinstance(item, CaseState) else None
