from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..utils.scoring import cosine_similarity
from .base_store import flatten_payload


@dataclass
class VectorRow:
    item_id: str
    text: str
    payload: dict[str, Any]


class SimpleVectorStore:
    def __init__(self):
        self._rows: dict[str, VectorRow] = {}

    def upsert(self, item_id: str, payload: dict[str, Any]) -> None:
        self._rows[item_id] = VectorRow(item_id=item_id, text=flatten_payload(payload), payload=payload)

    def remove(self, item_id: str) -> None:
        self._rows.pop(item_id, None)

    def search(self, query_payload: dict[str, Any], top_k: int = 5) -> list[tuple[str, float, dict[str, Any]]]:
        query_text = flatten_payload(query_payload)
        scored: list[tuple[str, float, dict[str, Any]]] = []
        for item_id, row in self._rows.items():
            scored.append((item_id, cosine_similarity(query_text, row.text), row.payload))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(1, int(top_k))]
