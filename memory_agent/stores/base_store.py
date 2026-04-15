from __future__ import annotations

import json
import math
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, List, Tuple, Type

from ..schemas import SerializableMixin


TOKEN_RE = re.compile(r"[a-z0-9_]+")


def flatten_signature(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, dict):
        return " ".join(f"{key} {flatten_signature(value)}" for key, value in payload.items())
    if isinstance(payload, list):
        return " ".join(flatten_signature(value) for value in payload)
    return str(payload)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def cosine_overlap(text_a: str, text_b: str) -> float:
    vec_a = Counter(tokenize(text_a))
    vec_b = Counter(tokenize(text_b))
    if not vec_a or not vec_b:
        return 0.0
    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[token] * vec_b[token] for token in common)
    norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class FlatMemoryStore:
    def __init__(self, root_dir: str | Path, filename: str, item_cls: Type[SerializableMixin]):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.root_dir / filename
        self.item_cls = item_cls
        self._lock = threading.Lock()
        if not self.path.exists():
            self._write_items([])

    def _read_items(self) -> list[SerializableMixin]:
        with self._lock:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        return [self.item_cls.from_dict(item) for item in raw]

    def _write_items(self, items: Iterable[SerializableMixin]) -> None:
        payload = [item.to_dict() if isinstance(item, SerializableMixin) else item for item in items]
        with self._lock:
            self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_items(self) -> list[SerializableMixin]:
        return self._read_items()

    def get_item(self, item_id: str) -> SerializableMixin | None:
        for item in self._read_items():
            if getattr(item, "item_id", "") == item_id:
                return item
        return None

    def upsert(self, item: SerializableMixin) -> str:
        items = self._read_items()
        replaced = False
        for idx, existing in enumerate(items):
            if getattr(existing, "item_id", "") == getattr(item, "item_id", ""):
                items[idx] = item
                replaced = True
                break
        if not replaced:
            items.append(item)
        self._write_items(items)
        return getattr(item, "item_id", "")

    def remove(self, item_id: str) -> None:
        items = [item for item in self._read_items() if getattr(item, "item_id", "") != item_id]
        self._write_items(items)

    def _metadata_filter(self, item: SerializableMixin, query_signature: dict[str, Any]) -> tuple[bool, list[str]]:
        state = getattr(item, "state_signature", None) or getattr(item, "trigger_signature", None) or {}
        matched_fields: list[str] = []
        for key in ("chief_complaint", "turn_stage"):
            query_value = str(query_signature.get(key, "") or "").lower().strip()
            state_value = str(state.get(key, "") or "").lower().strip()
            if query_value and state_value and query_value == state_value:
                matched_fields.append(key)

        for key in (
            "symptom_patterns",
            "uncertainty_patterns",
            "missing_slot_patterns",
            "hypothesis_patterns",
            "modality_patterns",
            "safety_patterns",
        ):
            query_vals = {str(v).lower() for v in query_signature.get(key, []) or []}
            state_vals = {str(v).lower() for v in state.get(key, []) or []}
            overlap = query_vals & state_vals
            if overlap:
                matched_fields.append(key)

        return (bool(matched_fields) or not any(query_signature.get(k) for k in query_signature), matched_fields)

    def search(self, query_signature: dict[str, Any], top_k: int = 10) -> list[tuple[SerializableMixin, float, list[str]]]:
        query_text = flatten_signature(query_signature)
        scored: list[tuple[SerializableMixin, float, list[str]]] = []
        for item in self._read_items():
            passes_filter, matched_fields = self._metadata_filter(item, query_signature)
            if not passes_filter:
                continue
            state = getattr(item, "state_signature", None) or getattr(item, "trigger_signature", None) or {}
            item_text = flatten_signature(state)
            score = cosine_overlap(query_text, item_text)
            if matched_fields:
                score += 0.05 * len(matched_fields)
            scored.append((item, score, matched_fields))

        scored.sort(key=lambda row: row[1], reverse=True)
        return scored[: max(1, int(top_k))]
