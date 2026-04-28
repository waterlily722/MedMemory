from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Iterable, Type

from ..schemas import SerializableMixin


class JsonMemoryStore:
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
            text = self.path.read_text(encoding="utf-8")
        raw = json.loads(text) if text.strip() else []
        return [self.item_cls.from_dict(item) for item in raw]

    def _write_items(self, items: Iterable[SerializableMixin]) -> None:
        payload = [item.to_dict() if isinstance(item, SerializableMixin) else item for item in items]
        with self._lock:
            self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_items(self) -> list[SerializableMixin]:
        return self._read_items()

    def upsert(self, item: SerializableMixin, id_field: str = "item_id") -> str:
        items = self._read_items()
        item_id = getattr(item, id_field)
        replaced = False
        for idx, existing in enumerate(items):
            if getattr(existing, id_field, None) == item_id:
                items[idx] = item
                replaced = True
                break
        if not replaced:
            items.append(item)
        self._write_items(items)
        return str(item_id)

    def remove(self, item_id: str, id_field: str = "item_id") -> None:
        items = [item for item in self._read_items() if getattr(item, id_field, None) != item_id]
        self._write_items(items)

    def get(self, item_id: str, id_field: str = "item_id") -> SerializableMixin | None:
        for item in self._read_items():
            if getattr(item, id_field, None) == item_id:
                return item
        return None


def flatten_payload(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, dict):
        return " ".join(f"{k} {flatten_payload(v)}" for k, v in payload.items())
    if isinstance(payload, list):
        return " ".join(flatten_payload(v) for v in payload)
    return str(payload)
