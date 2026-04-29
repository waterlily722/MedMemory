from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Type

from ..schemas import SerializableMixin


class JsonMemoryStore:
    def __init__(self, root_dir: str | Path, filename: str, item_cls: Type[SerializableMixin]):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.root_dir / filename
        self.item_cls = item_cls
        self._lock = threading.Lock()
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def _read_raw(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self._lock:
            text = self.path.read_text(encoding="utf-8")
        rows: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows

    def _write_raw(self, rows: list[dict[str, Any]]) -> None:
        with self._lock:
            if rows:
                content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
            else:
                content = ""
            self.path.write_text(content, encoding="utf-8")

    def list_all(self):
        return [self.item_cls.from_dict(row) for row in self._read_raw()]

    def append(self, item: SerializableMixin) -> str:
        rows = self._read_raw()
        row = item.to_dict() if isinstance(item, SerializableMixin) else dict(item)
        rows.append(row)
        self._write_raw(rows)
        return str(row.get("memory_id") or row.get("item_id") or "")

    def upsert(self, item: SerializableMixin, id_field: str = "memory_id") -> str:
        rows = self._read_raw()
        row = item.to_dict() if isinstance(item, SerializableMixin) else dict(item)
        item_id = str(row.get(id_field) or row.get("memory_id") or row.get("item_id") or "")
        replaced = False
        for index, existing in enumerate(rows):
            if str(existing.get(id_field) or existing.get("memory_id") or existing.get("item_id") or "") == item_id:
                rows[index] = row
                replaced = True
                break
        if not replaced:
            rows.append(row)
        self._write_raw(rows)
        return item_id

    def find_by_id(self, memory_id: str, id_field: str = "memory_id"):
        for row in self._read_raw():
            if str(row.get(id_field) or row.get("memory_id") or row.get("item_id") or "") == str(memory_id):
                return self.item_cls.from_dict(row)
        return None


def flatten_payload(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, dict):
        return " ".join(f"{key} {flatten_payload(value)}" for key, value in payload.items())
    if isinstance(payload, list):
        return " ".join(flatten_payload(value) for value in payload)
    return str(payload)
