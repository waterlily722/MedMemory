from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_memory_trace(log_root: str | Path, case_id: str, payload: dict[str, Any]) -> None:
    root = Path(log_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{case_id}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
