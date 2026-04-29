from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schemas import (
    ApplicabilityResult,
    CaseState,
    MemoryGuidance,
    MemoryQuery,
    MemoryRetrievalResult,
)


def _to_dict(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def build_trace_payload(
    case_state: CaseState,
    memory_query: MemoryQuery,
    retrieval_result: MemoryRetrievalResult,
    applicability_result: ApplicabilityResult,
    memory_guidance: MemoryGuidance,
    selected_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "case_id": case_state.case_id,
        "turn_id": case_state.turn_id,
        "case_state": case_state.to_dict(),
        "memory_query": memory_query.to_dict(),
        "retrieval_result": retrieval_result.to_dict(),
        "applicability_result": applicability_result.to_dict(),
        "memory_guidance": memory_guidance.to_dict(),
        "selected_action": selected_action or {},
        "blocked_actions": memory_guidance.blocked_actions,
    }


def append_memory_trace(
    trace_root: str | Path,
    payload: dict[str, Any],
) -> Path:
    root = Path(trace_root)
    root.mkdir(parents=True, exist_ok=True)

    case_id = str(payload.get("case_id") or "unknown_case")
    path = root / f"{case_id}.jsonl"

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_dict(payload), ensure_ascii=False) + "\n")

    return path