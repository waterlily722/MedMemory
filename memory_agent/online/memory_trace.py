from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schemas import ApplicabilityResult, CaseState, MemoryGuidance, MemoryQuery, MemoryRetrievalResult


def append_memory_trace(log_root: str | Path, case_id: str, payload: dict[str, Any]) -> None:
    root = Path(log_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{case_id}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_trace_payload(
    case_state: CaseState,
    memory_query: MemoryQuery,
    retrieval_result: MemoryRetrievalResult,
    applicability_result: ApplicabilityResult,
    memory_guidance: MemoryGuidance,
    selected_action: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case_id": case_state.case_id,
        "turn_id": case_state.turn_id,
        "case_state": case_state.to_dict(),
        "memory_query": memory_query.to_dict(),
        "retrieved_ids": {
            "positive_experience": [hit.memory_id for hit in retrieval_result.positive_experience_hits],
            "negative_experience": [hit.memory_id for hit in retrieval_result.negative_experience_hits],
            "skill": [hit.memory_id for hit in retrieval_result.skill_hits],
            "knowledge": [hit.memory_id for hit in retrieval_result.knowledge_hits],
        },
        "applicability_result": applicability_result.to_dict(),
        "memory_guidance": memory_guidance.to_dict(),
        "selected_action": selected_action,
        "blocked_actions": list(memory_guidance.blocked_actions),
    }
