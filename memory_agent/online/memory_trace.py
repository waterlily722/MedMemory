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
from ..utils.config import TRACE_CONFIG


def _to_dict(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _to_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_dict(item) for item in value]
    return value


def _clip(value: Any, max_chars: int | None = None) -> Any:
    max_chars = max_chars or int(TRACE_CONFIG.get("max_text_chars", 600))
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"


def _compact_case_state(case_state: dict[str, Any] | None) -> dict[str, Any]:
    case_state = case_state or {}
    return {
        "turn_id": case_state.get("turn_id"),
        "problem_summary": _clip(case_state.get("problem_summary")),
        "local_goal": case_state.get("local_goal"),
        "finalize_risk": case_state.get("finalize_risk"),
        "missing_info": case_state.get("missing_info") or [],
        "active_hypotheses": case_state.get("active_hypotheses") or [],
        "key_evidence_count": len(case_state.get("key_evidence") or []),
        "negative_evidence_count": len(case_state.get("negative_evidence") or []),
        "modality_flags": case_state.get("modality_flags") or [],
        "reviewed_modalities": case_state.get("reviewed_modalities") or [],
    }


def _hit_summary(hit: dict[str, Any]) -> dict[str, Any]:
    content = hit.get("content") or {}
    return {
        "memory_id": hit.get("memory_id") or content.get("memory_id"),
        "memory_type": hit.get("memory_type") or content.get("memory_type"),
        "score": hit.get("score"),
        "outcome_type": content.get("outcome_type"),
        "situation_text": _clip(content.get("situation_text")),
        "action_text": _clip(content.get("action_text")),
        "boundary_text": _clip(content.get("boundary_text")),
        "retrieval_tags": content.get("retrieval_tags") or [],
        "risk_tags": content.get("risk_tags") or [],
    }


def _compact_retrieval(retrieval: dict[str, Any] | None) -> dict[str, Any]:
    retrieval = retrieval or {}
    max_hits = int(TRACE_CONFIG.get("max_hits_per_type", 5))
    compact: dict[str, Any] = {}
    for key in [
        "positive_experience_hits",
        "negative_experience_hits",
        "skill_hits",
        "knowledge_hits",
    ]:
        hits = retrieval.get(key) or []
        compact[key] = [_hit_summary(hit) for hit in hits[:max_hits]]
        compact[f"{key}_count"] = len(hits)
    return compact


def _compact_applicability(applicability: dict[str, Any] | None) -> dict[str, Any]:
    applicability = applicability or {}
    return {
        "memory_assessments": applicability.get("memory_assessments") or [],
        "action_assessments": applicability.get("action_assessments") or [],
        "hard_blocked_actions": applicability.get("hard_blocked_actions") or [],
        "risk_warning": applicability.get("risk_warning") or "",
    }


def _compact_memory_debug(debug: dict[str, Any] | None) -> dict[str, Any]:
    debug = debug or {}
    include_llm_io = bool(TRACE_CONFIG.get("include_llm_io", False))
    include_prompt_payload = bool(TRACE_CONFIG.get("include_prompt_payload", False))
    compact: dict[str, Any] = {
        "memory_enabled": debug.get("memory_enabled"),
        "memory_llm": debug.get("memory_llm") or {},
        "candidate_actions": debug.get("candidate_actions") or [],
    }

    case_update = debug.get("case_state_update") or {}
    compact["case_state_update"] = {
        "mode": case_update.get("mode"),
        "llm_available": case_update.get("llm_available"),
        "used_fallback": case_update.get("used_fallback"),
        "fallback_reason": case_update.get("fallback_reason"),
        "validation_ok": case_update.get("validation_ok"),
        "validation_errors": case_update.get("validation_errors") or [],
        "rule_fallback_case_state": _compact_case_state(case_update.get("rule_fallback_case_state")),
        "final_case_state": _compact_case_state(case_update.get("final_case_state")),
    }

    query_builder = debug.get("query_builder") or {}
    compact["query_builder"] = {
        "mode": query_builder.get("mode"),
        "llm_available": query_builder.get("llm_available"),
        "used_fallback": query_builder.get("used_fallback"),
        "fallback_reason": query_builder.get("fallback_reason"),
        "rule_fallback_query": query_builder.get("rule_fallback_query"),
        "parsed_output": query_builder.get("parsed_output"),
        "final_query": query_builder.get("final_query"),
    }

    retrieval = debug.get("retrieval") or {}
    compact["retrieval"] = {
        "memory_root": retrieval.get("memory_root"),
        "embedding_available": retrieval.get("embedding_available"),
        "disable_experience_memory": retrieval.get("disable_experience_memory"),
        "disable_skill_memory": retrieval.get("disable_skill_memory"),
        "disable_knowledge_memory": retrieval.get("disable_knowledge_memory"),
        "result": _compact_retrieval(retrieval.get("result")),
    }

    applicability = debug.get("applicability") or {}
    compact["applicability"] = {
        "mode": applicability.get("mode"),
        "hard_blocked_actions": applicability.get("hard_blocked_actions") or [],
        "risk_warning": applicability.get("risk_warning") or "",
        "hit_assessments": [
            {
                "hit": _hit_summary(item.get("hit") or {}),
                "mode": item.get("mode"),
                "llm_available": item.get("llm_available"),
                "used_fallback": item.get("used_fallback"),
                "fallback_reason": item.get("fallback_reason"),
                "parsed_output": item.get("parsed_output"),
                "final_assessment": item.get("final_assessment"),
            }
            for item in (applicability.get("hit_assessments") or [])
        ],
        "final_applicability_result": _compact_applicability(
            applicability.get("final_applicability_result")
        ),
    }

    compact["guidance"] = debug.get("guidance") or {}

    if TRACE_CONFIG.get("include_observation_payload", False):
        compact["input_observation"] = debug.get("input_observation")
        compact["guidance_injection"] = debug.get("guidance_injection")

    if include_llm_io:
        for source, target in [
            (case_update, compact["case_state_update"]),
            (query_builder, compact["query_builder"]),
        ]:
            target["raw_output"] = source.get("raw_output")
            if include_prompt_payload:
                target["prompt"] = source.get("prompt")
                target["payload"] = source.get("payload")
        for source, target in zip(
            applicability.get("hit_assessments") or [],
            compact["applicability"]["hit_assessments"],
        ):
            target["raw_output"] = source.get("raw_output")
            if include_prompt_payload:
                target["prompt"] = source.get("prompt")
                target["payload"] = source.get("payload")

    return compact


def _compact_turn_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": record.get("episode_id"),
        "case_id": record.get("case_id"),
        "turn_id": record.get("turn_id"),
        "selected_action": record.get("selected_action") or {},
        "reward": record.get("reward"),
        "done": record.get("done"),
        "case_state": _compact_case_state(record.get("case_state")),
        "memory_query": record.get("memory_query") or {},
        "retrieval_result": _compact_retrieval(record.get("retrieval_result")),
        "applicability_result": _compact_applicability(record.get("applicability_result")),
        "memory_guidance": record.get("memory_guidance") or {},
        "memory_debug": _compact_memory_debug(record.get("memory_debug")),
    }


def build_trace_payload(
    case_state: CaseState,
    memory_query: MemoryQuery,
    retrieval_result: MemoryRetrievalResult,
    applicability_result: ApplicabilityResult,
    memory_guidance: MemoryGuidance,
    selected_action: dict[str, Any] | None = None,
    memory_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an online debug trace using current schema objects only."""
    return {
        "case_id": case_state.case_id,
        "turn_id": case_state.turn_id,
        "case_state": case_state.to_dict(),
        "memory_query": memory_query.to_dict(),
        "retrieval_result": retrieval_result.to_dict(),
        "applicability_result": applicability_result.to_dict(),
        "memory_guidance": memory_guidance.to_dict(),
        "selected_action": selected_action or {},
        "memory_debug": memory_debug or {},
        "blocked_actions": list(memory_guidance.blocked_actions),
    }


def append_memory_trace(
    trace_root: str | Path,
    payload: dict[str, Any],
) -> Path:
    root = Path(trace_root)
    root.mkdir(parents=True, exist_ok=True)
    case_id = str(payload.get("case_id") or "unknown_case")
    path = root / f"{case_id}.jsonl"
    if not TRACE_CONFIG.get("write_jsonl_snapshot", False):
        return path
    payload = _to_dict(payload)
    if not TRACE_CONFIG.get("include_full_turn", False):
        payload["case_state"] = _compact_case_state(payload.get("case_state"))
        payload["retrieval_result"] = _compact_retrieval(payload.get("retrieval_result"))
        payload["applicability_result"] = _compact_applicability(payload.get("applicability_result"))
        payload["memory_debug"] = _compact_memory_debug(payload.get("memory_debug"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def append_memory_turn_trace(
    trace_root: str | Path,
    turn_record: dict[str, Any] | Any,
) -> Path:
    """
    Maintain a human-readable JSON trace grouped by case/episode/turn.

    This complements the JSONL snapshot log above. Each turn contains the full
    TurnRecord payload: selected action, env observation/info, reward/done, and
    all memory pipeline artifacts used before that action.
    """
    root = Path(trace_root)
    root.mkdir(parents=True, exist_ok=True)

    raw_record = _to_dict(turn_record)
    record = (
        raw_record
        if TRACE_CONFIG.get("include_full_turn", False)
        else _compact_turn_record(raw_record)
    )
    case_id = str(record.get("case_id") or "unknown_case")
    episode_id = str(record.get("episode_id") or "")
    path = root / f"{case_id}.json"

    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            payload = {}
    else:
        payload = {}

    payload.setdefault("case_id", case_id)
    if episode_id:
        payload.setdefault("episode_id", episode_id)
    payload.setdefault("schema", "memory_turn_trace.v1")
    turns = payload.setdefault("turns", [])

    turn_key = (record.get("episode_id"), record.get("turn_id"))
    replaced = False
    for idx, existing in enumerate(turns):
        existing_key = (existing.get("episode_id"), existing.get("turn_id"))
        if existing_key == turn_key:
            turns[idx] = record
            replaced = True
            break
    if not replaced:
        turns.append(record)

    turns.sort(key=lambda item: (str(item.get("episode_id") or ""), int(item.get("turn_id") or 0)))

    tmp_path = path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    tmp_path.replace(path)
    return path
