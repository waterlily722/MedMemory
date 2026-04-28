from __future__ import annotations

import json
from typing import Any


def _json_dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


STRICT_JSON_RULES = (
    "Return one JSON object only. Do not use markdown fences, commentary, bullets, or free text. "
    "Follow the schema exactly. If a field is unsupported, keep the field and use an empty value."
)


def query_builder_prompt(payload: dict[str, Any]) -> str:
    return (
        "You are a medical memory query builder for schema-first retrieval.\n"
        f"{STRICT_JSON_RULES}\n"
        "Preserve situation_anchor, uncertainty_focus, retrieval_intent, finalize_risk_reason, and all evidence lists.\n"
        f"Input:\n{_json_dump(payload)}"
    )


def applicability_prompt(payload: dict[str, Any]) -> str:
    return (
        "You are an applicability judge for medical memory.\n"
        f"{STRICT_JSON_RULES}\n"
        "Judge one memory item at a time using the full memory_content. Do not invent missing facts.\n"
        f"Input:\n{_json_dump(payload)}"
    )


def experience_extraction_prompt(payload: dict[str, Any]) -> str:
    return (
        "You are a post-episode experience extractor for doctor memory.\n"
        f"{STRICT_JSON_RULES}\n"
        "Extract experience candidates from complete turn_records. Preserve key_evidence, missing_info, "
        "applicability_conditions, non_applicability_conditions, retrieval_tags, confidence, and outcome type.\n"
        f"Input:\n{_json_dump(payload)}"
    )


def experience_merge_prompt(payload: dict[str, Any]) -> str:
    return (
        "You are an experience dedup/merge controller.\n"
        f"{STRICT_JSON_RULES}\n"
        "Merge only when situation_anchor, local_goal, action_sequence, hypotheses, and boundary are compatible.\n"
        f"Input:\n{_json_dump(payload)}"
    )


def skill_mining_prompt(payload: dict[str, Any]) -> str:
    return (
        "You are a skill miner from repeated successful experiences.\n"
        f"{STRICT_JSON_RULES}\n"
        "Mine only from repeated cross-episode successful paths. Do not mine from a single episode batch.\n"
        f"Input:\n{_json_dump(payload)}"
    )
