from __future__ import annotations

import json
from typing import Any

from .schemas import APPLICABILITY_SCHEMA, EXPERIENCE_EXTRACTION_SCHEMA, EXPERIENCE_MERGE_SCHEMA, QUERY_BUILDER_SCHEMA, SKILL_SCHEMA

STRICT_JSON_RULES = "Return one JSON object only. Do not use markdown fences, commentary, bullets, or free text. Follow the schema exactly."


def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def query_builder_prompt(payload: dict[str, Any]) -> str:
    return (
        "You are a medical memory query builder.\n"
        f"{STRICT_JSON_RULES}\n"
        f"Schema:\n{_dump(QUERY_BUILDER_SCHEMA)}\n"
        f"Input:\n{_dump(payload)}"
    )


def applicability_prompt(payload: dict[str, Any]) -> str:
    return (
        "You are a structured medical memory applicability judge.\n"
        f"{STRICT_JSON_RULES}\n"
        "Judge one memory item at a time. Hard safety rules must be preserved.\n"
        f"Schema:\n{_dump(APPLICABILITY_SCHEMA)}\n"
        f"Input:\n{_dump(payload)}"
    )


def experience_extraction_prompt(payload: dict[str, Any]) -> str:
    return (
        "You are an experience extractor for medical memory.\n"
        f"{STRICT_JSON_RULES}\n"
        "Extract reusable decision fragments, not case summaries.\n"
        f"Schema:\n{_dump(EXPERIENCE_EXTRACTION_SCHEMA)}\n"
        f"Input:\n{_dump(payload)}"
    )


def experience_merge_prompt(payload: dict[str, Any]) -> str:
    return (
        "You are an experience merge controller.\n"
        f"{STRICT_JSON_RULES}\n"
        f"Schema:\n{_dump(EXPERIENCE_MERGE_SCHEMA)}\n"
        f"Input:\n{_dump(payload)}"
    )


def skill_consolidation_prompt(payload: dict[str, Any]) -> str:
    return (
        "You are a skill consolidator from repeated successful experiences.\n"
        f"{STRICT_JSON_RULES}\n"
        "Only aggregate across episodes; never create a skill from one episode.\n"
        f"Schema:\n{_dump(SKILL_SCHEMA)}\n"
        f"Input:\n{_dump(payload)}"
    )
