from __future__ import annotations

import json
from typing import Any


def _json_dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def query_builder_prompt(payload: dict[str, Any]) -> str:
    return (
        "You are a medical memory query builder. Return only a JSON object.\n"
        "Build a high-quality retrieval query under uncertainty.\n"
        f"Input:\n{_json_dump(payload)}"
    )


def applicability_prompt(payload: dict[str, Any]) -> str:
    return (
        "You are an applicability judge for medical memory. Return only JSON.\n"
        "Judge each memory item against current case and candidate actions.\n"
        f"Input:\n{_json_dump(payload)}"
    )


def experience_extraction_prompt(payload: dict[str, Any]) -> str:
    return (
        "You are a post-episode experience extractor for doctor memory. Return only JSON.\n"
        "Extract local decision pattern and boundary conditions, not case summary.\n"
        f"Input:\n{_json_dump(payload)}"
    )


def experience_merge_prompt(payload: dict[str, Any]) -> str:
    return (
        "You are an experience dedup/merge controller. Return only JSON.\n"
        f"Input:\n{_json_dump(payload)}"
    )


def skill_mining_prompt(payload: dict[str, Any]) -> str:
    return (
        "You are a skill miner from repeated successful experiences. Return only JSON.\n"
        f"Input:\n{_json_dump(payload)}"
    )
