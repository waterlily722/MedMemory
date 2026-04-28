from __future__ import annotations

import json
import re
from typing import Any


def _extract_json_fragment(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        return "{}"
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


def _ensure_type_defaults(data: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    for field in schema.get("required", []):
        data.setdefault(field, None)

    for field in schema.get("list_fields", []):
        if not isinstance(data.get(field), list):
            value = data.get(field)
            if value is None:
                data[field] = []
            elif isinstance(value, str):
                data[field] = [value] if value else []
            else:
                data[field] = list(value) if isinstance(value, (tuple, set)) else []

    for field in schema.get("dict_fields", []):
        if not isinstance(data.get(field), dict):
            data[field] = {}

    return data


def _validate(data: dict[str, Any], schema: dict[str, Any]) -> tuple[bool, str]:
    for field in schema.get("required", []):
        if field not in data:
            return False, f"missing required field: {field}"
    return True, "ok"


def parse_validate_repair(raw_text: str, schema: dict[str, Any], fallback: dict[str, Any]) -> tuple[dict[str, Any], bool, str]:
    try:
        payload = json.loads(_extract_json_fragment(raw_text))
        if not isinstance(payload, dict):
            payload = {}
    except Exception as exc:
        payload = {}
        reason = f"json_parse_error: {exc}"
    else:
        reason = "ok"

    payload = _ensure_type_defaults(payload, schema)
    valid, msg = _validate(payload, schema)
    if valid:
        return payload, False, "ok"

    repaired = _ensure_type_defaults(dict(payload), schema)
    valid, msg = _validate(repaired, schema)
    if valid:
        return repaired, True, f"repaired:{msg}"

    fb = _ensure_type_defaults(dict(fallback), schema)
    return fb, True, f"fallback:{reason};{msg}"
