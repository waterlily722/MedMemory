from __future__ import annotations

import json
import re
from typing import Any, Tuple


def _extract_json_fragment(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        return "{}"
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


def _ensure_defaults(data: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
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
    for field, allowed in schema.get("enum_fields", {}).items():
        if field in data and allowed and data.get(field) not in allowed:
            data[field] = allowed[0]
    for field, bounds in schema.get("range_fields", {}).items():
        if field not in data or data.get(field) is None:
            continue
        try:
            numeric = float(data.get(field))
        except Exception:
            continue
        lower = bounds.get("min")
        upper = bounds.get("max")
        if lower is not None:
            numeric = max(float(lower), numeric)
        if upper is not None:
            numeric = min(float(upper), numeric)
        data[field] = int(numeric) if float(numeric).is_integer() else round(numeric, 4)
    return data


def _validate(data: dict[str, Any], schema: dict[str, Any]) -> tuple[bool, str]:
    for field in schema.get("required", []):
        if field not in data:
            return False, f"missing required field: {field}"
    for field, allowed in schema.get("enum_fields", {}).items():
        if field in data and allowed and data.get(field) not in allowed:
            return False, f"invalid enum value for {field}"
    for field, bounds in schema.get("range_fields", {}).items():
        if field in data and data.get(field) is not None:
            try:
                numeric = float(data.get(field))
            except Exception:
                return False, f"invalid numeric value for {field}"
            if bounds.get("min") is not None and numeric < float(bounds["min"]):
                return False, f"below minimum for {field}"
            if bounds.get("max") is not None and numeric > float(bounds["max"]):
                return False, f"above maximum for {field}"
    return True, "ok"


def parse_validate_repair(raw_text: str, schema: dict[str, Any], fallback: dict[str, Any]) -> Tuple[dict[str, Any], bool, str]:
    reason = "ok"
    try:
        payload = json.loads(_extract_json_fragment(raw_text))
        if not isinstance(payload, dict):
            payload = {}
    except Exception as exc:
        payload = {}
        reason = f"json_parse_error: {exc}"
    payload = _ensure_defaults(payload, schema)
    valid, msg = _validate(payload, schema)
    if valid:
        return payload, False, "ok"
    repaired = _ensure_defaults(dict(payload), schema)
    valid, msg = _validate(repaired, schema)
    if valid:
        return repaired, True, f"repaired:{msg}"
    fb = _ensure_defaults(dict(fallback), schema)
    return fb, True, f"fallback:{reason};{msg}"
