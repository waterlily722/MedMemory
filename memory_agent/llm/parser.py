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

    for field, nested in schema.get("nested_fields", {}).items():
        if field not in data:
            continue
        value = data.get(field)
        if isinstance(value, dict) and nested:
            data[field] = _ensure_type_defaults(value, nested)
        elif isinstance(value, list) and nested.get("list_items"):
            repaired_items = []
            for item in value:
                repaired_items.append(_ensure_type_defaults(item if isinstance(item, dict) else {}, nested["list_items"]))
            data[field] = repaired_items

    return data


def _validate(data: dict[str, Any], schema: dict[str, Any]) -> tuple[bool, str]:
    for field in schema.get("required", []):
        if field not in data:
            return False, f"missing required field: {field}"

    for field, allowed in schema.get("enum_fields", {}).items():
        if field in data and allowed and data.get(field) not in allowed:
            return False, f"invalid enum value for {field}: {data.get(field)}"

    for field, bounds in schema.get("range_fields", {}).items():
        if field in data and data.get(field) is not None:
            try:
                numeric = float(data.get(field))
            except Exception:
                return False, f"invalid numeric value for {field}"
            lower = bounds.get("min")
            upper = bounds.get("max")
            if lower is not None and numeric < float(lower):
                return False, f"value below minimum for {field}"
            if upper is not None and numeric > float(upper):
                return False, f"value above maximum for {field}"

    for field, nested in schema.get("nested_fields", {}).items():
        if field not in data:
            continue
        value = data.get(field)
        if isinstance(value, dict) and nested:
            valid, msg = _validate(value, nested)
            if not valid:
                return False, f"{field}.{msg}"
        elif isinstance(value, list) and nested.get("list_items"):
            for idx, item in enumerate(value):
                if not isinstance(item, dict):
                    return False, f"{field}[{idx}] is not an object"
                valid, msg = _validate(item, nested["list_items"])
                if not valid:
                    return False, f"{field}[{idx}].{msg}"
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
