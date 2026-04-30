"""Tests for the LLM output parser (parse_validate_repair)."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PACKAGE = _HERE.parent
if str(_PACKAGE) not in sys.path:
    sys.path.insert(0, str(_PACKAGE))

from memory_agent.llm.parser import parse_validate_repair, _extract_json_text, _coerce_list


SIMPLE_SCHEMA = {
    "required": ["name", "value"],
    "list_fields": ["tags"],
    "dict_fields": ["meta"],
    "enum_fields": {"status": ["ok", "error"]},
    "range_fields": {"score": {"min": 0.0, "max": 1.0}},
}


def test_parse_clean_json():
    result, ok, errors = parse_validate_repair(
        '{"name": "test", "value": 42, "tags": ["a", "b"], "status": "ok", "score": 0.8}',
        SIMPLE_SCHEMA,
    )
    assert ok
    assert result["name"] == "test"
    assert result["value"] == 42
    assert result["tags"] == ["a", "b"]


def test_parse_missing_fields_filled_from_fallback():
    result, ok, errors = parse_validate_repair(
        '{"name": "test"}',
        SIMPLE_SCHEMA,
        {"value": 0, "tags": [], "meta": {}, "status": "ok", "score": 0.5},
    )
    # Fallback fills missing required field "value", so ok=True
    assert ok, f"Expected ok=True, got errors={errors}"
    assert result["name"] == "test"
    assert result["value"] == 0
    assert result["tags"] == []
    assert result["meta"] == {}


def test_parse_enum_violation():
    result, ok, errors = parse_validate_repair(
        '{"name": "test", "value": 1, "status": "invalid_status", "score": 0.5}',
        SIMPLE_SCHEMA,
        {"status": "ok", "tags": [], "meta": {}, "score": 0.5},
    )
    assert not ok
    assert "enum" in errors[0]


def test_parse_range_clamping():
    result, ok, errors = parse_validate_repair(
        '{"name": "test", "value": 1, "score": 5.0}',
        SIMPLE_SCHEMA,
        {"tags": [], "meta": {}, "status": "ok", "score": 0.5},
    )
    # score should be clamped to 1.0
    assert result["score"] == 1.0


def test_parse_markdown_fence():
    result, ok, _ = parse_validate_repair(
        '```json\n{"name": "test", "value": 1, "score": 0.5}\n```',
        SIMPLE_SCHEMA,
        {"tags": [], "meta": {}, "status": "ok", "score": 0.5},
    )
    assert result["name"] == "test"


def test_parse_coerce_list_from_scalar():
    result, ok, _ = parse_validate_repair(
        '{"name": "test", "value": 1, "tags": "single_tag", "score": 0.5}',
        SIMPLE_SCHEMA,
        {"tags": [], "meta": {}, "status": "ok", "score": 0.5},
    )
    assert result["tags"] == ["single_tag"]


def test_parse_invalid_json_returns_fallback():
    result, ok, errors = parse_validate_repair(
        "not json at all",
        SIMPLE_SCHEMA,
        {"name": "fallback", "value": 0, "tags": [], "meta": {}, "status": "ok", "score": 0.5},
    )
    assert result["name"] == "fallback"


# ═══════════════════════════════════════════════════════════════════════
# _extract_json_text
# ═══════════════════════════════════════════════════════════════════════

def test_extract_json_text_plain():
    assert _extract_json_text('{"a": 1}') == '{"a": 1}'


def test_extract_json_text_with_markdown():
    assert _extract_json_text('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_extract_json_text_empty():
    assert _extract_json_text("") == "{}"
    assert _extract_json_text(None) == "{}"
