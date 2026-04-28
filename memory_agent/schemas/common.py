from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import asdict, fields, is_dataclass
from typing import Any, Optional, Union, get_args, get_origin


def _serialize(value: Any) -> Any:
    if isinstance(value, SerializableMixin):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


def _unwrap_optional(tp: Any) -> Any:
    origin = get_origin(tp)
    if origin not in (Union, Optional):
        return tp
    args = [arg for arg in get_args(tp) if arg is not type(None)]
    return args[0] if len(args) == 1 else tp


def _deserialize(tp: Any, value: Any) -> Any:
    if value is None:
        return None

    tp = _unwrap_optional(tp)
    origin = get_origin(tp)

    if origin is list:
        item_type = get_args(tp)[0] if get_args(tp) else Any
        return [_deserialize(item_type, item) for item in value]

    if origin is dict:
        key_type, val_type = get_args(tp) if get_args(tp) else (Any, Any)
        return {
            _deserialize(key_type, key): _deserialize(val_type, item)
            for key, item in value.items()
        }

    if isinstance(tp, type) and issubclass(tp, SerializableMixin):
        if isinstance(value, tp):
            return value
        return tp.from_dict(value)

    return value


class SerializableMixin:
    source_field_refs: list[str]

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for f in fields(self):
            data[f.name] = _serialize(getattr(self, f.name))
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        data = data or {}
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            kwargs[f.name] = _deserialize(f.type, data.get(f.name))
        return cls(**kwargs)


@dataclass
class RankedIntent(SerializableMixin):
    intent_type: str
    score: float = 0.0
    rationale: str = ""
    source_field_refs: list[str] = field(default_factory=list)


@dataclass
class CanonicalEvidence(SerializableMixin):
    evidence_id: str
    turn_id: str
    source_type: str
    raw_field_refs: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    raw_structured: dict[str, Any] = field(default_factory=dict)
    raw_image_refs: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    negated_facts: list[str] = field(default_factory=list)
    uncertainty_patterns: list[str] = field(default_factory=list)
    symptom_patterns: list[str] = field(default_factory=list)
    test_patterns: list[str] = field(default_factory=list)
    route_flags: dict[str, bool] = field(default_factory=dict)
    source_field_refs: list[str] = field(default_factory=list)
