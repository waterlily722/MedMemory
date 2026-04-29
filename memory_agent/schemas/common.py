from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, TypeVar

T = TypeVar("T")


class SerializableMixin:
    def to_dict(self) -> dict[str, Any]:
        return _convert(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        if data is None:
            return cls()  # type: ignore[misc]
        return cls(**dict(data))  # type: ignore[misc]


def _convert(value: Any) -> Any:
    if is_dataclass(value):
        return _convert(asdict(value))
    if isinstance(value, dict):
        return {key: _convert(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_convert(item) for item in value]
    if isinstance(value, tuple):
        return [_convert(item) for item in value]
    return value
