from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def _convert(value: Any) -> Any:
    if is_dataclass(value):
        return _convert(asdict(value))

    if isinstance(value, dict):
        return {str(key): _convert(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_convert(item) for item in value]

    if isinstance(value, tuple):
        return [_convert(item) for item in value]

    return value


class SerializableMixin:
    def to_dict(self) -> dict[str, Any]:
        if not is_dataclass(self):
            raise TypeError(f"{self.__class__.__name__} must be a dataclass")
        return _convert(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        if data is None:
            raise ValueError(f"{cls.__name__}.from_dict received None")
        if not isinstance(data, dict):
            raise TypeError(f"{cls.__name__}.from_dict expected dict, got {type(data)}")
        return cls(**dict(data))