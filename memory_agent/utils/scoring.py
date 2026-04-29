from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def flatten_payload(payload: Any) -> str:
    if payload is None:
        return ""

    if isinstance(payload, dict):
        return " ".join(
            f"{key} {flatten_payload(value)}"
            for key, value in payload.items()
        )

    if isinstance(payload, list):
        return " ".join(flatten_payload(item) for item in payload)

    return str(payload)


def cosine_similarity(text_a: str, text_b: str) -> float:
    vec_a = Counter(tokenize(text_a))
    vec_b = Counter(tokenize(text_b))

    if not vec_a or not vec_b:
        return 0.0

    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[token] * vec_b[token] for token in common)

    norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vec_b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)