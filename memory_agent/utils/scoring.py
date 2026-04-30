from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

# English alphanumeric runs OR individual CJK characters (each char = 1 token)
TOKEN_RE = re.compile(
    r"[a-z0-9_]+|[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]"
)


def tokenize(text: str) -> list[str]:
    """Tokenize text into English word stems and individual CJK characters.

    English: lowercase alphanumeric runs (e.g. "chest" → "chest").
    CJK: each Chinese character becomes its own token (e.g. "胸痛" → ["胸", "痛"]).
    Punctuation and whitespace are discarded automatically.
    """
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