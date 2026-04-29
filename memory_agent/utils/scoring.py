from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from .config import SCORING_CONFIG

TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def flatten_payload(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, dict):
        return " ".join(f"{key} {flatten_payload(value)}" for key, value in payload.items())
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


def overlap_score(left: list[str], right: list[str]) -> float:
    left_set = {item.lower() for item in left if item}
    right_set = {item.lower() for item in right if item}
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / float(len(left_set | right_set))


def weighted_experience_score(
    semantic_similarity: float,
    goal_match: float,
    hypothesis_overlap: float,
    evidence_overlap: float,
    boundary_match: float,
) -> float:
    weights = SCORING_CONFIG["experience"]
    return (
        weights["semantic_weight"] * semantic_similarity
        + weights["goal_weight"] * goal_match
        + weights["hypothesis_weight"] * hypothesis_overlap
        + weights["evidence_weight"] * evidence_overlap
        + weights["boundary_weight"] * boundary_match
    )


def weighted_skill_score(
    trigger_match: float,
    goal_match: float,
    precondition_match: float,
    modality_match: float,
    success_rate: float,
) -> float:
    weights = SCORING_CONFIG["skill"]
    return (
        weights["trigger_weight"] * trigger_match
        + weights["goal_weight"] * goal_match
        + weights["precondition_weight"] * precondition_match
        + weights["modality_weight"] * modality_match
        + weights["success_weight"] * success_rate
    )
