from __future__ import annotations

import math
import re
from collections import Counter

from .config import SCORING_CONFIG

TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def cosine_similarity(text_a: str, text_b: str) -> float:
    vec_a = Counter(tokenize(text_a))
    vec_b = Counter(tokenize(text_b))
    if not vec_a or not vec_b:
        return 0.0
    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[token] * vec_b[token] for token in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def overlap_score(a: list[str], b: list[str]) -> float:
    sa = {x.lower() for x in a if x}
    sb = {x.lower() for x in b if x}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def weighted_experience_score(
    semantic_similarity: float,
    hypothesis_overlap: float,
    local_goal_match: float,
    modality_match: float,
    risk_match: float,
    evidence_overlap: float,
) -> float:
    w = SCORING_CONFIG["experience"]
    return (
        w["semantic_weight"] * semantic_similarity
        + w["hypothesis_weight"] * hypothesis_overlap
        + w["local_goal_weight"] * local_goal_match
        + w["modality_weight"] * modality_match
        + w["risk_weight"] * risk_match
        + w["evidence_weight"] * evidence_overlap
    )


def weighted_skill_score(
    trigger_match: float,
    clinical_goal_match: float,
    precondition_match: float,
    modality_match: float,
    success_rate: float,
    boundary_consistency: float,
) -> float:
    w = SCORING_CONFIG["skill"]
    return (
        w["trigger_weight"] * trigger_match
        + w["clinical_goal_weight"] * clinical_goal_match
        + w["precondition_weight"] * precondition_match
        + w["modality_weight"] * modality_match
        + w["success_rate_weight"] * success_rate
        + w["boundary_weight"] * boundary_consistency
    )
