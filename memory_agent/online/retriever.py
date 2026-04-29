from __future__ import annotations

from pathlib import Path
from typing import Any

from ..memory_store import ExperienceMemoryStore, KnowledgeMemoryStore, SkillMemoryStore
from ..schemas import MemoryQuery, MemoryRetrievalResult, RetrievalHit
from ..utils.config import MEMORY_ROOT_DIRNAME, RETRIEVAL_CONFIG
from ..utils.scoring import cosine_similarity, flatten_payload


DEFAULT_MEMORY_ROOT = Path(MEMORY_ROOT_DIRNAME)


def _root(root_dir: str | None) -> Path:
    root = Path(root_dir) if root_dir else DEFAULT_MEMORY_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def _join(values: list[Any]) -> str:
    return ", ".join(str(value) for value in values if value)


def experience_to_text(payload: dict[str, Any]) -> str:
    return "\n".join([
        f"Situation: {payload.get('situation_text', '')}",
        f"Action: {payload.get('action_text', '')}",
        f"Outcome: {payload.get('outcome_text', '')}",
        f"Boundary: {payload.get('boundary_text', '')}",
        f"Action sequence: {flatten_payload(payload.get('action_sequence', []))}",
        f"Retrieval tags: {_join(payload.get('retrieval_tags', []))}",
        f"Risk tags: {_join(payload.get('risk_tags', []))}",
        f"Failure mode: {payload.get('failure_mode', '')}",
    ])


def skill_to_text(payload: dict[str, Any]) -> str:
    return "\n".join([
        f"Skill: {payload.get('skill_name', '')}",
        f"Situation: {payload.get('situation_text', '')}",
        f"Goal: {payload.get('goal_text', '')}",
        f"Procedure: {payload.get('procedure_text', '')}",
        f"Boundary: {payload.get('boundary_text', '')}",
        f"Procedure steps: {flatten_payload(payload.get('procedure', []))}",
        f"Contraindications: {_join(payload.get('contraindications', []))}",
        f"Evidence count: {payload.get('evidence_count', '')}",
        f"Success rate: {payload.get('success_rate', '')}",
        f"Unsafe rate: {payload.get('unsafe_rate', '')}",
    ])


def knowledge_to_text(payload: dict[str, Any]) -> str:
    return "\n".join([
        str(payload.get("content", "")),
        f"Tags: {_join(payload.get('tags', []))}",
        f"Source: {payload.get('source', '')}",
    ])


def memory_to_text(memory_type: str, payload: dict[str, Any]) -> str:
    if memory_type == "experience":
        return experience_to_text(payload)
    if memory_type == "skill":
        return skill_to_text(payload)
    if memory_type == "knowledge":
        return knowledge_to_text(payload)
    return flatten_payload(payload)


def _score_memory(query: MemoryQuery, memory_type: str, payload: dict[str, Any]) -> float:
    text = memory_to_text(memory_type, payload)
    score = cosine_similarity(query.query_text, text)

    # Tiny tag bonus. Still text-based, no structured query dependency.
    tags = payload.get("retrieval_tags") or payload.get("tags") or []
    if isinstance(tags, list) and tags:
        score += 0.05 * cosine_similarity(query.query_text, " ".join(str(tag) for tag in tags))

    return max(0.0, min(1.0, score))


def _build_hit(
    memory_id: str,
    memory_type: str,
    content: dict[str, Any],
    score: float,
) -> RetrievalHit:
    return RetrievalHit(
        memory_id=memory_id,
        memory_type=memory_type,
        content=content,
        score=round(score, 4),
    )


def _experience_hits(
    query: MemoryQuery,
    root_dir: str | None,
) -> tuple[list[RetrievalHit], list[RetrievalHit]]:
    store = ExperienceMemoryStore(_root(root_dir))

    positive: list[RetrievalHit] = []
    negative: list[RetrievalHit] = []

    for card in store.list_all():
        payload = card.to_dict()
        score = _score_memory(query, "experience", payload)

        hit = _build_hit(
            memory_id=card.memory_id,
            memory_type="experience",
            content=payload,
            score=score,
        )

        if card.outcome_type in {"failure", "unsafe"}:
            negative.append(hit)
        else:
            positive.append(hit)

    positive.sort(key=lambda item: item.score, reverse=True)
    negative.sort(key=lambda item: item.score, reverse=True)

    return (
        positive[: RETRIEVAL_CONFIG["positive_experience_top_k"]],
        negative[: RETRIEVAL_CONFIG["negative_experience_top_k"]],
    )


def _skill_hits(query: MemoryQuery, root_dir: str | None) -> list[RetrievalHit]:
    store = SkillMemoryStore(_root(root_dir))

    hits: list[RetrievalHit] = []
    for card in store.list_all():
        payload = card.to_dict()
        score = _score_memory(query, "skill", payload)
        hits.append(_build_hit(card.memory_id, "skill", payload, score))

    hits.sort(key=lambda item: item.score, reverse=True)
    return hits[: RETRIEVAL_CONFIG["skill_top_k"]]


def _knowledge_hits(query: MemoryQuery, root_dir: str | None) -> list[RetrievalHit]:
    store = KnowledgeMemoryStore(_root(root_dir))

    hits: list[RetrievalHit] = []
    for item in store.list_all():
        payload = item.to_dict()
        score = _score_memory(query, "knowledge", payload)
        hits.append(_build_hit(item.memory_id, "knowledge", payload, score))

    hits.sort(key=lambda item: item.score, reverse=True)
    return hits[: RETRIEVAL_CONFIG["knowledge_top_k"]]


def retrieve_multi_memory(
    memory_query: MemoryQuery,
    root_dir: str | None = None,
    disable_experience_memory: bool = False,
    disable_skill_memory: bool = False,
    disable_knowledge_memory: bool = False,
) -> MemoryRetrievalResult:
    positive: list[RetrievalHit] = []
    negative: list[RetrievalHit] = []

    if not disable_experience_memory:
        positive, negative = _experience_hits(memory_query, root_dir)

    return MemoryRetrievalResult(
        positive_experience_hits=positive,
        negative_experience_hits=negative,
        skill_hits=[] if disable_skill_memory else _skill_hits(memory_query, root_dir),
        knowledge_hits=[] if disable_knowledge_memory else _knowledge_hits(memory_query, root_dir),
    )