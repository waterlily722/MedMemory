from __future__ import annotations

from pathlib import Path
from typing import Any

from ..memory_store import ExperienceMemoryStore, KnowledgeMemoryStore, SkillMemoryStore
from ..schemas import MemoryQuery, MemoryRetrievalResult, RetrievalHit
from ..utils.config import MEMORY_ROOT_DIRNAME, RETRIEVAL_CONFIG
from ..utils.scoring import cosine_similarity, flatten_payload

DEFAULT_MEMORY_ROOT = Path(MEMORY_ROOT_DIRNAME)

DEFAULT_MIN_SCORES = {
    "positive_experience": 0.18,
    "negative_experience": 0.25,
    "skill": 0.20,
    "knowledge": 0.12,
}


def _config_value(name: str, default: Any) -> Any:
    return RETRIEVAL_CONFIG.get(name, default)


def _root(root_dir: str | None) -> Path:
    root = Path(root_dir) if root_dir else DEFAULT_MEMORY_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def _join(values: list[Any]) -> str:
    return ", ".join(str(value) for value in values or [] if value)


def experience_to_text(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"situation_text: {payload.get('situation_text', '')}",
            f"action_text: {payload.get('action_text', '')}",
            f"outcome_text: {payload.get('outcome_text', '')}",
            f"boundary_text: {payload.get('boundary_text', '')}",
            f"action_sequence: {flatten_payload(payload.get('action_sequence', []))}",
            f"outcome_type: {payload.get('outcome_type', '')}",
            f"failure_mode: {payload.get('failure_mode', '')}",
            f"retrieval_tags: {_join(payload.get('retrieval_tags', []))}",
            f"risk_tags: {_join(payload.get('risk_tags', []))}",
        ]
    )


def skill_to_text(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"skill_name: {payload.get('skill_name', '')}",
            f"situation_text: {payload.get('situation_text', '')}",
            f"goal_text: {payload.get('goal_text', '')}",
            f"procedure_text: {payload.get('procedure_text', '')}",
            f"boundary_text: {payload.get('boundary_text', '')}",
            f"procedure: {flatten_payload(payload.get('procedure', []))}",
            f"contraindications: {_join(payload.get('contraindications', []))}",
            f"evidence_count: {payload.get('evidence_count', '')}",
            f"unique_case_count: {payload.get('unique_case_count', '')}",
            f"success_rate: {payload.get('success_rate', '')}",
            f"unsafe_rate: {payload.get('unsafe_rate', '')}",
            f"confidence: {payload.get('confidence', '')}",
        ]
    )


def knowledge_to_text(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"content: {payload.get('content', '')}",
            f"tags: {_join(payload.get('tags', []))}",
            f"source: {payload.get('source', '')}",
            f"confidence: {payload.get('confidence', '')}",
        ]
    )


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


def _filter_sort_top_k(
    hits: list[RetrievalHit],
    *,
    top_k: int,
    min_score: float,
) -> list[RetrievalHit]:
    filtered = [hit for hit in hits if hit.score >= min_score]
    filtered.sort(key=lambda item: item.score, reverse=True)
    return filtered[:top_k]


def _experience_hits(
    query: MemoryQuery,
    root_dir: str | None,
    *,
    positive_top_k: int,
    negative_top_k: int,
    positive_min_score: float,
    negative_min_score: float,
) -> tuple[list[RetrievalHit], list[RetrievalHit]]:
    store = ExperienceMemoryStore(_root(root_dir))
    positive: list[RetrievalHit] = []
    negative: list[RetrievalHit] = []

    for card in store.list_all():
        payload = card.to_dict()
        score = _score_memory(query, "experience", payload)
        hit = _build_hit(card.memory_id, "experience", payload, score)
        if card.outcome_type in {"failure", "unsafe"}:
            negative.append(hit)
        else:
            positive.append(hit)

    return (
        _filter_sort_top_k(positive, top_k=positive_top_k, min_score=positive_min_score),
        _filter_sort_top_k(negative, top_k=negative_top_k, min_score=negative_min_score),
    )


def _skill_hits(
    query: MemoryQuery,
    root_dir: str | None,
    *,
    top_k: int,
    min_score: float,
) -> list[RetrievalHit]:
    store = SkillMemoryStore(_root(root_dir))
    hits: list[RetrievalHit] = []
    for card in store.list_all():
        payload = card.to_dict()
        score = _score_memory(query, "skill", payload)
        hits.append(_build_hit(card.memory_id, "skill", payload, score))
    return _filter_sort_top_k(hits, top_k=top_k, min_score=min_score)


def _knowledge_hits(
    query: MemoryQuery,
    root_dir: str | None,
    *,
    top_k: int,
    min_score: float,
) -> list[RetrievalHit]:
    store = KnowledgeMemoryStore(_root(root_dir))
    hits: list[RetrievalHit] = []
    for item in store.list_all():
        payload = item.to_dict()
        score = _score_memory(query, "knowledge", payload)
        hits.append(_build_hit(item.memory_id, "knowledge", payload, score))
    return _filter_sort_top_k(hits, top_k=top_k, min_score=min_score)


def retrieve_multi_memory(
    memory_query: MemoryQuery,
    root_dir: str | None = None,
    disable_experience_memory: bool = False,
    disable_skill_memory: bool = False,
    disable_knowledge_memory: bool = False,
    positive_experience_min_score: float | None = None,
    negative_experience_min_score: float | None = None,
    skill_min_score: float | None = None,
    knowledge_min_score: float | None = None,
) -> MemoryRetrievalResult:
    positive: list[RetrievalHit] = []
    negative: list[RetrievalHit] = []

    if not disable_experience_memory:
        positive, negative = _experience_hits(
            memory_query,
            root_dir,
            positive_top_k=int(_config_value("positive_experience_top_k", 5)),
            negative_top_k=int(_config_value("negative_experience_top_k", 3)),
            positive_min_score=(
                positive_experience_min_score
                if positive_experience_min_score is not None
                else float(_config_value("positive_experience_min_score", DEFAULT_MIN_SCORES["positive_experience"]))
            ),
            negative_min_score=(
                negative_experience_min_score
                if negative_experience_min_score is not None
                else float(_config_value("negative_experience_min_score", DEFAULT_MIN_SCORES["negative_experience"]))
            ),
        )

    skills = []
    if not disable_skill_memory:
        skills = _skill_hits(
            memory_query,
            root_dir,
            top_k=int(_config_value("skill_top_k", 3)),
            min_score=(
                skill_min_score
                if skill_min_score is not None
                else float(_config_value("skill_min_score", DEFAULT_MIN_SCORES["skill"]))
            ),
        )

    knowledge = []
    if not disable_knowledge_memory:
        knowledge = _knowledge_hits(
            memory_query,
            root_dir,
            top_k=int(_config_value("knowledge_top_k", 3)),
            min_score=(
                knowledge_min_score
                if knowledge_min_score is not None
                else float(_config_value("knowledge_min_score", DEFAULT_MIN_SCORES["knowledge"]))
            ),
        )

    return MemoryRetrievalResult(
        positive_experience_hits=positive,
        negative_experience_hits=negative,
        skill_hits=skills,
        knowledge_hits=knowledge,
    )
