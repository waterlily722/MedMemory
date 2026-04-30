from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from ..llm import EmbeddingClient
from ..memory_store import ExperienceMemoryStore, KnowledgeMemoryStore, SkillMemoryStore
from ..schemas import MemoryQuery, MemoryRetrievalResult, OutcomeType, RetrievalHit
from ..utils.config import MEMORY_ROOT_DIRNAME, RETRIEVAL_CONFIG
from ..utils.scoring import cosine_similarity as token_cosine, flatten_payload

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_ROOT = Path(MEMORY_ROOT_DIRNAME)


def _root(root_dir: str | None) -> Path:
    root = Path(root_dir) if root_dir else DEFAULT_MEMORY_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def _join(values: list[Any]) -> str:
    return ", ".join(str(value) for value in values if value)


def experience_to_text(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Situation: {payload.get('situation_text', '')}",
            f"Action: {payload.get('action_text', '')}",
            f"Outcome: {payload.get('outcome_text', '')}",
            f"Boundary: {payload.get('boundary_text', '')}",
            f"Action sequence: {flatten_payload(payload.get('action_sequence', []))}",
            f"Retrieval tags: {_join(payload.get('retrieval_tags', []))}",
            f"Risk tags: {_join(payload.get('risk_tags', []))}",
            f"Failure mode: {payload.get('failure_mode', '')}",
        ]
    )


def skill_to_text(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
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
        ]
    )


def knowledge_to_text(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(payload.get("content", "")),
            f"Tags: {_join(payload.get('tags', []))}",
            f"Source: {payload.get('source', '')}",
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


def _embedding_cosine(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity between two dense embedding vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(v * v for v in vec_a))
    norm_b = math.sqrt(sum(v * v for v in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _precompute_embeddings(
    items: list[Any],
    memory_type: str,
    embedding_client: EmbeddingClient | None,
) -> dict[str, list[float]]:
    """Batch-embed all items. Returns {memory_id: vector}."""
    if embedding_client is None or not embedding_client.available():
        return {}

    texts: list[str] = []
    id_order: list[str] = []
    for item in items:
        payload = item.to_dict() if hasattr(item, "to_dict") else item
        mid = str(payload.get("memory_id") or "")
        text = memory_to_text(memory_type, payload)
        texts.append(text)
        id_order.append(mid)

    vectors = embedding_client.embed(texts)
    if vectors is None or len(vectors) != len(id_order):
        logger.warning(
            "Embedding batch returned %d vectors for %d items; falling back",
            len(vectors) if vectors else 0, len(id_order),
        )
        return {}

    return {mid: vec for mid, vec in zip(id_order, vectors)}


def _score_memory(
    query: MemoryQuery,
    memory_type: str,
    payload: dict[str, Any],
    query_embedding: list[float] | None = None,
    memory_embedding: list[float] | None = None,
) -> float:
    """
    Score a single memory against the query.

    Priority:
    1. If both embeddings available → embedding cosine similarity.
    2. Else → token-based cosine similarity on the flattened text.
    3. Tag bonus (+0.05) is applied on top of the text score.
    """
    if query_embedding is not None and memory_embedding is not None:
        score = _embedding_cosine(query_embedding, memory_embedding)
    else:
        text = memory_to_text(memory_type, payload)
        score = token_cosine(query.query_text, text)
        # Tag bonus (only for token-based, since tags are already in the embedding text)
        tags = payload.get("retrieval_tags") or payload.get("tags") or []
        if isinstance(tags, list) and tags:
            score += 0.05 * token_cosine(
                query.query_text, " ".join(str(tag) for tag in tags)
            )
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


def _threshold(name: str, override: float | None) -> float:
    if override is not None:
        return float(override)
    return float(RETRIEVAL_CONFIG.get(name, 0.0) or 0.0)


def _experience_hits(
    query: MemoryQuery,
    root_dir: str | None,
    positive_min_score: float,
    negative_min_score: float,
    embedding_vectors: dict[str, list[float]] | None = None,
    query_embedding: list[float] | None = None,
) -> tuple[list[RetrievalHit], list[RetrievalHit]]:
    store = ExperienceMemoryStore(_root(root_dir))
    all_cards = store.list_all()

    vectors = embedding_vectors or {}
    pos_top_k = int(RETRIEVAL_CONFIG.get("positive_experience_top_k", 5))
    neg_top_k = int(RETRIEVAL_CONFIG.get("negative_experience_top_k", 3))

    positive: list[RetrievalHit] = []
    negative: list[RetrievalHit] = []

    for card in all_cards:
        payload = card.to_dict()
        mid = card.memory_id
        mem_vec = vectors.get(mid)

        score = _score_memory(
            query, "experience", payload,
            query_embedding=query_embedding,
            memory_embedding=mem_vec,
        )
        is_negative = card.outcome_type in {
            OutcomeType.FAILURE.value, OutcomeType.UNSAFE.value
        }

        if is_negative and score < negative_min_score:
            continue
        if not is_negative and score < positive_min_score:
            continue

        hit = _build_hit(mid, "experience", payload, score)
        if is_negative:
            negative.append(hit)
        else:
            positive.append(hit)

    positive.sort(key=lambda h: h.score, reverse=True)
    negative.sort(key=lambda h: h.score, reverse=True)
    return positive[:pos_top_k], negative[:neg_top_k]


def _skill_hits(
    query: MemoryQuery,
    root_dir: str | None,
    min_score: float,
    embedding_vectors: dict[str, list[float]] | None = None,
    query_embedding: list[float] | None = None,
) -> list[RetrievalHit]:
    store = SkillMemoryStore(_root(root_dir))
    all_cards = store.list_all()
    vectors = embedding_vectors or {}
    top_k = int(RETRIEVAL_CONFIG.get("skill_top_k", 3))
    hits: list[RetrievalHit] = []

    for card in all_cards:
        payload = card.to_dict()
        mid = card.memory_id
        mem_vec = vectors.get(mid)
        score = _score_memory(
            query, "skill", payload,
            query_embedding=query_embedding,
            memory_embedding=mem_vec,
        )
        if score >= min_score:
            hits.append(_build_hit(mid, "skill", payload, score))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]


def _knowledge_hits(
    query: MemoryQuery,
    root_dir: str | None,
    min_score: float,
    embedding_vectors: dict[str, list[float]] | None = None,
    query_embedding: list[float] | None = None,
) -> list[RetrievalHit]:
    store = KnowledgeMemoryStore(_root(root_dir))
    all_items = store.list_all()
    vectors = embedding_vectors or {}
    top_k = int(RETRIEVAL_CONFIG.get("knowledge_top_k", 3))
    hits: list[RetrievalHit] = []

    for item in all_items:
        payload = item.to_dict()
        mid = item.memory_id
        mem_vec = vectors.get(mid)
        score = _score_memory(
            query, "knowledge", payload,
            query_embedding=query_embedding,
            memory_embedding=mem_vec,
        )
        if score >= min_score:
            hits.append(_build_hit(mid, "knowledge", payload, score))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]


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
    embedding_client: EmbeddingClient | None = None,
) -> MemoryRetrievalResult:
    """
    Retrieve relevant memories for a query.

    When ``embedding_client`` is provided and available, all stored items are
    batch-embedded once and scored via embedding cosine similarity. Otherwise
    falls back to token-based Bag-of-Words cosine similarity.
    """
    # ── Pre-compute embeddings for all stores ──────────────────────────
    query_embedding: list[float] | None = None
    exp_vectors: dict[str, list[float]] = {}
    skill_vectors: dict[str, list[float]] = {}
    kn_vectors: dict[str, list[float]] = {}

    if embedding_client is not None and embedding_client.available():
        # Embed query
        query_embedding = embedding_client.embed_one(memory_query.query_text)

        if query_embedding and not disable_experience_memory:
            store = ExperienceMemoryStore(_root(root_dir))
            exp_vectors = _precompute_embeddings(
                store.list_all(), "experience", embedding_client
            ) or {}

        if query_embedding and not disable_skill_memory:
            store = SkillMemoryStore(_root(root_dir))
            skill_vectors = _precompute_embeddings(
                store.list_all(), "skill", embedding_client
            ) or {}

        if query_embedding and not disable_knowledge_memory:
            store = KnowledgeMemoryStore(_root(root_dir))
            kn_vectors = _precompute_embeddings(
                store.list_all(), "knowledge", embedding_client
            ) or {}

        if not query_embedding:
            logger.warning("Query embedding failed; falling back to token scoring")
            query_embedding = None

    # ── Retrieve ───────────────────────────────────────────────────────
    positive: list[RetrievalHit] = []
    negative: list[RetrievalHit] = []

    if not disable_experience_memory:
        positive, negative = _experience_hits(
            memory_query, root_dir,
            _threshold("positive_experience_min_score", positive_experience_min_score),
            _threshold("negative_experience_min_score", negative_experience_min_score),
            embedding_vectors=exp_vectors or None,
            query_embedding=query_embedding,
        )

    return MemoryRetrievalResult(
        positive_experience_hits=positive,
        negative_experience_hits=negative,
        skill_hits=[]
        if disable_skill_memory
        else _skill_hits(
            memory_query, root_dir,
            _threshold("skill_min_score", skill_min_score),
            embedding_vectors=skill_vectors or None,
            query_embedding=query_embedding,
        ),
        knowledge_hits=[]
        if disable_knowledge_memory
        else _knowledge_hits(
            memory_query, root_dir,
            _threshold("knowledge_min_score", knowledge_min_score),
            embedding_vectors=kn_vectors or None,
            query_embedding=query_embedding,
        ),
    )
