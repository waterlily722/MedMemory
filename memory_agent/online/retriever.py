from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..memory_store import ExperienceMemoryStore, KnowledgeMemoryStore, SkillMemoryStore, flatten_payload
from ..schemas import MemoryQuery, MemoryRetrievalResult, RetrievalHit
from ..utils.config import RETRIEVAL_LIMITS
from ..utils.scoring import cosine_similarity, overlap_score, weighted_experience_score, weighted_skill_score

DEFAULT_MEMORY_ROOT = Path(__file__).resolve().parent.parent / "memory_data"


def _root(root_dir: str | None) -> Path:
    root = Path(root_dir) if root_dir else DEFAULT_MEMORY_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def _query_text(query: MemoryQuery) -> str:
    return flatten_payload(query.to_dict())


def _score_fields(query: MemoryQuery, content: dict, memory_type: str) -> float:
    query_text = _query_text(query)
    content_text = flatten_payload(content)
    semantic = cosine_similarity(query_text, content_text)
    goal = 1.0 if query.local_goal and query.local_goal.lower() in content_text.lower() else 0.0
    hypotheses = overlap_score(query.active_hypotheses, content.get("active_hypotheses", []) if isinstance(content.get("active_hypotheses"), list) else [])
    evidence = overlap_score(query.positive_evidence, content.get("key_evidence", []) if isinstance(content.get("key_evidence"), list) else [])
    boundary = 1.0 if query.finalize_risk == "high" and content.get("outcome_type") in {"unsafe", "failure"} else 0.0
    if memory_type == "experience":
        return weighted_experience_score(semantic, goal, hypotheses, evidence, boundary)
    if memory_type == "skill":
        trigger = semantic
        precondition = overlap_score(query.active_hypotheses, content.get("trigger_conditions", []) if isinstance(content.get("trigger_conditions"), list) else [])
        modality = overlap_score(query.modality_need, content.get("required_modalities", []) if isinstance(content.get("required_modalities"), list) else [])
        success_rate = float(content.get("confidence", 0.5))
        return weighted_skill_score(trigger, goal, precondition, modality, success_rate)
    knowledge = semantic
    return knowledge


def _build_hit(memory_id: str, memory_type: str, content: dict, score: float, matched_fields: list[str]) -> RetrievalHit:
    return RetrievalHit(memory_id=memory_id, memory_type=memory_type, content=content, score=round(score, 4), matched_fields=matched_fields)


def _experience_hits(query: MemoryQuery, root_dir: str | None) -> tuple[list[RetrievalHit], list[RetrievalHit]]:
    store = ExperienceMemoryStore(_root(root_dir))
    positive: list[RetrievalHit] = []
    negative: list[RetrievalHit] = []
    for card in store.list_all():
        payload = card.to_dict()
        score = _score_fields(query, payload, "experience")
        matched = ["semantic"]
        if query.local_goal and query.local_goal.lower() in flatten_payload(payload).lower():
            matched.append("goal")
        hit = _build_hit(card.memory_id, "negative_experience" if card.outcome_type in {"failure", "unsafe"} else "experience", payload, score, matched)
        if card.outcome_type in {"failure", "unsafe"}:
            negative.append(hit)
        else:
            positive.append(hit)
    positive.sort(key=lambda item: item.score, reverse=True)
    negative.sort(key=lambda item: item.score, reverse=True)
    return positive[: RETRIEVAL_LIMITS["positive_experience_top_k"]], negative[: RETRIEVAL_LIMITS["negative_experience_top_k"]]


def _skill_hits(query: MemoryQuery, root_dir: str | None) -> list[RetrievalHit]:
    store = SkillMemoryStore(_root(root_dir))
    hits: list[RetrievalHit] = []
    for card in store.list_all():
        payload = card.to_dict()
        score = _score_fields(query, payload, "skill")
        hits.append(_build_hit(card.memory_id, "skill", payload, score, ["semantic"]))
    hits.sort(key=lambda item: item.score, reverse=True)
    return hits[: RETRIEVAL_LIMITS["skill_top_k"]]


def _knowledge_hits(query: MemoryQuery, root_dir: str | None) -> list[RetrievalHit]:
    store = KnowledgeMemoryStore(_root(root_dir))
    hits: list[RetrievalHit] = []
    for item in store.list_all():
        payload = item.to_dict()
        score = _score_fields(query, payload, "knowledge")
        hits.append(_build_hit(item.memory_id, "knowledge", payload, score, ["semantic"]))
    hits.sort(key=lambda item: item.score, reverse=True)
    return hits[: RETRIEVAL_LIMITS["knowledge_top_k"]]


def retrieve_multi_memory(
    memory_query: MemoryQuery,
    root_dir: str | None = None,
    disable_experience_memory: bool = False,
    disable_skill_memory: bool = False,
    disable_knowledge_memory: bool = False,
) -> MemoryRetrievalResult:
    positive, negative = ([], [])
    if not disable_experience_memory:
        positive, negative = _experience_hits(memory_query, root_dir)
    return MemoryRetrievalResult(
        positive_experience_hits=positive,
        negative_experience_hits=negative,
        skill_hits=[] if disable_skill_memory else _skill_hits(memory_query, root_dir),
        knowledge_hits=[] if disable_knowledge_memory else _knowledge_hits(memory_query, root_dir),
    )
