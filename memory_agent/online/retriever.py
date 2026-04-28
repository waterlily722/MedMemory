from __future__ import annotations

from pathlib import Path

from ..memory_store import ExperienceMemoryStore, KnowledgeMemoryStore, SkillMemoryStore
from ..memory_store.base_store import flatten_payload
from ..schemas import MemoryQuery, MemoryRetrievalResult, RetrievalHit
from ..utils.config import RETRIEVAL_CONFIG
from ..utils.scoring import cosine_similarity, overlap_score, weighted_experience_score, weighted_skill_score

DEFAULT_MEMORY_ROOT = Path(__file__).resolve().parent.parent / "store_data"


def _root(root: str | None) -> Path:
    base = Path(root) if root else DEFAULT_MEMORY_ROOT
    base.mkdir(parents=True, exist_ok=True)
    return base


def _experience_hits(query: MemoryQuery, root: str | None) -> tuple[list[RetrievalHit], list[RetrievalHit]]:
    store = ExperienceMemoryStore(_root(root))
    q = query.structured
    q_text = query.query_text
    positive_hits: list[RetrievalHit] = []
    negative_hits: list[RetrievalHit] = []
    for card in store.list_items():
        semantic = cosine_similarity(q_text, flatten_payload(card.to_dict()))
        hypothesis = overlap_score(q.active_hypotheses, [card.situation_anchor])
        local_goal = 1.0 if q.local_goal and q.local_goal in card.local_goal else 0.0
        modality = 1.0 if not q.modality_flags else overlap_score(q.modality_flags, list(card.visual_signature.values()))
        risk = 1.0 if q.finalize_risk != "high" or card.outcome_type != "unsafe" else 0.0
        evidence = overlap_score(q.key_positive_evidence, [card.outcome_shift, card.boundary])
        score = weighted_experience_score(semantic, hypothesis, local_goal, modality, risk, evidence)
        matched = ["semantic"]
        if hypothesis > 0:
            matched.append("hypothesis")
        if local_goal > 0:
            matched.append("local_goal")
        hit = RetrievalHit(
            item_id=card.item_id,
            retrieval_score=round(score, 4),
            matched_fields=matched,
            payload={
                "memory_id": card.item_id,
                "memory_type": "experience",
                "outcome_type": card.outcome_type,
                "content": card.to_dict(),
                "source": "experience_memory_store",
            },
            source_field_refs=card.source_field_refs,
        )
        if card.outcome_type in {"unsafe", "failure"} or card.error_tag:
            negative_hits.append(hit)
        else:
            positive_hits.append(hit)
    positive_hits.sort(key=lambda x: x.retrieval_score, reverse=True)
    negative_hits.sort(key=lambda x: x.retrieval_score, reverse=True)
    return positive_hits[: RETRIEVAL_CONFIG["experience_top_k"]], negative_hits[: RETRIEVAL_CONFIG["negative_experience_top_k"]]


def _skill_hits(query: MemoryQuery, root: str | None) -> list[RetrievalHit]:
    store = SkillMemoryStore(_root(root))
    q = query.structured
    q_text = query.query_text
    hits: list[RetrievalHit] = []
    for card in store.list_items():
        trigger = cosine_similarity(q_text, card.skill_trigger)
        goal = 1.0 if q.local_goal and q.local_goal in card.clinical_goal else 0.0
        precond = overlap_score(q.active_hypotheses, card.preconditions)
        modality = overlap_score(q.modality_flags, list(card.visual_trigger.values())) if q.modality_flags else 1.0
        score = weighted_skill_score(
            trigger_match=trigger,
            clinical_goal_match=goal,
            precondition_match=precond,
            modality_match=modality,
            success_rate=card.success_rate,
            boundary_consistency=max(0.0, 1.0 - card.unsafe_rate),
        )
        matched = ["trigger"]
        if precond > 0:
            matched.append("precondition")
        hits.append(
            RetrievalHit(
                item_id=card.skill_id,
                retrieval_score=round(score, 4),
                matched_fields=matched,
                payload={
                    "memory_id": card.skill_id,
                    "memory_type": "skill",
                    "content": card.to_dict(),
                    "source": "skill_memory_store",
                },
                source_field_refs=card.source_field_refs,
            )
        )
    hits.sort(key=lambda x: x.retrieval_score, reverse=True)
    return hits[: RETRIEVAL_CONFIG["skill_top_k"]]


def _knowledge_hits(query: MemoryQuery, root: str | None) -> list[RetrievalHit]:
    store = KnowledgeMemoryStore(_root(root))
    q = query.structured
    hits: list[RetrievalHit] = []
    for item in store.list_items():
        score = cosine_similarity(query.query_text, item.content)
        score += 0.1 * overlap_score(q.active_hypotheses, item.disease_tags)
        score += 0.05 * overlap_score(q.modality_flags, item.modality_tags)
        hits.append(
            RetrievalHit(
                item_id=item.item_id,
                retrieval_score=round(score, 4),
                matched_fields=["semantic"],
                payload={
                    "memory_id": item.item_id,
                    "memory_type": "knowledge",
                    "content": item.to_dict(),
                    "source": item.source,
                },
                source_field_refs=item.source_field_refs,
            )
        )
    hits.sort(key=lambda x: x.retrieval_score, reverse=True)
    return hits[: RETRIEVAL_CONFIG["knowledge_top_k"]]


def retrieve_multi_memory(
    memory_query: MemoryQuery,
    turn_id: int,
    root_dir: str | None = None,
    disable_experience_memory: bool = False,
    disable_skill_memory: bool = False,
    disable_knowledge_memory: bool = False,
) -> MemoryRetrievalResult:
    experience_hits, negative_experience_hits = ([], [])
    if not disable_experience_memory:
        experience_hits, negative_experience_hits = _experience_hits(memory_query, root_dir)
    return MemoryRetrievalResult(
        turn_id=turn_id,
        experience_hits=experience_hits,
        negative_experience_hits=negative_experience_hits,
        skill_hits=[] if disable_skill_memory else _skill_hits(memory_query, root_dir),
        knowledge_hits=[] if disable_knowledge_memory else _knowledge_hits(memory_query, root_dir),
        source_field_refs=memory_query.source_field_refs,
    )
