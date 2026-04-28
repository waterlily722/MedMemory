from __future__ import annotations

from .online.retriever import DEFAULT_MEMORY_ROOT, retrieve_multi_memory
from .online.reranker import rerank_retrieval_result
from .schemas import MemoryQuery, MemoryRetrievalResult


def _to_memory_query(query_signature_or_query) -> MemoryQuery:
    if isinstance(query_signature_or_query, MemoryQuery):
        return query_signature_or_query
    if isinstance(query_signature_or_query, dict):
        if "structured" in query_signature_or_query:
            return MemoryQuery.from_dict(query_signature_or_query)
        query_text = " | ".join(f"{k}:{v}" for k, v in query_signature_or_query.items())
        return MemoryQuery(query_text=query_text)
    return MemoryQuery(query_text=str(query_signature_or_query))


def retrieve_experience(query_signature, top_k: int = 5, root_dir: str | None = None):
    query = _to_memory_query(query_signature)
    result = retrieve_multi_memory(query, turn_id=0, root_dir=root_dir)
    return result.experience_hits[:top_k]


def retrieve_skill(query_signature, top_k: int = 3, root_dir: str | None = None):
    query = _to_memory_query(query_signature)
    result = retrieve_multi_memory(query, turn_id=0, root_dir=root_dir)
    return result.skill_hits[:top_k]


def retrieve_guardrail(query_signature, top_k: int = 3, root_dir: str | None = None):
    # Guardrail memory is represented as unsafe experience/knowledge hints in the new design.
    query = _to_memory_query(query_signature)
    result = retrieve_multi_memory(query, turn_id=0, root_dir=root_dir)
    merged = [
        hit
        for hit in result.experience_hits
        if (hit.payload.get("content", {}) or {}).get("outcome_type") == "unsafe"
    ]
    if len(merged) < top_k:
        merged.extend(result.knowledge_hits[: top_k - len(merged)])
    return merged[:top_k]


def retrieve_all(
    query_signature,
    top_k_each: int = 5,
    root_dir: str | None = None,
    disable_experience_memory: bool = False,
    disable_skill_memory: bool = False,
    disable_knowledge_memory: bool = False,
) -> MemoryRetrievalResult:
    query = _to_memory_query(query_signature)
    result = retrieve_multi_memory(
        query,
        turn_id=0,
        root_dir=root_dir,
        disable_experience_memory=disable_experience_memory,
        disable_skill_memory=disable_skill_memory,
        disable_knowledge_memory=disable_knowledge_memory,
    )
    reranked = rerank_retrieval_result(result)
    reranked.experience_hits = reranked.experience_hits[:top_k_each]
    reranked.skill_hits = reranked.skill_hits[:top_k_each]
    reranked.knowledge_hits = reranked.knowledge_hits[:top_k_each]
    # Backward compatibility for older controller code paths expecting guardrail_hits.
    setattr(reranked, "guardrail_hits", retrieve_guardrail(query, top_k=top_k_each, root_dir=root_dir))
    return reranked
