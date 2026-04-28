from __future__ import annotations

from ..schemas import MemoryRetrievalResult


def rerank_retrieval_result(result: MemoryRetrievalResult) -> MemoryRetrievalResult:
    reranked = MemoryRetrievalResult.from_dict(result.to_dict())
    reranked.experience_hits.sort(key=lambda x: (x.retrieval_score, len(x.matched_fields)), reverse=True)
    reranked.skill_hits.sort(key=lambda x: (x.retrieval_score, len(x.matched_fields)), reverse=True)
    reranked.knowledge_hits.sort(key=lambda x: (x.retrieval_score, len(x.matched_fields)), reverse=True)
    return reranked
