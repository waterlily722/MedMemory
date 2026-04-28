from __future__ import annotations

from ..schemas import ApplicabilityResult, MemoryRetrievalResult


def build_memory_guidance(applicability: ApplicabilityResult, retrieval: MemoryRetrievalResult) -> dict:
    by_id = {a.action_id: a for a in applicability.action_assessments}
    recommended = [a.action_id for a in applicability.action_assessments if a.decision == "apply"]
    discouraged = [a.action_id for a in applicability.action_assessments if a.decision in {"escalate", "hint"}]
    blocked = [a.action_id for a in applicability.action_assessments if a.decision == "block"]

    used_memory_ids = []
    for assessment in applicability.action_assessments:
        if assessment.decision == "apply":
            used_memory_ids.extend(assessment.supporting_experience_ids)
            used_memory_ids.extend(assessment.supporting_skill_ids)
            used_memory_ids.extend(assessment.supporting_knowledge_ids)

    used_memory_ids = list(dict.fromkeys(used_memory_ids))
    rationale = "; ".join(by_id[x].rationale for x in recommended[:2]) if recommended else "No high-confidence memory guidance."
    risk_warning = "Finalize is blocked by controller." if any("finalize" in x.lower() for x in blocked) else ""

    return {
        "recommended_actions": recommended,
        "discouraged_actions": discouraged,
        "blocked_actions": blocked,
        "used_memory_ids": used_memory_ids,
        "memory_rationale": rationale,
        "risk_warning": risk_warning,
        "why_not_finalize": risk_warning,
        "retrieved_experience_ids": [h.item_id for h in retrieval.experience_hits],
        "retrieved_skill_ids": [h.item_id for h in retrieval.skill_hits],
        "retrieved_knowledge_ids": [h.item_id for h in retrieval.knowledge_hits],
    }
