from __future__ import annotations

import uuid
from pathlib import Path

from .retriever import DEFAULT_MEMORY_ROOT
from .schemas import DistilledEpisode, ExperienceItem, MemoryUpdateOperation, MemoryUpdatePlan, SkillItem
from .stores.base_store import cosine_overlap, flatten_signature
from .stores.experience_store import ExperienceStore
from .stores.guardrail_store import GuardrailStore
from .stores.skill_store import SkillStore


def _root(root_dir: str | None = None) -> Path:
    base = Path(root_dir) if root_dir else DEFAULT_MEMORY_ROOT
    base.mkdir(parents=True, exist_ok=True)
    return base


def _merge_experience(existing: ExperienceItem, candidate: ExperienceItem) -> ExperienceItem:
    merged = ExperienceItem.from_dict(existing.to_dict())
    merged.source_episode_ids = list(dict.fromkeys(existing.source_episode_ids + candidate.source_episode_ids))
    merged.source_case_ids = list(dict.fromkeys(existing.source_case_ids + candidate.source_case_ids))
    merged.source_field_refs = list(dict.fromkeys(existing.source_field_refs + candidate.source_field_refs))
    merged.utility_stats["support_count"] = int(existing.utility_stats.get("support_count", 1)) + 1
    merged.utility_stats["avg_local_gain"] = round(
        (
            float(existing.utility_stats.get("avg_local_gain", 0.0))
            + float(candidate.utility_stats.get("avg_local_gain", 0.0))
        )
        / 2.0,
        4,
    )
    merged.applicability["useful_when"] = list(dict.fromkeys((existing.applicability.get("useful_when", []) + candidate.applicability.get("useful_when", []))))
    return merged


def _merge_skill(existing: SkillItem, candidate: SkillItem) -> SkillItem:
    merged = SkillItem.from_dict(existing.to_dict())
    merged.source_experience_ids = list(dict.fromkeys(existing.source_experience_ids + candidate.source_experience_ids))
    merged.source_case_ids = list(dict.fromkeys(existing.source_case_ids + candidate.source_case_ids))
    merged.source_field_refs = list(dict.fromkeys(existing.source_field_refs + candidate.source_field_refs))
    support_count = int(existing.reliability.get("support_count", 1)) + int(candidate.reliability.get("support_count", 1))
    fail_count = int(existing.reliability.get("fail_count", 0))
    merged.reliability.update(
        {
            "support_count": support_count,
            "fail_count": fail_count,
            "success_rate": round(support_count / max(1, support_count + fail_count), 4),
            "coverage": max(float(existing.reliability.get("coverage", 0.0)), float(candidate.reliability.get("coverage", 0.0))),
        }
    )
    return merged


def _best_match(store, signature: dict, threshold: float = 0.78):
    matches = store.search(signature, top_k=1)
    if not matches:
        return None, 0.0
    item, score, _ = matches[0]
    return (item, score) if score >= threshold else (None, score)


def update_memory(distilled_episode: DistilledEpisode | dict, root_dir: str | None = None) -> MemoryUpdatePlan:
    distilled = distilled_episode if isinstance(distilled_episode, DistilledEpisode) else DistilledEpisode.from_dict(distilled_episode)
    root = _root(root_dir)
    exp_store = ExperienceStore(root)
    skill_store = SkillStore(root)
    guardrail_store = GuardrailStore(root)

    operations: list[MemoryUpdateOperation] = []
    backend_updates = {"flat_store_updates": [], "embedding_index_updates": []}

    for item in distilled.candidate_experience_items:
        existing, score = _best_match(exp_store, item.state_signature)
        if existing:
            merged = _merge_experience(existing, item)
            exp_store.upsert(merged)
            operations.append(MemoryUpdateOperation("merge", "experience", merged.item_id, [existing.item_id, item.item_id], f"merged similar experience (score={score:.3f})", item.source_field_refs))
            backend_updates["flat_store_updates"].append({"store": "experience", "item_id": merged.item_id})
            if int(merged.utility_stats.get("support_count", 0)) >= 3 and float(merged.utility_stats.get("avg_local_gain", 0.0)) >= 0.2:
                promoted_skill = SkillItem(
                    item_id=f"skill_{uuid.uuid4().hex[:10]}",
                    source_experience_ids=[merged.item_id],
                    source_case_ids=merged.source_case_ids,
                    skill_meta={"skill_name": f"promoted_{merged.action_sequence[0]['action_type']}", "skill_family": merged.action_sequence[0]["action_type"]},
                    trigger_signature=merged.trigger_signature,
                    action_sequence=merged.action_sequence,
                    success_criteria=[{"criterion": "average local gain stays positive"}],
                    contraindications=merged.applicability.get("contraindications", []),
                    reliability={
                        "support_count": int(merged.utility_stats.get("support_count", 0)),
                        "fail_count": int(merged.utility_stats.get("fail_count", 0)),
                        "success_rate": 1.0,
                        "coverage": 0.6,
                    },
                    source_field_refs=merged.source_field_refs,
                )
                skill_store.upsert(promoted_skill)
                operations.append(MemoryUpdateOperation("promote", "skill", promoted_skill.item_id, [merged.item_id], "promoted stable experience to skill", promoted_skill.source_field_refs))
        else:
            exp_store.upsert(item)
            operations.append(MemoryUpdateOperation("write", "experience", item.item_id, [], "new experience item", item.source_field_refs))
            backend_updates["flat_store_updates"].append({"store": "experience", "item_id": item.item_id})

    for item in distilled.candidate_skill_items:
        existing, score = _best_match(skill_store, item.trigger_signature)
        if existing:
            merged = _merge_skill(existing, item)
            skill_store.upsert(merged)
            operations.append(MemoryUpdateOperation("revise", "skill", merged.item_id, [existing.item_id, item.item_id], f"revised similar skill (score={score:.3f})", item.source_field_refs))
        else:
            skill_store.upsert(item)
            operations.append(MemoryUpdateOperation("write", "skill", item.item_id, [], "new skill candidate", item.source_field_refs))

    for item in distilled.candidate_guardrail_items:
        existing, score = _best_match(guardrail_store, item.trigger_signature, threshold=0.7)
        if existing:
            existing_item = existing
            merged = existing_item.from_dict(existing_item.to_dict())
            merged.source_episode_ids = list(dict.fromkeys(existing_item.source_episode_ids + item.source_episode_ids))
            merged.source_case_ids = list(dict.fromkeys(existing_item.source_case_ids + item.source_case_ids))
            merged.source_field_refs = list(dict.fromkeys(existing_item.source_field_refs + item.source_field_refs))
            merged.risk_stats["occurrence_count"] = int(existing_item.risk_stats.get("occurrence_count", 1)) + 1
            merged.risk_stats["severity"] = "high" if item.risk_stats.get("severity") == "high" else existing_item.risk_stats.get("severity", "mid")
            guardrail_store.upsert(merged)
            operations.append(MemoryUpdateOperation("merge", "guardrail", merged.item_id, [existing_item.item_id, item.item_id], f"merged similar guardrail (score={score:.3f})", item.source_field_refs))
        else:
            guardrail_store.upsert(item)
            operations.append(MemoryUpdateOperation("write", "guardrail", item.item_id, [], "new guardrail item", item.source_field_refs))

    if len(exp_store.list_items()) > 500:
        for stale in exp_store.list_items()[:-500]:
            exp_store.remove(stale.item_id)
            operations.append(MemoryUpdateOperation("prune", "experience", stale.item_id, [], "pruned store to bounded size", stale.source_field_refs))

    return MemoryUpdatePlan(
        episode_id=distilled.episode_id,
        operations=operations,
        backend_updates=backend_updates,
        source_field_refs=distilled.source_field_refs,
    )
