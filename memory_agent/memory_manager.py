from __future__ import annotations

from pathlib import Path

from .memory_store import ExperienceMemoryStore, SkillMemoryStore
from .offline.experience_merger import can_merge, merge_experience
from .offline.skill_abstractor import promote_experiences_to_skill
from .schemas import DistilledEpisode, ExperienceCard, MemoryUpdateOperation, MemoryUpdatePlan, SkillCard
from .online.retriever import DEFAULT_MEMORY_ROOT


def _root(root_dir: str | None = None) -> Path:
    base = Path(root_dir) if root_dir else DEFAULT_MEMORY_ROOT
    base.mkdir(parents=True, exist_ok=True)
    return base


def _upsert_experience_with_merge(store: ExperienceMemoryStore, card: ExperienceCard, operations: list[MemoryUpdateOperation]) -> ExperienceCard:
    existing_items = store.list_items()
    for existing in existing_items:
        if can_merge(existing, card):
            merged = merge_experience(existing, card)
            store.upsert(merged)
            operations.append(
                MemoryUpdateOperation(
                    op_type="merge",
                    target_memory="experience",
                    target_item_id=merged.item_id,
                    source_item_ids=[existing.item_id, card.item_id],
                    reason="merged similar experience pattern",
                    source_field_refs=card.source_field_refs,
                )
            )
            return merged

    store.upsert(card)
    operations.append(
        MemoryUpdateOperation(
            op_type="write",
            target_memory="experience",
            target_item_id=card.item_id,
            source_item_ids=[],
            reason="new experience card",
            source_field_refs=card.source_field_refs,
        )
    )
    return card


def update_memory(distilled_episode: DistilledEpisode | dict, root_dir: str | None = None) -> MemoryUpdatePlan:
    distilled = distilled_episode if isinstance(distilled_episode, DistilledEpisode) else DistilledEpisode.from_dict(distilled_episode)
    root = _root(root_dir)
    exp_store = ExperienceMemoryStore(root)
    skill_store = SkillMemoryStore(root)

    operations: list[MemoryUpdateOperation] = []
    merged_batch: list[ExperienceCard] = []

    for raw in distilled.candidate_experience_items:
        card = raw if isinstance(raw, ExperienceCard) else ExperienceCard.from_dict(raw)
        merged = _upsert_experience_with_merge(exp_store, card, operations)
        merged_batch.append(merged)

    for raw in distilled.candidate_skill_items:
        card = raw if isinstance(raw, SkillCard) else SkillCard.from_dict(raw)
        skill_store.upsert(card)
        operations.append(
            MemoryUpdateOperation(
                op_type="write",
                target_memory="skill",
                target_item_id=card.skill_id,
                source_item_ids=card.source_experience_ids,
                reason="new distilled skill candidate",
                source_field_refs=card.source_field_refs,
            )
        )

    promoted = promote_experiences_to_skill(merged_batch)
    if promoted is not None:
        skill_store.upsert(promoted)
        operations.append(
            MemoryUpdateOperation(
                op_type="promote",
                target_memory="skill",
                target_item_id=promoted.skill_id,
                source_item_ids=promoted.source_experience_ids,
                reason="promoted repeated successful experiences",
                source_field_refs=promoted.source_field_refs,
            )
        )

    return MemoryUpdatePlan(
        episode_id=distilled.episode_id,
        operations=operations,
        backend_updates={
            "experience_items": len(exp_store.list_items()),
            "skill_items": len(skill_store.list_items()),
        },
        source_field_refs=distilled.source_field_refs,
    )
