from __future__ import annotations

from pathlib import Path
from typing import Any

from ..memory_store import ExperienceMemoryStore
from ..online.retriever import DEFAULT_MEMORY_ROOT
from ..schemas import DistilledEpisode, ExperienceCard
from .experience_extractor import extract_experiences
from .experience_merger import decide_merge_llm, decide_merge_rule, merge_experience


def _root(root_dir: str | None) -> Path:
    root = Path(root_dir) if root_dir else DEFAULT_MEMORY_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_memory_from_distilled_episode(
    distilled_episode: DistilledEpisode | dict[str, Any],
    root_dir: str | None = None,
    experience_extraction_mode: str = "rule",
    experience_merge_mode: str = "rule",
    llm_client=None,
) -> dict[str, Any]:
    distilled = distilled_episode if isinstance(distilled_episode, DistilledEpisode) else DistilledEpisode.from_dict(distilled_episode)
    root = _root(root_dir)
    store = ExperienceMemoryStore(root)
    existing = store.list_all()
    extracted = extract_experiences(distilled, mode=experience_extraction_mode, llm_client=llm_client)

    written_ids: list[str] = []
    merged_count = 0
    conflict_count = 0
    discarded_count = 0

    for experience in extracted:
        decision = decide_merge_llm(experience, existing, llm_client) if experience_merge_mode == "llm" and llm_client is not None else decide_merge_rule(experience, existing)
        merge_decision = str(decision.get("merge_decision", "insert_new"))
        if merge_decision == "discard":
            discarded_count += 1
            continue
        if merge_decision == "conflict":
            conflict_count += 1
            experience.conflict_group_id = str(decision.get("conflict_group_id") or experience.conflict_group_id or "") or None
            store.upsert(experience)
            written_ids.append(experience.memory_id)
            existing.append(experience)
            continue
        if merge_decision == "merge":
            merged_count += 1
            merged = ExperienceCard.from_dict(decision.get("merged_experience") or experience.to_dict())
            store.upsert(merged)
            written_ids.append(merged.memory_id)
            existing = [item for item in existing if item.memory_id != merged.memory_id]
            existing.append(merged)
            continue
        store.upsert(experience)
        written_ids.append(experience.memory_id)
        existing.append(experience)

    return {
        "episode_id": distilled.episode_id,
        "written_experience_ids": written_ids,
        "merged_count": merged_count,
        "conflict_count": conflict_count,
        "discarded_count": discarded_count,
        "experience_store_count": len(store.list_all()),
    }
