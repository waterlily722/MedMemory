from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..memory_store import ExperienceMemoryStore

logger = logging.getLogger(__name__)
from ..schemas import DistilledEpisode, ExperienceCard
from ..utils.config import MEMORY_ROOT_DIRNAME
from ..utils.scoring import cosine_similarity
from .experience_extractor import extract_experiences
from .experience_merger import decide_merge_llm, decide_merge_rule


def _root(root_dir: str | None) -> Path:
    root = Path(root_dir) if root_dir else Path(MEMORY_ROOT_DIRNAME)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _similar_existing(
    experience: ExperienceCard,
    existing: list[ExperienceCard],
    limit: int = 20,
) -> list[ExperienceCard]:
    scored: list[tuple[float, ExperienceCard]] = []

    for item in existing:
        score = (
            0.7 * cosine_similarity(experience.situation_text, item.situation_text)
            + 0.3 * cosine_similarity(experience.action_text, item.action_text)
        )
        scored.append((score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for score, item in scored[:limit] if score > 0.0]


def _find_by_id(items: list[ExperienceCard], memory_id: str) -> ExperienceCard | None:
    for item in items:
        if item.memory_id == memory_id:
            return item
    return None


def _apply_conflict_group(
    store: ExperienceMemoryStore,
    existing_items: list[ExperienceCard],
    experience: ExperienceCard,
    target_ids: list[str],
    conflict_group_id: str,
) -> None:
    conflict_group_id = conflict_group_id or experience.conflict_group_id or ""

    for target_id in target_ids:
        existing = _find_by_id(existing_items, target_id)
        if existing is None:
            continue
        existing.conflict_group_id = conflict_group_id
        store.upsert(existing)

    experience.conflict_group_id = conflict_group_id
    store.upsert(experience)


def write_memory_from_distilled_episode(
    distilled_episode: DistilledEpisode | dict[str, Any],
    root_dir: str | None = None,
    experience_extraction_mode: str = "llm",
    experience_merge_mode: str = "rule",
    llm_client=None,
) -> dict[str, Any]:
    """
    Write ExperienceCards from a distilled episode.

    Important:
      - Experience extraction defaults to LLM.
      - If no LLM is available, extraction returns [] and nothing is written.
      - Skill mining is not done here.
    """
    distilled = (
        distilled_episode
        if isinstance(distilled_episode, DistilledEpisode)
        else DistilledEpisode.from_dict(distilled_episode)
    )

    root = _root(root_dir)
    store = ExperienceMemoryStore(root)

    existing = store.list_all()
    extracted = extract_experiences(
        distilled,
        mode=experience_extraction_mode,
        llm_client=llm_client,
    )

    written_ids: list[str] = []
    merged_count = 0
    conflict_count = 0
    discarded_count = 0

    for experience in extracted:
        candidates = _similar_existing(experience, existing)

        if experience_merge_mode == "llm" and llm_client is not None:
            decision = decide_merge_llm(experience, candidates, llm_client)
        else:
            decision = decide_merge_rule(experience, candidates)

        merge_decision = str(decision.get("merge_decision") or "insert_new")

        if merge_decision == "discard":
            discarded_count += 1
            continue

        if merge_decision == "conflict":
            conflict_count += 1
            conflict_group_id = str(decision.get("conflict_group_id") or "")
            target_ids = [str(item) for item in decision.get("target_memory_ids", [])]
            _apply_conflict_group(store, existing, experience, target_ids, conflict_group_id)
            written_ids.append(experience.memory_id)
            existing = store.list_all()
            continue

        if merge_decision == "merge":
            merged_payload = decision.get("merged_experience") or experience.to_dict()
            merged = ExperienceCard.from_dict(merged_payload)
            store.upsert(merged)
            written_ids.append(merged.memory_id)
            merged_count += 1
            existing = store.list_all()
            continue

        store.upsert(experience)
        written_ids.append(experience.memory_id)
        existing.append(experience)

    result = {
        "episode_id": distilled.episode_id,
        "written_experience_ids": written_ids,
        "extracted_count": len(extracted),
        "merged_count": merged_count,
        "conflict_count": conflict_count,
        "discarded_count": discarded_count,
        "experience_store_count": len(store.list_all()),
    }

    logger.info(
        "Memory write done — episode=%s extracted=%d merged=%d conflict=%d discarded=%d "
        "total_store=%d",
        distilled.episode_id, len(extracted), merged_count,
        conflict_count, discarded_count, result["experience_store_count"],
    )
    return result