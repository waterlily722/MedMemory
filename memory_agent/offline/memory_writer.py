from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..memory_store import ExperienceMemoryStore

logger = logging.getLogger(__name__)
from ..schemas import DistilledEpisode, ExperienceCard
from ..utils.config import MEMORY_ROOT_DIRNAME, MERGE_CONFIG
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
    limit: int | None = None,
) -> list[ExperienceCard]:
    if limit is None:
        limit = int(MERGE_CONFIG.get("candidate_top_k", 20) or 20)
    scored: list[tuple[float, ExperienceCard]] = []

    for item in existing:
        score = (
            0.7 * cosine_similarity(experience.situation_text, item.situation_text)
            + 0.3 * cosine_similarity(experience.action_text, item.action_text)
        )
        scored.append((score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for score, item in scored[:limit] if score > 0.0]


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
    inserted_count = 0

    for experience in extracted:
        candidates = _similar_existing(experience, existing)

        if experience_merge_mode == "llm" and llm_client is not None:
            decision = decide_merge_llm(experience, candidates, llm_client)
        else:
            decision = decide_merge_rule(experience, candidates)

        merge_decision = str(decision.get("merge_decision") or "insert_new")

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
        inserted_count += 1
        existing.append(experience)

    result = {
        "episode_id": distilled.episode_id,
        "written_experience_ids": written_ids,
        "extracted_count": len(extracted),
        "merged_count": merged_count,
        "inserted_count": inserted_count,
        "experience_store_count": len(store.list_all()),
    }

    logger.info(
        "Memory write done — episode=%s extracted=%d merged=%d inserted=%d "
        "total_store=%d",
        distilled.episode_id, len(extracted), merged_count,
        inserted_count, result["experience_store_count"],
    )
    return result
