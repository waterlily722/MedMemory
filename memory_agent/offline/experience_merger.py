from __future__ import annotations

import logging
import os
from typing import Any

from ..llm import LLMClient, experience_merge_prompt, parse_validate_repair

logger = logging.getLogger(__name__)
from ..llm.schemas import EXPERIENCE_MERGE_SCHEMA
from ..schemas import ExperienceCard, OutcomeType
from ..utils.config import MERGE_CONFIG
from ..utils.scoring import bm25_similarity, cosine_similarity as token_cosine


def _threshold(name: str, default: float) -> float:
    try:
        return float(MERGE_CONFIG.get(name, default))
    except Exception:
        return default


def _same_direction(left: ExperienceCard, right: ExperienceCard) -> bool:
    left_positive = left.outcome_type in {
        OutcomeType.SUCCESS.value, OutcomeType.PARTIAL_SUCCESS.value
    }
    right_positive = right.outcome_type in {
        OutcomeType.SUCCESS.value, OutcomeType.PARTIAL_SUCCESS.value
    }
    return left_positive == right_positive


def _same_trigger(left: ExperienceCard, right: ExperienceCard) -> bool:
    situation_threshold = _threshold("semantic_threshold", 0.80)
    action_threshold = _threshold("action_threshold", 0.75)
    if _merge_scoring_mode() != "fielded_bm25":
        situation_score = token_cosine(left.situation_text, right.situation_text)
        action_score = token_cosine(left.action_text, right.action_text)
    else:
        situation_score = bm25_similarity(left.situation_text, right.situation_text)
        action_score = bm25_similarity(left.action_text, right.action_text)

    return (
        situation_score >= situation_threshold
        and action_score >= action_threshold
    )


def _merge_scoring_mode() -> str:
    return str(
        os.environ.get("MEDGYM_MERGE_SCORING")
        or MERGE_CONFIG.get("candidate_scoring")
        or "cosine"
    ).strip().lower()


def _can_merge(left: ExperienceCard, right: ExperienceCard) -> bool:
    return (
        _same_trigger(left, right)
        and _same_direction(left, right)
    )


def _unique(values: list[Any]) -> list[Any]:
    seen = set()
    output = []
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _choose_longer(left: str, right: str) -> str:
    left = left or ""
    right = right or ""
    return left if len(left) >= len(right) else right


def merge_experience(base: ExperienceCard, incoming: ExperienceCard) -> ExperienceCard:
    """
    Merge two experiences with same trigger and same outcome direction.
    Keep base memory_id stable.
    """
    merged = ExperienceCard.from_dict(base.to_dict())

    merged.support_count = max(1, base.support_count) + max(1, incoming.support_count)

    merged.situation_text = _choose_longer(base.situation_text, incoming.situation_text)
    merged.action_text = _choose_longer(base.action_text, incoming.action_text)
    merged.outcome_text = _choose_longer(base.outcome_text, incoming.outcome_text)
    merged.boundary_text = _choose_longer(base.boundary_text, incoming.boundary_text)

    merged.action_sequence = (
        base.action_sequence
        if len(base.action_sequence) >= len(incoming.action_sequence)
        else incoming.action_sequence
    )

    merged.tags = _unique(base.tags + incoming.tags)

    merged.confidence = max(base.confidence, incoming.confidence)

    merged.source = {
        "episode_ids": _unique(
            list((base.source or {}).get("episode_ids") or [])
            + list((incoming.source or {}).get("episode_ids") or [])
        ),
        "case_ids": _unique(
            list((base.source or {}).get("case_ids") or [])
            + list((incoming.source or {}).get("case_ids") or [])
        ),
        "turn_ids": _unique(
            list((base.source or {}).get("turn_ids") or [])
            + list((incoming.source or {}).get("turn_ids") or [])
        ),
    }

    return merged


def decide_merge_rule(
    new_experience: ExperienceCard,
    similar_existing: list[ExperienceCard],
) -> dict[str, Any]:
    for existing in similar_existing:
        if _can_merge(existing, new_experience):
            merged = merge_experience(existing, new_experience)
            return {
                "merge_decision": "merge",
                "target_memory_ids": [existing.memory_id],
                "reason": "same situation/action/outcome direction",
                "merged_experience": merged.to_dict(),
            }

    return {
        "merge_decision": "insert_new",
        "target_memory_ids": [],
        "reason": "no compatible existing memory",
        "merged_experience": new_experience.to_dict(),
    }


def decide_merge_llm(
    new_experience: ExperienceCard,
    similar_existing: list[ExperienceCard],
    llm_client: LLMClient,
) -> dict[str, Any]:
    fallback = decide_merge_rule(new_experience, similar_existing)

    if not similar_existing:
        return fallback

    if not llm_client.available():
        return fallback

    payload = {
        "new_experience": new_experience.to_dict(),
        "similar_existing": [item.to_dict() for item in similar_existing],
        "rule_decision": fallback,
        "instruction": (
            "Decide whether to merge the new experience into one retrieved "
            "candidate or insert it as a separate new memory. "
            "Never merge opposite outcomes or incompatible boundaries; choose "
            "insert_new when uncertain."
        ),
    }

    parsed, _, _ = parse_validate_repair(
        llm_client.generate_json(experience_merge_prompt(payload), max_tokens=1200),
        EXPERIENCE_MERGE_SCHEMA,
        fallback,
    )

    decision = str(parsed.get("merge_decision") or "insert_new")
    if decision not in {"insert_new", "merge"}:
        logger.warning(
            "LLM merge returned invalid decision=%r; falling back to rule", decision
        )
        return fallback

    if decision == "merge" and not isinstance(parsed.get("merged_experience"), dict):
        logger.warning("LLM merge decided 'merge' but merged_experience is not dict; fallback")
        return fallback

    if decision == "merge":
        candidate_ids = {item.memory_id for item in similar_existing}
        target_ids = [str(item) for item in parsed.get("target_memory_ids") or []]
        target_id = next((mid for mid in target_ids if mid in candidate_ids), "")
        merged = parsed.get("merged_experience") or {}
        merged_id = str(merged.get("memory_id") or "")
        if not target_id or merged_id != target_id:
            logger.warning(
                "LLM merge did not preserve a retrieved candidate memory_id; fallback"
            )
            return fallback

    return parsed
