from __future__ import annotations

import logging
import uuid
from typing import Any

from ..llm import LLMClient, experience_extraction_prompt, parse_validate_repair
from ..llm.schemas import EXPERIENCE_EXTRACTION_SCHEMA
from ..schemas import DistilledEpisode, ExperienceCard, OutcomeType

logger = logging.getLogger(__name__)

MAX_HIGH_VALUE_TURNS = 6
MAX_EXPERIENCES_PER_EPISODE = 3
MIN_HIGH_VALUE_REWARD = 0.25


def _as_distilled(distilled_episode: DistilledEpisode | dict[str, Any]) -> DistilledEpisode:
    return (
        distilled_episode
        if isinstance(distilled_episode, DistilledEpisode)
        else DistilledEpisode.from_dict(distilled_episode)
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _selected_action(turn_record: dict[str, Any]) -> dict[str, Any]:
    action = turn_record.get("selected_action") or {}
    return action if isinstance(action, dict) else {}


def _selected_action_blocked(turn_record: dict[str, Any]) -> bool:
    if bool(turn_record.get("selected_action_blocked")):
        return True
    action = _selected_action(turn_record)
    return bool(action.get("blocked_by_memory") or action.get("blocked"))


def _has_large_outcome_shift(turn_record: dict[str, Any]) -> bool:
    env_info = turn_record.get("env_info") or {}
    if not isinstance(env_info, dict):
        return False
    keys = [
        "outcome_shift",
        "diagnostic_shift",
        "uncertainty_reduction",
        "resolved_uncertainty",
        "new_key_evidence",
        "important_feedback",
    ]
    return any(bool(env_info.get(key)) for key in keys)


def _is_high_value_turn(turn_record: dict[str, Any]) -> bool:
    reward = _safe_float(turn_record.get("reward"), 0.0)
    return (
        reward >= MIN_HIGH_VALUE_REWARD
        or _selected_action_blocked(turn_record)
        or _has_large_outcome_shift(turn_record)
    )


def _turn_priority(turn_record: dict[str, Any]) -> float:
    reward = _safe_float(turn_record.get("reward"), 0.0)
    bonus = 0.0
    if _selected_action_blocked(turn_record):
        bonus += 1.0
    if _has_large_outcome_shift(turn_record):
        bonus += 0.7
    return reward + bonus


def select_high_value_turns(
    turn_records: list[dict[str, Any]],
    limit: int = MAX_HIGH_VALUE_TURNS,
) -> list[dict[str, Any]]:
    selected = [record for record in turn_records if _is_high_value_turn(record)]
    selected.sort(key=_turn_priority, reverse=True)
    return selected[:limit]


def _compact_turn(turn: dict[str, Any]) -> dict[str, Any]:
    action = _selected_action(turn)
    return {
        "turn_id": turn.get("turn_id", 0),
        "case_state": turn.get("case_state", {}),
        "memory_query": turn.get("memory_query", {}),
        "retrieval_result": turn.get("retrieval_result", {}),
        "applicability_result": turn.get("applicability_result", {}),
        "memory_guidance": turn.get("memory_guidance", {}),
        "selected_action": action,
        "selected_action_blocked": _selected_action_blocked(turn),
        "env_observation": turn.get("env_observation", {}),
        "env_info": turn.get("env_info", {}),
        "reward": _safe_float(turn.get("reward"), 0.0),
        "done": bool(turn.get("done", False)),
    }


def _fallback_memory_id() -> str:
    return f"exp_{uuid.uuid4().hex[:12]}"


def _normalize_action_sequence(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    sequence: list[dict[str, str]] = []
    for item in value[:4]:
        if isinstance(item, dict):
            action_type = str(item.get("action_type") or "").upper()
            action_label = str(item.get("action_label") or item.get("label") or "")
        else:
            action_type = ""
            action_label = str(item)
        if action_type or action_label:
            sequence.append({"action_type": action_type, "action_label": action_label})
    return sequence


def _card_from_raw(
    raw: dict[str, Any],
    distilled: DistilledEpisode,
) -> ExperienceCard | None:
    if not isinstance(raw, dict):
        return None

    boundary_text = str(raw.get("boundary_text") or "").strip()
    if not boundary_text:
        # The boundary is the core safety condition of Experience Memory.
        return None

    outcome_type = str(raw.get("outcome_type") or OutcomeType.PARTIAL_SUCCESS.value)
    try:
        outcome_type = OutcomeType(outcome_type).value
    except Exception:
        outcome_type = OutcomeType.PARTIAL_SUCCESS.value

    source_turn_ids: list[int] = []
    for value in raw.get("source_turn_ids") or []:
        try:
            source_turn_ids.append(int(value))
        except Exception:
            pass

    card = ExperienceCard(
        memory_id=str(raw.get("memory_id") or _fallback_memory_id()),
        memory_type="experience",
        situation_text=str(raw.get("situation_text") or "").strip(),
        action_text=str(raw.get("action_text") or "").strip(),
        outcome_text=str(raw.get("outcome_text") or "").strip(),
        boundary_text=boundary_text,
        action_sequence=_normalize_action_sequence(raw.get("action_sequence")),
        outcome_type=outcome_type,
        failure_mode=str(raw.get("failure_mode") or "").strip(),
        retrieval_tags=[str(item) for item in raw.get("retrieval_tags") or [] if str(item)],
        risk_tags=[str(item) for item in raw.get("risk_tags") or [] if str(item)],
        confidence=max(0.0, min(1.0, _safe_float(raw.get("confidence"), 0.5))),
        support_count=max(1, int(_safe_float(raw.get("support_count"), 1))),
        conflict_group_id=str(raw.get("conflict_group_id") or ""),
        source_episode_ids=[str(item) for item in raw.get("source_episode_ids") or [] if str(item)],
        source_case_ids=[str(item) for item in raw.get("source_case_ids") or [] if str(item)],
        source_turn_ids=source_turn_ids,
    )

    if distilled.episode_id and distilled.episode_id not in card.source_episode_ids:
        card.source_episode_ids.append(distilled.episode_id)
    if distilled.case_id and distilled.case_id not in card.source_case_ids:
        card.source_case_ids.append(distilled.case_id)

    if not card.situation_text or not card.action_text or not card.outcome_text:
        return None
    return card


def extract_experiences(
    distilled_episode: DistilledEpisode | dict[str, Any],
    mode: str = "llm",
    llm_client: LLMClient | None = None,
) -> list[ExperienceCard]:
    """
    Extract ExperienceCards only via LLM from selected high-value turns.

    Rule extraction is intentionally disabled because boundary_text should be
    generated by clinical reasoning rather than string templates.
    """
    distilled = _as_distilled(distilled_episode)
    if mode != "llm" or llm_client is None or not llm_client.available():
        logger.warning(
            "Experience extraction skipped — llm_client available=%s, mode=%s",
            llm_client.available() if llm_client else False,
            mode,
        )
        return []

    high_value_turns = select_high_value_turns(distilled.turn_records)
    if not high_value_turns:
        logger.info(
            "No high-value turns in episode %s — skipping extraction",
            distilled.episode_id,
        )
        return []

    payload = {
        "episode_id": distilled.episode_id,
        "case_id": distilled.case_id,
        "episode_summary": distilled.summary,
        "feedback": distilled.feedback,
        "high_value_turns": [_compact_turn(turn) for turn in high_value_turns],
        "max_experiences": MAX_EXPERIENCES_PER_EPISODE,
    }
    parsed, _, _ = parse_validate_repair(
        llm_client.generate_json(experience_extraction_prompt(payload), max_tokens=2000),
        EXPERIENCE_EXTRACTION_SCHEMA,
        {"experiences": []},
    )

    cards: list[ExperienceCard] = []
    for raw in (parsed.get("experiences") or [])[:MAX_EXPERIENCES_PER_EPISODE]:
        card = _card_from_raw(raw, distilled)
        if card is not None:
            cards.append(card)

    logger.info(
        "Extracted %d experience cards from episode %s (high-value turns: %d)",
        len(cards), distilled.episode_id, len(high_value_turns),
    )
    return cards
