from __future__ import annotations

import logging
import uuid
from typing import Any

from ..llm import LLMClient, experience_extraction_prompt, parse_validate_repair
from ..llm.schemas import EXPERIENCE_EXTRACTION_SCHEMA
from ..schemas import DistilledEpisode, ExperienceCard, OutcomeType
from ..utils.config import LLM_CONFIG

logger = logging.getLogger(__name__)

MAX_EPISODE_TURNS_FOR_EXTRACTION = int(
    LLM_CONFIG.get("experience_extraction_max_turns", 15)
)
MAX_EXPERIENCES_PER_EPISODE = 3
MAX_EXPERIENCE_EXTRACTION_OUTPUT_TOKENS = int(
    LLM_CONFIG.get("experience_extraction_max_output_tokens", 1200)
)
MAX_PROMPT_TEXT_CHARS = int(
    LLM_CONFIG.get("experience_extraction_max_text_chars", 1800)
)


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


def select_episode_turns(
    turn_records: list[dict[str, Any]],
    limit: int = MAX_EPISODE_TURNS_FOR_EXTRACTION,
) -> list[dict[str, Any]]:
    """Use the complete episode context, capped only to protect prompt length."""
    return list(turn_records or [])[-limit:]


def _clip_text(value: Any, max_chars: int = MAX_PROMPT_TEXT_CHARS) -> str:
    text = str(value or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def _compact_case_state(case_state: Any) -> dict[str, Any]:
    state = case_state if isinstance(case_state, dict) else {}
    acquired = state.get("acquired_information") or []
    compact_acquired: list[dict[str, Any]] = []
    for item in acquired[-3:]:
        if not isinstance(item, dict):
            continue
        compact_acquired.append(
            {
                "turn_id": item.get("turn_id"),
                "source_path": item.get("source_path"),
                "content": _clip_text(item.get("content"), 300),
            }
        )
    return {
        "chief_complaint": _clip_text(state.get("chief_complaint"), 300),
        "acquired_information_recent": compact_acquired,
    }


def _compact_memory_query(query: Any) -> dict[str, Any]:
    q = query if isinstance(query, dict) else {}
    return {
        "query_text": _clip_text(q.get("query_text"), 400),
        "query_facets": q.get("query_facets") or {},
    }


def _compact_memory_guidance(guidance: Any) -> dict[str, Any]:
    g = guidance if isinstance(guidance, dict) else {}
    selected = []
    for item in (g.get("selected_memories") or [])[:2]:
        if not isinstance(item, dict):
            continue
        content = item.get("content") if isinstance(item.get("content"), dict) else {}
        selected.append(
            {
                "memory_id": item.get("memory_id"),
                "memory_type": item.get("memory_type"),
                "score": item.get("score"),
                "content": {
                    key: _clip_text(content.get(key), 300)
                    for key in (
                        "situation",
                        "action",
                        "outcome",
                        "boundary",
                        "skill_name",
                        "goal",
                        "procedure",
                        "content",
                    )
                    if content.get(key)
                },
            }
        )
    return {"selected_memories": selected}


def _compact_observation(value: Any) -> Any:
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"memory_guidance", "memory_guidance_structured"}:
                continue
            if isinstance(item, (dict, list)):
                compact[key] = _clip_text(item, 500)
            else:
                compact[key] = _clip_text(item, 500)
        return compact
    if isinstance(value, list):
        return [_clip_text(item, 400) for item in value[-3:]]
    return _clip_text(value, 600)


def _compact_env_info(value: Any) -> dict[str, Any]:
    info = value if isinstance(value, dict) else {}
    keep_keys = (
        "response",
        "metadata",
        "turn_type",
        "tool_name",
        "diagnosis",
        "final_response",
        "case_id",
    )
    compact: dict[str, Any] = {}
    for key in keep_keys:
        if key in info:
            compact[key] = _clip_text(info.get(key), 500)
    return compact


def _compact_selected_action(action: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        "action_type",
        "tool_name",
        "name",
        "arguments",
        "final_response",
        "blocked_by_memory",
        "blocked",
    ):
        if key not in action:
            continue
        value = action.get(key)
        if isinstance(value, (dict, list)):
            compact[key] = _clip_text(value, 400)
        else:
            compact[key] = value if isinstance(value, bool) else _clip_text(value, 400)
    if not compact:
        compact["raw"] = _clip_text(action, 500)
    return compact


def _compact_turn(turn: dict[str, Any]) -> dict[str, Any]:
    action = _selected_action(turn)
    return {
        "turn_id": turn.get("turn_id", 0),
        "case_state": _compact_case_state(turn.get("case_state")),
        "memory_query": _compact_memory_query(turn.get("memory_query")),
        "memory_guidance": _compact_memory_guidance(turn.get("memory_guidance")),
        "selected_action": _compact_selected_action(action),
        "selected_action_blocked": bool(
            turn.get("selected_action_blocked")
            or action.get("blocked_by_memory")
            or action.get("blocked")
        ),
        "env_observation": _compact_observation(turn.get("env_observation")),
        "env_info": _compact_env_info(turn.get("env_info")),
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

    raw_source = raw.get("source") if isinstance(raw.get("source"), dict) else {}
    source_case_ids = [str(item) for item in raw_source.get("case_ids") or [] if str(item)]
    source_episode_ids = [str(item) for item in raw_source.get("episode_ids") or [] if str(item)]
    source_turn_ids = []
    for item in raw_source.get("turn_ids") or []:
        try:
            source_turn_ids.append(str(int(item)))
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
        tags=[str(item) for item in raw.get("tags") or [] if str(item)],
        confidence=max(0.0, min(1.0, _safe_float(raw.get("confidence"), 0.5))),
        support_count=max(1, int(_safe_float(raw.get("support_count"), 1))),
        source={
            "episode_ids": source_episode_ids,
            "case_ids": source_case_ids,
            "turn_ids": source_turn_ids,
        },
    )

    card.source.setdefault("episode_ids", [])
    card.source.setdefault("case_ids", [])
    card.source.setdefault("turn_ids", [])
    if distilled.episode_id and distilled.episode_id not in card.source["episode_ids"]:
        card.source["episode_ids"].append(distilled.episode_id)
    if distilled.case_id and distilled.case_id not in card.source["case_ids"]:
        card.source["case_ids"].append(distilled.case_id)

    if not card.situation_text or not card.action_text or not card.outcome_text:
        return None
    return card


def extract_experiences(
    distilled_episode: DistilledEpisode | dict[str, Any],
    mode: str = "llm",
    llm_client: LLMClient | None = None,
) -> list[ExperienceCard]:
    """
    Extract ExperienceCards via LLM from the full episode context.

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

    episode_turns = select_episode_turns(distilled.turn_records)
    if not episode_turns:
        logger.info(
            "No turns in episode %s — skipping extraction",
            distilled.episode_id,
        )
        return []

    payload = {
        "episode_id": distilled.episode_id,
        "case_id": distilled.case_id,
        "episode_summary": _clip_text(distilled.summary, 2000),
        "feedback": distilled.feedback,
        "episode_turns": [_compact_turn(turn) for turn in episode_turns],
        "max_experiences": MAX_EXPERIENCES_PER_EPISODE,
    }
    prompt = experience_extraction_prompt(payload)
    logger.info(
        "Experience extraction prompt size for episode %s: %d chars, %d turns, max_output_tokens=%d",
        distilled.episode_id,
        len(prompt),
        len(episode_turns),
        MAX_EXPERIENCE_EXTRACTION_OUTPUT_TOKENS,
    )
    parsed, _, _ = parse_validate_repair(
        llm_client.generate_json(
            prompt,
            max_tokens=MAX_EXPERIENCE_EXTRACTION_OUTPUT_TOKENS,
        ),
        EXPERIENCE_EXTRACTION_SCHEMA,
        {"experiences": []},
    )

    cards: list[ExperienceCard] = []
    for raw in (parsed.get("experiences") or [])[:MAX_EXPERIENCES_PER_EPISODE]:
        card = _card_from_raw(raw, distilled)
        if card is not None:
            cards.append(card)

    logger.info(
        "Extracted %d experience cards from episode %s (episode turns used: %d)",
        len(cards), distilled.episode_id, len(episode_turns),
    )
    return cards
