from __future__ import annotations

import logging
import uuid
from typing import Any

from ..llm import LLMClient, parse_validate_repair, skill_extraction_prompt
from ..llm.schemas import SKILL_EXTRACTION_SCHEMA
from ..schemas import DistilledEpisode, SkillCard
from ..utils.config import LLM_CONFIG
from .experience_extractor import build_clinical_episode_trace

logger = logging.getLogger(__name__)

MAX_SKILLS_PER_EPISODE = 2
MAX_SKILL_EXTRACTION_OUTPUT_TOKENS = int(
    LLM_CONFIG.get("skill_extraction_max_output_tokens", 1600)
)


def _as_distilled(distilled_episode: DistilledEpisode | dict[str, Any]) -> DistilledEpisode:
    return (
        distilled_episode
        if isinstance(distilled_episode, DistilledEpisode)
        else DistilledEpisode.from_dict(distilled_episode)
    )


def _safe_float(value: Any, default: float = 0.5) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _fallback_skill_id() -> str:
    return f"skill_{uuid.uuid4().hex[:12]}"


def _normalize_procedure(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    procedure: list[dict[str, str]] = []
    for item in value[:5]:
        if isinstance(item, dict):
            action_type = str(item.get("action_type") or "").upper()
            action_label = str(item.get("action_label") or item.get("label") or "")
        else:
            action_type = ""
            action_label = str(item)
        if action_type or action_label:
            procedure.append({"action_type": action_type, "action_label": action_label})
    return procedure


def _skill_from_raw(raw: dict[str, Any], distilled: DistilledEpisode) -> SkillCard | None:
    if not isinstance(raw, dict):
        return None

    procedure = _normalize_procedure(raw.get("procedure"))
    if not procedure:
        return None

    raw_source = raw.get("source") if isinstance(raw.get("source"), dict) else {}
    source = {
        "episode_ids": [str(item) for item in raw_source.get("episode_ids") or [] if str(item)],
        "case_ids": [str(item) for item in raw_source.get("case_ids") or [] if str(item)],
        "turn_ids": [str(item) for item in raw_source.get("turn_ids") or [] if str(item)],
    }
    if distilled.episode_id and distilled.episode_id not in source["episode_ids"]:
        source["episode_ids"].append(distilled.episode_id)
    if distilled.case_id and distilled.case_id not in source["case_ids"]:
        source["case_ids"].append(distilled.case_id)

    skill = SkillCard(
        memory_id=str(raw.get("memory_id") or _fallback_skill_id()),
        memory_type="skill",
        skill_name=str(raw.get("skill_name") or "").strip(),
        situation_text=str(raw.get("situation_text") or "").strip(),
        goal_text=str(raw.get("goal_text") or "").strip(),
        procedure_text=str(raw.get("procedure_text") or "").strip(),
        procedure=procedure,
        boundary_text=str(raw.get("boundary_text") or "").strip(),
        confidence=max(0.0, min(1.0, _safe_float(raw.get("confidence"), 0.5))),
        support_count=max(1, int(_safe_float(raw.get("support_count"), 1))),
        source=source,
    )

    if not (
        skill.skill_name
        and skill.situation_text
        and skill.goal_text
        and skill.procedure_text
        and skill.boundary_text
    ):
        return None
    return skill


def extract_skills_from_distilled_episode(
    distilled_episode: DistilledEpisode | dict[str, Any],
    mode: str = "llm",
    llm_client: LLMClient | None = None,
) -> list[SkillCard]:
    distilled = _as_distilled(distilled_episode)
    feedback = distilled.feedback if isinstance(distilled.feedback, dict) else {}
    if not bool(feedback.get("success", False)):
        return []

    if mode != "llm" or llm_client is None or not llm_client.available():
        logger.warning(
            "Skill extraction skipped — llm_client available=%s, mode=%s",
            llm_client.available() if llm_client else False,
            mode,
        )
        return []

    clinical_trace = build_clinical_episode_trace(distilled.turn_records)
    if not clinical_trace:
        return []

    payload = {
        "episode_id": distilled.episode_id,
        "case_id": distilled.case_id,
        "episode_outcome": {
            "success": True,
            "final_diagnosis": feedback.get("final_diagnosis") or "",
            "gold_diagnosis": feedback.get("gold_diagnosis") or "",
            "total_reward": feedback.get("total_reward") or 0.0,
            "summary": feedback.get("summary") or "",
        },
        "clinical_episode_trace": clinical_trace,
        "max_skills": MAX_SKILLS_PER_EPISODE,
    }

    parsed, _, _ = parse_validate_repair(
        llm_client.generate_json(
            skill_extraction_prompt(payload),
            max_tokens=MAX_SKILL_EXTRACTION_OUTPUT_TOKENS,
        ),
        SKILL_EXTRACTION_SCHEMA,
        {"skills": []},
    )

    skills: list[SkillCard] = []
    for raw in (parsed.get("skills") or [])[:MAX_SKILLS_PER_EPISODE]:
        skill = _skill_from_raw(raw, distilled)
        if skill is not None:
            skills.append(skill)

    logger.info(
        "Extracted %d skill cards from successful episode %s",
        len(skills),
        distilled.episode_id,
    )
    return skills
