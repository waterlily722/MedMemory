from __future__ import annotations

import uuid
from typing import Any

from ..llm import LLMClient, experience_extraction_prompt, parse_validate_repair
from ..llm.schemas import EXPERIENCE_EXTRACTION_SCHEMA
from ..schemas import DistilledEpisode, ExperienceCard


def _selected_action(turn_record: dict[str, Any]) -> dict[str, Any]:
    selected = turn_record.get("selected_action") or {}
    return selected if isinstance(selected, dict) else {}


def _outcome_type(turn_record: dict[str, Any], feedback: dict[str, Any]) -> str:
    if turn_record.get("selected_action_blocked"):
        return "unsafe"
    reward = float(turn_record.get("reward", feedback.get("total_reward", 0.0)))
    if reward >= 0.7:
        return "success"
    if reward >= 0.25:
        return "partial_success"
    return "failure"


def _build_experience(turn_record: dict[str, Any], feedback: dict[str, Any], outcome_type: str) -> ExperienceCard:
    before = turn_record.get("case_state_before") or {}
    selected = _selected_action(turn_record)
    action_type = str(selected.get("action_type", "ASK"))
    action_label = str(selected.get("action_label", action_type.lower()))
    return ExperienceCard(
        memory_id=str(turn_record.get("experience_id") or f"exp_{uuid.uuid4().hex[:10]}"),
        situation_anchor=str(before.get("problem_summary", ""))[:240],
        local_goal=str(before.get("local_goal", "")),
        uncertainty_state=str(before.get("uncertainty_summary", "")),
        key_evidence=list(before.get("key_evidence") or [])[:8],
        missing_info=list(before.get("missing_info") or [])[:8],
        active_hypotheses=list(before.get("active_hypotheses") or [])[:8],
        action_sequence=[{"action_type": action_type, "action_label": action_label}],
        outcome_shift=str((turn_record.get("env_info") or {}).get("outcome_shift") or feedback.get("notes") or ""),
        outcome_type=outcome_type,
        failure_mode="safety_block" if outcome_type == "unsafe" else None,
        boundary=", ".join((before.get("missing_info") or [])[:3]),
        applicability_conditions=list(before.get("modality_flags") or []),
        non_applicability_conditions=list(before.get("negative_evidence") or [])[:4],
        modality_flags=list(before.get("modality_flags") or []),
        retrieval_tags=[action_type],
        risk_tags=["blocked_action"] if turn_record.get("selected_action_blocked") else [],
        confidence=max(0.2, min(1.0, float(turn_record.get("reward", 0.0)) if turn_record.get("reward", 0.0) else 0.5)),
        support_count=1,
        source_episode_ids=[str(feedback.get("episode_id", ""))],
        source_case_ids=[str(feedback.get("case_id", ""))],
    )


def extract_experiences_rule(distilled_episode: DistilledEpisode | dict[str, Any]) -> list[ExperienceCard]:
    distilled = distilled_episode if isinstance(distilled_episode, DistilledEpisode) else DistilledEpisode.from_dict(distilled_episode)
    feedback = distilled.episode_feedback if isinstance(distilled.episode_feedback, dict) else {}
    experiences: list[ExperienceCard] = []
    for turn_record in distilled.turn_records:
        if not isinstance(turn_record, dict):
            continue
        outcome_type = _outcome_type(turn_record, feedback)
        experiences.append(_build_experience(turn_record, feedback, outcome_type))
    return experiences[:3]


def extract_experiences_llm(distilled_episode: DistilledEpisode | dict[str, Any], llm_client: LLMClient) -> list[ExperienceCard]:
    rule_experiences = extract_experiences_rule(distilled_episode)
    if not llm_client.available():
        return rule_experiences
    distilled = distilled_episode if isinstance(distilled_episode, DistilledEpisode) else DistilledEpisode.from_dict(distilled_episode)
    payload = {"distilled_episode": distilled.to_dict(), "candidate_rule_experiences": [experience.to_dict() for experience in rule_experiences]}
    fallback = {"memory_type": "experience", "memory_id": rule_experiences[0].memory_id if rule_experiences else f"exp_{uuid.uuid4().hex[:10]}", "situation_anchor": rule_experiences[0].situation_anchor if rule_experiences else "", "local_goal": rule_experiences[0].local_goal if rule_experiences else "", "uncertainty_state": rule_experiences[0].uncertainty_state if rule_experiences else "", "key_evidence": rule_experiences[0].key_evidence if rule_experiences else [], "missing_info": rule_experiences[0].missing_info if rule_experiences else [], "active_hypotheses": rule_experiences[0].active_hypotheses if rule_experiences else [], "action_sequence": rule_experiences[0].action_sequence if rule_experiences else [], "outcome_shift": rule_experiences[0].outcome_shift if rule_experiences else "", "outcome_type": rule_experiences[0].outcome_type if rule_experiences else "partial_success", "failure_mode": rule_experiences[0].failure_mode if rule_experiences else None, "boundary": rule_experiences[0].boundary if rule_experiences else "", "applicability_conditions": rule_experiences[0].applicability_conditions if rule_experiences else [], "non_applicability_conditions": rule_experiences[0].non_applicability_conditions if rule_experiences else [], "modality_flags": rule_experiences[0].modality_flags if rule_experiences else [], "retrieval_tags": rule_experiences[0].retrieval_tags if rule_experiences else [], "risk_tags": rule_experiences[0].risk_tags if rule_experiences else [], "confidence": rule_experiences[0].confidence if rule_experiences else 0.5, "support_count": 1, "source_episode_ids": rule_experiences[0].source_episode_ids if rule_experiences else [], "source_case_ids": rule_experiences[0].source_case_ids if rule_experiences else []}
    parsed, _, _ = parse_validate_repair(llm_client.generate_json(experience_extraction_prompt(payload)), EXPERIENCE_EXTRACTION_SCHEMA, fallback)
    return [ExperienceCard.from_dict(parsed)]


def extract_experiences(distilled_episode: DistilledEpisode | dict[str, Any], mode: str = "rule", llm_client: LLMClient | None = None) -> list[ExperienceCard]:
    if mode == "llm" and llm_client is not None:
        return extract_experiences_llm(distilled_episode, llm_client)
    return extract_experiences_rule(distilled_episode)
