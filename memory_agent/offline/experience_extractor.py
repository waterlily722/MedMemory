from __future__ import annotations

import uuid
from typing import Any

from ..llm import EXPERIENCE_EXTRACT_SCHEMA, LLMClient, experience_extraction_prompt, parse_validate_repair
from ..schemas import DistilledEpisode, EpisodeFeedback, ExperienceCard, TurnFeedback


def extract_experiences_rule(trajectory: Any, turn_feedback_list: list[TurnFeedback], episode_feedback: EpisodeFeedback) -> list[ExperienceCard]:
    memory_info = ((getattr(trajectory, "info", {}) or {}).get("memory_agent", {}) or {})
    turn_records = memory_info.get("turn_records", [])
    case_id = str((getattr(trajectory, "task", {}) or {}).get("case_id", ""))

    out: list[ExperienceCard] = []
    for idx, feedback in enumerate(turn_feedback_list):
        gain = float((feedback.local_gain or {}).get("gain_value", 0.0))
        if gain <= 0.0:
            continue
        record = turn_records[idx] if idx < len(turn_records) else {}
        chosen = ((record.get("action_decision") or {}).get("chosen_action") or {})
        before = record.get("case_before") or {}
        out.append(
            ExperienceCard(
                item_id=f"exp_{uuid.uuid4().hex[:10]}",
                situation_anchor=str(before.get("problem_summary", ""))[:180],
                local_goal=str(before.get("local_goal", "")),
                action_sequence=[
                    {
                        "action_type": str(chosen.get("action_type", "")),
                        "action_label": str(chosen.get("action_label", chosen.get("action_id", ""))),
                    }
                ],
                outcome_shift=str(feedback.local_gain),
                boundary=",".join((before.get("missing_info") or [])[:4]),
                outcome_type="success" if gain > 0.6 else "partial_success",
                error_tag=list((feedback.safety_signal or {}).keys()) if feedback.safety_signal else [],
                support_count=1,
                source_episode_ids=[episode_feedback.episode_id],
                source_case_ids=[case_id],
                source_field_refs=list(record.get("source_field_refs") or []),
            )
        )
    return out


def extract_experiences_llm(
    trajectory: Any,
    turn_feedback_list: list[TurnFeedback],
    episode_feedback: EpisodeFeedback,
    llm_client: LLMClient,
) -> list[ExperienceCard]:
    fallback_card = {
        "memory_type": "experience",
        "experience_id": f"exp_{uuid.uuid4().hex[:10]}",
        "source_episode_id": episode_feedback.episode_id,
        "situation_anchor": "",
        "local_goal": "",
        "uncertainty_state": "",
        "key_evidence": [],
        "missing_info": [],
        "action_sequence": [],
        "outcome_shift": "",
        "success_signal": "partial",
        "failure_mode": None,
        "boundary": "",
        "applicability_conditions": [],
        "non_applicability_conditions": [],
        "modality_flags": [],
        "risk_tags": [],
        "retrieval_tags": [],
        "confidence": 0.5,
    }

    prompt = experience_extraction_prompt(
        {
            "episode_feedback": episode_feedback.to_dict(),
            "turn_feedback_list": [x.to_dict() for x in turn_feedback_list],
            "trajectory_uid": getattr(trajectory, "uid", ""),
        }
    )
    parsed, _, _ = parse_validate_repair(llm_client.generate_json(prompt), EXPERIENCE_EXTRACT_SCHEMA, fallback_card)

    card = ExperienceCard(
        item_id=str(parsed.get("experience_id") or fallback_card["experience_id"]),
        situation_anchor=str(parsed.get("situation_anchor", "")),
        local_goal=str(parsed.get("local_goal", "")),
        action_sequence=[
            {
                "action_type": str(step.get("action_type", "")),
                "action_label": str(step.get("action_label", step.get("action_template", ""))),
            }
            for step in (parsed.get("action_sequence") or [])
            if isinstance(step, dict)
        ],
        outcome_shift=str(parsed.get("outcome_shift", "")),
        boundary=str(parsed.get("boundary", "")),
        outcome_type={"success": "success", "partial": "partial_success", "failure": "failure"}.get(
            str(parsed.get("success_signal", "partial")), "partial_success"
        ),
        error_tag=[str(x) for x in parsed.get("risk_tags", [])],
        support_count=1,
        source_episode_ids=[str(parsed.get("source_episode_id", episode_feedback.episode_id))],
        source_case_ids=[],
        source_field_refs=episode_feedback.source_field_refs,
    )
    return [card]
