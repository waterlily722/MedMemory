from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from ..schemas import DistilledEpisode, EpisodeFeedback, ExperienceCard, SkillCard, SkillProcedureStep, TurnFeedback
from ..utils.config import EXPERIENCE_WRITE_CONFIG


def _as_turn_feedbacks(turn_feedback_list: list[Any]) -> list[TurnFeedback]:
    output: list[TurnFeedback] = []
    for item in turn_feedback_list or []:
        output.append(item if isinstance(item, TurnFeedback) else TurnFeedback.from_dict(item))
    return output


def distill_from_trajectory(trajectory: Any, turn_feedback_list: list[Any], episode_feedback: EpisodeFeedback | dict[str, Any]) -> DistilledEpisode:
    epi = episode_feedback if isinstance(episode_feedback, EpisodeFeedback) else EpisodeFeedback.from_dict(episode_feedback)
    turn_feedbacks = _as_turn_feedbacks(turn_feedback_list)

    memory_info = ((getattr(trajectory, "info", {}) or {}).get("memory_agent", {}) or {})
    turn_records = memory_info.get("turn_records", [])
    case_id = str((getattr(trajectory, "task", {}) or {}).get("case_id", ""))

    experiences: list[ExperienceCard] = []
    for idx, feedback in enumerate(turn_feedbacks):
        gain_value = float((feedback.local_gain or {}).get("gain_value", 0.0))
        if gain_value < EXPERIENCE_WRITE_CONFIG["min_outcome_shift_score"]:
            continue
        record = turn_records[idx] if idx < len(turn_records) else {}
        chosen = ((record.get("action_decision") or {}).get("chosen_action") or {})
        before = record.get("case_before") or {}

        experiences.append(
            ExperienceCard(
                item_id=f"exp_{uuid.uuid4().hex[:10]}",
                situation_anchor=str((before.get("problem_summary") or "")[:160]),
                local_goal=str(before.get("local_goal", "")),
                action_sequence=[
                    {
                        "action_type": str(chosen.get("action_type", "")),
                        "action_label": str(chosen.get("action_label", chosen.get("action_id", ""))),
                    }
                ],
                outcome_shift=str(feedback.local_gain),
                boundary=",".join((before.get("missing_info") or [])[:4]),
                outcome_type="success" if gain_value > 0.7 else "partial_success",
                error_tag=list((feedback.safety_signal or {}).keys()) if feedback.safety_signal else [],
                support_count=1,
                source_episode_ids=[epi.episode_id],
                source_case_ids=[case_id],
                source_field_refs=list(record.get("source_field_refs") or []),
            )
        )

    skills: list[SkillCard] = []
    if len(experiences) >= 2 and float((epi.reward or {}).get("total_score", 0.0)) >= 0.7:
        first = experiences[0]
        steps = []
        for i, act in enumerate(first.action_sequence[:3], start=1):
            steps.append(
                SkillProcedureStep(
                    step_id=i,
                    action_type=act.get("action_type", ""),
                    action_label=act.get("action_label", ""),
                    expected_observation="reduced uncertainty",
                    fallback_action="REQUEST_EXAM",
                    source_field_refs=first.source_field_refs,
                )
            )
        skills.append(
            SkillCard(
                skill_id=f"skill_{uuid.uuid4().hex[:10]}",
                skill_name=f"skill_{first.action_sequence[0].get('action_type', 'generic').lower()}",
                skill_trigger=first.situation_anchor,
                clinical_goal=first.local_goal,
                preconditions=[first.local_goal],
                procedure_template=steps,
                stop_condition=["finalize_risk low"],
                boundary=[first.boundary],
                contraindications=[],
                source_experience_ids=[x.item_id for x in experiences[:3]],
                support_count=len(experiences),
                success_rate=1.0,
                unsafe_rate=0.0,
                confidence=0.6,
                source_case_ids=[case_id],
                source_field_refs=first.source_field_refs,
            )
        )

    return DistilledEpisode(
        episode_id=epi.episode_id,
        summary={
            "timestamp": datetime.utcnow().isoformat(),
            "experience_count": len(experiences),
            "skill_count": len(skills),
            "reward": epi.reward,
        },
        candidate_experience_items=[x.to_dict() for x in experiences],
        candidate_skill_items=[x.to_dict() for x in skills],
        revision_signals={"item_ids_to_revise": [], "item_ids_to_deprecate": []},
        source_field_refs=epi.source_field_refs,
    )
