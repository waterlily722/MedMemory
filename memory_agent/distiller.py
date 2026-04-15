from __future__ import annotations

import uuid
from typing import Any

from .schemas import DistilledEpisode, EpisodeFeedback, ExperienceItem, GuardrailItem, SkillItem, TurnFeedback


def distill_episode(
    trajectory,
    turn_feedback_list,
    episode_feedback,
) -> DistilledEpisode:
    episode_feedback = episode_feedback if isinstance(episode_feedback, EpisodeFeedback) else EpisodeFeedback.from_dict(episode_feedback)
    turn_feedback_objects = [item if isinstance(item, TurnFeedback) else TurnFeedback.from_dict(item) for item in (turn_feedback_list or [])]
    memory_info = ((getattr(trajectory, "info", {}) or {}).get("memory_agent", {}) or {})
    turn_records = memory_info.get("turn_records", [])
    case_id = ((getattr(trajectory, "task", {}) or {}).get("case_id", ""))

    candidate_experience_items: list[ExperienceItem] = []
    candidate_skill_items: list[SkillItem] = []
    candidate_guardrail_items: list[GuardrailItem] = []
    pivotal_turns: list[int] = []

    for idx, feedback in enumerate(turn_feedback_objects):
        record = turn_records[idx] if idx < len(turn_records) else {}
        action_decision = (record.get("action_decision") or {}).get("chosen_action", {})
        case_before = record.get("case_before", {})
        case_after = record.get("case_after", {})

        if (feedback.local_gain or {}).get("gain_value", 0.0) >= 0.15 and action_decision:
            pivotal_turns.append(idx)
            candidate_experience_items.append(
                ExperienceItem(
                    item_id=f"exp_{uuid.uuid4().hex[:10]}",
                    source_episode_ids=[episode_feedback.episode_id],
                    source_case_ids=[case_id],
                    state_signature={
                        "chief_complaint": ((case_before.get("raw_snapshot") or {}).get("history") or {}).get("chief_complaint", ""),
                        "symptom_patterns": (case_before.get("derived_state") or {}).get("confirmed_facts", [])[:6],
                        "uncertainty_patterns": (case_before.get("derived_state") or {}).get("active_uncertainties", []),
                        "missing_slot_patterns": (case_before.get("derived_state") or {}).get("missing_critical_slots", []),
                        "hypothesis_patterns": (case_before.get("derived_state") or {}).get("tentative_differential", []),
                        "modality_patterns": list(((case_before.get("derived_state") or {}).get("modality_state", {}) or {}).get("available", {}).keys()),
                        "safety_patterns": ((case_before.get("derived_state") or {}).get("safety_state", {}) or {}).get("dangerous_alternatives_not_ruled_out", []),
                        "turn_stage": "mid",
                    },
                    trigger_signature=(record.get("intent_plan") or {}).get("query_signature", {}),
                    action_sequence=[
                        {
                            "step_index": idx,
                            "action_type": action_decision.get("action_type", ""),
                            "action_text": action_decision.get("action_text", ""),
                            "action_args": action_decision.get("action_args", {}),
                        }
                    ],
                    effect_summary=feedback.local_gain,
                    utility_stats={
                        "support_count": 1,
                        "fail_count": 0,
                        "avg_local_gain": (feedback.local_gain or {}).get("gain_value", 0.0),
                        "avg_turns_saved": 1 if (feedback.local_gain or {}).get("resolved_slots") else 0,
                    },
                    applicability={
                        "useful_when": (case_before.get("derived_state") or {}).get("missing_critical_slots", [])[:3],
                        "not_useful_when": [],
                        "contraindications": [],
                    },
                    source_field_refs=(record.get("source_field_refs") or []),
                )
            )

        if (feedback.safety_signal or {}).get("premature_finalize_signal") or (feedback.local_cost or {}).get("wasted_tool"):
            candidate_guardrail_items.append(
                GuardrailItem(
                    item_id=f"guard_{uuid.uuid4().hex[:10]}",
                    source_episode_ids=[episode_feedback.episode_id],
                    source_case_ids=[case_id],
                    trigger_signature=(record.get("intent_plan") or {}).get("query_signature", {}),
                    risky_action={
                        "action_type": action_decision.get("action_type", ""),
                        "action_text": action_decision.get("action_text", ""),
                        "action_args": action_decision.get("action_args", {}),
                    },
                    failure_mechanism={
                        "failure_type": "premature_finalize" if (feedback.safety_signal or {}).get("premature_finalize_signal") else "low_yield_action",
                        "root_cause": (case_before.get("derived_state") or {}).get("missing_critical_slots", [])[:3],
                        "misleading_surface_patterns": (case_before.get("derived_state") or {}).get("confirmed_facts", [])[:4],
                        "hidden_required_evidence": (case_before.get("derived_state") or {}).get("dangerous_alternatives_not_ruled_out", []),
                    },
                    safer_alternative={
                        "action_sequence": [
                            {
                                "action_type": "request_exam",
                                "action_text": "Gather more evidence before finalizing.",
                                "action_args": {"exam_type": "targeted diagnostic workup"},
                            }
                        ],
                        "escalation_rules": ["block_finalize_until_low_risk"],
                        "repair_hints": ["fill missing critical slots", "rule out dangerous alternatives"],
                    },
                    risk_stats={
                        "occurrence_count": 1,
                        "avg_penalty": 0.2,
                        "harmful_accept_rate": 1.0,
                        "severity": "high" if (feedback.safety_signal or {}).get("premature_finalize_signal") else "mid",
                    },
                    source_field_refs=(record.get("source_field_refs") or []),
                )
            )

    if episode_feedback.reward.get("total_score", 0.0) >= 0.7 and len(candidate_experience_items) >= 2:
        first = candidate_experience_items[0]
        candidate_skill_items.append(
            SkillItem(
                item_id=f"skill_{uuid.uuid4().hex[:10]}",
                source_experience_ids=[item.item_id for item in candidate_experience_items[:3]],
                source_case_ids=[case_id],
                skill_meta={
                    "skill_name": f"Stabilize_{first.action_sequence[0]['action_type']}",
                    "skill_family": first.action_sequence[0]["action_type"],
                },
                trigger_signature=first.trigger_signature,
                action_sequence=first.action_sequence,
                success_criteria=[{"criterion": "reduces uncertainty or fills missing slots"}],
                contraindications=[],
                reliability={
                    "support_count": len(candidate_experience_items),
                    "fail_count": 0,
                    "success_rate": 1.0,
                    "coverage": 0.5,
                },
                source_field_refs=first.source_field_refs,
            )
        )

    return DistilledEpisode(
        episode_id=episode_feedback.episode_id,
        summary={
            "key_state_transitions": [feedback.local_gain for feedback in turn_feedback_objects if (feedback.local_gain or {}).get("gain_value", 0.0) > 0.0],
            "pivotal_turns": pivotal_turns,
            "final_outcome_summary": episode_feedback.reward,
        },
        candidate_experience_items=candidate_experience_items,
        candidate_skill_items=candidate_skill_items,
        candidate_guardrail_items=candidate_guardrail_items,
        revision_signals={
            "item_ids_to_revise": [item.item_id for item in candidate_guardrail_items if item.risk_stats.get("severity") == "high"],
            "item_ids_to_deprecate": [],
        },
        source_field_refs=episode_feedback.source_field_refs,
    )
