from __future__ import annotations

import uuid
from typing import Any

from ..llm import EXPERIENCE_EXTRACT_SCHEMA, LLMClient, experience_extraction_prompt, parse_validate_repair
from ..schemas import DistilledEpisode, EpisodeFeedback, ExperienceCard, TurnFeedback


def _turn_records(trajectory: Any) -> list[dict[str, Any]]:
    memory_info = ((getattr(trajectory, "info", {}) or {}).get("memory_agent", {}) or {})
    records = memory_info.get("turn_records", [])
    return [item for item in records if isinstance(item, dict)]


def _is_significant_record(record: dict[str, Any], feedback: TurnFeedback | None) -> bool:
    if feedback is None:
        return True
    gain_value = float((feedback.local_gain or {}).get("gain_value", 0.0))
    safety_signal = feedback.safety_signal or {}
    return gain_value > 0.0 or bool(safety_signal.get("premature_finalize_signal")) or bool(safety_signal.get("high_finalize_risk"))


def _collect_evidence_text(case_before: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for item in case_before.get("evidence_items", []) or []:
        if isinstance(item, dict):
            text = str(item.get("content", ""))
            if text:
                texts.append(text)
    if case_before.get("problem_summary"):
        texts.append(str(case_before.get("problem_summary")))
    if case_before.get("uncertainty_summary"):
        texts.append(str(case_before.get("uncertainty_summary")))
    return texts[:10]


def _build_card_from_record(
    record: dict[str, Any],
    feedback: TurnFeedback | None,
    episode_feedback: EpisodeFeedback,
    outcome_type: str,
    confidence: float,
) -> ExperienceCard:
    experience_id = f"exp_{uuid.uuid4().hex[:10]}"
    case_before = record.get("case_before") or {}
    action_decision = record.get("action_decision") or {}
    chosen = action_decision.get("chosen_action") or {}
    retrieval = record.get("retrieval_result") or {}
    applicability = record.get("applicability_result") or {}
    selected_memory_ids = list(record.get("selected_memory_ids") or [])
    rejected_memory_ids = list(record.get("rejected_memory_ids") or [])
    retrieved_ids = list((retrieval.get("experience_hits") or []))
    if not retrieved_ids and isinstance(retrieval, dict):
        retrieved_ids = [hit.get("item_id", "") for hit in (retrieval.get("experience_hits") or []) if isinstance(hit, dict)]

    action_sequence = []
    if isinstance(chosen, dict):
        action_sequence.append({"action_type": str(chosen.get("action_type", "")), "action_label": str(chosen.get("action_label", chosen.get("action_id", "")))})

    positive_evidence = _collect_evidence_text(case_before)
    missing_info = list(case_before.get("missing_info") or [])[:8]
    hypotheses = [str(h.get("name", h)) if isinstance(h, dict) else str(h) for h in case_before.get("active_hypotheses", [])]
    applicability_conditions = [str(case_before.get("local_goal", ""))] + [str(x) for x in case_before.get("modality_flags", [])]
    non_applicability_conditions = list(case_before.get("next_action_constraints") or []) + missing_info[:4]
    retrieval_tags = list(dict.fromkeys(selected_memory_ids + rejected_memory_ids + [str(x.get("item_id", "")) for x in (retrieval.get("knowledge_hits") or []) if isinstance(x, dict)]))
    if not retrieval_tags:
        retrieval_tags = [str(x) for x in retrieved_ids if x]

    return ExperienceCard(
        item_id=experience_id,
        experience_id=experience_id,
        situation_anchor=str(case_before.get("problem_summary", ""))[:220],
        local_goal=str(case_before.get("local_goal", "")),
        action_sequence=action_sequence,
        outcome_shift=str((feedback.local_gain if feedback else {}) or record.get("turn_feedback", {}).get("local_gain", {})),
        boundary=",".join(missing_info[:4]),
        outcome_type=outcome_type,
        key_evidence=positive_evidence,
        missing_info=missing_info,
        applicability_conditions=applicability_conditions,
        non_applicability_conditions=non_applicability_conditions,
        retrieval_tags=retrieval_tags,
        confidence=confidence,
        uncertainty_state=str(case_before.get("uncertainty_summary", "")),
        success_signal=str((feedback.local_gain or {}).get("gain_type", "partial") if feedback else "partial"),
        failure_mode="unsafe_finalize" if outcome_type == "unsafe" else ("low_yield" if outcome_type == "failure" else ""),
        error_tag=list((feedback.safety_signal or {}).keys()) if feedback and feedback.safety_signal else [],
        support_count=1,
        conflict_group_id=str(record.get("conflict_group_id", "")),
        hypotheses=hypotheses,
        source_turn_ids=[str(record.get("turn_id", ""))],
        source_episode_ids=[episode_feedback.episode_id],
        source_case_ids=[str(case_before.get("case_id", ""))],
        visual_signature={"pattern": str(case_before.get("modality_flags", []))},
        source_field_refs=list(record.get("source_field_refs") or []),
    )


def extract_experiences_rule(trajectory: Any, turn_feedback_list: list[TurnFeedback], episode_feedback: EpisodeFeedback) -> list[ExperienceCard]:
    turn_records = _turn_records(trajectory)
    feedback_by_turn = {feedback.turn_id: feedback for feedback in turn_feedback_list}

    out: list[ExperienceCard] = []
    for idx, record in enumerate(turn_records):
        turn_id = record.get("turn_id", idx)
        feedback = feedback_by_turn.get(turn_id)
        if not _is_significant_record(record, feedback):
            continue
        gain_value = float((feedback.local_gain or {}).get("gain_value", 0.0)) if feedback else 0.0
        outcome_type = "success" if gain_value >= 0.6 else "partial_success" if gain_value > 0.0 else "failure"
        if feedback and feedback.safety_signal and feedback.safety_signal.get("high_finalize_risk"):
            outcome_type = "unsafe"
        confidence = max(0.1, min(1.0, gain_value if gain_value > 0 else 0.35))
        out.append(_build_card_from_record(record, feedback, episode_feedback, outcome_type, confidence))
    return out


def extract_experiences_llm(
    trajectory: Any,
    turn_feedback_list: list[TurnFeedback],
    episode_feedback: EpisodeFeedback,
    llm_client: LLMClient,
) -> list[ExperienceCard]:
    turn_records = _turn_records(trajectory)
    feedback_by_turn = {feedback.turn_id: feedback for feedback in turn_feedback_list}
    cards: list[ExperienceCard] = []

    for idx, record in enumerate(turn_records):
        turn_id = record.get("turn_id", idx)
        feedback = feedback_by_turn.get(turn_id)
        if not _is_significant_record(record, feedback):
            continue

        prompt = experience_extraction_prompt(
            {
                "episode_feedback": episode_feedback.to_dict(),
                "turn_feedback": feedback.to_dict() if feedback else {},
                "turn_record": record,
                "trajectory_uid": getattr(trajectory, "uid", ""),
            }
        )
        fallback_card = {
            "memory_type": "experience",
            "experience_id": f"exp_{uuid.uuid4().hex[:10]}",
            "source_episode_id": episode_feedback.episode_id,
            "situation_anchor": str((record.get("case_before") or {}).get("problem_summary", "")),
            "local_goal": str((record.get("case_before") or {}).get("local_goal", "")),
            "uncertainty_state": str((record.get("case_before") or {}).get("uncertainty_summary", "")),
            "key_evidence": _collect_evidence_text(record.get("case_before") or {}),
            "missing_info": list((record.get("case_before") or {}).get("missing_info") or []),
            "action_sequence": [],
            "outcome_shift": str((feedback.local_gain if feedback else {}) or {}),
            "success_signal": "partial",
            "failure_mode": None,
            "boundary": ",".join(list((record.get("case_before") or {}).get("missing_info") or [])[:4]),
            "applicability_conditions": list((record.get("case_before") or {}).get("modality_flags") or []),
            "non_applicability_conditions": list((record.get("case_before") or {}).get("next_action_constraints") or []),
            "modality_flags": list((record.get("case_before") or {}).get("modality_flags") or []),
            "risk_tags": [],
            "retrieval_tags": list((record.get("selected_memory_ids") or [])),
            "confidence": 0.5,
        }
        parsed, _, _ = parse_validate_repair(llm_client.generate_json(prompt), EXPERIENCE_EXTRACT_SCHEMA, fallback_card)

        step_list = []
        for step in parsed.get("action_sequence") or []:
            if isinstance(step, dict):
                step_list.append(
                    {
                        "action_type": str(step.get("action_type", "")),
                        "action_label": str(step.get("action_label", step.get("action_template", ""))),
                    }
                )

        cards.append(
            ExperienceCard(
                item_id=str(parsed.get("experience_id") or fallback_card["experience_id"]),
                experience_id=str(parsed.get("experience_id") or fallback_card["experience_id"]),
                situation_anchor=str(parsed.get("situation_anchor", fallback_card["situation_anchor"])),
                local_goal=str(parsed.get("local_goal", fallback_card["local_goal"])),
                action_sequence=step_list,
                outcome_shift=str(parsed.get("outcome_shift", "")),
                boundary=str(parsed.get("boundary", "")),
                outcome_type={"success": "success", "partial": "partial_success", "failure": "failure", "unsafe": "unsafe"}.get(
                    str(parsed.get("success_signal", "partial")), "partial_success"
                ),
                key_evidence=[str(x) for x in parsed.get("key_evidence", [])],
                missing_info=[str(x) for x in parsed.get("missing_info", [])],
                applicability_conditions=[str(x) for x in parsed.get("applicability_conditions", [])],
                non_applicability_conditions=[str(x) for x in parsed.get("non_applicability_conditions", [])],
                retrieval_tags=[str(x) for x in parsed.get("retrieval_tags", [])],
                confidence=float(parsed.get("confidence", 0.5)),
                uncertainty_state=str(parsed.get("uncertainty_state", fallback_card["uncertainty_state"])),
                success_signal=str(parsed.get("success_signal", "partial")),
                failure_mode=str(parsed.get("failure_mode") or ""),
                error_tag=[str(x) for x in parsed.get("risk_tags", [])],
                support_count=1,
                source_turn_ids=[str(turn_id)],
                source_episode_ids=[episode_feedback.episode_id],
                source_case_ids=[str((record.get("case_before") or {}).get("case_id", ""))],
                visual_signature={"pattern": str((record.get("case_before") or {}).get("modality_flags", []))},
                source_field_refs=episode_feedback.source_field_refs,
            )
        )

    return cards
