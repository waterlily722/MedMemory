from __future__ import annotations

import re
from typing import Any

from .schemas import ActionDecision, CaseMemory, EpisodeFeedback, MedEnvCaseBundle, TurnFeedback


FINAL_DIAGNOSIS_RE = re.compile(r"\\box(?:ed)?\{(.+?)\}")


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _extract_final_diagnosis(trajectory) -> str:
    for step in reversed(getattr(trajectory, "steps", []) or []):
        action = getattr(step, "action", None)
        if isinstance(action, list):
            for call in action:
                fn = (call.get("function") or {}) if isinstance(call, dict) else {}
                if fn.get("name") == "diagnosis":
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        match = FINAL_DIAGNOSIS_RE.search(args)
                        if match:
                            return match.group(1).strip()
                    elif isinstance(args, dict):
                        text = str(args.get("final_response", ""))
                        match = FINAL_DIAGNOSIS_RE.search(text)
                        if match:
                            return match.group(1).strip()
        text = getattr(step, "model_response", "") or ""
        match = FINAL_DIAGNOSIS_RE.search(text)
        if match:
            return match.group(1).strip()
    return ""


def build_turn_feedback(
    case_before: CaseMemory,
    action_decision: ActionDecision,
    execution_result,
    case_after: CaseMemory,
) -> TurnFeedback:
    before_missing = set(case_before.derived_state.get("missing_critical_slots", []))
    after_missing = set(case_after.derived_state.get("missing_critical_slots", []))
    before_uncertain = set(case_before.derived_state.get("active_uncertainties", []))
    after_uncertain = set(case_after.derived_state.get("active_uncertainties", []))

    resolved_slots = sorted(before_missing - after_missing)
    uncertainty_reduction = sorted(before_uncertain - after_uncertain)
    used_refs = case_after.derived_state.get("interaction_state", {}).get("recent_useful_evidence_refs", [])
    gain_type = "slot_fill" if resolved_slots else "uncertainty_reduce" if uncertainty_reduction else "useful_tool" if execution_result.execution_status == "success" and used_refs else "safe_defer"
    gain_value = round(0.2 * len(resolved_slots) + 0.15 * len(uncertainty_reduction) + (0.1 if used_refs else 0.0), 4)

    asked_before = case_before.derived_state.get("interaction_state", {}).get("asked_questions", [])
    chosen_action = action_decision.chosen_action
    redundant_question = bool(chosen_action.get("action_type") == "ask" and chosen_action.get("action_text") in asked_before)
    wasted_turn = gain_value == 0.0 and execution_result.execution_status != "success"
    wasted_tool = execution_result.execution_status == "no_effect"
    risk_before = case_before.derived_state.get("safety_state", {}).get("premature_finalize_risk", "mid")
    after_conflict = case_after.derived_state.get("safety_state", {}).get("evidence_conflict_level", "low")
    before_conflict = case_before.derived_state.get("safety_state", {}).get("evidence_conflict_level", "low")

    return TurnFeedback(
        turn_id=action_decision.turn_id,
        local_gain={
            "gain_type": gain_type,
            "gain_value": gain_value,
            "resolved_slots": resolved_slots,
            "uncertainty_reduction": uncertainty_reduction,
            "useful_evidence_refs": used_refs[-5:],
        },
        local_cost={
            "wasted_turn": wasted_turn,
            "wasted_tool": wasted_tool,
            "redundant_question": redundant_question,
        },
        safety_signal={
            "premature_finalize_signal": chosen_action.get("action_type") == "finalize" and risk_before != "low",
            "evidence_conflict_increase": before_conflict == "low" and after_conflict in {"mid", "high"},
            "guardrail_hit": "guardrail" in (action_decision.final_rationale or "").lower(),
        },
        source_field_refs=action_decision.source_field_refs,
    )


def build_episode_feedback(
    bundle,
    trajectory,
) -> EpisodeFeedback:
    bundle = bundle if isinstance(bundle, MedEnvCaseBundle) else MedEnvCaseBundle.from_dict(bundle)
    steps = getattr(trajectory, "steps", []) or []
    final_diagnosis = _extract_final_diagnosis(trajectory)

    counts = {"retrieve": 0, "request_exam": 0, "cxr": 0, "cxr_grounding": 0, "diagnosis": 0}
    premature_finalize_events = []
    guardrail_violations = []
    low_yield_segments = []

    turn_records = ((getattr(trajectory, "info", {}) or {}).get("memory_agent", {}) or {}).get("turn_records", [])
    for idx, record in enumerate(turn_records):
        feedback = record.get("turn_feedback", {})
        decision = ((record.get("action_decision") or {}).get("chosen_action") or {})
        action_type = decision.get("action_type", "")
        if action_type in counts:
            counts[action_type] += 1
        if (feedback.get("safety_signal") or {}).get("premature_finalize_signal"):
            premature_finalize_events.append(idx)
        if (feedback.get("safety_signal") or {}).get("guardrail_hit"):
            guardrail_violations.append(idx)
        if (feedback.get("local_cost") or {}).get("wasted_turn"):
            low_yield_segments.append(idx)

    gold_candidates = [str((bundle.ehr or {}).get("Final_Result") or "")]
    normalized_pred = _normalize(final_diagnosis)
    utility_score = 0.0
    for gold in gold_candidates:
        normalized_gold = _normalize(gold)
        if normalized_pred and normalized_gold and (normalized_gold in normalized_pred or normalized_pred in normalized_gold):
            utility_score = 1.0
            break

    total_turns = len(turn_records) or len(steps)
    efficiency_score = max(0.0, 1.0 - max(total_turns - 4, 0) * 0.08)
    safety_penalty = 0.2 * len(premature_finalize_events) + 0.1 * len(guardrail_violations)
    safety_score = max(0.0, 1.0 - safety_penalty)
    total_score = round((utility_score + efficiency_score + safety_score) / 3.0, 4)

    return EpisodeFeedback(
        episode_id=getattr(trajectory, "uid", ""),
        offline_supervision={
            "ehr_final_result": (bundle.ehr or {}).get("Final_Result"),
        },
        trajectory_metrics={
            "total_turns": total_turns,
            "retrieve_count": counts["retrieve"],
            "request_exam_count": counts["request_exam"],
            "cxr_count": counts["cxr"],
            "grounding_count": counts["cxr_grounding"],
            "repeated_low_yield_segments": low_yield_segments,
            "premature_finalize_events": premature_finalize_events,
            "guardrail_violations": guardrail_violations,
        },
        reward={
            "utility_score": utility_score,
            "efficiency_score": round(efficiency_score, 4),
            "safety_score": round(safety_score, 4),
            "total_score": total_score,
            "predicted_diagnosis": final_diagnosis,
        },
        source_field_refs=["ehr.Final_Result"],
    )
