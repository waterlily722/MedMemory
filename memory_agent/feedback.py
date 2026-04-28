from __future__ import annotations

import re
from typing import Any

from .schemas import ActionDecision, CaseMemory, EpisodeFeedback, MedEnvCaseBundle, TurnFeedback
from .utils.bench_adapter import extract_gold_diagnosis, unwrap_osce_examination

FINAL_DIAGNOSIS_RE = re.compile(r"\\box(?:ed)?\{(.+?)\}")


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", (text or "").lower()).split())


def _extract_final_diagnosis(trajectory) -> str:
    for step in reversed(getattr(trajectory, "steps", []) or []):
        action = getattr(step, "action", None)
        if isinstance(action, list):
            for call in action:
                fn = (call.get("function") or {}) if isinstance(call, dict) else {}
                if fn.get("name") == "diagnosis":
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        m = FINAL_DIAGNOSIS_RE.search(args)
                        if m:
                            return m.group(1).strip()
                    if isinstance(args, dict):
                        text = str(args.get("final_response", ""))
                        m = FINAL_DIAGNOSIS_RE.search(text)
                        if m:
                            return m.group(1).strip()
        text = getattr(step, "model_response", "") or ""
        m = FINAL_DIAGNOSIS_RE.search(text)
        if m:
            return m.group(1).strip()
    return ""


def build_turn_feedback(
    case_before: CaseMemory,
    action_decision: ActionDecision,
    execution_result,
    case_after: CaseMemory,
) -> TurnFeedback:
    before_missing = set(case_before.missing_info)
    after_missing = set(case_after.missing_info)
    resolved_slots = sorted(before_missing - after_missing)

    before_h = {h.name for h in case_before.active_hypotheses}
    after_h = {h.name for h in case_after.active_hypotheses}
    hypothesis_refined = bool(after_h and after_h != before_h)

    gain_value = 0.2 * len(resolved_slots) + (0.2 if hypothesis_refined else 0.0)
    if execution_result.execution_status == "success" and case_after.evidence_items:
        gain_value += 0.1

    wasted_turn = gain_value == 0.0 and execution_result.execution_status != "success"
    wasted_tool = execution_result.execution_status == "no_effect"
    redundant_question = action_decision.chosen_action.get("action_type") == "ASK" and not resolved_slots

    return TurnFeedback(
        turn_id=action_decision.turn_id,
        local_gain={
            "gain_type": "slot_fill" if resolved_slots else "hypothesis_refine" if hypothesis_refined else "low_gain",
            "gain_value": round(gain_value, 4),
            "resolved_slots": resolved_slots,
            "hypothesis_refined": hypothesis_refined,
        },
        local_cost={
            "wasted_turn": wasted_turn,
            "wasted_tool": wasted_tool,
            "redundant_question": redundant_question,
        },
        safety_signal={
            "premature_finalize_signal": action_decision.chosen_action.get("action_type") == "FINALIZE_DIAGNOSIS" and case_before.finalize_risk != "low",
            "high_finalize_risk": case_after.finalize_risk == "high",
        },
        source_field_refs=action_decision.source_field_refs,
    )


def build_episode_feedback(bundle, trajectory) -> EpisodeFeedback:
    bundle = bundle if isinstance(bundle, MedEnvCaseBundle) else MedEnvCaseBundle.from_dict(bundle)
    turn_records = ((getattr(trajectory, "info", {}) or {}).get("memory_agent", {}) or {}).get("turn_records", [])
    final_diagnosis = _extract_final_diagnosis(trajectory)

    premature = 0
    low_yield = 0
    for record in turn_records:
        feedback = record.get("turn_feedback") or {}
        if (feedback.get("safety_signal") or {}).get("premature_finalize_signal"):
            premature += 1
        if (feedback.get("local_cost") or {}).get("wasted_turn"):
            low_yield += 1

    gold = extract_gold_diagnosis(bundle.ehr)
    osce = unwrap_osce_examination(bundle.ehr)
    pred = _normalize(final_diagnosis)
    utility = 1.0 if pred and (_normalize(gold) in pred or pred in _normalize(gold)) else 0.0

    total_turns = len(turn_records) or len(getattr(trajectory, "steps", []) or [])
    efficiency = max(0.0, 1.0 - max(total_turns - 4, 0) * 0.08)
    safety = max(0.0, 1.0 - 0.2 * premature)
    total = round((utility + efficiency + safety) / 3.0, 4)

    return EpisodeFeedback(
        episode_id=getattr(trajectory, "uid", ""),
        offline_supervision={"ehr_final_result": gold},
        trajectory_metrics={
            "total_turns": total_turns,
            "repeated_low_yield_segments": low_yield,
            "premature_finalize_events": premature,
            "principal_diagnosis": str((osce.get("Principal_Diagnosis") or {}).get("icd_title", "")),
        },
        reward={
            "utility_score": utility,
            "efficiency_score": round(efficiency, 4),
            "safety_score": round(safety, 4),
            "total_score": total,
            "predicted_diagnosis": final_diagnosis,
        },
        source_field_refs=["OSCE_Examination.Correct_Diagnosis", "OSCE_Examination.Principal_Diagnosis"],
    )
