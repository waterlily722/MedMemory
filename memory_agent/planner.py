from __future__ import annotations

from typing import Any, Dict, List

from .schemas import ActionCandidate, CaseMemory, IntentPlan, RankedIntent


def _turn_stage(turn_index: int) -> str:
    if turn_index <= 2:
        return "early"
    if turn_index <= 6:
        return "mid"
    return "late"


def _build_query_signature(case_memory: CaseMemory) -> dict[str, Any]:
    history = case_memory.raw_snapshot.get("history", {})
    derived = case_memory.derived_state
    interaction = derived.get("interaction_state", {})
    modality_state = derived.get("modality_state", {})
    safety_state = derived.get("safety_state", {})
    return {
        "chief_complaint": history.get("chief_complaint", ""),
        "symptom_patterns": derived.get("confirmed_facts", [])[:8] + derived.get("active_hypotheses", [])[:4],
        "uncertainty_patterns": derived.get("active_uncertainties", []),
        "missing_slot_patterns": derived.get("missing_critical_slots", []),
        "hypothesis_patterns": derived.get("tentative_differential", []),
        "modality_patterns": [name for name, enabled in (modality_state.get("available", {}) or {}).items() if enabled],
        "safety_patterns": safety_state.get("dangerous_alternatives_not_ruled_out", []),
        "turn_stage": _turn_stage(int(interaction.get("turn_index", 0))),
        "budget_patterns": [f"repeated_low_yield:{safety_state.get('repeated_low_yield_streak', 0)}"],
    }


def _candidate(action_id: str, action_type: str, action_text: str, action_args: dict[str, Any], planner_score: float, refs: list[str]) -> ActionCandidate:
    return ActionCandidate(
        action_id=action_id,
        action_type=action_type,
        action_text=action_text,
        action_args=action_args,
        planner_score=planner_score,
        source_field_refs=refs,
    )


def _chief_complaint(case_memory: CaseMemory) -> str:
    return str((case_memory.raw_snapshot.get("history") or {}).get("chief_complaint", "") or "")


def plan_intent(case_memory: CaseMemory) -> IntentPlan:
    interaction = case_memory.derived_state.get("interaction_state", {})
    turn_id = f"turn_{interaction.get('turn_index', 0)}"
    chief = _chief_complaint(case_memory)
    missing_slots = case_memory.derived_state.get("missing_critical_slots", [])
    uncertainties = case_memory.derived_state.get("active_uncertainties", [])
    differential = case_memory.derived_state.get("tentative_differential", [])
    modality_state = case_memory.derived_state.get("modality_state", {})
    safety_state = case_memory.derived_state.get("safety_state", {})
    asked_questions = interaction.get("asked_questions", [])

    ranked_intents: List[RankedIntent] = []
    action_candidates: List[ActionCandidate] = []
    refs = case_memory.source_field_refs or ["ehr.History"]

    if missing_slots:
        ranked_intents.append(RankedIntent("clarify_history", 0.92, f"Need to fill: {missing_slots[0]}", refs))
        ask_text = f"Please tell me more about {missing_slots[0]}."
        action_candidates.append(_candidate("ask_missing_slot", "ask", ask_text, {"question": ask_text}, 0.92, refs))

    if uncertainties:
        ranked_intents.append(RankedIntent("reduce_uncertainty", 0.84, f"Uncertainty signals: {', '.join(uncertainties[:2])}", refs))

    if differential:
        ranked_intents.append(RankedIntent("test_hypothesis", 0.78, f"Current lead diagnosis: {differential[0]}", refs))

    if modality_state.get("available", {}).get("retrieve"):
        query = chief or (differential[0] if differential else "undifferentiated patient presentation")
        retrieve_text = f"Retrieve guidance for {query}"
        action_candidates.append(_candidate("retrieve_guidance", "retrieve", retrieve_text, {"query": query}, 0.63, refs))
        ranked_intents.append(RankedIntent("use_retrieval", 0.63, "External knowledge may reduce ambiguity.", refs))

    complaint_lower = chief.lower()
    if any(keyword in complaint_lower for keyword in ("altered mental status", "confusion", "aphasia")):
        exam_type = "CT head and relevant labs"
    elif any(keyword in complaint_lower for keyword in ("chest pain",)):
        exam_type = "troponin and ECG"
    elif any(keyword in complaint_lower for keyword in ("shortness of breath", "cough", "dyspnea")):
        exam_type = "CXR and basic labs"
    else:
        exam_type = "basic labs"
    action_candidates.append(_candidate("request_exam", "request_exam", exam_type, {"exam_type": exam_type}, 0.67, refs))

    if modality_state.get("available", {}).get("cxr"):
        action_candidates.append(_candidate("open_cxr", "cxr", "Review the available chest X-ray.", {}, 0.55, refs))
        ranked_intents.append(RankedIntent("use_cxr", 0.55, "CXR is available in this case bundle.", refs))

    if modality_state.get("available", {}).get("cxr_grounding") and any("limited" in u for u in uncertainties):
        prompts = ["heart", "lung opacity"]
        action_candidates.append(
            _candidate("ground_cxr", "cxr_grounding", "Ground suspicious CXR regions.", {"text_prompts": prompts}, 0.49, refs)
        )
        ranked_intents.append(RankedIntent("use_grounding", 0.49, "Grounding may resolve imaging ambiguity.", refs))

    if differential:
        final_response = f"The final diagnosis is: \\boxed{{{differential[0]}}}."
        finalize_score = 0.4
        if safety_state.get("premature_finalize_risk") == "low":
            finalize_score = 0.88
        elif safety_state.get("premature_finalize_risk") == "mid":
            finalize_score = 0.52
        action_candidates.append(
            _candidate("finalize", "finalize", final_response, {"final_response": final_response}, finalize_score, refs)
        )
        if finalize_score < 0.7:
            ranked_intents.append(RankedIntent("defer_finalize", 0.8, "More evidence is still needed before final diagnosis.", refs))

    if not asked_questions and not missing_slots:
        question = f"When did your {chief or 'symptoms'} start?"
        action_candidates.append(_candidate("ask_timeline", "ask", question, {"question": question}, 0.71, refs))

    ranked_intents = sorted(ranked_intents, key=lambda item: item.score, reverse=True)
    action_candidates = sorted(action_candidates, key=lambda item: item.planner_score, reverse=True)
    query_signature = _build_query_signature(case_memory)
    return IntentPlan(
        turn_id=turn_id,
        ranked_intents=ranked_intents,
        action_candidates=action_candidates,
        query_signature=query_signature,
        source_field_refs=refs,
    )
