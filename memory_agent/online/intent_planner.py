from __future__ import annotations

from ..schemas import ActionCandidate, CaseState, IntentPlan, RankedIntent
from .query_builder import build_memory_query_with_mode


def _candidate(
    action_id: str,
    action_type: str,
    action_label: str,
    action_content: str,
    planner_score: float,
    refs: list[str],
) -> ActionCandidate:
    return ActionCandidate(
        action_id=action_id,
        action_type=action_type,
        action_label=action_label,
        action_content=action_content,
        planner_score=planner_score,
        source_field_refs=refs,
    )


def _to_ranked_intents(case_memory: CaseState) -> list[RankedIntent]:
    intents: list[RankedIntent] = []
    refs = case_memory.source_field_refs
    if case_memory.missing_info:
        intents.append(RankedIntent("collect_missing_info", 0.9, f"missing: {case_memory.missing_info[:2]}", refs))
    if case_memory.active_hypotheses:
        intents.append(RankedIntent("disambiguate_hypothesis", 0.82, case_memory.active_hypotheses[0].name, refs))
    if "image" in case_memory.modality_flags:
        intents.append(RankedIntent("verify_image_pattern", 0.66, "image modality available", refs))
    if case_memory.finalize_risk == "low":
        intents.append(RankedIntent("consider_finalize", 0.75, "risk is low", refs))
    return sorted(intents, key=lambda x: x.score, reverse=True)


def plan_intent(case_memory: CaseState) -> IntentPlan:
    refs = case_memory.source_field_refs or ["ehr.History"]
    candidates: list[ActionCandidate] = []

    if case_memory.missing_info:
        slot = case_memory.missing_info[0]
        candidates.append(
            _candidate(
                action_id="ask_missing",
                action_type="ASK",
                action_label="ask_onset",
                action_content=f"Please clarify {slot}.",
                planner_score=0.92,
                refs=refs,
            )
        )

    if "lab" in case_memory.modality_flags:
        candidates.append(
            _candidate(
                action_id="request_lab",
                action_type="REQUEST_LAB",
                action_label="order_CBC",
                action_content="Order CBC for differential narrowing.",
                planner_score=0.7,
                refs=refs,
            )
        )

    if "image" in case_memory.modality_flags:
        candidates.append(
            _candidate(
                action_id="review_image",
                action_type="REVIEW_IMAGE",
                action_label="review_opacity_pattern",
                action_content="Review chest image for key patterns.",
                planner_score=0.68,
                refs=refs,
            )
        )

    candidates.append(
        _candidate(
            action_id="update_hypothesis",
            action_type="UPDATE_HYPOTHESIS",
            action_label="update_ranked_differential",
            action_content="Update the ranked hypotheses with latest evidence.",
            planner_score=0.62,
            refs=refs,
        )
    )

    candidates.append(
        _candidate(
            action_id="defer_finalize",
            action_type="DEFER_FINALIZE",
            action_label="defer_until_safe",
            action_content="Defer final diagnosis and gather more evidence.",
            planner_score=0.58,
            refs=refs,
        )
    )

    if case_memory.active_hypotheses:
        label = case_memory.active_hypotheses[0].name
        candidates.append(
            _candidate(
                action_id="finalize",
                action_type="FINALIZE_DIAGNOSIS",
                action_label="finalize_primary_diagnosis",
                action_content=f"Primary diagnosis likely: {label}",
                planner_score=0.55 if case_memory.finalize_risk != "low" else 0.88,
                refs=refs,
            )
        )

    ranked = _to_ranked_intents(case_memory)
    memory_query = build_memory_query_with_mode(case_memory, [c.action_type for c in candidates], mode="rule")
    plan = IntentPlan(
        turn_id=case_memory.turn_id,
        action_candidates=sorted(candidates, key=lambda c: c.planner_score, reverse=True),
        memory_query=memory_query,
        source_field_refs=refs,
    )
    setattr(plan, "ranked_intents", ranked)
    return plan


def plan_intent_with_mode(
    case_memory: CaseState,
    query_builder_mode: str = "rule",
    llm_client=None,
    observation: dict | None = None,
    interaction_history_summary: str = "",
) -> IntentPlan:
    plan = plan_intent(case_memory)
    plan.memory_query = build_memory_query_with_mode(
        case_memory,
        [c.action_type for c in plan.action_candidates],
        mode=query_builder_mode,
        llm_client=llm_client,
        observation=observation,
        interaction_history_summary=interaction_history_summary,
    )
    return plan