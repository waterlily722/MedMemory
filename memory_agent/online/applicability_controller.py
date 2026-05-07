from __future__ import annotations

import logging

from ..llm import LLMClient, applicability_prompt, parse_validate_repair

logger = logging.getLogger(__name__)
from ..llm.schemas import APPLICABILITY_SCHEMA
from ..schemas import (
    ActionAssessment,
    ApplicabilityResult,
    CaseState,
    MemoryApplicabilityAssessment,
    MemoryQuery,
    MemoryRetrievalResult,
    OutcomeType,
    RetrievalHit,
)
from ..utils.config import APPLICABILITY_CONFIG, MEMORY_ACTION_CONFIG

DEFAULT_ACTIONS = list(MEMORY_ACTION_CONFIG["default_actions"])
FINALIZE_ACTION = str(MEMORY_ACTION_CONFIG["finalize_action"])


def _all_hits(retrieval_result: MemoryRetrievalResult) -> list[RetrievalHit]:
    return (
        retrieval_result.positive_experience_hits
        + retrieval_result.negative_experience_hits
        + retrieval_result.skill_hits
        + retrieval_result.knowledge_hits
    )


def _outcome_type(hit: RetrievalHit) -> str:
    return str(hit.content.get("outcome_type") or "").lower()


_NEGATIVE_OUTCOMES = {OutcomeType.FAILURE.value, OutcomeType.UNSAFE.value}
_POSITIVE_OUTCOMES = {OutcomeType.SUCCESS.value, OutcomeType.PARTIAL_SUCCESS.value}


def _is_negative_experience(hit: RetrievalHit) -> bool:
    return hit.memory_type == "experience" and _outcome_type(hit) in _NEGATIVE_OUTCOMES


def _infer_action_from_steps(steps: object) -> str:
    if isinstance(steps, list) and steps:
        first = steps[0]
        if isinstance(first, dict):
            return str(first.get("action_type") or "").upper()
    return ""


def _infer_action_from_text(text: str) -> str:
    upper = text.upper()
    for action in DEFAULT_ACTIONS:
        if action in upper:
            return action
    return ""


def _infer_action_from_memory(hit: RetrievalHit) -> str:
    if hit.memory_type == "experience":
        action = _infer_action_from_steps(hit.content.get("action_sequence") or [])
        return action or _infer_action_from_text(str(hit.content.get("action_text") or ""))
    if hit.memory_type == "skill":
        action = _infer_action_from_steps(hit.content.get("procedure") or [])
        return action or _infer_action_from_text(str(hit.content.get("procedure_text") or ""))
    return ""


def _unsafe_block_threshold() -> float:
    return float(APPLICABILITY_CONFIG.get("unsafe_block_score", 0.35))


def _skill_apply_threshold() -> float:
    return float(APPLICABILITY_CONFIG.get("skill_apply_score", 0.40))


def _rule_memory_assessment(hit: RetrievalHit) -> MemoryApplicabilityAssessment:
    action = _infer_action_from_memory(hit)

    if _is_negative_experience(hit):
        outcome = _outcome_type(hit)
        if outcome == "unsafe" and action and hit.score >= _unsafe_block_threshold():
            return MemoryApplicabilityAssessment(
                memory_id=hit.memory_id,
                memory_type=hit.memory_type,
                decision="block",
                reason=(
                    "Retrieved high-confidence unsafe experience with overlapping action; "
                    "block repeating this risky path."
                ),
                action_bias={action: -0.9},
                blocked_actions=[action],
            )
        return MemoryApplicabilityAssessment(
            memory_id=hit.memory_id,
            memory_type=hit.memory_type,
            decision="hint",
            reason=(
                "Retrieved negative experience. Treat as a cautionary signal; "
                "discourage rather than block unless clearly unsafe and high-confidence."
            ),
            action_bias={action: -0.45} if action else {},
            blocked_actions=[],
        )

    if hit.memory_type == "experience":
        return MemoryApplicabilityAssessment(
            memory_id=hit.memory_id,
            memory_type=hit.memory_type,
            decision="hint",
            reason="Retrieved positive experience may provide a useful local action hint.",
            action_bias={action: 0.35} if action else {},
            blocked_actions=[],
        )

    if hit.memory_type == "skill":
        return MemoryApplicabilityAssessment(
            memory_id=hit.memory_id,
            memory_type=hit.memory_type,
            decision="hint",
            reason=(
                "Retrieved skill is potentially useful, but retrieval alone is not enough "
                "to automatically apply it; check boundary conditions."
            ),
            action_bias={action: 0.30} if action else {},
            blocked_actions=[],
        )

    return MemoryApplicabilityAssessment(
        memory_id=hit.memory_id,
        memory_type=hit.memory_type,
        decision="hint",
        reason="Retrieved knowledge may provide relevant background guidance.",
        action_bias={},
        blocked_actions=[],
    )


def _postprocess_llm_assessment(
    hit: RetrievalHit,
    assessment: MemoryApplicabilityAssessment,
) -> MemoryApplicabilityAssessment:
    action = _infer_action_from_memory(hit)
    outcome = _outcome_type(hit)

    if assessment.decision not in {"apply", "hint", "block", "ignore"}:
        return _rule_memory_assessment(hit)

    assessment.memory_id = hit.memory_id
    assessment.memory_type = hit.memory_type

    if hit.memory_type == "skill" and assessment.decision == "apply" and hit.score < _skill_apply_threshold():
        assessment.decision = "hint"
        assessment.reason = (
            assessment.reason
            or "Skill score is below apply threshold; downgrade to hint."
        )

    if hit.memory_type == "experience" and outcome == "failure" and assessment.decision == "block":
        assessment.decision = "hint"
        assessment.blocked_actions = []
        if action and not assessment.action_bias:
            assessment.action_bias = {action: -0.45}
        assessment.reason = (
            assessment.reason
            or "Failure experience should discourage, not hard-block, unless unsafe."
        )

    if hit.memory_type == "experience" and outcome == "unsafe" and assessment.decision == "block":
        if hit.score < _unsafe_block_threshold() or not action:
            assessment.decision = "hint"
            assessment.blocked_actions = []
            if action and not assessment.action_bias:
                assessment.action_bias = {action: -0.45}
            assessment.reason = (
                assessment.reason
                or "Unsafe experience was not similar enough for hard block; downgrade to hint."
            )

    assessment.blocked_actions = [
        str(item).upper()
        for item in assessment.blocked_actions
        if str(item).upper() in DEFAULT_ACTIONS
    ]
    assessment.action_bias = {
        str(key).upper(): float(value)
        for key, value in (assessment.action_bias or {}).items()
        if str(key).upper() in DEFAULT_ACTIONS
    }
    return assessment


def _llm_memory_assessment(
    hit: RetrievalHit,
    case_state: CaseState,
    memory_query: MemoryQuery,
    llm_client: LLMClient,
    debug: dict[str, object] | None = None,
) -> MemoryApplicabilityAssessment:
    fallback = _rule_memory_assessment(hit)
    payload = {
        "case_state": case_state.to_dict(),
        "memory_query": memory_query.to_dict(),
        "retrieved_memory": {
            "memory_id": hit.memory_id,
            "memory_type": hit.memory_type,
            "score": hit.score,
            "content": hit.content,
        },
        "allowed_decisions": ["apply", "hint", "block", "ignore"],
        "allowed_actions": DEFAULT_ACTIONS,
        "instruction": (
            "Judge whether this memory applies to the current case. "
            "Retrieved skill should not automatically apply; use apply only when boundary conditions fit. "
            "Failure experiences should discourage; unsafe experiences may block only when highly applicable."
        ),
    }
    prompt = applicability_prompt(payload)
    if debug is not None:
        debug["mode"] = "llm"
        debug["hit"] = payload["retrieved_memory"]
        debug["rule_fallback_assessment"] = fallback.to_dict()
        debug["llm_available"] = llm_client.available()
        debug["payload"] = payload
        debug["prompt"] = prompt
    if not llm_client.available():
        if debug is not None:
            debug["used_fallback"] = True
            debug["fallback_reason"] = "llm_unavailable"
            debug["final_assessment"] = fallback.to_dict()
        return fallback

    raw_output = llm_client.generate_json(prompt, max_tokens=900)
    parsed, _, _ = parse_validate_repair(
        raw_output,
        APPLICABILITY_SCHEMA,
        fallback.to_dict(),
    )
    if debug is not None:
        debug["raw_output"] = raw_output
        debug["parsed_output"] = parsed
    try:
        assessment = MemoryApplicabilityAssessment.from_dict(parsed)
    except Exception as exc:
        logger.warning(
            "LLM applicability assessment parse failed for memory %s: %s",
            hit.memory_id, exc,
        )
        if debug is not None:
            debug["used_fallback"] = True
            debug["fallback_reason"] = f"parse_exception: {exc}"
            debug["final_assessment"] = fallback.to_dict()
        return fallback
    result = _postprocess_llm_assessment(hit, assessment)
    if debug is not None:
        debug["used_fallback"] = False
        debug["final_assessment"] = result.to_dict()
    return result


def _hard_block_actions(case_state: CaseState) -> tuple[list[str], str]:
    blocked: list[str] = []
    warnings: list[str] = []

    warning_text = APPLICABILITY_CONFIG.get("risk_warning_text", {})

    if (
        APPLICABILITY_CONFIG.get("hard_block_finalize_on_high_risk", True)
        and case_state.finalize_risk == "high"
    ):
        blocked.append(FINALIZE_ACTION)
        warnings.append(str(warning_text.get("high_finalize_risk", "finalize_risk is high")))

    missing_info_min = int(APPLICABILITY_CONFIG.get("hard_block_finalize_missing_info_min", 3))
    if missing_info_min > 0 and len(case_state.missing_info) >= missing_info_min:
        if FINALIZE_ACTION not in blocked:
            blocked.append(FINALIZE_ACTION)
        warnings.append(str(warning_text.get("missing_info", "multiple critical missing information slots remain unresolved")))

    if (
        APPLICABILITY_CONFIG.get("image_unreviewed_warning", True)
        and "image" in case_state.modality_flags
        and "image" not in case_state.reviewed_modalities
    ):
        warnings.append(str(warning_text.get("image_unreviewed", "image modality is available but not yet reviewed")))

    return blocked, "; ".join(warnings)


def _aggregate_action_assessments(
    memory_assessments: list[MemoryApplicabilityAssessment],
    hard_blocked_actions: list[str],
) -> list[ActionAssessment]:
    action_map: dict[str, ActionAssessment] = {}

    def ensure(action: str) -> ActionAssessment:
        action = str(action).upper()
        if action not in action_map:
            action_map[action] = ActionAssessment(action_type=action)
        return action_map[action]

    for action in hard_blocked_actions:
        item = ensure(action)
        item.blocked = True
        item.reason = str(APPLICABILITY_CONFIG.get("hard_block_reason", "Configured safety rule blocked this action."))

    for assessment in memory_assessments:
        if assessment.decision == "ignore":
            continue

        for action, delta in assessment.action_bias.items():
            action = str(action).upper()
            if not action:
                continue
            item = ensure(action)
            try:
                item.score_delta += float(delta)
            except Exception:
                pass
            if assessment.decision in {"apply", "hint"} and delta > 0:
                item.supporting_memory_ids.append(assessment.memory_id)
            if delta < 0:
                item.warning_memory_ids.append(assessment.memory_id)

        if assessment.decision == "block":
            for action in assessment.blocked_actions:
                action = str(action).upper()
                if not action:
                    continue
                item = ensure(action)
                item.blocked = True
                item.warning_memory_ids.append(assessment.memory_id)
                item.reason = assessment.reason or "Blocked by retrieved memory."

    for item in action_map.values():
        item.supporting_memory_ids = list(dict.fromkeys(item.supporting_memory_ids))
        item.warning_memory_ids = list(dict.fromkeys(item.warning_memory_ids))
        if not item.reason:
            if item.score_delta > 0:
                item.reason = "Supported by retrieved memory."
            elif item.score_delta < 0:
                item.reason = "Discouraged by retrieved memory."
            else:
                item.reason = "Neutral."

    return sorted(action_map.values(), key=lambda item: item.action_type)


def apply_applicability_control(
    case_state: CaseState,
    memory_query: MemoryQuery,
    retrieval_result: MemoryRetrievalResult,
    mode: str = "rule",
    llm_client: LLMClient | None = None,
    debug: dict[str, object] | None = None,
) -> ApplicabilityResult:
    memory_assessments: list[MemoryApplicabilityAssessment] = []
    if debug is not None:
        debug["mode"] = mode
        debug["case_state"] = case_state.to_dict()
        debug["memory_query"] = memory_query.to_dict()
        debug["retrieval_result"] = retrieval_result.to_dict()
        debug["hit_assessments"] = []

    for hit in _all_hits(retrieval_result):
        hit_debug: dict[str, object] = {}
        if mode in {"llm", "hybrid"} and llm_client is not None:
            assessment = _llm_memory_assessment(
                hit,
                case_state,
                memory_query,
                llm_client,
                debug=hit_debug if debug is not None else None,
            )
        else:
            assessment = _rule_memory_assessment(hit)
            if debug is not None:
                hit_debug["mode"] = "rule"
                hit_debug["hit"] = hit.to_dict()
                hit_debug["final_assessment"] = assessment.to_dict()
        memory_assessments.append(assessment)
        if debug is not None:
            debug["hit_assessments"].append(hit_debug)

    hard_blocked, risk_warning = _hard_block_actions(case_state)
    action_assessments = _aggregate_action_assessments(memory_assessments, hard_blocked)
    if debug is not None:
        debug["hard_blocked_actions"] = hard_blocked
        debug["risk_warning"] = risk_warning

    result = ApplicabilityResult(
        memory_assessments=memory_assessments,
        action_assessments=action_assessments,
        hard_blocked_actions=hard_blocked,
        risk_warning=risk_warning,
    )
    if debug is not None:
        debug["final_applicability_result"] = result.to_dict()
    return result
