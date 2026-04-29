from __future__ import annotations

from ..schemas import ApplicabilityResult, MemoryGuidance


def build_memory_guidance(
    applicability_result: ApplicabilityResult,
) -> MemoryGuidance:
    recommended: list[str] = []
    discouraged: list[str] = []
    blocked: list[str] = list(applicability_result.hard_blocked_actions)

    used_memory_ids: list[str] = []
    warning_memory_ids: list[str] = []

    rationale_parts: list[str] = []

    for action in applicability_result.action_assessments:
        if action.blocked:
            blocked.append(action.action_type)
        elif action.score_delta > 0:
            recommended.append(action.action_type)
        elif action.score_delta < 0:
            discouraged.append(action.action_type)

        used_memory_ids.extend(action.supporting_memory_ids)
        warning_memory_ids.extend(action.warning_memory_ids)

        if action.reason:
            rationale_parts.append(f"{action.action_type}: {action.reason}")

    for assessment in applicability_result.memory_assessments:
        if assessment.decision in {"apply", "hint"}:
            used_memory_ids.append(assessment.memory_id)
        if assessment.decision == "block":
            warning_memory_ids.append(assessment.memory_id)

    blocked = list(dict.fromkeys(blocked))
    recommended = [item for item in dict.fromkeys(recommended) if item not in blocked]
    discouraged = [item for item in dict.fromkeys(discouraged) if item not in blocked]
    used_memory_ids = list(dict.fromkeys(used_memory_ids))
    warning_memory_ids = list(dict.fromkeys(warning_memory_ids))

    why_not_finalize = ""
    if "FINALIZE_DIAGNOSIS" in blocked:
        why_not_finalize = (
            applicability_result.risk_warning
            or "Diagnosis finalization is blocked by safety rules or negative memory."
        )

    return MemoryGuidance(
        recommended_actions=recommended,
        discouraged_actions=discouraged,
        blocked_actions=blocked,
        used_memory_ids=used_memory_ids,
        warning_memory_ids=warning_memory_ids,
        rationale=" ".join(rationale_parts[:6]),
        risk_warning=applicability_result.risk_warning,
        why_not_finalize=why_not_finalize,
    )


def guidance_to_text(guidance: MemoryGuidance) -> str:
    parts: list[str] = []

    if guidance.recommended_actions:
        parts.append(
            "Recommended actions: "
            + ", ".join(guidance.recommended_actions)
        )

    if guidance.discouraged_actions:
        parts.append(
            "Discouraged actions: "
            + ", ".join(guidance.discouraged_actions)
        )

    if guidance.blocked_actions:
        parts.append(
            "Blocked actions: "
            + ", ".join(guidance.blocked_actions)
        )

    if guidance.risk_warning:
        parts.append(f"Risk warning: {guidance.risk_warning}")

    if guidance.why_not_finalize:
        parts.append(f"Why not finalize: {guidance.why_not_finalize}")

    if guidance.used_memory_ids:
        parts.append(
            "Used memory ids: "
            + ", ".join(guidance.used_memory_ids[:8])
        )

    if guidance.warning_memory_ids:
        parts.append(
            "Warning memory ids: "
            + ", ".join(guidance.warning_memory_ids[:8])
        )

    return "\n".join(parts)