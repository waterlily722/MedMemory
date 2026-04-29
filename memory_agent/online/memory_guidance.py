from __future__ import annotations

from ..schemas import ApplicabilityResult, MemoryGuidance


def build_memory_guidance(applicability: ApplicabilityResult) -> MemoryGuidance:
    recommended = [assessment.action_type for assessment in applicability.action_assessments if not assessment.blocked and assessment.score_delta >= 0.1]
    discouraged = [assessment.action_type for assessment in applicability.action_assessments if not assessment.blocked and assessment.score_delta < 0.1]
    blocked = list(dict.fromkeys(applicability.hard_blocked_actions))
    used_memory_ids = []
    warning_memory_ids = []
    for assessment in applicability.action_assessments:
        if assessment.blocked:
            warning_memory_ids.extend(assessment.warning_memory_ids)
        if assessment.score_delta > 0:
            used_memory_ids.extend(assessment.supporting_memory_ids)
    rationale = "; ".join(assessment.reason for assessment in applicability.action_assessments if assessment.score_delta > 0) or "No strong memory support."
    why_not_finalize = "Finalize is blocked by hard safety rules." if "FINALIZE_DIAGNOSIS" in blocked else ""
    return MemoryGuidance(
        recommended_actions=list(dict.fromkeys(recommended)),
        discouraged_actions=list(dict.fromkeys(discouraged)),
        blocked_actions=list(dict.fromkeys(blocked)),
        used_memory_ids=list(dict.fromkeys(used_memory_ids)),
        warning_memory_ids=list(dict.fromkeys(warning_memory_ids)),
        rationale=rationale,
        risk_warning=applicability.risk_warning,
        why_not_finalize=why_not_finalize,
    )


def guidance_to_text(guidance: MemoryGuidance) -> str:
    lines = ["Memory guidance:"]
    if guidance.recommended_actions:
        lines.append("Recommended: " + ", ".join(guidance.recommended_actions))
    if guidance.discouraged_actions:
        lines.append("Discouraged: " + ", ".join(guidance.discouraged_actions))
    if guidance.blocked_actions:
        lines.append("Blocked: " + ", ".join(guidance.blocked_actions))
    if guidance.used_memory_ids:
        lines.append("Used memories: " + ", ".join(guidance.used_memory_ids))
    if guidance.warning_memory_ids:
        lines.append("Warnings: " + ", ".join(guidance.warning_memory_ids))
    if guidance.risk_warning:
        lines.append("Risk: " + guidance.risk_warning)
    if guidance.why_not_finalize:
        lines.append("Why not finalize: " + guidance.why_not_finalize)
    lines.append("Rationale: " + guidance.rationale)
    return "\n".join(lines)
