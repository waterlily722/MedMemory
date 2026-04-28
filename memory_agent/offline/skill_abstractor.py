from __future__ import annotations

import uuid

from ..schemas import ExperienceCard, SkillCard, SkillProcedureStep
from ..utils.config import SKILL_ABSTRACTION_CONFIG


def promote_experiences_to_skill(experiences: list[ExperienceCard]) -> SkillCard | None:
    if not experiences:
        return None

    support = sum(x.support_count for x in experiences)
    success_count = sum(1 for x in experiences if x.outcome_type in {"success", "partial_success"})
    unsafe_count = sum(1 for x in experiences if x.outcome_type == "unsafe")
    success_rate = success_count / max(1, len(experiences))
    unsafe_rate = unsafe_count / max(1, len(experiences))
    unique_cases = {cid for x in experiences for cid in x.source_case_ids}

    if support < SKILL_ABSTRACTION_CONFIG["min_support_count"]:
        return None
    if success_rate < SKILL_ABSTRACTION_CONFIG["min_success_rate"]:
        return None
    if unsafe_rate > SKILL_ABSTRACTION_CONFIG["max_unsafe_rate"]:
        return None
    if len(unique_cases) < SKILL_ABSTRACTION_CONFIG["min_cross_case_support"]:
        return None

    seed = experiences[0]
    steps = []
    for idx, action in enumerate(seed.action_sequence[:3], start=1):
        steps.append(
            SkillProcedureStep(
                step_id=idx,
                action_type=action.get("action_type", ""),
                action_label=action.get("action_label", ""),
                expected_observation="information gain improves",
                fallback_action="REQUEST_EXAM",
                source_field_refs=seed.source_field_refs,
            )
        )

    return SkillCard(
        skill_id=f"skill_{uuid.uuid4().hex[:10]}",
        skill_name=f"abstracted_{steps[0].action_type.lower() if steps else 'clinical'}",
        skill_trigger=seed.situation_anchor,
        clinical_goal=seed.local_goal,
        preconditions=[seed.local_goal],
        procedure_template=steps,
        stop_condition=["uncertainty reduced", "finalize risk low"],
        boundary=[seed.boundary],
        contraindications=[],
        source_experience_ids=[x.item_id for x in experiences],
        support_count=support,
        success_rate=round(success_rate, 4),
        unsafe_rate=round(unsafe_rate, 4),
        confidence=round(min(0.99, success_rate * (1.0 - unsafe_rate)), 4),
        source_case_ids=list(unique_cases),
        source_field_refs=seed.source_field_refs,
    )


def should_deactivate_skill(skill: SkillCard, recent_failures: int, boundary_conflicts: int, reject_count: int) -> bool:
    if recent_failures >= 3:
        return True
    if skill.unsafe_rate > SKILL_ABSTRACTION_CONFIG["max_unsafe_rate"]:
        return True
    if boundary_conflicts >= 3:
        return True
    if reject_count >= 5:
        return True
    return False
