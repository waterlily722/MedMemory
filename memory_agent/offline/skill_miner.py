from __future__ import annotations

import uuid

from ..llm import LLMClient, SKILL_MINER_SCHEMA, parse_validate_repair, skill_mining_prompt
from ..schemas import ExperienceCard, SkillCard, SkillProcedureStep
from .skill_abstractor import promote_experiences_to_skill


def mine_skill_rule(experiences: list[ExperienceCard]) -> SkillCard | None:
    return promote_experiences_to_skill(experiences)


def mine_skill_llm(experiences: list[ExperienceCard], llm_client: LLMClient) -> SkillCard | None:
    if not experiences:
        return None

    fallback = {
        "memory_type": "skill",
        "skill_id": f"skill_{uuid.uuid4().hex[:10]}",
        "source_experience_ids": [x.item_id for x in experiences],
        "skill_name": "",
        "clinical_situation": experiences[0].situation_anchor,
        "local_goal": experiences[0].local_goal,
        "trigger_conditions": [experiences[0].local_goal],
        "procedure": [],
        "stop_conditions": [],
        "success_criteria": [],
        "failure_modes": [],
        "contraindications": [],
        "required_modalities": [],
        "applicability_boundary": experiences[0].boundary,
        "evidence_count": len(experiences),
        "confidence": 0.6,
        "version": 1,
    }

    prompt = skill_mining_prompt({"experiences": [x.to_dict() for x in experiences]})
    parsed, _, _ = parse_validate_repair(llm_client.generate_json(prompt), SKILL_MINER_SCHEMA, fallback)
    steps = []
    for idx, step in enumerate(parsed.get("procedure", []), start=1):
        if not isinstance(step, dict):
            continue
        steps.append(
            SkillProcedureStep(
                step_id=idx,
                action_type=str(step.get("action_type", "")),
                action_label=str(step.get("action_template", step.get("action_label", ""))),
                expected_observation=str(step.get("expected_information_gain", "")),
                fallback_action="REQUEST_EXAM",
            )
        )

    return SkillCard(
        skill_id=str(parsed.get("skill_id") or fallback["skill_id"]),
        skill_name=str(parsed.get("skill_name", "")),
        skill_trigger=str(parsed.get("clinical_situation", "")),
        clinical_goal=str(parsed.get("local_goal", "")),
        preconditions=[str(x) for x in parsed.get("trigger_conditions", [])],
        procedure_template=steps,
        stop_condition=[str(x) for x in parsed.get("stop_conditions", [])],
        boundary=[str(parsed.get("applicability_boundary", ""))],
        contraindications=[str(x) for x in parsed.get("contraindications", [])],
        source_experience_ids=[str(x) for x in parsed.get("source_experience_ids", [])],
        support_count=int(parsed.get("evidence_count", len(experiences))),
        success_rate=0.8,
        unsafe_rate=0.0,
        confidence=float(parsed.get("confidence", 0.6)),
        source_case_ids=[],
        source_field_refs=[],
    )
