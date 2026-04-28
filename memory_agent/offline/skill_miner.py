from __future__ import annotations

import uuid

from ..llm import LLMClient, SKILL_MINER_SCHEMA, parse_validate_repair, skill_mining_prompt
from ..schemas import ExperienceCard, SkillCard, SkillProcedureStep
from .skill_abstractor import promote_experiences_to_skill
from ..utils.scoring import cosine_similarity


def mine_skill_rule(experiences: list[ExperienceCard]) -> SkillCard | None:
    successful = [item for item in experiences if item.outcome_type in {"success", "partial_success"} and item.confidence >= 0.5]
    if not successful:
        return None

    clusters: list[list[ExperienceCard]] = []
    for experience in successful:
        placed = False
        for cluster in clusters:
            seed = cluster[0]
            same_goal = seed.local_goal == experience.local_goal or cosine_similarity(seed.local_goal, experience.local_goal) > 0.85
            same_anchor = cosine_similarity(seed.situation_anchor, experience.situation_anchor) > 0.82
            same_action = [step.get("action_label", "") for step in seed.action_sequence[:2]] == [step.get("action_label", "") for step in experience.action_sequence[:2]]
            if same_goal and same_anchor and same_action:
                cluster.append(experience)
                placed = True
                break
        if not placed:
            clusters.append([experience])

    best_cluster = max(clusters, key=lambda cluster: (len(cluster), sum(item.support_count for item in cluster), sum(item.confidence for item in cluster)))
    if len(best_cluster) < 2:
        return None
    return promote_experiences_to_skill(best_cluster)


def mine_skill_llm(experiences: list[ExperienceCard], llm_client: LLMClient) -> SkillCard | None:
    if not experiences:
        return None

    clustered = [item for item in experiences if item.outcome_type in {"success", "partial_success"}]
    if len(clustered) < 2:
        return None

    fallback = {
        "memory_type": "skill",
        "skill_id": f"skill_{uuid.uuid4().hex[:10]}",
        "source_experience_ids": [x.item_id for x in clustered],
        "skill_name": "",
        "clinical_situation": clustered[0].situation_anchor,
        "local_goal": clustered[0].local_goal,
        "trigger_conditions": [clustered[0].local_goal],
        "procedure": [],
        "stop_conditions": [],
        "success_criteria": [],
        "failure_modes": [],
        "contraindications": [],
        "required_modalities": [],
        "applicability_boundary": clustered[0].boundary,
        "evidence_count": len(clustered),
        "confidence": 0.6,
        "version": 1,
    }

    prompt = skill_mining_prompt({"experiences": [x.to_dict() for x in clustered]})
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
        required_modalities=[str(x) for x in parsed.get("required_modalities", [])],
        source_experience_ids=[str(x) for x in parsed.get("source_experience_ids", [])],
        support_count=int(parsed.get("evidence_count", len(clustered))),
        success_rate=0.8,
        unsafe_rate=0.0,
        confidence=float(parsed.get("confidence", 0.6)),
        skill_pattern_id=str(parsed.get("skill_pattern_id", "")),
        cross_episode_support_count=int(parsed.get("evidence_count", len(clustered))),
        source_case_ids=[],
        source_field_refs=[],
    )
