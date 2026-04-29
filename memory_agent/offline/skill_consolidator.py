from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..llm import LLMClient, parse_validate_repair, skill_consolidation_prompt
from ..llm.schemas import SKILL_SCHEMA
from ..memory_store import ExperienceMemoryStore, SkillMemoryStore
from ..online.retriever import DEFAULT_MEMORY_ROOT
from ..schemas import ExperienceCard, SkillCard
from ..utils.config import SKILL_CONFIG
from ..utils.scoring import cosine_similarity


def _root(memory_root: str | None) -> Path:
    root = Path(memory_root) if memory_root else DEFAULT_MEMORY_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cluster_experiences(experiences: list[ExperienceCard]) -> list[list[ExperienceCard]]:
    clusters: list[list[ExperienceCard]] = []
    for experience in experiences:
        placed = False
        for cluster in clusters:
            seed = cluster[0]
            if (
                cosine_similarity(seed.situation_anchor, experience.situation_anchor) >= 0.8
                and cosine_similarity(seed.local_goal, experience.local_goal) >= 0.8
                and cosine_similarity(" ".join(step.get("action_label", step.get("action_type", "")) for step in seed.action_sequence), " ".join(step.get("action_label", step.get("action_type", "")) for step in experience.action_sequence)) >= 0.8
            ):
                cluster.append(experience)
                placed = True
                break
        if not placed:
            clusters.append([experience])
    return clusters


def _build_skill_from_cluster(cluster: list[ExperienceCard], skill_index: int, llm_client: LLMClient | None = None, mode: str = "rule") -> SkillCard | None:
    support_count = sum(item.support_count for item in cluster)
    unique_cases = {case_id for item in cluster for case_id in item.source_case_ids if case_id}
    success_count = sum(1 for item in cluster if item.outcome_type in {"success", "partial_success"})
    unsafe_count = sum(1 for item in cluster if item.outcome_type == "unsafe")
    success_rate = success_count / max(1, len(cluster))
    unsafe_rate = unsafe_count / max(1, len(cluster))
    if support_count < SKILL_CONFIG["min_support_count"]:
        return None
    if len(unique_cases) < SKILL_CONFIG["min_unique_cases"]:
        return None
    if success_rate < SKILL_CONFIG["min_success_rate"]:
        return None
    if unsafe_rate > SKILL_CONFIG["max_unsafe_rate"]:
        return None

    seed = cluster[0]
    rule_skill = SkillCard(
        memory_id=f"skill_{seed.memory_id}_{skill_index}",
        skill_name=f"skill_{skill_index}",
        clinical_situation=seed.situation_anchor,
        local_goal=seed.local_goal,
        trigger_conditions=list(dict.fromkeys(seed.active_hypotheses + [seed.situation_anchor]))[:6],
        procedure=seed.action_sequence,
        stop_conditions=["risk low", "uncertainty reduced"],
        success_criteria=["goal achieved"],
        failure_modes=["unsafe finalization"],
        contraindications=["high finalize risk"],
        required_modalities=list(dict.fromkeys(seed.modality_flags)),
        applicability_boundary=seed.boundary,
        source_experience_ids=[item.memory_id for item in cluster],
        evidence_count=support_count,
        confidence=min(0.99, round(success_rate * (1.0 - unsafe_rate), 4)),
        version=1,
    )

    if mode != "llm" or llm_client is None or not llm_client.available():
        return rule_skill

    payload = {"experiences": [item.to_dict() for item in cluster], "candidate_skill": rule_skill.to_dict()}
    fallback = rule_skill.to_dict()
    parsed, _, _ = parse_validate_repair(llm_client.generate_json(skill_consolidation_prompt(payload)), SKILL_SCHEMA, fallback)
    return SkillCard.from_dict(parsed)


def consolidate_skills_from_store(memory_root: str | None, mode: str = "rule", llm_client: LLMClient | None = None) -> list[SkillCard]:
    root = _root(memory_root)
    exp_store = ExperienceMemoryStore(root)
    skill_store = SkillMemoryStore(root)
    experiences = [item for item in exp_store.list_all() if item.outcome_type in {"success", "partial_success"}]
    if len(experiences) < SKILL_CONFIG["min_support_count"]:
        return []

    clusters = _cluster_experiences(experiences)
    skills: list[SkillCard] = []
    for index, cluster in enumerate(clusters, start=1):
        skill = _build_skill_from_cluster(cluster, index, llm_client=llm_client, mode=mode)
        if skill is None:
            continue
        skill_store.upsert(skill)
        skills.append(skill)
    return skills


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Consolidate skills from cross-episode successful experiences.")
    parser.add_argument("--memory_root", default=None)
    parser.add_argument("--mode", default="rule", choices=["rule", "llm"])
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    skills = consolidate_skills_from_store(args.memory_root, mode=args.mode, llm_client=LLMClient())
    print(f"generated={len(skills)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
