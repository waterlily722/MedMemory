from __future__ import annotations

import argparse
from pathlib import Path

from ..llm import LLMClient, parse_validate_repair, skill_consolidation_prompt
from ..llm.schemas import SKILL_SCHEMA
from ..memory_store import ExperienceMemoryStore, SkillMemoryStore
from ..schemas import ExperienceCard, SkillCard
from ..utils.config import MEMORY_ROOT_DIRNAME, SKILL_CONFIG
from ..utils.scoring import cosine_similarity


def _root(memory_root: str | None) -> Path:
    root = Path(memory_root) if memory_root else Path(MEMORY_ROOT_DIRNAME)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _is_positive(experience: ExperienceCard) -> bool:
    return experience.outcome_type in {"success", "partial_success"}


def _is_unsafe(experience: ExperienceCard) -> bool:
    return experience.outcome_type == "unsafe"


def _cluster_positive_experiences(
    experiences: list[ExperienceCard],
) -> list[list[ExperienceCard]]:
    clusters: list[list[ExperienceCard]] = []

    for exp in experiences:
        if not _is_positive(exp):
            continue

        placed = False
        for cluster in clusters:
            seed = cluster[0]
            if (
                cosine_similarity(seed.situation_text, exp.situation_text) >= 0.80
                and cosine_similarity(seed.action_text, exp.action_text) >= 0.75
            ):
                cluster.append(exp)
                placed = True
                break

        if not placed:
            clusters.append([exp])

    return clusters


def _matching_unsafe_support(
    cluster: list[ExperienceCard],
    all_experiences: list[ExperienceCard],
) -> int:
    seed = cluster[0]
    support = 0

    for exp in all_experiences:
        if not _is_unsafe(exp):
            continue
        if (
            cosine_similarity(seed.situation_text, exp.situation_text) >= 0.80
            and cosine_similarity(seed.action_text, exp.action_text) >= 0.75
        ):
            support += max(1, exp.support_count)

    return support


def _unique_case_count(cluster: list[ExperienceCard]) -> int:
    case_ids = {
        case_id
        for exp in cluster
        for case_id in exp.source_case_ids
        if case_id
    }
    return len(case_ids)


def _source_experience_ids(cluster: list[ExperienceCard]) -> list[str]:
    seen = set()
    ids = []
    for exp in cluster:
        if exp.memory_id in seen:
            continue
        seen.add(exp.memory_id)
        ids.append(exp.memory_id)
    return ids


def _build_rule_skill(
    cluster: list[ExperienceCard],
    all_experiences: list[ExperienceCard],
    skill_index: int,
) -> SkillCard | None:
    support_count = sum(max(1, item.support_count) for item in cluster)
    unsafe_support = _matching_unsafe_support(cluster, all_experiences)
    total_support = max(1, support_count + unsafe_support)

    success_rate = support_count / total_support
    unsafe_rate = unsafe_support / total_support
    unique_cases = _unique_case_count(cluster)

    if support_count < SKILL_CONFIG["min_support_count"]:
        return None
    if unique_cases < SKILL_CONFIG["min_unique_cases"]:
        return None
    if success_rate < SKILL_CONFIG["min_success_rate"]:
        return None
    if unsafe_rate > SKILL_CONFIG["max_unsafe_rate"]:
        return None

    seed = cluster[0]

    return SkillCard(
        memory_id=f"skill_{seed.memory_id}_{skill_index}",
        skill_name=f"skill_{skill_index}",
        situation_text=seed.situation_text,
        goal_text="Select a high-yield local action path before finalizing diagnosis.",
        procedure_text=seed.action_text,
        boundary_text=seed.boundary_text,
        procedure=seed.action_sequence,
        contraindications=[],
        source_experience_ids=_source_experience_ids(cluster),
        evidence_count=support_count,
        unique_case_count=unique_cases,
        success_rate=round(success_rate, 4),
        unsafe_rate=round(unsafe_rate, 4),
        confidence=min(0.99, round(success_rate * (1.0 - unsafe_rate), 4)),
        version=1,
    )


def _build_skill_from_cluster(
    cluster: list[ExperienceCard],
    all_experiences: list[ExperienceCard],
    skill_index: int,
    mode: str = "rule",
    llm_client: LLMClient | None = None,
) -> SkillCard | None:
    rule_skill = _build_rule_skill(cluster, all_experiences, skill_index)
    if rule_skill is None:
        return None

    if mode != "llm" or llm_client is None or not llm_client.available():
        return rule_skill

    payload = {
        "experiences": [item.to_dict() for item in cluster],
        "candidate_skill": rule_skill.to_dict(),
        "instruction": (
            "Refine the candidate skill using only the repeated successful experiences. "
            "Do not create a skill from one episode. "
            "Keep the schema exactly."
        ),
    }

    parsed, _, _ = parse_validate_repair(
        llm_client.generate_json(skill_consolidation_prompt(payload), max_tokens=1800),
        SKILL_SCHEMA,
        rule_skill.to_dict(),
    )

    return SkillCard.from_dict(parsed)


def consolidate_skills_from_store(
    memory_root: str | None,
    mode: str = "rule",
    llm_client: LLMClient | None = None,
) -> list[SkillCard]:
    root = _root(memory_root)

    exp_store = ExperienceMemoryStore(root)
    skill_store = SkillMemoryStore(root)

    all_experiences = exp_store.list_all()
    clusters = _cluster_positive_experiences(all_experiences)

    skills: list[SkillCard] = []
    for index, cluster in enumerate(clusters, start=1):
        skill = _build_skill_from_cluster(
            cluster=cluster,
            all_experiences=all_experiences,
            skill_index=index,
            mode=mode,
            llm_client=llm_client,
        )
        if skill is None:
            continue

        skill_store.upsert(skill)
        skills.append(skill)

    return skills


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Consolidate skills from cross-episode successful experiences."
    )
    parser.add_argument("--memory_root", default=None)
    parser.add_argument("--mode", default="rule", choices=["rule", "llm"])
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    llm_client = LLMClient() if args.mode == "llm" else None
    skills = consolidate_skills_from_store(
        memory_root=args.memory_root,
        mode=args.mode,
        llm_client=llm_client,
    )
    print(f"generated={len(skills)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())