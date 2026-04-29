from __future__ import annotations

import argparse
from pathlib import Path

from ..llm import LLMClient
from ..memory_store import ExperienceMemoryStore, SkillMemoryStore
from ..online.retriever import DEFAULT_MEMORY_ROOT
from .memory_consolidator import consolidate_skill


def consolidate_skills(memory_root: str | None = None, mode: str = "rule") -> dict[str, str | int]:
    root = Path(memory_root) if memory_root else DEFAULT_MEMORY_ROOT
    root.mkdir(parents=True, exist_ok=True)

    exp_store = ExperienceMemoryStore(root)
    skill_store = SkillMemoryStore(root)

    experiences = exp_store.list_items()
    successful = [item for item in experiences if item.outcome_type in {"success", "partial_success"}]
    unique_episode_ids = {eid for item in successful for eid in (item.source_episode_ids or [])}
    if len(unique_episode_ids) < 2:
        return {
            "status": "skipped",
            "reason": "not enough cross-episode successes",
            "experience_count": len(successful),
            "skill_count": len(skill_store.list_items()),
        }

    llm_client = LLMClient() if mode == "llm" else None
    if mode == "llm" and (llm_client is None or not llm_client.available()):
        return {
            "status": "skipped",
            "reason": "llm client unavailable",
            "experience_count": len(successful),
            "skill_count": len(skill_store.list_items()),
        }

    promoted = consolidate_skill(
        clustered_success_experiences=successful,
        mode=mode,
        llm_client=llm_client,
    )
    if promoted is None:
        return {
            "status": "skipped",
            "reason": "no eligible clusters",
            "experience_count": len(successful),
            "skill_count": len(skill_store.list_items()),
        }

    skill_store.upsert(promoted)
    return {
        "status": "ok",
        "skill_id": promoted.skill_id,
        "experience_count": len(successful),
        "skill_count": len(skill_store.list_items()),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Consolidate skills from cross-episode experiences.")
    parser.add_argument("--memory_root", default=None, help="Root directory for memory stores.")
    parser.add_argument("--mode", default="rule", choices=["rule", "llm"], help="Skill consolidation mode.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = consolidate_skills(memory_root=args.memory_root, mode=args.mode)
    status = str(result.get("status", ""))
    if status == "ok":
        print(f"skill_id={result.get('skill_id')} skill_count={result.get('skill_count')}")
        return 0
    print(f"status={status} reason={result.get('reason')}")
    return 1 if status == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())