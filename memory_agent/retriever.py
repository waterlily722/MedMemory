from __future__ import annotations

from pathlib import Path
from typing import Any

from .schemas import MemoryRetrievalResult, RetrievalHit
from .stores.experience_store import ExperienceStore
from .stores.guardrail_store import GuardrailStore
from .stores.skill_store import SkillStore


DEFAULT_MEMORY_ROOT = Path(__file__).resolve().parent / "store_data"


def _store_root(root: str | None = None) -> Path:
    base = Path(root) if root else DEFAULT_MEMORY_ROOT
    base.mkdir(parents=True, exist_ok=True)
    return base


def _as_hit(row) -> RetrievalHit:
    item, score, matched_fields = row
    return RetrievalHit(
        item_id=getattr(item, "item_id", ""),
        retrieval_score=round(float(score), 4),
        matched_fields=matched_fields,
        source_field_refs=getattr(item, "source_field_refs", []) or [],
    )


def retrieve_experience(query_signature, top_k: int = 10, root_dir: str | None = None):
    store = ExperienceStore(_store_root(root_dir))
    return [_as_hit(row) for row in store.search(query_signature, top_k=top_k)]


def retrieve_skill(query_signature, top_k: int = 10, root_dir: str | None = None):
    store = SkillStore(_store_root(root_dir))
    return [_as_hit(row) for row in store.search(query_signature, top_k=top_k)]


def retrieve_guardrail(query_signature, top_k: int = 10, root_dir: str | None = None):
    store = GuardrailStore(_store_root(root_dir))
    return [_as_hit(row) for row in store.search(query_signature, top_k=top_k)]


def retrieve_all(query_signature, top_k_each: int = 10, root_dir: str | None = None) -> MemoryRetrievalResult:
    return MemoryRetrievalResult(
        turn_id=str(query_signature.get("turn_id", "")),
        query_signature=query_signature,
        experience_hits=retrieve_experience(query_signature, top_k=top_k_each, root_dir=root_dir),
        skill_hits=retrieve_skill(query_signature, top_k=top_k_each, root_dir=root_dir),
        guardrail_hits=retrieve_guardrail(query_signature, top_k=top_k_each, root_dir=root_dir),
        source_field_refs=[],
    )
