"""Tests for JsonMemoryStore and its subclasses."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PACKAGE = _HERE.parent
if str(_PACKAGE) not in sys.path:
    sys.path.insert(0, str(_PACKAGE))

from memory_agent.memory_store.base_store import JsonMemoryStore
from memory_agent.memory_store import ExperienceMemoryStore, SkillMemoryStore, KnowledgeMemoryStore
from memory_agent.schemas import ExperienceCard, SkillCard, KnowledgeItem


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _card_factory(data: dict) -> ExperienceCard:
    return ExperienceCard.from_dict(data)


def _make_store(tmp: str):
    return JsonMemoryStore(tmp, "test.jsonl", _card_factory)


# ═══════════════════════════════════════════════════════════════════════
# JsonMemoryStore
# ═══════════════════════════════════════════════════════════════════════

def test_append_and_list():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        card = ExperienceCard(memory_id="exp_001", situation_text="test")
        store.append(card)
        all_items = store.list_all()
        assert len(all_items) == 1
        assert all_items[0].memory_id == "exp_001"


def test_append_duplicate_raises():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        store.append(ExperienceCard(memory_id="exp_001"))
        try:
            store.append(ExperienceCard(memory_id="exp_001"))
            assert False, "Expected ValueError"
        except ValueError:
            pass


def test_upsert_insert():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        store.upsert(ExperienceCard(memory_id="exp_001"))
        assert len(store.list_all()) == 1


def test_upsert_update():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        store.append(ExperienceCard(memory_id="exp_001", situation_text="old"))
        store.upsert(ExperienceCard(memory_id="exp_001", situation_text="new"))
        found = store.find_by_id("exp_001")
        assert found is not None
        assert found.situation_text == "new"


def test_find_by_id():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        store.append(ExperienceCard(memory_id="exp_001"))
        found = store.find_by_id("exp_001")
        assert found is not None
        assert found.memory_id == "exp_001"
        assert store.find_by_id("nonexistent") is None


def test_find_by_id_empty_string():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        assert store.find_by_id("") is None


def test_clear():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        store.append(ExperienceCard(memory_id="exp_001"))
        store.append(ExperienceCard(memory_id="exp_002"))
        store.clear()
        assert store.list_all() == []


def test_persistence():
    """Data survives store instance recreation."""
    with tempfile.TemporaryDirectory() as tmp:
        s1 = JsonMemoryStore(tmp, "test.jsonl", _card_factory)
        s1.append(ExperienceCard(memory_id="persist_001"))
        s2 = JsonMemoryStore(tmp, "test.jsonl", _card_factory)
        assert len(s2.list_all()) == 1
        assert s2.find_by_id("persist_001") is not None


def test_non_strict_skips_bad_lines():
    """When strict_json=False, bad lines are silently skipped."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.jsonl"
        path.write_text(
            '{"memory_id": "good", "memory_type": "experience"}\n'
            'not json\n'
            '{"memory_id": "also_good", "memory_type": "experience"}\n',
            encoding="utf-8",
        )
        store = JsonMemoryStore(tmp, "test.jsonl", _card_factory, strict_json=False)
        items = store.list_all()
        assert len(items) == 2


def test_strict_raises_on_bad_lines():
    """When strict_json=True, bad lines raise ValueError."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.jsonl"
        path.write_text(
            '{"memory_id": "good"}\n' 'not json\n',
            encoding="utf-8",
        )
        store = JsonMemoryStore(tmp, "test.jsonl", _card_factory, strict_json=True)
        try:
            store.list_all()
            assert False, "Expected ValueError"
        except ValueError:
            pass


# ═══════════════════════════════════════════════════════════════════════
# ExperienceMemoryStore
# ═══════════════════════════════════════════════════════════════════════

def test_experience_store():
    with tempfile.TemporaryDirectory() as tmp:
        s = ExperienceMemoryStore(tmp)
        card = ExperienceCard(
            memory_id="exp_001",
            situation_text="chest pain",
            action_text="order ECG",
            outcome_type="success",
            boundary_text="use when cardiac risk",
        )
        s.append(card)
        loaded = s.list_all()
        assert len(loaded) == 1
        assert loaded[0].situation_text == "chest pain"


# ═══════════════════════════════════════════════════════════════════════
# SkillMemoryStore
# ═══════════════════════════════════════════════════════════════════════

def test_skill_store():
    with tempfile.TemporaryDirectory() as tmp:
        s = SkillMemoryStore(tmp)
        skill = SkillCard(
            memory_id="skill_001",
            skill_name="chest pain workup",
            situation_text="chest pain",
            goal_text="rule out ACS",
            procedure_text="ECG, troponin",
            boundary_text="use when chest pain",
        )
        s.append(skill)
        loaded = s.list_all()
        assert len(loaded) == 1
        assert loaded[0].skill_name == "chest pain workup"


# ═══════════════════════════════════════════════════════════════════════
# KnowledgeMemoryStore
# ═══════════════════════════════════════════════════════════════════════

def test_knowledge_store():
    with tempfile.TemporaryDirectory() as tmp:
        s = KnowledgeMemoryStore(tmp)
        item = KnowledgeItem(
            memory_id="kn_001",
            content="Pneumonia is a lung infection.",
            tags=["pneumonia", "infection"],
        )
        s.append(item)
        loaded = s.list_all()
        assert len(loaded) == 1
        assert loaded[0].content == "Pneumonia is a lung infection."
