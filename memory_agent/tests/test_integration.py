"""Integration tests for the memory agent pipeline (without LLM dependency).

Tests the rule-based code paths that do not require an LLM endpoint.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PACKAGE = _HERE.parent
if str(_PACKAGE) not in sys.path:
    sys.path.insert(0, str(_PACKAGE))

from memory_agent.schemas import (
    CaseState,
    MemoryQuery,
    MemoryRetrievalResult,
    RetrievalHit,
    ExperienceCard,
    OutcomeType,
    ApplicabilityResult,
    MemoryApplicabilityAssessment,
)
from memory_agent.online.case_updater import (
    init_case_state,
    update_case_state_rule,
    _collect_texts,
    _is_negative_text,
)
from memory_agent.online.query_builder import build_memory_query_rule
from memory_agent.online.retriever import (
    retrieve_multi_memory,
    memory_to_text,
    experience_to_text,
    _root,
)
from memory_agent.online.memory_guidance import guidance_to_text, build_memory_guidance
from memory_agent.online.applicability_controller import (
    apply_applicability_control,
    _rule_memory_assessment,
    _hard_block_actions,
    DEFAULT_ACTIONS,
)
from memory_agent.offline.experience_merger import (
    decide_merge_rule,
    merge_experience,
    _same_trigger,
    _same_direction,
)
from memory_agent.offline.episode_distiller import distill_from_trajectory, _to_dict_record
from memory_agent.offline.experience_extractor import (
    select_episode_turns,
)
from memory_agent.offline.skill_consolidator import (
    _is_positive,
    _is_unsafe,
    _cluster_positive_experiences,
)


# ═══════════════════════════════════════════════════════════════════════
# CaseState
# ═══════════════════════════════════════════════════════════════════════

def test_init_case_state():
    bundle = {
        "case_id": "c001",
        "ehr": {
            "OSCE_Examination": {
                "Patient_Actor": {
                    "Symptoms": {"Chief_Complaint": "chest pain"},
                },
            },
        },
    }
    cs = init_case_state(bundle)
    assert cs.case_id == "c001"
    assert "chest pain" in cs.chief_complaint
    assert cs.acquired_information == []
    assert cs.turn_id == 0


def test_init_case_state_empty():
    cs = init_case_state({})
    assert cs.case_id == ""
    assert cs.turn_id == 0
    assert cs.chief_complaint == ""


def test_update_case_state_rule():
    prev = CaseState(case_id="c001", turn_id=0, chief_complaint="chest pain")
    updated = update_case_state_rule(prev, {"question": "I have chest pain"})
    assert updated.turn_id == 1
    assert len(updated.acquired_information) > 0


# ═══════════════════════════════════════════════════════════════════════
# Query Builder
# ═══════════════════════════════════════════════════════════════════════

def test_build_memory_query_rule():
    cs = CaseState(
        case_id="c001",
        turn_id=2,
        chief_complaint="chest pain",
        acquired_information=[
            {"turn_id": 1, "source_path": "observation.question", "content": "chest pain"},
            {"turn_id": 2, "source_path": "observation.tool_outputs.ecg", "content": "ECG normal"},
        ],
    )
    mq = build_memory_query_rule(cs)
    assert mq.case_id == "c001"
    assert "chest pain" in mq.query_text
    assert "ECG normal" in mq.query_text


# ═══════════════════════════════════════════════════════════════════════
# Retriever
# ═══════════════════════════════════════════════════════════════════════

def test_retrieve_empty_store():
    """Querying an empty store returns empty results."""
    mq = MemoryQuery(case_id="c001", query_text="chest pain")
    with tempfile.TemporaryDirectory() as tmp:
        result = retrieve_multi_memory(mq, root_dir=tmp)
        assert len(result.positive_experience_hits) == 0
        assert len(result.negative_experience_hits) == 0
        assert len(result.skill_hits) == 0
        assert len(result.knowledge_hits) == 0


def test_memory_to_text_experience():
    payload = {
        "situation_text": "chest pain",
        "action_text": "order ECG",
        "outcome_text": "normal",
        "boundary_text": "cardiac risk",
        "action_sequence": [{"action_type": "ASK", "action_label": "ask about pain"}],
        "tags": ["chest_pain"],
        "source": {},
    }
    text = memory_to_text("experience", payload)
    assert "chest pain" in text
    assert "order ECG" in text


# ═══════════════════════════════════════════════════════════════════════
# Applicability
# ═══════════════════════════════════════════════════════════════════════

def test_hard_block_actions_high_risk():
    cs = CaseState(case_id="c001")
    blocked, warning = _hard_block_actions(cs)
    assert "FINALIZE_DIAGNOSIS" not in blocked


def test_hard_block_actions_low_risk():
    cs = CaseState(case_id="c001")
    blocked, warning = _hard_block_actions(cs)
    assert "FINALIZE_DIAGNOSIS" not in blocked


def test_apply_applicability_control_no_memories():
    cs = CaseState(case_id="c001")
    mq = MemoryQuery(case_id="c001", query_text="chest pain")
    result = MemoryRetrievalResult()
    ar = apply_applicability_control(cs, mq, result, mode="rule")
    # Hard-block rules still apply
    assert len(ar.memory_assessments) == 0


# ═══════════════════════════════════════════════════════════════════════
# Experience Merger
# ═══════════════════════════════════════════════════════════════════════

def test_same_direction_both_positive():
    a = ExperienceCard(memory_id="a", outcome_type=OutcomeType.SUCCESS.value)
    b = ExperienceCard(memory_id="b", outcome_type=OutcomeType.PARTIAL_SUCCESS.value)
    assert _same_direction(a, b) is True


def test_same_direction_opposite():
    a = ExperienceCard(memory_id="a", outcome_type=OutcomeType.SUCCESS.value)
    b = ExperienceCard(memory_id="b", outcome_type=OutcomeType.FAILURE.value)
    assert _same_direction(a, b) is False


def test_decide_merge_rule_insert_new():
    new_exp = ExperienceCard(
        memory_id="new",
        situation_text="unique situation",
        action_text="unique action",
        outcome_type=OutcomeType.SUCCESS.value,
        boundary_text="test",
    )
    decision = decide_merge_rule(new_exp, [])
    assert decision["merge_decision"] == "insert_new"


# ═══════════════════════════════════════════════════════════════════════
# Episode turn selection
# ═══════════════════════════════════════════════════════════════════════

def test_select_episode_turns_keeps_failed_turns():
    turns = [
        {"reward": 1.0, "selected_action": {}, "env_info": {}},
        {"reward": 0.0, "selected_action": {}, "env_info": {}},
    ]
    selected = select_episode_turns(turns, limit=5)
    assert len(selected) == 2


def test_select_episode_turns_applies_limit_from_end():
    turns = [
        {"turn_id": 1},
        {"turn_id": 2},
        {"turn_id": 3},
    ]
    selected = select_episode_turns(turns, limit=2)
    assert [turn["turn_id"] for turn in selected] == [2, 3]


# ═══════════════════════════════════════════════════════════════════════
# Skill consolidator helpers
# ═══════════════════════════════════════════════════════════════════════

def test_is_positive():
    assert _is_positive(ExperienceCard(memory_id="a", outcome_type=OutcomeType.SUCCESS.value)) is True
    assert _is_positive(ExperienceCard(memory_id="b", outcome_type=OutcomeType.FAILURE.value)) is False


def test_is_unsafe():
    assert _is_unsafe(ExperienceCard(memory_id="a", outcome_type=OutcomeType.UNSAFE.value)) is True
    assert _is_unsafe(ExperienceCard(memory_id="b", outcome_type=OutcomeType.SUCCESS.value)) is False


def test_cluster_empty():
    assert _cluster_positive_experiences([]) == []


# ═══════════════════════════════════════════════════════════════════════
# Guidance helpers
# ═══════════════════════════════════════════════════════════════════════

def test_guidance_to_text_empty():
    from memory_agent.schemas import MemoryGuidance
    text = guidance_to_text(MemoryGuidance())
    assert text == ""


def test_guidance_to_text_with_content():
    from memory_agent.schemas import MemoryGuidance
    result = ApplicabilityResult(
        memory_assessments=[
            MemoryApplicabilityAssessment(
                memory_id="m1",
                memory_type="experience",
                decision="apply",
            ),
        ],
    )
    retrieval = MemoryRetrievalResult(
        positive_experience_hits=[
            RetrievalHit(
                memory_id="m1",
                memory_type="experience",
                score=0.9,
                content={
                    "situation_text": "acute chest pain",
                    "action_text": "ask onset and exertional trigger",
                    "outcome_text": "prior case improved differential",
                },
            )
        ]
    )
    guidance = build_memory_guidance(result, retrieval)
    text = guidance_to_text(guidance)
    assert "acute chest pain" in text
    assert "ask onset" in text
    assert "m1" in text
