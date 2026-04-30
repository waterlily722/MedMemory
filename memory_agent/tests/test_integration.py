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
    ExperienceCard,
    OutcomeType,
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
    select_high_value_turns,
    _is_high_value_turn,
    _turn_priority,
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
    assert "chest pain" in cs.problem_summary
    assert cs.finalize_risk == "high"
    assert cs.turn_id == 0


def test_init_case_state_empty():
    cs = init_case_state({})
    assert cs.case_id == ""
    assert cs.turn_id == 0
    assert cs.problem_summary


def test_update_case_state_rule():
    prev = CaseState(case_id="c001", turn_id=0, problem_summary="chest pain")
    updated = update_case_state_rule(prev, {"question": "I have chest pain"})
    assert updated.turn_id == 1
    assert len(updated.key_evidence) > 0


# ═══════════════════════════════════════════════════════════════════════
# Query Builder
# ═══════════════════════════════════════════════════════════════════════

def test_build_memory_query_rule():
    cs = CaseState(
        case_id="c001",
        turn_id=2,
        problem_summary="chest pain",
        key_evidence=["ECG normal"],
        missing_info=["troponin"],
        finalize_risk="high",
    )
    mq = build_memory_query_rule(cs)
    assert mq.case_id == "c001"
    assert "chest pain" in mq.query_text
    assert "ECG normal" in mq.query_text
    assert "troponin" in mq.query_text


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
        "retrieval_tags": ["chest_pain"],
        "risk_tags": [],
        "failure_mode": "",
    }
    text = memory_to_text("experience", payload)
    assert "chest pain" in text
    assert "order ECG" in text


# ═══════════════════════════════════════════════════════════════════════
# Applicability
# ═══════════════════════════════════════════════════════════════════════

def test_hard_block_actions_high_risk():
    cs = CaseState(case_id="c001", finalize_risk="high", missing_info=["a", "b", "c"])
    blocked, warning = _hard_block_actions(cs)
    assert "FINALIZE_DIAGNOSIS" in blocked


def test_hard_block_actions_low_risk():
    cs = CaseState(case_id="c001", finalize_risk="low", missing_info=[])
    blocked, warning = _hard_block_actions(cs)
    assert "FINALIZE_DIAGNOSIS" not in blocked


def test_apply_applicability_control_no_memories():
    cs = CaseState(case_id="c001", finalize_risk="medium")
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
# High value turn selection
# ═══════════════════════════════════════════════════════════════════════

def test_select_high_value_turns_high_reward():
    turns = [
        {"reward": 1.0, "selected_action": {}, "env_info": {}},
        {"reward": 0.0, "selected_action": {}, "env_info": {}},
    ]
    selected = select_high_value_turns(turns, limit=5)
    assert len(selected) == 1


def test_select_high_value_turns_blocked_action():
    turns = [
        {
            "reward": 0.0,
            "selected_action": {"blocked_by_memory": True},
            "env_info": {},
        },
    ]
    selected = select_high_value_turns(turns, limit=5)
    assert len(selected) == 1


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
    from memory_agent.schemas import MemoryGuidance, ApplicabilityResult, ActionAssessment
    result = ApplicabilityResult(
        hard_blocked_actions=["FINALIZE_DIAGNOSIS"],
        action_assessments=[
            ActionAssessment(action_type="ASK", score_delta=0.5),
        ],
        risk_warning="high finalize risk",
    )
    guidance = build_memory_guidance(result)
    text = guidance_to_text(guidance)
    assert "ASK" in text
    assert "FINALIZE_DIAGNOSIS" in text
