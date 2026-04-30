"""Tests for the schema layer (dataclasses + SerializableMixin + OutcomeType)."""

from __future__ import annotations

import pickle
import sys
import tempfile
from pathlib import Path

# ── Ensure the package is importable ──────────────────────────────────
_HERE = Path(__file__).resolve().parent
_PACKAGE = _HERE.parent
if str(_PACKAGE) not in sys.path:
    sys.path.insert(0, str(_PACKAGE))


from memory_agent.schemas import (
    OutcomeType,
    SerializableMixin,
    CaseState,
    MemoryQuery,
    ExperienceCard,
    SkillCard,
    KnowledgeItem,
    RetrievalHit,
    MemoryRetrievalResult,
    MemoryApplicabilityAssessment,
    ActionAssessment,
    ApplicabilityResult,
    MemoryGuidance,
    TurnRecord,
    EpisodeFeedback,
    DistilledEpisode,
)
from memory_agent.schemas.common import _convert


# ═══════════════════════════════════════════════════════════════════════
# OutcomeType
# ═══════════════════════════════════════════════════════════════════════

def test_outcome_type_values():
    assert OutcomeType.SUCCESS.value == "success"
    assert OutcomeType.PARTIAL_SUCCESS.value == "partial_success"
    assert OutcomeType.FAILURE.value == "failure"
    assert OutcomeType.UNSAFE.value == "unsafe"


def test_outcome_type_from_string():
    assert OutcomeType("success") == OutcomeType.SUCCESS
    assert OutcomeType("failure") == OutcomeType.FAILURE
    assert OutcomeType("partial_success") == OutcomeType.PARTIAL_SUCCESS


def test_outcome_type_unknown_defaults_to_partial_success():
    assert OutcomeType("unknown_garbage") == OutcomeType.PARTIAL_SUCCESS
    assert OutcomeType("") == OutcomeType.PARTIAL_SUCCESS


def test_outcome_type_str_equality():
    """OutcomeType members compare equal to their string value."""
    assert OutcomeType.SUCCESS == "success"
    assert OutcomeType.FAILURE == "failure"


# ═══════════════════════════════════════════════════════════════════════
# SerializableMixin
# ═══════════════════════════════════════════════════════════════════════

def test_serializable_mixin_to_dict_roundtrip():
    card = ExperienceCard(
        memory_id="exp_001",
        situation_text="patient with chest pain",
        action_text="order ECG",
        outcome_text="ECG normal, pain resolved",
        boundary_text="use when chest pain with cardiac risk factors",
        outcome_type=OutcomeType.SUCCESS.value,
        retrieval_tags=["chest_pain", "ecg"],
        source_turn_ids=[1, 2, 3],
    )
    d = card.to_dict()
    assert d["memory_id"] == "exp_001"
    assert d["outcome_type"] == "success"
    assert d["retrieval_tags"] == ["chest_pain", "ecg"]
    assert d["source_turn_ids"] == [1, 2, 3]

    restored = ExperienceCard.from_dict(d)
    assert restored.memory_id == "exp_001"
    assert restored.outcome_type == "success"
    assert restored.retrieval_tags == ["chest_pain", "ecg"]
    assert restored.source_turn_ids == [1, 2, 3]


def test_serializable_mixin_from_dict_none():
    try:
        ExperienceCard.from_dict(None)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_serializable_mixin_nested():
    """TurnRecord with typed nested fields serializes correctly."""
    cs = CaseState(case_id="case_001", problem_summary="chest pain")
    tr = TurnRecord(
        episode_id="ep_001",
        case_id="case_001",
        turn_id=1,
        case_state=cs,
    )
    d = tr.to_dict()
    assert d["episode_id"] == "ep_001"
    assert d["case_state"]["case_id"] == "case_001"
    assert d["case_state"]["problem_summary"] == "chest pain"


def test_memory_retrieval_result_custom_from_dict():
    result = MemoryRetrievalResult.from_dict({
        "positive_experience_hits": [
            {"memory_id": "exp_1", "memory_type": "experience", "content": {"a": 1}, "score": 0.9},
        ],
    })
    assert len(result.positive_experience_hits) == 1
    assert result.positive_experience_hits[0].memory_id == "exp_1"
    assert result.positive_experience_hits[0].score == 0.9


def test_memory_retrieval_result_from_dict_none():
    result = MemoryRetrievalResult.from_dict(None)
    assert len(result.positive_experience_hits) == 0


def test_applicability_result_custom_from_dict():
    result = ApplicabilityResult.from_dict({
        "memory_assessments": [
            {"memory_id": "m1", "memory_type": "experience", "decision": "apply", "reason": "good match", "action_bias": {"ASK": 0.5}, "blocked_actions": []},
        ],
        "action_assessments": [],
        "hard_blocked_actions": ["FINALIZE_DIAGNOSIS"],
        "risk_warning": "high risk",
    })
    assert len(result.memory_assessments) == 1
    assert result.memory_assessments[0].memory_id == "m1"
    assert result.hard_blocked_actions == ["FINALIZE_DIAGNOSIS"]


# ═══════════════════════════════════════════════════════════════════════
# CaseState
# ═══════════════════════════════════════════════════════════════════════

def test_case_state_creation():
    cs = CaseState(case_id="c001", turn_id=0, problem_summary="chest pain")
    assert cs.case_id == "c001"
    assert cs.finalize_risk == "high"
    assert cs.modality_flags == []


def test_case_state_roundtrip():
    cs = CaseState(
        case_id="c001",
        turn_id=3,
        problem_summary="shortness of breath",
        key_evidence=["low O2 sat", "cough"],
        active_hypotheses=["pneumonia", "COPD"],
        finalize_risk="medium",
        modality_flags=["text", "lab", "image"],
    )
    d = cs.to_dict()
    restored = CaseState.from_dict(d)
    assert restored.key_evidence == ["low O2 sat", "cough"]
    assert restored.finalize_risk == "medium"
    assert restored.modality_flags == ["text", "lab", "image"]


# ═══════════════════════════════════════════════════════════════════════
# MemoryQuery
# ═══════════════════════════════════════════════════════════════════════

def test_memory_query():
    mq = MemoryQuery(case_id="c001", turn_id=2, query_text="chest pain with ECG")
    assert mq.query_text == "chest pain with ECG"


# ═══════════════════════════════════════════════════════════════════════
# Guidance
# ═══════════════════════════════════════════════════════════════════════

def test_memory_guidance():
    mg = MemoryGuidance(
        recommended_actions=["ASK", "REQUEST_LAB"],
        blocked_actions=["FINALIZE_DIAGNOSIS"],
        rationale="missing critical info",
    )
    assert mg.recommended_actions == ["ASK", "REQUEST_LAB"]
    assert "FINALIZE_DIAGNOSIS" in mg.blocked_actions
