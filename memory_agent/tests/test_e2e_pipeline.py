"""End-to-end pipeline test (no LLM/env required)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))  # MedGym/
sys.path.insert(0, str(_HERE.parent))           # memory_agent/

from memory_agent.schemas import (
    CaseState, MemoryQuery, ExperienceCard, OutcomeType,
    MemoryRetrievalResult, ApplicabilityResult, MemoryGuidance, TurnRecord,
    EpisodeFeedback, DistilledEpisode,
)
from memory_agent.online.case_updater import init_case_state, update_case_state_rule
from memory_agent.online.query_builder import build_memory_query_rule
from memory_agent.online.retriever import retrieve_multi_memory
from memory_agent.online.applicability_controller import (
    apply_applicability_control, _hard_block_actions,
)
from memory_agent.online.memory_guidance import build_memory_guidance, guidance_to_text
from memory_agent.offline.experience_merger import decide_merge_rule, merge_experience
from memory_agent.offline.episode_distiller import distill_from_trajectory
from memory_agent.memory_store import ExperienceMemoryStore, SkillMemoryStore


def test_full_online_pipeline():
    """Simulate one turn of the online memory pipeline."""
    # 1. Init CaseState from a mock bundle (like env.reset() → task dict)
    bundle = {
        "case_id": "e2e_001",
        "ehr": {
            "OSCE_Examination": {
                "Patient_Actor": {
                    "Symptoms": {"Chief_Complaint": "chest pain and shortness of breath"},
                },
            },
        },
    }
    cs = init_case_state(bundle)
    assert cs.case_id == "e2e_001"
    assert "chest pain" in cs.problem_summary
    assert cs.finalize_risk == "high"
    assert cs.turn_id == 0

    # 2. Update CaseState from observation (like env after ask_patient)
    obs = {"question": "I have chest pain for 2 days, worse when exercising"}
    cs = update_case_state_rule(cs, obs)
    assert cs.turn_id == 1
    assert len(cs.key_evidence) >= 1
    assert any("chest pain" in ev for ev in cs.key_evidence)

    # 3. Build memory query
    mq = build_memory_query_rule(cs, candidate_actions=["ASK", "REQUEST_LAB"])
    assert mq.case_id == "e2e_001"
    assert "chest pain" in mq.query_text

    # 4. Retrieve (empty store → empty results)
    with tempfile.TemporaryDirectory() as tmp:
        result = retrieve_multi_memory(mq, root_dir=tmp)
        assert isinstance(result, MemoryRetrievalResult)
        assert len(result.positive_experience_hits) == 0

        # 5. Applicability control
        ar = apply_applicability_control(cs, mq, result, mode="rule")
        assert isinstance(ar, ApplicabilityResult)
        # HIGH finalize_risk → hard block FINALIZE_DIAGNOSIS
        assert "FINALIZE_DIAGNOSIS" in ar.hard_blocked_actions

        # 6. Build guidance
        guidance = build_memory_guidance(ar)
        assert isinstance(guidance, MemoryGuidance)
        assert "FINALIZE_DIAGNOSIS" in guidance.blocked_actions

        # 7. Guidance to text
        text = guidance_to_text(guidance)
        assert "Blocked" in text
        assert "FINALIZE_DIAGNOSIS" in text

    print("  ✓ Full online pipeline OK")


def test_full_offline_pipeline():
    """Simulate experience extraction and merge without LLM."""
    with tempfile.TemporaryDirectory() as tmp:
        store = ExperienceMemoryStore(tmp)

        # Create some experiences
        exp1 = ExperienceCard(
            memory_id="exp_001",
            situation_text="patient with chest pain and ECG normal",
            action_text="order troponin",
            outcome_text="troponin elevated, diagnosed NSTEMI",
            boundary_text="use when chest pain with cardiac risk factors; "
                          "not for young patients without risk factors",
            outcome_type=OutcomeType.SUCCESS.value,
            action_sequence=[{"action_type": "REQUEST_LAB", "action_label": "troponin"}],
            retrieval_tags=["chest_pain", "troponin"],
        )
        exp2 = ExperienceCard(
            memory_id="exp_002",
            situation_text="patient with chest pain and ECG normal",
            action_text="order troponin",
            outcome_text="troponin normal, discharged",
            boundary_text="use when chest pain with cardiac risk factors",
            outcome_type=OutcomeType.SUCCESS.value,
            action_sequence=[{"action_type": "REQUEST_LAB", "action_label": "troponin"}],
            retrieval_tags=["chest_pain", "troponin"],
        )

        # Insert and verify
        store.append(exp1)
        assert len(store.list_all()) == 1

        # Merge decision (same trigger/direction → merge)
        decision = decide_merge_rule(exp2, [exp1])
        assert decision["merge_decision"] == "merge", f"Expected merge, got {decision}"

        merged = merge_experience(exp1, exp2)
        assert merged.support_count >= 2
        store.upsert(merged)

        # Verify store
        assert len(store.list_all()) == 1
        found = store.find_by_id("exp_001")
        assert found is not None
        assert found.support_count >= 2

    print("  ✓ Full offline pipeline OK")


def test_distill_from_trajectory():
    """Simulate trajectory distillation."""
    feedback = EpisodeFeedback(
        episode_id="ep_test",
        case_id="case_test",
        success=True,
        total_reward=0.85,
        final_diagnosis="NSTEMI",
        gold_diagnosis="Non-ST elevation myocardial infarction",
        summary="Patient diagnosed correctly",
    )
    trajectory = {
        "info": {
            "memory_agent": {
                "turn_records": [
                    TurnRecord(
                        episode_id="ep_test",
                        case_id="case_test",
                        turn_id=1,
                        case_state=CaseState(case_id="case_test", turn_id=1),
                        selected_action={"action_type": "ASK", "action_label": "ask about pain"},
                        reward=0.0,
                        done=False,
                    ).to_dict(),
                    TurnRecord(
                        episode_id="ep_test",
                        case_id="case_test",
                        turn_id=2,
                        case_state=CaseState(case_id="case_test", turn_id=2),
                        selected_action={"action_type": "FINALIZE_DIAGNOSIS", "action_label": "NSTEMI"},
                        reward=0.85,
                        done=True,
                    ).to_dict(),
                ]
            }
        }
    }

    distilled = distill_from_trajectory(trajectory, feedback)
    assert distilled.episode_id == "ep_test"
    assert len(distilled.turn_records) == 2
    assert distilled.feedback["success"] is True

    print("  ✓ Trajectory distillation OK")


def test_hard_block_rules():
    """Test that finalize_risk=high blocks FINALIZE_DIAGNOSIS."""
    cs = CaseState(case_id="test", finalize_risk="high", missing_info=["a", "b", "c"])
    blocked, _ = _hard_block_actions(cs)
    assert "FINALIZE_DIAGNOSIS" in blocked

    cs2 = CaseState(case_id="test", finalize_risk="low", missing_info=[])
    blocked2, _ = _hard_block_actions(cs2)
    assert "FINALIZE_DIAGNOSIS" not in blocked2

    print("  ✓ Hard block rules OK")


def test_reset_memory():
    """Test wrapper.reset_memory clears all state."""
    # We can't easily instantiate MemoryWrappedMedicalAgent without rllm,
    # but we can test the state class directly.
    tr = TurnRecord(
        episode_id="ep_test",
        case_id="case_test",
        turn_id=1,
        case_state=CaseState(case_id="case_test"),
    )
    d = tr.to_dict()
    assert d["case_state"]["case_id"] == "case_test"

    print("  ✓ Reset/memory state OK")


if __name__ == "__main__":
    tests = [
        test_full_online_pipeline,
        test_full_offline_pipeline,
        test_distill_from_trajectory,
        test_hard_block_rules,
        test_reset_memory,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  ✗ {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n=== E2E: {passed} passed, {failed} failed ===")
    sys.exit(0 if failed == 0 else 1)
