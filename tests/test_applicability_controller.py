from __future__ import annotations

import tempfile
import unittest

from memory_agent.memory_store import ExperienceMemoryStore, SkillMemoryStore
from memory_agent.online.applicability_controller import apply_applicability_control
from memory_agent.online.query_builder import build_memory_query_rule
from memory_agent.online.retriever import retrieve_multi_memory
from memory_agent.schemas import CaseState, ExperienceCard, SkillCard


class ApplicabilityControllerTests(unittest.TestCase):
    def test_finalize_is_hard_blocked_from_case_state(self):
        case_state = CaseState(
            case_id="case-1",
            problem_summary="chest pain",
            missing_info=["ECG", "troponin", "risk factors"],
            finalize_risk="high",
        )
        query = build_memory_query_rule(case_state, ["REQUEST_LAB", "FINALIZE_DIAGNOSIS"])
        retrieval = retrieve_multi_memory(
            query,
            root_dir=tempfile.mkdtemp(),
            disable_experience_memory=True,
            disable_skill_memory=True,
            disable_knowledge_memory=True,
        )
        applicability = apply_applicability_control(case_state, query, retrieval, mode="rule")
        self.assertIn("FINALIZE_DIAGNOSIS", applicability.hard_blocked_actions)
        by_action = {item.action_type: item for item in applicability.action_assessments}
        self.assertTrue(by_action["FINALIZE_DIAGNOSIS"].blocked)

    def test_retrieved_skill_is_hint_not_default_apply(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            SkillMemoryStore(tmpdir).upsert(
                SkillCard(
                    memory_id="skill-1",
                    skill_name="ACS workup",
                    situation_text="chest pain with unresolved ACS risk",
                    goal_text="rule out ACS",
                    procedure_text="request ECG and troponin before finalizing",
                    procedure=[{"action_type": "REQUEST_LAB", "action_label": "ECG and troponin"}],
                    boundary_text="applies to adult chest pain with ACS concern",
                    evidence_count=5,
                    unique_case_count=3,
                    success_rate=0.8,
                    confidence=0.9,
                )
            )
            case_state = CaseState(
                case_id="case-1",
                problem_summary="adult chest pain",
                missing_info=["ECG", "troponin"],
                local_goal="rule out ACS",
                uncertainty_summary="ACS not excluded",
                finalize_risk="high",
            )
            query = build_memory_query_rule(case_state, ["REQUEST_LAB", "FINALIZE_DIAGNOSIS"])
            retrieval = retrieve_multi_memory(query, root_dir=tmpdir)
            applicability = apply_applicability_control(case_state, query, retrieval, mode="rule")
            skill_assessments = [a for a in applicability.memory_assessments if a.memory_id == "skill-1"]
            self.assertEqual(skill_assessments[0].decision, "hint")

    def test_failure_experience_discourages_before_blocking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ExperienceMemoryStore(tmpdir).upsert(
                ExperienceCard(
                    memory_id="exp-fail",
                    situation_text="chest pain before ACS workup",
                    action_text="FINALIZE_DIAGNOSIS: finalize too early",
                    outcome_text="missed ACS",
                    boundary_text="applies when ECG and troponin are missing",
                    action_sequence=[{"action_type": "FINALIZE_DIAGNOSIS", "action_label": "finalize"}],
                    outcome_type="failure",
                )
            )
            case_state = CaseState(
                case_id="case-1",
                problem_summary="chest pain",
                missing_info=["ECG", "troponin"],
                finalize_risk="medium",
            )
            query = build_memory_query_rule(case_state, ["FINALIZE_DIAGNOSIS"])
            retrieval = retrieve_multi_memory(
                query,
                root_dir=tmpdir,
                negative_experience_min_score=0.0,
            )
            applicability = apply_applicability_control(case_state, query, retrieval, mode="rule")
            fail = [a for a in applicability.memory_assessments if a.memory_id == "exp-fail"][0]
            self.assertIn(fail.decision, {"hint", "block"})
            self.assertLess(fail.action_bias.get("FINALIZE_DIAGNOSIS", 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
