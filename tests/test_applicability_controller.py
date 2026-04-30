from __future__ import annotations

import tempfile
import unittest

from memory_agent.memory_store import ExperienceMemoryStore, SkillMemoryStore
from memory_agent.online.applicability_controller import apply_applicability_control
from memory_agent.online.query_builder import build_memory_query_rule
from memory_agent.online.retriever import retrieve_multi_memory
from memory_agent.schemas import CaseState, ExperienceCard, SkillCard


class ApplicabilityControllerTests(unittest.TestCase):
    def test_finalize_is_hard_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ExperienceMemoryStore(tmpdir).upsert(
                ExperienceCard(
                    memory_id="exp-1",
                    situation_text="chest pain ACS uncertainty",
                    action_text="REQUEST_LAB: request troponin",
                    outcome_text="some improvement",
                    boundary_text="adult with chest pain and incomplete ACS workup",
                    action_sequence=[{"action_type": "REQUEST_LAB", "action_label": "request lab"}],
                    outcome_type="success",
                    source_episode_ids=["ep-1"],
                    source_case_ids=["case-1"],
                )
            )
            case_state = CaseState(
                case_id="case-1",
                problem_summary="chest pain",
                key_evidence=["troponin pending"],
                missing_info=["troponin", "ECG", "onset"],
                active_hypotheses=["ACS"],
                local_goal="rule out ACS",
                uncertainty_summary="need more evidence",
                finalize_risk="high",
                modality_flags=["text", "lab"],
            )
            query = build_memory_query_rule(case_state, ["REQUEST_LAB", "FINALIZE_DIAGNOSIS"])
            retrieval = retrieve_multi_memory(query, root_dir=tmpdir)
            applicability = apply_applicability_control(case_state, query, retrieval, mode="rule")
            self.assertIn("FINALIZE_DIAGNOSIS", applicability.hard_blocked_actions)
            by_action = {item.action_type: item for item in applicability.action_assessments}
            self.assertTrue(by_action["FINALIZE_DIAGNOSIS"].blocked)

    def test_failure_discourages_but_unsafe_can_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ExperienceMemoryStore(tmpdir)
            store.upsert(
                ExperienceCard(
                    memory_id="failure-exp",
                    situation_text="chest pain ACS uncertainty",
                    action_text="FINALIZE_DIAGNOSIS: premature closure",
                    outcome_text="missed ACS",
                    boundary_text="applies when chest pain workup is incomplete",
                    action_sequence=[{"action_type": "FINALIZE_DIAGNOSIS", "action_label": "finalize"}],
                    outcome_type="failure",
                )
            )
            store.upsert(
                ExperienceCard(
                    memory_id="unsafe-exp",
                    situation_text="chest pain ACS uncertainty high risk",
                    action_text="FINALIZE_DIAGNOSIS: premature closure",
                    outcome_text="unsafe missed MI",
                    boundary_text="applies when chest pain has unresolved ACS evidence",
                    action_sequence=[{"action_type": "FINALIZE_DIAGNOSIS", "action_label": "finalize"}],
                    outcome_type="unsafe",
                )
            )
            case_state = CaseState(
                case_id="case-1",
                problem_summary="chest pain ACS uncertainty",
                missing_info=[],
                active_hypotheses=["ACS"],
                local_goal="avoid premature closure",
                uncertainty_summary="ACS evidence incomplete",
                finalize_risk="low",
                modality_flags=["text"],
            )
            query = build_memory_query_rule(case_state, ["FINALIZE_DIAGNOSIS"])
            retrieval = retrieve_multi_memory(query, root_dir=tmpdir)
            applicability = apply_applicability_control(case_state, query, retrieval, mode="rule")

            by_id = {item.memory_id: item for item in applicability.memory_assessments}
            self.assertEqual(by_id["failure-exp"].decision, "hint")
            self.assertEqual(by_id["unsafe-exp"].decision, "block")

    def test_skill_retrieval_is_hint_not_apply_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            SkillMemoryStore(tmpdir).upsert(
                SkillCard(
                    memory_id="skill-1",
                    skill_name="ACS workup",
                    situation_text="chest pain ACS uncertainty",
                    goal_text="rule out ACS",
                    procedure_text="ask key history then request ECG and troponin",
                    procedure=[{"action_type": "REQUEST_LAB", "action_label": "request ECG/troponin"}],
                    source_experience_ids=["exp-1"],
                    evidence_count=5,
                    confidence=0.9,
                )
            )
            case_state = CaseState(
                case_id="case-1",
                problem_summary="chest pain ACS uncertainty",
                active_hypotheses=["ACS"],
                local_goal="rule out ACS",
                uncertainty_summary="need more evidence",
                finalize_risk="medium",
            )
            query = build_memory_query_rule(case_state, ["REQUEST_LAB"])
            retrieval = retrieve_multi_memory(query, root_dir=tmpdir)
            applicability = apply_applicability_control(case_state, query, retrieval, mode="rule")
            self.assertEqual(applicability.memory_assessments[0].decision, "hint")


if __name__ == "__main__":
    unittest.main()
