from __future__ import annotations

import tempfile
import unittest

from memory_agent.memory_store import ExperienceMemoryStore
from memory_agent.online.applicability_controller import apply_applicability_control
from memory_agent.online.query_builder import build_memory_query_rule
from memory_agent.online.retriever import retrieve_multi_memory
from memory_agent.schemas import CaseState, ExperienceCard


class ApplicabilityControllerTests(unittest.TestCase):
    def test_finalize_is_hard_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ExperienceMemoryStore(tmpdir).upsert(
                ExperienceCard(
                    memory_id="exp-1",
                    situation_anchor="chest pain",
                    local_goal="rule out ACS",
                    key_evidence=["troponin"],
                    outcome_type="success",
                    action_sequence=[{"action_type": "REQUEST_LAB", "action_label": "request lab"}],
                    source_episode_ids=["ep-1"],
                    source_case_ids=["case-1"],
                )
            )
            case_state = CaseState(case_id="case-1", problem_summary="chest pain", key_evidence=["troponin"], missing_info=["troponin"], active_hypotheses=["ACS"], local_goal="rule out ACS", uncertainty_summary="need more evidence", finalize_risk="high", modality_flags=["text", "lab"])
            query = build_memory_query_rule(case_state, ["REQUEST_LAB", "FINALIZE_DIAGNOSIS"])
            retrieval = retrieve_multi_memory(query, root_dir=tmpdir)
            applicability = apply_applicability_control(case_state, query, retrieval, ["REQUEST_LAB", "FINALIZE_DIAGNOSIS"], mode="rule")
            self.assertIn("FINALIZE_DIAGNOSIS", applicability.hard_blocked_actions)
            by_action = {item.action_type: item for item in applicability.action_assessments}
            self.assertTrue(by_action["FINALIZE_DIAGNOSIS"].blocked)
