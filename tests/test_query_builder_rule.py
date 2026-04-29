from __future__ import annotations

import unittest

from memory_agent.online.query_builder import build_memory_query_rule
from memory_agent.schemas import CaseState


class QueryBuilderRuleTests(unittest.TestCase):
    def test_rule_query_builder(self):
        case_state = CaseState(
            case_id="case-1",
            problem_summary="patient with chest pain",
            key_evidence=["chest pain"],
            negative_evidence=["no fever"],
            missing_info=["troponin"],
            active_hypotheses=["ACS"],
            local_goal="rule out ACS",
            uncertainty_summary="need more evidence",
            finalize_risk="high",
            modality_flags=["text", "lab"],
        )
        query = build_memory_query_rule(case_state, ["REQUEST_LAB", "FINALIZE_DIAGNOSIS"])
        self.assertEqual(query.situation_anchor, case_state.problem_summary)
        self.assertEqual(query.retrieval_intent, "mixed")
        self.assertIn("rule out ACS", query.query_text)
        self.assertIn("troponin", query.missing_info)
