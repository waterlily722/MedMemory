from __future__ import annotations

import unittest

from memory_agent.online.query_builder import build_memory_query_rule
from memory_agent.schemas import CaseState


class QueryBuilderRuleTests(unittest.TestCase):
    def test_rule_query_builder(self):
        case_state = CaseState(
            case_id="case-1",
            problem_summary="patient with chest pain",
            uncertainty_summary="need more evidence",
        )
        query = build_memory_query_rule(case_state, ["REQUEST_LAB", "FINALIZE_DIAGNOSIS"])
        self.assertIn(case_state.problem_summary, query.query_text)
        self.assertIn("REQUEST_LAB", query.query_text or "")
        # Ensure MemoryQuery only contains minimal fields
        keys = set(query.to_dict().keys())
        self.assertEqual(keys, {"case_id", "turn_id", "query_text"})
