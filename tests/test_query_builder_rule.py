from __future__ import annotations

import unittest

from memory_agent.online.query_builder import build_memory_query_rule
from memory_agent.schemas import CaseState


class QueryBuilderRuleTests(unittest.TestCase):
    def test_rule_query_builder_uses_schema_fields_only(self):
        case_state = CaseState(
            case_id="case-1",
            problem_summary="patient with chest pain",
            key_evidence=["substernal pain"],
            negative_evidence=["denies fever"],
            missing_info=["troponin"],
            active_hypotheses=["ACS"],
            local_goal="rule out ACS",
            uncertainty_summary="need more evidence",
            finalize_risk="high",
            modality_flags=["text", "lab"],
            reviewed_modalities=["text"],
            interaction_history_summary="turn_1: asked about pain",
        )
        query = build_memory_query_rule(
            case_state,
            ["REQUEST_LAB", {"action_type": "FINALIZE_DIAGNOSIS", "action_label": "finalize"}],
        )
        self.assertIn("problem_summary: patient with chest pain", query.query_text)
        self.assertIn("missing_info: troponin", query.query_text)
        self.assertIn("finalize_risk: high", query.query_text)
        self.assertIn("candidate_actions:", query.query_text)
        self.assertIn("REQUEST_LAB", query.query_text)

        # Ensure MemoryQuery schema stays minimal.
        self.assertEqual(set(query.to_dict().keys()), {"case_id", "turn_id", "query_text"})


if __name__ == "__main__":
    unittest.main()
