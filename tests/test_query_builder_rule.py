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
            negative_evidence=["no fever"],
            missing_info=["ECG or troponin"],
            active_hypotheses=["ACS"],
            local_goal="rule out high-risk cardiac cause",
            uncertainty_summary="need more evidence",
            finalize_risk="high",
            modality_flags=["text", "lab"],
            reviewed_modalities=["text"],
            interaction_history_summary="turn_1: asked onset",
        )
        query = build_memory_query_rule(
            case_state,
            [{"action_type": "REQUEST_LAB", "action_label": "troponin"}, "FINALIZE_DIAGNOSIS"],
        )
        self.assertIn("problem_summary", query.query_text)
        self.assertIn(case_state.problem_summary, query.query_text)
        self.assertIn("REQUEST_LAB", query.query_text)
        self.assertIn("missing_info", query.query_text)

        # MemoryQuery schema remains minimal.
        keys = set(query.to_dict().keys())
        self.assertEqual(keys, {"case_id", "turn_id", "query_text"})


if __name__ == "__main__":
    unittest.main()
