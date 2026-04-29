from __future__ import annotations

import json
import unittest

from memory_agent.llm import LLMClient
from memory_agent.online.query_builder import build_memory_query
from memory_agent.schemas import CaseState


class MockLLMClient(LLMClient):
    def __init__(self, payload: dict[str, object]):
        super().__init__(model="mock", base_url="http://example.com", api_key="")
        self.payload = payload

    def available(self) -> bool:
        return True

    def generate_json(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1200) -> str:
        _ = prompt, temperature, max_tokens
        return json.dumps(self.payload, ensure_ascii=False)


class QueryBuilderMockLLMTests(unittest.TestCase):
    def test_llm_query_builder(self):
        case_state = CaseState(case_id="case-1", problem_summary="chest pain", local_goal="rule out ACS", uncertainty_summary="need more evidence")
        client = MockLLMClient(
            {
                "query_text": "llm query",
                "situation_anchor": "anchor",
                "local_goal": "goal",
                "uncertainty_focus": "uncertain",
                "positive_evidence": ["a"],
                "negative_evidence": ["b"],
                "missing_info": ["c"],
                "active_hypotheses": ["ACS"],
                "modality_need": ["lab"],
                "candidate_action_need": ["REQUEST_LAB"],
                "finalize_risk": "high",
                "finalize_risk_reason": "missing_critical_info",
                "retrieval_intent": "mixed",
            }
        )
        query = build_memory_query(case_state, ["REQUEST_LAB"], mode="llm", llm_client=client)
        self.assertEqual(query.query_text, "llm query")
        self.assertEqual(query.finalize_risk_reason, "missing_critical_info")
