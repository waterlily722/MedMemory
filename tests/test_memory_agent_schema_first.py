from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parents[4]
for path in (str(CODE_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from memory_agent.memory_manager import update_memory
from memory_agent.online.applicability_controller import apply_applicability_control
from memory_agent.online.query_builder import build_memory_query, build_memory_query_with_mode
from memory_agent.online.retriever import retrieve_multi_memory
from memory_agent.online.doctor_policy import choose_next_action
from memory_agent.schemas import (
    ActionCandidate,
    CaseState,
    DistilledEpisode,
    EpisodeFeedback,
    ExperienceCard,
    HypothesisState,
    IntentPlan,
    MemoryQuery,
    MemoryRetrievalResult,
    RetrievalHit,
)


class MockLLMClient:
    def __init__(self, payload: dict[str, object]):
        self.payload = payload

    def generate_json(self, prompt: str, temperature: float = 0.2, max_tokens: int = 900) -> str:
        _ = prompt, temperature, max_tokens
        return json.dumps(self.payload, ensure_ascii=False)


class MemoryAgentSchemaFirstTests(unittest.TestCase):
    def _case_state(self) -> CaseState:
        return CaseState(
            case_id="case-001",
            turn_id=1,
            problem_summary="Patient with chest pain and shortness of breath",
            evidence_items=[],
            missing_info=["troponin", "cxr"],
            active_hypotheses=[HypothesisState(name="ACS", probability_hint="medium", supporting_evidence=["chest pain"])],
            local_goal="rule out ACS",
            uncertainty_summary="need more evidence before final diagnosis",
            finalize_risk="high",
            modality_flags=["text", "lab"],
            next_action_constraints=["do not finalize without troponin"],
        )

    def test_rule_query_builder_preserves_structured_fields(self):
        case_state = self._case_state()
        query = build_memory_query(case_state, ["REQUEST_LAB", "FINALIZE_DIAGNOSIS"])

        self.assertEqual(query.structured.situation_anchor, case_state.problem_summary)
        self.assertEqual(query.structured.uncertainty_focus, case_state.uncertainty_summary)
        self.assertEqual(query.structured.finalize_risk_reason, "missing critical info")
        self.assertEqual(query.structured.retrieval_intent, "skill")
        self.assertIn("rule out ACS", query.query_text)

    def test_mock_llm_query_and_applicability(self):
        case_state = self._case_state()
        llm_query_client = MockLLMClient(
            {
                "situation_anchor": case_state.problem_summary,
                "local_goal": case_state.local_goal,
                "uncertainty_focus": case_state.uncertainty_summary,
                "positive_evidence": ["chest pain"],
                "negative_evidence": ["no fever"],
                "missing_info": case_state.missing_info,
                "active_hypotheses": ["ACS"],
                "modality_need": ["lab"],
                "candidate_action_need": ["REQUEST_LAB"],
                "finalize_risk_reason": "missing_evidence",
                "retrieval_intent": "mixed",
                "query_text": "ACS query",
                "finalize_risk": "high",
            }
        )
        query = build_memory_query_with_mode(
            case_state,
            ["REQUEST_LAB"],
            mode="llm",
            llm_client=llm_query_client,
            observation={"note": "mock"},
            interaction_history_summary="turn 1",
        )
        self.assertEqual(query.structured.retrieval_intent, "mixed")
        self.assertEqual(query.structured.finalize_risk_reason, "missing_evidence")

        memory_hit = RetrievalHit(
            item_id="exp_1",
            retrieval_score=0.91,
            matched_fields=["semantic"],
            payload={
                "memory_id": "exp_1",
                "memory_type": "experience",
                "content": {
                    "item_id": "exp_1",
                    "experience_id": "exp_1",
                    "situation_anchor": "Patient with chest pain and shortness of breath",
                    "local_goal": "rule out ACS",
                    "action_sequence": [{"action_type": "REQUEST_LAB", "action_label": "order_troponin"}],
                    "outcome_shift": "better evidence",
                    "boundary": "missing troponin",
                    "outcome_type": "success",
                    "key_evidence": ["chest pain"],
                    "missing_info": ["troponin"],
                    "applicability_conditions": ["lab"],
                    "non_applicability_conditions": ["finalize risk high"],
                    "retrieval_tags": ["REQUEST_LAB"],
                    "confidence": 0.9,
                    "uncertainty_state": "need troponin",
                    "success_signal": "success",
                    "failure_mode": "",
                    "error_tag": [],
                    "support_count": 1,
                    "conflict_group_id": "",
                    "hypotheses": ["ACS"],
                    "source_turn_ids": ["1"],
                    "source_episode_ids": ["ep_1"],
                    "source_case_ids": ["case-001"],
                    "visual_signature": {},
                    "source_field_refs": ["turn_1"],
                },
            },
            source_field_refs=["turn_1"],
        )
        retrieval = MemoryRetrievalResult(
            turn_id=1,
            experience_hits=[memory_hit],
            negative_experience_hits=[],
            skill_hits=[],
            knowledge_hits=[],
        )
        plan = IntentPlan(
            turn_id=1,
            action_candidates=[ActionCandidate(action_id="a1", action_type="REQUEST_LAB", action_label="order_troponin", action_content="Order troponin")],
            memory_query=query,
        )
        llm_app_client = MockLLMClient(
            {
                "memory_id": "exp_1",
                "memory_type": "experience",
                "memory_content": memory_hit.payload["content"],
                "applicability": "high",
                "reason": "supports ordering troponin",
                "matched_aspects": ["REQUEST_LAB"],
                "mismatched_aspects": [],
                "boundary_violation": False,
                "action_bias": {"REQUEST_LAB": 0.5},
                "blocked_actions": [],
                "controller_decision": "apply",
            }
        )
        applicability = apply_applicability_control(case_state, plan, retrieval, mode="llm", llm_client=llm_app_client)
        self.assertTrue(applicability.memory_assessments)
        self.assertEqual(applicability.action_assessments[0].decision, "apply")

    def test_end_to_end_write_then_retrieve_then_decide(self):
        case_state = self._case_state()
        with tempfile.TemporaryDirectory() as tmpdir:
            experience = ExperienceCard(
                item_id="exp_case_001",
                situation_anchor=case_state.problem_summary,
                local_goal=case_state.local_goal,
                action_sequence=[{"action_type": "REQUEST_LAB", "action_label": "order_troponin"}],
                outcome_shift="resolved uncertainty",
                boundary="missing troponin",
                outcome_type="success",
                key_evidence=["chest pain", "shortness of breath"],
                missing_info=["troponin"],
                applicability_conditions=["lab"],
                non_applicability_conditions=["finalize risk high"],
                retrieval_tags=["REQUEST_LAB"],
                confidence=0.9,
                uncertainty_state="need more evidence",
                success_signal="success",
                failure_mode="",
                support_count=1,
                source_turn_ids=["1"],
                source_episode_ids=["ep_001"],
                source_case_ids=[case_state.case_id],
            )
            distilled = DistilledEpisode(
                episode_id="ep_001",
                summary={"note": "smoke test"},
                candidate_experience_items=[experience.to_dict()],
                candidate_skill_items=[],
            )
            update_memory(distilled, root_dir=tmpdir, experience_merge_mode="rule", skill_mining_mode="rule")

            query = build_memory_query(case_state, ["REQUEST_LAB", "FINALIZE_DIAGNOSIS"])
            retrieval = retrieve_multi_memory(query, turn_id=1, root_dir=tmpdir)
            self.assertGreaterEqual(len(retrieval.experience_hits), 1)

            plan = IntentPlan(
                turn_id=1,
                action_candidates=[
                    ActionCandidate(action_id="a1", action_type="REQUEST_LAB", action_label="order_troponin", action_content="Order troponin"),
                    ActionCandidate(action_id="a2", action_type="FINALIZE_DIAGNOSIS", action_label="finalize", action_content="Finalize diagnosis"),
                ],
                memory_query=query,
            )
            applicability = apply_applicability_control(case_state, plan, retrieval, mode="rule")
            decision = choose_next_action(case_state, plan, applicability)

            self.assertEqual(decision.chosen_action["action_type"], "REQUEST_LAB")
            self.assertFalse(any(a.decision == "block" for a in applicability.action_assessments if a.action_id == "a1"))


if __name__ == "__main__":
    unittest.main()
