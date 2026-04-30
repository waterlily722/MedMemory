from __future__ import annotations

import tempfile
import unittest

from memory_agent.memory_store import ExperienceMemoryStore, KnowledgeMemoryStore, SkillMemoryStore
from memory_agent.online.query_builder import build_memory_query_rule
from memory_agent.online.retriever import retrieve_multi_memory
from memory_agent.schemas import CaseState, ExperienceCard, KnowledgeItem, SkillCard


class RetrieverTests(unittest.TestCase):
    def test_retrieve_written_memory_above_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_store = ExperienceMemoryStore(tmpdir)
            skill_store = SkillMemoryStore(tmpdir)
            kn_store = KnowledgeMemoryStore(tmpdir)

            exp_store.upsert(
                ExperienceCard(
                    memory_id="exp-1",
                    situation_text="chest pain ACS uncertainty",
                    action_text="REQUEST_LAB: troponin",
                    outcome_text="improved",
                    boundary_text="adult with chest pain and incomplete ACS workup",
                    action_sequence=[{"action_type": "REQUEST_LAB", "action_label": "request lab"}],
                    outcome_type="success",
                    retrieval_tags=["chest pain", "ACS", "troponin"],
                    source_episode_ids=["ep-1"],
                    source_case_ids=["case-1"],
                )
            )
            exp_store.upsert(
                ExperienceCard(
                    memory_id="exp-2",
                    situation_text="chest pain ACS uncertainty",
                    action_text="FINALIZE_DIAGNOSIS: premature finalization",
                    outcome_text="unsafe missed ACS",
                    boundary_text="applies when ACS workup is incomplete",
                    action_sequence=[{"action_type": "FINALIZE_DIAGNOSIS", "action_label": "finalize"}],
                    outcome_type="unsafe",
                    retrieval_tags=["chest pain", "ACS", "premature closure"],
                    source_episode_ids=["ep-2"],
                    source_case_ids=["case-2"],
                )
            )
            skill_store.upsert(
                SkillCard(
                    memory_id="skill-1",
                    skill_name="ACS workup",
                    situation_text="chest pain ACS uncertainty",
                    goal_text="rule out ACS",
                    procedure_text="request ECG and troponin",
                    procedure=[{"action_type": "REQUEST_LAB", "action_label": "ask"}],
                    source_experience_ids=["exp-1"],
                    evidence_count=5,
                    confidence=0.9,
                )
            )
            kn_store.upsert(
                KnowledgeItem(
                    memory_id="kn-1",
                    content="troponin and ECG are key in ACS chest pain evaluation",
                    tags=["ACS", "chest pain"],
                    source="wiki",
                )
            )
            case_state = CaseState(
                case_id="case-1",
                problem_summary="chest pain ACS uncertainty",
                key_evidence=["troponin pending"],
                missing_info=["troponin"],
                active_hypotheses=["ACS"],
                local_goal="rule out ACS",
                uncertainty_summary="need more evidence",
                finalize_risk="high",
                modality_flags=["text", "lab"],
            )
            query = build_memory_query_rule(case_state, ["REQUEST_LAB", "FINALIZE_DIAGNOSIS"])
            result = retrieve_multi_memory(query, root_dir=tmpdir)
            self.assertGreaterEqual(len(result.positive_experience_hits), 1)
            self.assertGreaterEqual(len(result.negative_experience_hits), 1)
            self.assertGreaterEqual(len(result.skill_hits), 1)
            self.assertGreaterEqual(len(result.knowledge_hits), 1)
            self.assertEqual(result.negative_experience_hits[0].memory_type, "experience")

    def test_unrelated_memory_below_threshold_is_filtered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ExperienceMemoryStore(tmpdir).upsert(
                ExperienceCard(
                    memory_id="unrelated",
                    situation_text="ankle sprain after sports injury",
                    action_text="ASK: mechanism of injury",
                    outcome_text="clarified orthopedic diagnosis",
                    boundary_text="applies to ankle trauma",
                    action_sequence=[{"action_type": "ASK", "action_label": "ask mechanism"}],
                    outcome_type="success",
                )
            )
            case_state = CaseState(
                case_id="case-1",
                problem_summary="chest pain ACS uncertainty",
                active_hypotheses=["ACS"],
                uncertainty_summary="need troponin",
                finalize_risk="high",
            )
            query = build_memory_query_rule(case_state, ["REQUEST_LAB"])
            result = retrieve_multi_memory(query, root_dir=tmpdir, positive_experience_min_score=0.9)
            self.assertEqual(result.positive_experience_hits, [])


if __name__ == "__main__":
    unittest.main()
