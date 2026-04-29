from __future__ import annotations

import tempfile
import unittest

from memory_agent.memory_store import ExperienceMemoryStore, KnowledgeMemoryStore, SkillMemoryStore
from memory_agent.online.query_builder import build_memory_query_rule
from memory_agent.online.retriever import retrieve_multi_memory
from memory_agent.schemas import CaseState, ExperienceCard, KnowledgeItem, SkillCard


class RetrieverTests(unittest.TestCase):
    def test_retrieve_written_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_store = ExperienceMemoryStore(tmpdir)
            skill_store = SkillMemoryStore(tmpdir)
            kn_store = KnowledgeMemoryStore(tmpdir)
            exp_store.upsert(
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
            exp_store.upsert(
                ExperienceCard(
                    memory_id="exp-2",
                    situation_anchor="wrong diagnosis",
                    local_goal="avoid harm",
                    key_evidence=["unsafe"],
                    outcome_type="unsafe",
                    action_sequence=[{"action_type": "FINALIZE_DIAGNOSIS", "action_label": "finalize"}],
                    source_episode_ids=["ep-2"],
                    source_case_ids=["case-2"],
                )
            )
            skill_store.upsert(
                SkillCard(
                    memory_id="skill-1",
                    skill_name="ask more",
                    clinical_situation="chest pain",
                    local_goal="rule out ACS",
                    trigger_conditions=["chest pain"],
                    procedure=[{"action_type": "ASK", "action_label": "ask"}],
                    source_experience_ids=["exp-1"],
                    evidence_count=5,
                    confidence=0.9,
                )
            )
            kn_store.upsert(
                KnowledgeItem(
                    memory_id="kn-1",
                    title="ACS",
                    content="troponin and ECG are key",
                    disease_tags=["ACS"],
                    source="wiki",
                )
            )

            case_state = CaseState(case_id="case-1", problem_summary="chest pain", key_evidence=["troponin"], missing_info=["troponin"], active_hypotheses=["ACS"], local_goal="rule out ACS", uncertainty_summary="need more evidence", finalize_risk="high", modality_flags=["text", "lab"])
            query = build_memory_query_rule(case_state, ["REQUEST_LAB", "FINALIZE_DIAGNOSIS"])
            result = retrieve_multi_memory(query, root_dir=tmpdir)

            self.assertGreaterEqual(len(result.positive_experience_hits), 1)
            self.assertGreaterEqual(len(result.negative_experience_hits), 1)
            self.assertGreaterEqual(len(result.skill_hits), 1)
            self.assertGreaterEqual(len(result.knowledge_hits), 1)
            self.assertEqual(result.negative_experience_hits[0].memory_type, "negative_experience")
