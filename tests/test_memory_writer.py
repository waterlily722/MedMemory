from __future__ import annotations

import tempfile
import unittest

from memory_agent.offline.memory_writer import write_memory_from_distilled_episode
from memory_agent.memory_store import ExperienceMemoryStore
from memory_agent.schemas import DistilledEpisode


class MemoryWriterTests(unittest.TestCase):
    def test_write_experience(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # With default mode='llm' and no llm_client provided, extractor returns []
            # and nothing should be written. This validates the safe default.
            distilled = DistilledEpisode(
                episode_id="ep-1",
                case_id="case-1",
                turn_records=[
                    {
                        "turn_id": 1,
                        "case_state": {
                            "problem_summary": "chest pain",
                            "uncertainty_summary": "need more evidence",
                            "key_evidence": ["troponin"],
                            "missing_info": ["troponin"],
                            "active_hypotheses": ["ACS"],
                            "modality_flags": ["lab"],
                            "negative_evidence": [],
                        },
                        "selected_action": {"action_type": "REQUEST_LAB", "action_label": "request lab", "action_content": "Order troponin"},
                        "selected_action_blocked": False,
                        "reward": 0.8,
                        "done": False,
                    }
                ],
                feedback={"episode_id": "ep-1", "case_id": "case-1", "total_reward": 0.8, "success": False, "final_diagnosis": "", "gold_diagnosis": "", "summary": ""},
            )
            result = write_memory_from_distilled_episode(distilled, root_dir=tmpdir)
            # No experiences should be written without an available LLM.
            self.assertEqual(result.get("extracted_count", 0), 0)
            self.assertEqual(len(ExperienceMemoryStore(tmpdir).list_all()), 0)
