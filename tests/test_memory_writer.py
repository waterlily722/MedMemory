from __future__ import annotations

import tempfile
import unittest

from memory_agent.offline.memory_writer import write_memory_from_distilled_episode
from memory_agent.memory_store import ExperienceMemoryStore
from memory_agent.schemas import DistilledEpisode


class MemoryWriterTests(unittest.TestCase):
    def test_write_experience(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            distilled = DistilledEpisode(
                episode_id="ep-1",
                case_id="case-1",
                turn_records=[
                    {
                        "turn_id": 1,
                        "case_state_before": {
                            "problem_summary": "chest pain",
                            "local_goal": "rule out ACS",
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
                episode_feedback={"episode_id": "ep-1", "case_id": "case-1", "total_reward": 0.8},
            )
            result = write_memory_from_distilled_episode(distilled, root_dir=tmpdir)
            self.assertEqual(result["written_experience_ids"], [result["written_experience_ids"][0]])
            self.assertEqual(len(ExperienceMemoryStore(tmpdir).list_all()), 1)
