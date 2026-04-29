from __future__ import annotations

import tempfile
import unittest

from memory_agent.memory_store import ExperienceMemoryStore, SkillMemoryStore
from memory_agent.offline.skill_consolidator import consolidate_skills_from_store
from memory_agent.schemas import ExperienceCard


class SkillConsolidatorTests(unittest.TestCase):
    def test_cross_episode_skill_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ExperienceMemoryStore(tmpdir)
            for index in range(5):
                store.upsert(
                    ExperienceCard(
                        memory_id=f"exp-{index}",
                        situation_anchor="chest pain",
                        local_goal="rule out ACS",
                        action_sequence=[{"action_type": "REQUEST_LAB", "action_label": "request lab"}],
                        outcome_type="success",
                        support_count=1,
                        source_episode_ids=[f"ep-{index}"],
                        source_case_ids=[f"case-{index % 3}"],
                    )
                )
            skills = consolidate_skills_from_store(tmpdir, mode="rule")
            self.assertGreaterEqual(len(skills), 1)
            self.assertGreaterEqual(len(SkillMemoryStore(tmpdir).list_all()), 1)
