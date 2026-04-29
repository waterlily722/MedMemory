from __future__ import annotations

import unittest

from memory_agent.wrapper import MemoryWrappedMedicalAgent


class WrapperSmokeTests(unittest.TestCase):
    def test_import_and_basic_flow(self):
        agent = MemoryWrappedMedicalAgent(tools=[], parser_name="qwen", disable_memory=True)
        agent.reset_memory()
        # basic env update + model update flow shouldn't crash
        agent.update_from_env({"case_id": "case-1", "question": "hello"}, reward=0.0, done=False, info={})
        action = agent.update_from_model("hello")
        self.assertIsNotNone(action)
        # finalize episode via internal helper and ensure episode state flips
        agent._finalize_episode_if_needed(reward=0.0, info={"success": False, "total_reward": 0.0})
        self.assertTrue(agent.episode_finalized)
