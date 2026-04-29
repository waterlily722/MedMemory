from __future__ import annotations

import unittest

from memory_agent.wrapper import MemoryWrappedMedicalAgent


class WrapperSmokeTests(unittest.TestCase):
    def test_import_and_basic_flow(self):
        agent = MemoryWrappedMedicalAgent(tools=[], parser_name="qwen", enable_memory=False)
        agent.reset()
        agent.update_from_env({"case_id": "case-1", "question": "hello"}, reward=0.0, done=False, info={})
        action = agent.update_from_model("hello")
        self.assertIsNotNone(action)
        result = agent.finalize_episode(agent.trajectory)
        self.assertTrue(result.get("disabled"))
