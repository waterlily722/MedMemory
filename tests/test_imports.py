from __future__ import annotations

import unittest


class ImportTests(unittest.TestCase):
    def test_memory_agent_import(self):
        from memory_agent import MemoryWrappedMedicalAgent

        self.assertIsNotNone(MemoryWrappedMedicalAgent)
