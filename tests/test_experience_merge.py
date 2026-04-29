from __future__ import annotations

import unittest

from memory_agent.offline.experience_merger import decide_merge_rule
from memory_agent.schemas import ExperienceCard


class ExperienceMergeTests(unittest.TestCase):
    def test_merge_and_conflict(self):
        base = ExperienceCard(
            memory_id="exp-1",
            situation_anchor="chest pain",
            local_goal="rule out ACS",
            active_hypotheses=["ACS"],
            boundary="ed",
            action_sequence=[{"action_type": "REQUEST_LAB", "action_label": "request lab"}],
            outcome_type="success",
            source_episode_ids=["ep-1"],
            source_case_ids=["case-1"],
        )
        incoming = ExperienceCard(
            memory_id="exp-2",
            situation_anchor="chest pain",
            local_goal="rule out ACS",
            active_hypotheses=["ACS"],
            boundary="ed",
            action_sequence=[{"action_type": "REQUEST_LAB", "action_label": "request lab"}],
            outcome_type="failure",
            source_episode_ids=["ep-2"],
            source_case_ids=["case-2"],
        )
        conflict = decide_merge_rule(incoming, [base])
        self.assertEqual(conflict["merge_decision"], "conflict")

        same = ExperienceCard(
            memory_id="exp-3",
            situation_anchor="chest pain",
            local_goal="rule out ACS",
            active_hypotheses=["ACS"],
            boundary="ed",
            action_sequence=[{"action_type": "REQUEST_LAB", "action_label": "request lab"}],
            outcome_type="success",
            source_episode_ids=["ep-3"],
            source_case_ids=["case-3"],
        )
        merge = decide_merge_rule(same, [base])
        self.assertEqual(merge["merge_decision"], "merge")
