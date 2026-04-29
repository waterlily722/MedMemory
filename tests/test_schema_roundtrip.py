from __future__ import annotations

import unittest

from memory_agent.schemas import ApplicabilityResult, CaseState, ExperienceCard, MemoryGuidance, MemoryQuery, MemoryRetrievalResult, RetrievalHit, SkillCard


class SchemaRoundTripTests(unittest.TestCase):
    def test_roundtrip_case_state(self):
        case_state = CaseState(case_id="case-1", problem_summary="chest pain", key_evidence=["pain"], missing_info=["troponin"], active_hypotheses=["ACS"])
        self.assertEqual(CaseState.from_dict(case_state.to_dict()), case_state)

    def test_roundtrip_query(self):
        query = MemoryQuery(
            query_text="query",
            situation_anchor="anchor",
            local_goal="goal",
            uncertainty_focus="uncertain",
            positive_evidence=["a"],
            negative_evidence=["b"],
            missing_info=["c"],
            active_hypotheses=["h"],
            modality_need=["lab"],
            candidate_action_need=["REQUEST_LAB"],
            finalize_risk="high",
            finalize_risk_reason="missing_critical_info",
            retrieval_intent="mixed",
        )
        self.assertEqual(MemoryQuery.from_dict(query.to_dict()), query)

    def test_roundtrip_retrieval_and_guidance(self):
        hit = RetrievalHit(memory_id="m1", memory_type="experience", content={"x": 1}, score=0.9, matched_fields=["semantic"])
        retrieval = MemoryRetrievalResult(positive_experience_hits=[hit], negative_experience_hits=[], skill_hits=[], knowledge_hits=[])
        self.assertEqual(MemoryRetrievalResult.from_dict(retrieval.to_dict()), retrieval)

        guidance = MemoryGuidance(recommended_actions=["ASK"], discouraged_actions=["FINALIZE_DIAGNOSIS"], blocked_actions=["FINALIZE_DIAGNOSIS"], used_memory_ids=["m1"], warning_memory_ids=["m2"], rationale="r", risk_warning="w", why_not_finalize="n")
        self.assertEqual(MemoryGuidance.from_dict(guidance.to_dict()), guidance)

    def test_roundtrip_experience_skill(self):
        experience = ExperienceCard(memory_id="e1", situation_anchor="anchor", local_goal="goal", action_sequence=[{"action_type": "ASK", "action_label": "ask"}], source_episode_ids=["ep1"], source_case_ids=["case1"])
        skill = SkillCard(memory_id="s1", skill_name="skill", clinical_situation="anchor", local_goal="goal", trigger_conditions=["a"], procedure=[{"step": 1}], source_experience_ids=["e1"])
        self.assertEqual(ExperienceCard.from_dict(experience.to_dict()), experience)
        self.assertEqual(SkillCard.from_dict(skill.to_dict()), skill)

    def test_roundtrip_applicability(self):
        applicability = ApplicabilityResult(memory_assessments=[], action_assessments=[], hard_blocked_actions=["FINALIZE_DIAGNOSIS"], risk_warning="blocked")
        self.assertEqual(ApplicabilityResult.from_dict(applicability.to_dict()), applicability)
