from __future__ import annotations

QUERY_BUILDER_SCHEMA = {
    "required": [
        "situation_anchor",
        "local_goal",
        "uncertainty_focus",
        "positive_evidence",
        "negative_evidence",
        "missing_info",
        "active_hypotheses",
        "modality_need",
        "candidate_action_need",
        "finalize_risk_reason",
        "retrieval_intent",
        "query_text",
    ],
    "list_fields": [
        "positive_evidence",
        "negative_evidence",
        "missing_info",
        "active_hypotheses",
        "modality_need",
        "candidate_action_need",
    ],
}

APPLICABILITY_JUDGE_SCHEMA = {
    "required": [
        "memory_id",
        "applicability",
        "reason",
        "matched_aspects",
        "mismatched_aspects",
        "boundary_violation",
        "action_bias",
        "blocked_actions",
        "controller_decision",
    ],
    "list_fields": ["matched_aspects", "mismatched_aspects", "blocked_actions"],
    "dict_fields": ["action_bias"],
}

MEMORY_GUIDANCE_SCHEMA = {
    "required": [
        "recommended_actions",
        "discouraged_actions",
        "blocked_actions",
        "used_memory_ids",
        "memory_rationale",
        "risk_warning",
        "why_not_finalize",
    ],
    "list_fields": ["recommended_actions", "discouraged_actions", "blocked_actions", "used_memory_ids"],
}

EXPERIENCE_EXTRACT_SCHEMA = {
    "required": [
        "memory_type",
        "experience_id",
        "source_episode_id",
        "situation_anchor",
        "local_goal",
        "uncertainty_state",
        "key_evidence",
        "missing_info",
        "action_sequence",
        "outcome_shift",
        "success_signal",
        "failure_mode",
        "boundary",
        "applicability_conditions",
        "non_applicability_conditions",
        "modality_flags",
        "risk_tags",
        "retrieval_tags",
        "confidence",
    ],
    "list_fields": [
        "key_evidence",
        "missing_info",
        "action_sequence",
        "applicability_conditions",
        "non_applicability_conditions",
        "modality_flags",
        "risk_tags",
        "retrieval_tags",
    ],
}

EXPERIENCE_MERGE_SCHEMA = {
    "required": [
        "merge_decision",
        "target_memory_ids",
        "reason",
        "merged_experience",
        "discard_reason",
    ],
    "list_fields": ["target_memory_ids"],
    "dict_fields": ["merged_experience"],
}

SKILL_MINER_SCHEMA = {
    "required": [
        "memory_type",
        "skill_id",
        "source_experience_ids",
        "skill_name",
        "clinical_situation",
        "local_goal",
        "trigger_conditions",
        "procedure",
        "stop_conditions",
        "success_criteria",
        "failure_modes",
        "contraindications",
        "required_modalities",
        "applicability_boundary",
        "evidence_count",
        "confidence",
        "version",
    ],
    "list_fields": [
        "source_experience_ids",
        "trigger_conditions",
        "procedure",
        "stop_conditions",
        "success_criteria",
        "failure_modes",
        "contraindications",
        "required_modalities",
    ],
}
