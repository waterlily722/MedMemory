from __future__ import annotations

QUERY_BUILDER_SCHEMA = {
    "required": ["query_text"],
    "list_fields": [],
    "dict_fields": [],
}

APPLICABILITY_SCHEMA = {
    "required": [
        "memory_id",
        "memory_type",
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
    "enum_fields": {
        "memory_type": ["experience", "negative_experience", "skill", "knowledge"],
        "applicability": ["high", "medium", "low", "reject"],
        "controller_decision": ["apply", "hint", "block", "ignore"],
    },
}

EXPERIENCE_EXTRACTION_SCHEMA = {
    "required": [
        "memory_type",
        "memory_id",
        "situation_anchor",
        "local_goal",
        "uncertainty_state",
        "key_evidence",
        "missing_info",
        "active_hypotheses",
        "action_sequence",
        "outcome_shift",
        "outcome_type",
        "failure_mode",
        "boundary",
        "applicability_conditions",
        "non_applicability_conditions",
        "modality_flags",
        "retrieval_tags",
        "risk_tags",
        "confidence",
        "support_count",
        "source_episode_ids",
        "source_case_ids",
    ],
    "list_fields": [
        "key_evidence",
        "missing_info",
        "active_hypotheses",
        "action_sequence",
        "applicability_conditions",
        "non_applicability_conditions",
        "modality_flags",
        "retrieval_tags",
        "risk_tags",
        "source_episode_ids",
        "source_case_ids",
    ],
    "enum_fields": {
        "memory_type": ["experience"],
        "outcome_type": ["success", "partial_success", "failure", "unsafe"],
    },
    "range_fields": {"confidence": {"min": 0.0, "max": 1.0}, "support_count": {"min": 1, "max": 999999}},
}

EXPERIENCE_MERGE_SCHEMA = {
    "required": [
        "merge_decision",
        "target_memory_ids",
        "reason",
        "merged_experience",
        "conflict_group_id",
    ],
    "list_fields": ["target_memory_ids"],
    "dict_fields": ["merged_experience"],
    "enum_fields": {
        "merge_decision": ["insert_new", "merge", "keep_separate", "discard", "conflict"],
    },
}

SKILL_SCHEMA = {
    "required": [
        "memory_type",
        "memory_id",
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
        "source_experience_ids",
        "evidence_count",
        "confidence",
        "version",
    ],
    "list_fields": [
        "trigger_conditions",
        "procedure",
        "stop_conditions",
        "success_criteria",
        "failure_modes",
        "contraindications",
        "required_modalities",
        "source_experience_ids",
    ],
    "enum_fields": {"memory_type": ["skill"]},
    "range_fields": {"evidence_count": {"min": 0, "max": 999999}, "confidence": {"min": 0.0, "max": 1.0}, "version": {"min": 1, "max": 999999}},
}
