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
        "decision",
        "reason",
        "action_bias",
        "blocked_actions",
    ],
    "list_fields": ["blocked_actions"],
    "dict_fields": ["action_bias"],
    "enum_fields": {
        "memory_type": ["experience", "skill", "knowledge"],
        "decision": ["apply", "hint", "block", "ignore"],
    },
}

EXPERIENCE_EXTRACTION_SCHEMA = {
    "required": [
        "memory_id",
        "memory_type",
        "situation_text",
        "action_text",
        "outcome_text",
        "boundary_text",
        "action_sequence",
        "outcome_type",
        "failure_mode",
        "retrieval_tags",
        "risk_tags",
        "confidence",
        "support_count",
        "conflict_group_id",
        "source_episode_ids",
        "source_case_ids",
        "source_turn_ids",
    ],
    "list_fields": [
        "action_sequence",
        "retrieval_tags",
        "risk_tags",
        "source_episode_ids",
        "source_case_ids",
        "source_turn_ids",
    ],
    "enum_fields": {
        "memory_type": ["experience"],
        "outcome_type": ["success", "partial_success", "failure", "unsafe"],
    },
    "range_fields": {
        "confidence": {"min": 0.0, "max": 1.0},
        "support_count": {"min": 1, "max": 999999},
    },
}

SKILL_SCHEMA = {
    "required": [
        "memory_id",
        "memory_type",
        "skill_name",
        "situation_text",
        "goal_text",
        "procedure_text",
        "boundary_text",
        "procedure",
        "contraindications",
        "source_experience_ids",
        "evidence_count",
        "unique_case_count",
        "success_rate",
        "unsafe_rate",
        "confidence",
        "version",
    ],
    "list_fields": [
        "procedure",
        "contraindications",
        "source_experience_ids",
    ],
    "enum_fields": {"memory_type": ["skill"]},
    "range_fields": {
        "evidence_count": {"min": 0, "max": 999999},
        "unique_case_count": {"min": 0, "max": 999999},
        "success_rate": {"min": 0.0, "max": 1.0},
        "unsafe_rate": {"min": 0.0, "max": 1.0},
        "confidence": {"min": 0.0, "max": 1.0},
        "version": {"min": 1, "max": 999999},
    },
}