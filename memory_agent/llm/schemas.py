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
    "enum_fields": {
        "retrieval_intent": ["experience", "skill", "knowledge", "mixed"],
        "finalize_risk_reason": ["low_confidence", "missing_evidence", "image_needed", "lab_needed", "history_needed", "other"],
    },
}

APPLICABILITY_JUDGE_SCHEMA = {
    "required": [
        "memory_id",
        "memory_type",
        "memory_content",
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
        "applicability": ["low", "medium", "high"],
        "controller_decision": ["apply", "hint", "escalate", "block"],
    },
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
    "enum_fields": {
        "memory_type": ["experience"],
        "success_signal": ["success", "partial", "failure", "unsafe"],
    },
    "range_fields": {
        "confidence": {"min": 0.0, "max": 1.0},
    },
    "nested_fields": {
        "action_sequence": {
            "list_items": {
                "required": ["action_type", "action_label"],
                "enum_fields": {
                    "action_type": ["ASK", "REQUEST_EXAM", "REQUEST_LAB", "REQUEST_IMAGING", "REVIEW_IMAGE", "REVIEW_HISTORY", "UPDATE_HYPOTHESIS", "DEFER_FINALIZE", "FINALIZE_DIAGNOSIS"],
                },
            }
        },
        "visual_signature": {},
    },
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
    "enum_fields": {
        "merge_decision": ["insert_new", "merge_with_existing", "discard", "conflict"],
    },
    "optional_fields": ["conflict_group_id"],
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
    "enum_fields": {
        "memory_type": ["skill"],
    },
    "range_fields": {
        "confidence": {"min": 0.0, "max": 1.0},
        "evidence_count": {"min": 0, "max": 999999},
        "version": {"min": 1, "max": 999999},
    },
    "nested_fields": {
        "procedure": {
            "list_items": {
                "required": ["action_type", "action_label", "expected_observation", "fallback_action"],
                "enum_fields": {
                    "action_type": ["ASK", "REQUEST_EXAM", "REQUEST_LAB", "REQUEST_IMAGING", "REVIEW_IMAGE", "REVIEW_HISTORY", "UPDATE_HYPOTHESIS", "DEFER_FINALIZE", "FINALIZE_DIAGNOSIS"],
                },
            }
        },
    },
}
