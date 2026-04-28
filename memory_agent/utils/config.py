from __future__ import annotations

RETRIEVAL_CONFIG = {
    "experience_top_k": 5,
    "negative_experience_top_k": 3,
    "skill_top_k": 3,
    "knowledge_top_k": 3,
}

SCORING_CONFIG = {
    "experience": {
        "semantic_weight": 0.35,
        "hypothesis_weight": 0.20,
        "local_goal_weight": 0.15,
        "modality_weight": 0.10,
        "risk_weight": 0.10,
        "evidence_weight": 0.10,
    },
    "skill": {
        "trigger_weight": 0.25,
        "clinical_goal_weight": 0.20,
        "precondition_weight": 0.20,
        "modality_weight": 0.15,
        "success_rate_weight": 0.10,
        "boundary_weight": 0.10,
    },
}

APPLICABILITY_CONFIG = {
    "accept_threshold": 0.75,
    "weak_hint_threshold": 0.45,
    "reject_on_boundary_conflict": True,
    "reject_on_missing_modality": True,
    "block_premature_finalize": True,
}

EXPERIENCE_WRITE_CONFIG = {
    "max_action_sequence_len": 4,
    "min_outcome_shift_score": 0.5,
    "allow_failure_memory": True,
    "allow_unsafe_memory": True,
}

MERGE_CONFIG = {
    "semantic_threshold": 0.85,
    "action_edit_distance": 1,
    "min_hypothesis_overlap": 0.5,
}

SKILL_ABSTRACTION_CONFIG = {
    "min_support_count": 5,
    "min_success_rate": 0.75,
    "max_unsafe_rate": 0.05,
    "min_cross_case_support": 3,
}
