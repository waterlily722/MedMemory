from __future__ import annotations

MEMORY_ROOT_DIRNAME = "memory_data"
MEMORY_TRACE_DIRNAME = "logs/memory_trace"

RETRIEVAL_LIMITS = {
    "positive_experience_top_k": 5,
    "negative_experience_top_k": 3,
    "skill_top_k": 3,
    "knowledge_top_k": 3,
}

SCORING_CONFIG = {
    "experience": {
        "semantic_weight": 0.40,
        "goal_weight": 0.15,
        "hypothesis_weight": 0.15,
        "evidence_weight": 0.15,
        "boundary_weight": 0.15,
    },
    "skill": {
        "trigger_weight": 0.35,
        "goal_weight": 0.20,
        "precondition_weight": 0.20,
        "modality_weight": 0.10,
        "success_weight": 0.15,
    },
    "knowledge": {
        "semantic_weight": 0.65,
        "hypothesis_weight": 0.20,
        "modality_weight": 0.15,
    },
}

APPLICABILITY_CONFIG = {
    "accept_threshold": 0.7,
    "hint_threshold": 0.35,
    "block_threshold": 0.85,
}

MERGE_CONFIG = {
    "semantic_threshold": 0.82,
    "goal_threshold": 0.85,
    "action_threshold": 0.85,
}

SKILL_CONFIG = {
    "min_support_count": 5,
    "min_unique_cases": 3,
    "min_success_rate": 0.75,
    "max_unsafe_rate": 0.05,
}

MEMORY_RUNTIME_DEFAULTS = {
    "enable_memory": False,
    "query_builder_mode": "rule",
    "applicability_mode": "rule",
    "experience_extraction_mode": "rule",
    "experience_merge_mode": "rule",
    "memory_top_k": 5,
    "log_memory_trace": False,
    "disable_experience_memory": False,
    "disable_skill_memory": False,
    "disable_knowledge_memory": False,
}
