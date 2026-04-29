from __future__ import annotations

MEMORY_ROOT_DIRNAME = "memory_data"

RETRIEVAL_CONFIG = {
    "positive_experience_top_k": 5,
    "negative_experience_top_k": 3,
    "skill_top_k": 3,
    "knowledge_top_k": 3,
}

MERGE_CONFIG = {
    "semantic_threshold": 0.80,
    "action_threshold": 0.75,
    "boundary_threshold": 0.50,
}

SKILL_CONFIG = {
    "min_support_count": 5,
    "min_unique_cases": 3,
    "min_success_rate": 0.75,
    "max_unsafe_rate": 0.05,
}