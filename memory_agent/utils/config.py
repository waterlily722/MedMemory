from __future__ import annotations

from pathlib import Path


MEMORY_ROOT_DIRNAME = str(Path(__file__).resolve().parents[1] / "memory_data")

RETRIEVAL_CONFIG = {
    "positive_experience_top_k": 5,
    "negative_experience_top_k": 3,
    "skill_top_k": 3,
    "knowledge_top_k": 3,
    # Retrieval thresholds. Negative memory uses a higher threshold because it can
    # discourage or block actions downstream.
    "positive_experience_min_score": 0.18,
    "negative_experience_min_score": 0.25,
    "skill_min_score": 0.20,
    "knowledge_min_score": 0.12,
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

MEMORY_ACTION_CONFIG = {
    "tool_to_action": {
        "ask_patient": "ASK",
        "request_exam": "REQUEST_LAB",
        "retrieve": "REQUEST_LAB",
        "cxr": "REVIEW_IMAGE",
        "cxr_grounding": "REVIEW_IMAGE",
        "diagnosis": "FINALIZE_DIAGNOSIS",
    },
    "default_actions": [
        "ASK",
        "REQUEST_LAB",
        "REVIEW_IMAGE",
        "UPDATE_HYPOTHESIS",
        "FINALIZE_DIAGNOSIS",
    ],
    "always_include_actions": ["UPDATE_HYPOTHESIS"],
    "finalize_action": "FINALIZE_DIAGNOSIS",
}

CASE_STATE_CONFIG = {
    "initial_finalize_risk": "high",
    "default_local_goal": "collect_missing_critical_info",
    "default_uncertainty_summary": "initial state with unresolved diagnostic uncertainty",
    "fallback_problem_summary": "initial clinical problem unclear",
    "risk_levels": ["low", "medium", "high"],
    "critical_slot_templates": {
        "chest pain": [
            "onset",
            "radiation",
            "associated dyspnea",
            "exertional trigger",
            "cardiac history",
            "ECG or troponin",
        ],
        "shortness of breath": [
            "onset",
            "progression",
            "fever or cough",
            "oxygen saturation",
            "cardiac history",
            "CXR or imaging",
        ],
        "altered mental status": [
            "onset",
            "baseline mental status",
            "focal neurologic symptoms",
            "fever",
            "medication exposure",
            "trauma",
        ],
        "abdominal pain": [
            "onset",
            "location",
            "duration",
            "fever",
            "vomiting",
            "stool or urinary symptoms",
        ],
        "bleeding": [
            "amount",
            "duration",
            "hemodynamic symptoms",
            "medication exposure",
            "prior bleeding history",
        ],
    },
    "default_critical_slots": [
        "timeline",
        "associated symptoms",
        "severity",
        "risk factors",
        "targeted exam",
    ],
}

# Applicability heuristics and thresholds used by the online controller
APPLICABILITY_CONFIG = {
    "unsafe_block_score": 0.35,
    "skill_apply_score": 0.40,
    "hard_block_finalize_on_high_risk": True,
    "hard_block_finalize_missing_info_min": 3,
    "image_unreviewed_warning": True,
    "hard_block_reason": "Configured safety rule blocked this action.",
    "risk_warning_text": {
        "high_finalize_risk": "finalize_risk is high",
        "missing_info": "multiple critical missing information slots remain unresolved",
        "image_unreviewed": "image modality is available but not yet reviewed",
    },
}

TRACE_CONFIG = {
    # Compact trace is for memory debugging. Full turn details remain in trajectory_*.json.
    "format": "compact",
    "write_jsonl_snapshot": False,
    "include_full_turn": False,
    "include_observation_payload": False,
    "include_llm_io": False,
    "include_prompt_payload": False,
    "max_text_chars": 600,
    "max_hits_per_type": 5,
}
