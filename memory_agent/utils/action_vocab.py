from __future__ import annotations

ACTION_TYPES = {
    "ASK",
    "REQUEST_EXAM",
    "REQUEST_LAB",
    "REQUEST_IMAGING",
    "REVIEW_IMAGE",
    "REVIEW_HISTORY",
    "UPDATE_HYPOTHESIS",
    "DEFER_FINALIZE",
    "FINALIZE_DIAGNOSIS",
}

ASK_LABELS = {
    "ask_onset",
    "ask_duration",
    "ask_pain_location",
    "ask_fever",
    "ask_dyspnea",
    "ask_orthopnea",
    "ask_medication_history",
}

LAB_LABELS = {
    "order_CBC",
    "order_BNP",
    "order_D_dimer",
    "order_troponin",
}

IMAGE_REVIEW_LABELS = {
    "review_opacity_pattern",
    "review_fracture_line",
    "review_cardiomegaly",
    "review_effusion",
}

FINALIZE_LABELS = {"finalize_primary_diagnosis"}
