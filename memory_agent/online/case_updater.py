from __future__ import annotations

from typing import Any

from ..schemas import CaseState, EvidenceItem, HypothesisState
from ..utils.bench_adapter import nested_get, unwrap_osce_examination

CRITICAL_SLOT_TEMPLATES = {
    "brbpr": [
        "bleeding onset",
        "bleeding amount",
        "stool color",
        "abdominal pain",
        "dizziness or syncope",
        "anticoagulant or antiplatelet use",
    ],
    "gi bleed": [
        "bleeding onset",
        "bleeding amount",
        "stool color",
        "hemodynamic symptoms",
        "medication exposure",
        "prior bleeding history",
    ],
    "altered mental status": [
        "symptom onset",
        "baseline mental status",
        "focal neurologic symptoms",
        "fever or infection symptoms",
        "medication or substance exposure",
        "recent trauma",
    ],
    "chest pain": [
        "pain onset",
        "radiation",
        "exertional trigger",
        "associated dyspnea",
        "cardiac history",
        "troponin or ECG",
    ],
    "shortness of breath": [
        "onset and progression",
        "fever or cough",
        "leg swelling",
        "oxygen saturation",
        "cardiac history",
        "CXR or imaging",
    ],
}

HYPOTHESIS_MAP = {
    "brbpr": ["lower gastrointestinal bleeding", "thrombocytopenia", "anemia"],
    "gi bleed": ["upper gastrointestinal bleeding", "lower gastrointestinal bleeding", "anemia"],
    "altered mental status": ["stroke", "toxic-metabolic encephalopathy", "infection or sepsis"],
    "aphasia": ["stroke", "transient ischemic attack"],
    "chest pain": ["acute coronary syndrome", "pulmonary embolism", "aortic dissection"],
    "shortness of breath": ["pneumonia", "heart failure", "pulmonary embolism"],
    "fever": ["infection", "sepsis"],
    "cough": ["pneumonia", "bronchitis", "pulmonary edema"],
}


def _to_case_id(bundle: Any) -> str:
    if isinstance(bundle, dict):
        return str(bundle.get("case_id", ""))
    return str(getattr(bundle, "case_id", ""))


def _to_ehr(bundle: Any) -> dict[str, Any]:
    if isinstance(bundle, dict):
        return dict(bundle.get("ehr") or {})
    return dict(getattr(bundle, "ehr", {}) or {})


def init_case_state(bundle: Any, no_cxr: bool) -> CaseState:
    case_id = _to_case_id(bundle)
    ehr = _to_ehr(bundle)
    osce = unwrap_osce_examination(ehr)
    demographics = nested_get(osce, ["Patient_Actor", "Demographics"], {})
    history = nested_get(osce, ["Patient_Actor", "History"], {})
    symptoms = nested_get(osce, ["Patient_Actor", "Symptoms"], {})
    chief = str(symptoms.get("Chief_Complaint") or history.get("Chief_Complaint") or osce.get("Objective_for_Doctor", ""))
    objective = str(osce.get("Objective_for_Doctor", ""))

    modality_flags = ["text", "lab"]
    test_results = nested_get(osce, ["Test_Results"], {})
    if not no_cxr and bool(nested_get(test_results, ["CXR", "_present"], False)):
        modality_flags.append("image")

    return CaseState(
        case_id=case_id,
        turn_id=0,
        problem_summary=" | ".join([text for text in [chief, objective] if text]),
        evidence_items=[],
        missing_info=_critical_slots(chief or objective),
        active_hypotheses=[],
        local_goal="gather_high_value_evidence" if chief else "clarify_problem",
        uncertainty_summary="initial state",
        finalize_risk="high" if chief else "medium",
        modality_flags=modality_flags,
        next_action_constraints=["avoid_premature_finalize"],
        source_field_refs=[
            "OSCE_Examination.Objective_for_Doctor",
            "OSCE_Examination.Patient_Actor.Symptoms.Chief_Complaint",
            "OSCE_Examination.Patient_Actor.History.Chief_Complaint",
        ],
    )


def _critical_slots(chief: str) -> list[str]:
    lowered = (chief or "").lower()
    for key, slots in CRITICAL_SLOT_TEMPLATES.items():
        if key in lowered:
            return list(slots)
    return ["timeline", "associated symptoms", "targeted exam"]


def _append_evidence(case_state: CaseState, evidence_list: list[Any]) -> None:
    next_idx = len(case_state.evidence_items)
    for idx, evidence in enumerate(evidence_list):
        content = str(getattr(evidence, "content", "") or getattr(evidence, "raw_text", "") or "")
        if not content and getattr(evidence, "raw_structured", None):
            content = str(getattr(evidence, "raw_structured"))
        if not content:
            continue
        source = str(getattr(evidence, "source", "tool") or getattr(evidence, "source_type", "tool"))
        modality = str(getattr(evidence, "modality", "text"))
        item = EvidenceItem(
            evidence_id=str(getattr(evidence, "evidence_id", f"ev_{next_idx + idx}")),
            content=content[:400],
            source=source,
            modality=modality,
            polarity=str(getattr(evidence, "polarity", "positive")),
            confidence=float(getattr(evidence, "confidence", 0.7)),
            linked_hypotheses=list(getattr(evidence, "linked_hypotheses", []) or []),
            turn_id=case_state.turn_id,
            source_field_refs=list(getattr(evidence, "source_field_refs", []) or []),
        )
        case_state.evidence_items.append(item)


def _refresh_hypotheses(case_state: CaseState) -> None:
    text = " ".join(item.content.lower() for item in case_state.evidence_items[-30:])
    chief = (case_state.problem_summary or "").lower()
    names: list[str] = []
    for key, mapped in HYPOTHESIS_MAP.items():
        if key in chief or key in text:
            names.extend(mapped)

    unique = list(dict.fromkeys(names))[:6]
    hypotheses: list[HypothesisState] = []
    for name in unique:
        support = [item.evidence_id for item in case_state.evidence_items if name.lower() in item.content.lower()][:5]
        prob = "high" if len(support) >= 2 else "medium"
        hypotheses.append(
            HypothesisState(
                name=name,
                probability_hint=prob,
                supporting_evidence=support,
                conflicting_evidence=[],
                source_field_refs=["derived.hypothesis"],
            )
        )
    case_state.active_hypotheses = hypotheses


def _refresh_missing_and_risk(case_state: CaseState) -> None:
    text = " ".join(item.content.lower() for item in case_state.evidence_items)
    current = case_state.missing_info or _critical_slots(case_state.problem_summary)
    case_state.missing_info = [slot for slot in current if slot.lower() not in text][:8]

    if len(case_state.missing_info) >= 3:
        case_state.finalize_risk = "high"
        case_state.local_goal = "collect_missing_critical_info"
        case_state.uncertainty_summary = "multiple critical slots unresolved"
    elif len(case_state.missing_info) >= 1:
        case_state.finalize_risk = "medium"
        case_state.local_goal = "reduce_uncertainty"
        case_state.uncertainty_summary = "some critical slots unresolved"
    else:
        case_state.finalize_risk = "low"
        case_state.local_goal = "verify_before_finalize"
        case_state.uncertainty_summary = "core evidence mostly available"


def update_case_state(case_state: CaseState, evidence_list: list[Any]) -> CaseState:
    state = CaseState.from_dict(case_state.to_dict())
    state.turn_id += 1
    _append_evidence(state, evidence_list)
    _refresh_hypotheses(state)
    _refresh_missing_and_risk(state)
    return state
