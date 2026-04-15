from __future__ import annotations

from typing import Any, Dict, List

from .schemas import CanonicalEvidence, CaseMemory, MedEnvCaseBundle


CRITICAL_SLOT_TEMPLATES = {
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
    "abdominal pain": [
        "pain location",
        "pain onset",
        "nausea or vomiting",
        "bowel habit changes",
        "fever",
        "abdominal imaging",
    ],
}

HYPOTHESIS_MAP = {
    "altered mental status": ["stroke", "toxic-metabolic encephalopathy", "infection or sepsis"],
    "aphasia": ["stroke", "transient ischemic attack"],
    "chest pain": ["acute coronary syndrome", "pulmonary embolism", "aortic dissection"],
    "shortness of breath": ["pneumonia", "heart failure", "pulmonary embolism"],
    "fever": ["infection", "sepsis"],
    "cough": ["pneumonia", "bronchitis", "pulmonary edema"],
    "abdominal pain": ["appendicitis", "cholecystitis", "bowel obstruction"],
}


def _bundle(bundle: MedEnvCaseBundle | dict[str, Any]) -> MedEnvCaseBundle:
    return bundle if isinstance(bundle, MedEnvCaseBundle) else MedEnvCaseBundle.from_dict(bundle)


def _deep_copy_case_memory(case_memory: CaseMemory) -> CaseMemory:
    return CaseMemory.from_dict(case_memory.to_dict())


def _extract_cxr_snapshot(ehr: dict[str, Any], no_cxr: bool) -> dict[str, Any]:
    studies = ehr.get("CXR") or []
    labels: list[str] = []
    report_texts: list[str] = []
    image_refs: list[str] = []
    for study in studies if isinstance(studies, list) else []:
        if not isinstance(study, dict):
            continue
        if study.get("report_text"):
            report_texts.append(str(study.get("report_text")))
        for dicom in study.get("dicoms", []) or []:
            if not isinstance(dicom, dict):
                continue
            if dicom.get("view_position"):
                labels.append(str(dicom.get("view_position")))
            if dicom.get("jpg_path_abs"):
                image_refs.append(str(dicom["jpg_path_abs"]))
            elif dicom.get("jpg_path"):
                image_refs.append(str(dicom["jpg_path"]))
    return {
        "present": bool(studies) and not no_cxr,
        "studies": studies if isinstance(studies, list) else [],
        "labels": labels,
        "report_texts": report_texts,
        "image_refs": image_refs,
    }


def _initial_derived_state(raw_snapshot: dict[str, Any], no_cxr: bool) -> dict[str, Any]:
    chief = (raw_snapshot.get("history") or {}).get("chief_complaint", "")
    return {
        "confirmed_facts": [],
        "ruled_out_facts": [],
        "missing_critical_slots": [],
        "active_uncertainties": [],
        "active_hypotheses": [],
        "tentative_differential": [],
        "differential_confidence": [],
        "modality_state": {
            "available": {
                "retrieve": True,
                "cxr": not no_cxr and bool((raw_snapshot.get("tests") or {}).get("cxr", {}).get("present")),
                "cxr_report": bool((raw_snapshot.get("tests") or {}).get("cxr", {}).get("report_texts")),
                "cxr_grounding": not no_cxr and bool((raw_snapshot.get("tests") or {}).get("cxr", {}).get("present")),
            },
            "needed": {"retrieve": False, "cxr": False, "cxr_grounding": False},
        },
        "safety_state": {
            "premature_finalize_risk": "mid" if chief else "high",
            "evidence_conflict_level": "low",
            "dangerous_alternatives_not_ruled_out": [],
            "repeated_low_yield_streak": 0,
        },
        "interaction_state": {
            "asked_questions": [],
            "tool_calls_so_far": [],
            "recent_actions": [],
            "recent_useful_evidence_refs": [],
            "recent_local_gain": [],
            "turn_index": 0,
        },
    }


def init_case_memory(bundle: MedEnvCaseBundle | dict[str, Any], no_cxr: bool) -> CaseMemory:
    bundle = _bundle(bundle)
    ehr = bundle.ehr or {}
    history = ehr.get("History") or {}

    raw_snapshot = {
        "patient": {
            "demographics": (ehr.get("Patient_info") or {}),
        },
        "history": {
            "chief_complaint": history.get("Chief_Complaint", ""),
            "hpi": history.get("HPI", ""),
            "past_medical_history": history.get("Past_Medical_History", ""),
            "social_history": history.get("Social_History", ""),
        },
        "exam": {
            "physical_exam_text": ehr.get("Physical_Examination_Findings", ""),
            "triage_vitals": {},
            "ed_vitalsign_series": [],
        },
        "tests": {
            "imaging_reports": ehr.get("Test_Results-Imaging") or [],
            "cxr": _extract_cxr_snapshot(ehr, no_cxr=no_cxr),
        },
        "actions": {
            "orders_general": [],
            "medication_requests": [],
        },
        "medication_context": {
            "medrecon": ((ehr.get("Medrecon") or {}).get("medrecon", [])),
            "pyxis": ((ehr.get("Medrecon") or {}).get("pyxis", [])),
        },
        "tool_outputs": [],
    }

    meta = {
        "subject_id": ((ehr.get("Meta") or {}).get("subject_id")),
        "hadm_id": ((ehr.get("Meta") or {}).get("hadm_id")),
        "stay_id": ((ehr.get("Meta") or {}).get("stay_id")),
        "objective_for_doctor": ehr.get("Objective_for_Doctor", ""),
        "source_meta_refs": ["ehr.Meta"],
    }

    return CaseMemory(
        meta=meta,
        raw_snapshot=raw_snapshot,
        derived_state=_initial_derived_state(raw_snapshot, no_cxr=no_cxr),
        source_field_refs=["ehr.Meta"],
    )


def _choose_slot_template(chief_complaint: str, symptom_patterns: list[str]) -> list[str]:
    lowered = (chief_complaint or "").lower()
    for key, template in CRITICAL_SLOT_TEMPLATES.items():
        if key in lowered:
            return template
    for symptom in symptom_patterns:
        for key, template in CRITICAL_SLOT_TEMPLATES.items():
            if key in symptom:
                return template
    return ["timeline", "associated symptoms", "targeted exam or test"]


def _recompute_derived_state(case_memory: CaseMemory, new_evidence: list[CanonicalEvidence], execution_result=None) -> None:
    derived = case_memory.derived_state
    confirmed = list(dict.fromkeys(derived.get("confirmed_facts", [])))
    ruled_out = list(dict.fromkeys(derived.get("ruled_out_facts", [])))
    uncertainties = list(dict.fromkeys(derived.get("active_uncertainties", [])))
    symptom_patterns = []
    test_patterns = []
    useful_refs = list(derived.get("interaction_state", {}).get("recent_useful_evidence_refs", []))

    for evidence in new_evidence:
        confirmed.extend(evidence.facts)
        ruled_out.extend(evidence.negated_facts)
        uncertainties.extend(evidence.uncertainty_patterns)
        symptom_patterns.extend(evidence.symptom_patterns)
        test_patterns.extend(evidence.test_patterns)
        if evidence.facts or evidence.negated_facts or evidence.uncertainty_patterns:
            useful_refs.append(evidence.evidence_id)

    confirmed = list(dict.fromkeys(confirmed))[:80]
    ruled_out = list(dict.fromkeys(ruled_out))[:60]
    uncertainties = list(dict.fromkeys(uncertainties))[:20]
    symptom_patterns = list(dict.fromkeys(symptom_patterns))

    chief = (case_memory.raw_snapshot.get("history") or {}).get("chief_complaint", "")
    slot_template = _choose_slot_template(chief, symptom_patterns)
    evidence_text = " ".join(confirmed + ruled_out + symptom_patterns + test_patterns).lower()
    missing_slots = [slot for slot in slot_template if slot.lower() not in evidence_text]

    hypotheses: list[str] = []
    for keyword, mapped in HYPOTHESIS_MAP.items():
        if keyword in chief.lower() or keyword in " ".join(symptom_patterns):
            hypotheses.extend(mapped)
    if "aphasia" in evidence_text or "stroke" in evidence_text:
        hypotheses.extend(["stroke", "transient ischemic attack"])
    if "troponin" in evidence_text or "stent" in evidence_text:
        hypotheses.extend(["acute coronary syndrome"])
    hypotheses = list(dict.fromkeys(hypotheses))[:6]

    differential = hypotheses[:3]
    differential_confidence = []
    for idx, label in enumerate(differential):
        score = max(0.25, 0.7 - 0.15 * idx)
        if label.lower() in evidence_text:
            score = min(0.9, score + 0.15)
        differential_confidence.append({"label": label, "score": round(score, 2)})

    modality_state = derived.get("modality_state", {})
    modality_state["needed"] = {
        "retrieve": bool(uncertainties or missing_slots),
        "cxr": bool(modality_state.get("available", {}).get("cxr") and ("cough" in symptom_patterns or "shortness of breath" in symptom_patterns)),
        "cxr_grounding": bool(modality_state.get("available", {}).get("cxr_grounding") and any("limited" in u for u in uncertainties)),
    }

    dangerous = []
    if "acute coronary syndrome" in hypotheses and not any("troponin" in fact.lower() for fact in confirmed):
        dangerous.append("acute coronary syndrome")
    if "stroke" in hypotheses and not any("ct" in fact.lower() or "neuro" in fact.lower() for fact in confirmed):
        dangerous.append("stroke")
    if "pulmonary embolism" in hypotheses and not any("oxygen" in fact.lower() or "cxr" in fact.lower() for fact in confirmed):
        dangerous.append("pulmonary embolism")

    repeated_low_yield = derived.get("safety_state", {}).get("repeated_low_yield_streak", 0)
    if execution_result is not None and execution_result.execution_status == "no_effect":
        repeated_low_yield += 1
    elif new_evidence:
        repeated_low_yield = 0

    conflict_level = "high" if len(uncertainties) >= 3 else "mid" if uncertainties else "low"
    finalize_risk = "high" if len(missing_slots) >= 3 or dangerous else "mid" if len(missing_slots) >= 1 or uncertainties else "low"

    derived["confirmed_facts"] = confirmed
    derived["ruled_out_facts"] = ruled_out
    derived["missing_critical_slots"] = missing_slots
    derived["active_uncertainties"] = uncertainties
    derived["active_hypotheses"] = hypotheses
    derived["tentative_differential"] = differential
    derived["differential_confidence"] = differential_confidence
    derived["modality_state"] = modality_state
    derived["safety_state"] = {
        "premature_finalize_risk": finalize_risk,
        "evidence_conflict_level": conflict_level,
        "dangerous_alternatives_not_ruled_out": dangerous,
        "repeated_low_yield_streak": repeated_low_yield,
    }
    derived.setdefault("interaction_state", {})
    derived["interaction_state"]["recent_useful_evidence_refs"] = useful_refs[-12:]


def _update_raw_snapshot(case_memory: CaseMemory, evidence_list: list[CanonicalEvidence], execution_result=None) -> None:
    snapshot = case_memory.raw_snapshot
    tool_outputs = snapshot.setdefault("tool_outputs", [])
    for evidence in evidence_list:
        if evidence.source_type == "patient_reply" and evidence.raw_text:
            tool_outputs.append({"type": "patient_reply", "text": evidence.raw_text, "evidence_id": evidence.evidence_id})
        elif evidence.source_type in {"retrieve_output", "request_exam_output", "cxr_output", "cxr_grounding_output"}:
            tool_outputs.append(
                {
                    "type": evidence.source_type,
                    "text": evidence.raw_text[:500],
                    "structured": evidence.raw_structured,
                    "evidence_id": evidence.evidence_id,
                }
            )
            if evidence.source_type == "request_exam_output":
                if evidence.raw_structured.get("source") == "ehr":
                    snapshot["tests"].setdefault("imaging_reports", []).append(evidence.raw_structured)
            if evidence.source_type == "cxr_output":
                snapshot["tests"]["cxr"]["report_texts"].append(str(evidence.raw_structured.get("report_text", "")))
                snapshot["tests"]["cxr"]["image_refs"].extend(evidence.raw_image_refs)

    if execution_result is not None and execution_result.executed_action:
        interaction = case_memory.derived_state.setdefault("interaction_state", {})
        action_type = execution_result.executed_action.get("action_type", "")
        action_text = execution_result.executed_action.get("action_text", "")
        interaction.setdefault("recent_actions", []).append({"action_type": action_type, "action_text": action_text})
        interaction["recent_actions"] = interaction["recent_actions"][-10:]


def _update_interaction_state(case_memory: CaseMemory, executed_action=None, evidence_list: list[CanonicalEvidence] | None = None) -> None:
    interaction = case_memory.derived_state.setdefault("interaction_state", {})
    interaction.setdefault("asked_questions", [])
    interaction.setdefault("tool_calls_so_far", [])
    interaction.setdefault("recent_actions", [])
    interaction.setdefault("recent_local_gain", [])
    if executed_action is not None:
        interaction["turn_index"] = int(interaction.get("turn_index", 0)) + 1
    else:
        interaction["turn_index"] = int(interaction.get("turn_index", 0))

    if isinstance(executed_action, dict):
        action_type = executed_action.get("action_type", "")
        action_text = executed_action.get("action_text", "")
        if action_type == "ask" and action_text:
            interaction["asked_questions"].append(action_text)
            interaction["asked_questions"] = interaction["asked_questions"][-20:]
        if action_type:
            interaction["tool_calls_so_far"].append(action_type)
            interaction["tool_calls_so_far"] = interaction["tool_calls_so_far"][-20:]

    evidence_list = evidence_list or []
    local_gain = {
        "turn_index": interaction["turn_index"],
        "evidence_gain": sum(1 for ev in evidence_list if ev.facts or ev.negated_facts or ev.uncertainty_patterns),
    }
    interaction["recent_local_gain"].append(local_gain)
    interaction["recent_local_gain"] = interaction["recent_local_gain"][-10:]


def update_case_memory(
    prev_case_memory: CaseMemory,
    evidence_list: list[CanonicalEvidence],
    executed_action=None,
    execution_result=None,
) -> CaseMemory:
    case_memory = _deep_copy_case_memory(prev_case_memory)
    _update_raw_snapshot(case_memory, evidence_list=evidence_list, execution_result=execution_result)
    _recompute_derived_state(case_memory, new_evidence=evidence_list, execution_result=execution_result)
    _update_interaction_state(case_memory, executed_action=executed_action, evidence_list=evidence_list)
    return case_memory
