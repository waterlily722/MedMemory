from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List

from .schemas import CanonicalEvidence, MedEnvCaseBundle


NEGATION_PATTERNS = [
    re.compile(r"\bno evidence of ([^.;,\n]+)", re.IGNORECASE),
    re.compile(r"\bnegative for ([^.;,\n]+)", re.IGNORECASE),
    re.compile(r"\bdenies? ([^.;,\n]+)", re.IGNORECASE),
    re.compile(r"\bwithout ([^.;,\n]+)", re.IGNORECASE),
]

UNCERTAINTY_PATTERNS = [
    re.compile(r"\bpossible\b", re.IGNORECASE),
    re.compile(r"\bunclear\b", re.IGNORECASE),
    re.compile(r"\blimited by\b", re.IGNORECASE),
    re.compile(r"\bwithin these limitations\b", re.IGNORECASE),
    re.compile(r"\bcannot exclude\b", re.IGNORECASE),
    re.compile(r"\bconcern for\b", re.IGNORECASE),
]

SYMPTOM_KEYWORDS = [
    "altered mental status",
    "confusion",
    "aphasia",
    "chest pain",
    "shortness of breath",
    "dyspnea",
    "fever",
    "cough",
    "abdominal pain",
    "vomiting",
    "weakness",
    "syncope",
    "headache",
    "dizziness",
    "nausea",
    "palpitations",
]

TEST_KEYWORDS = [
    "ct",
    "mri",
    "xray",
    "cxr",
    "troponin",
    "lab",
    "labs",
    "microbiology",
    "culture",
    "glucose",
    "imaging",
    "ecg",
    "ekg",
    "vitals",
]


def _new_evidence_id(turn_id: str) -> str:
    return f"{turn_id}:{uuid.uuid4().hex[:10]}"


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _flatten_fact_candidates(text: str, structured: Any) -> list[str]:
    candidates: list[str] = []
    if text:
        for chunk in re.split(r"[\n.;]+", text):
            chunk = " ".join(chunk.strip().split())
            if len(chunk) < 4:
                continue
            candidates.append(chunk[:200])
    if isinstance(structured, dict):
        for key, value in structured.items():
            if isinstance(value, (str, int, float)) and str(value).strip():
                candidates.append(f"{key}: {value}")
            elif isinstance(value, list) and value:
                candidates.append(f"{key}: {_to_text(value[:3])[:200]}")
    elif isinstance(structured, list) and structured:
        for item in structured[:5]:
            candidates.append(_to_text(item)[:200])
    return candidates[:12]


def _extract_negations(text: str) -> list[str]:
    hits: list[str] = []
    for pattern in NEGATION_PATTERNS:
        hits.extend(match.group(1).strip() for match in pattern.finditer(text or ""))
    return list(dict.fromkeys(hit[:120] for hit in hits if hit.strip()))


def _extract_uncertainties(text: str) -> list[str]:
    lowered = text or ""
    hits = [pattern.pattern.strip("\\b") for pattern in UNCERTAINTY_PATTERNS if pattern.search(lowered)]
    return list(dict.fromkeys(hits))


def _extract_symptoms(text: str) -> list[str]:
    lowered = (text or "").lower()
    return [keyword for keyword in SYMPTOM_KEYWORDS if keyword in lowered]


def _extract_tests(text: str) -> list[str]:
    lowered = (text or "").lower()
    return [keyword for keyword in TEST_KEYWORDS if keyword in lowered]


def _build_evidence(
    turn_id: str,
    source_type: str,
    bundle_side: str,
    field_paths: list[str],
    raw_text: str = "",
    raw_structured: dict[str, Any] | None = None,
    raw_image_refs: list[str] | None = None,
) -> CanonicalEvidence:
    raw_structured = raw_structured or {}
    raw_image_refs = raw_image_refs or []
    fusion_text = " ".join([raw_text, _to_text(raw_structured), " ".join(raw_image_refs)]).strip()
    negated = _extract_negations(fusion_text)
    facts = [fact for fact in _flatten_fact_candidates(raw_text, raw_structured) if fact not in negated]
    return CanonicalEvidence(
        evidence_id=_new_evidence_id(turn_id),
        turn_id=turn_id,
        source_type=source_type,
        raw_field_refs={"bundle_side": bundle_side, "field_paths": field_paths},
        raw_text=raw_text,
        raw_structured=raw_structured,
        raw_image_refs=raw_image_refs,
        facts=facts,
        negated_facts=negated,
        uncertainty_patterns=_extract_uncertainties(fusion_text),
        symptom_patterns=_extract_symptoms(fusion_text),
        test_patterns=_extract_tests(fusion_text),
        route_flags={
            "for_case_update": True,
            "for_intent_planning": source_type != "static_env_field" or bool(raw_text),
            "for_memory_matching": True,
        },
        source_field_refs=field_paths,
    )


def canonicalize_static_case(bundle: MedEnvCaseBundle | dict[str, Any]) -> list[CanonicalEvidence]:
    if isinstance(bundle, dict):
        bundle = MedEnvCaseBundle.from_dict(bundle)

    evidences: list[CanonicalEvidence] = []
    turn_id = "static"
    static_specs = [
        ("ehr", "ehr.Meta", bundle.ehr.get("Meta"), ""),
        ("ehr", "ehr.Patient_info", bundle.ehr.get("Patient_info"), ""),
        ("ehr", "ehr.Objective_for_Doctor", None, bundle.ehr.get("Objective_for_Doctor", "")),
        ("ehr", "ehr.History.Chief_Complaint", None, ((bundle.ehr.get("History") or {}).get("Chief_Complaint", ""))),
        ("ehr", "ehr.History.HPI", None, ((bundle.ehr.get("History") or {}).get("HPI", ""))),
        ("ehr", "ehr.History.Past_Medical_History", None, ((bundle.ehr.get("History") or {}).get("Past_Medical_History", ""))),
        ("ehr", "ehr.History.Social_History", None, ((bundle.ehr.get("History") or {}).get("Social_History", ""))),
        ("ehr", "ehr.Physical_Examination_Findings", None, bundle.ehr.get("Physical_Examination_Findings", "")),
        ("ehr", "ehr.Test_Results-Labs", bundle.ehr.get("Test_Results-Labs"), ""),
        ("ehr", "ehr.Test_Results-Microbiology", bundle.ehr.get("Test_Results-Microbiology"), ""),
        ("ehr", "ehr.Test_Results-Imaging", bundle.ehr.get("Test_Results-Imaging"), ""),
        ("ehr", "ehr.CXR", bundle.ehr.get("CXR"), ""),
        ("ehr", "ehr.Medrecon", bundle.ehr.get("Medrecon"), ""),
    ]

    for bundle_side, field_path, raw_structured, raw_text in static_specs:
        if not raw_text and not raw_structured:
            continue
        raw_image_refs: list[str] = []
        if field_path == "ehr.CXR" and isinstance(raw_structured, list):
            for study in raw_structured:
                if not isinstance(study, dict):
                    continue
                for dicom in study.get("dicoms", []) or []:
                    if isinstance(dicom, dict):
                        if dicom.get("jpg_path_abs"):
                            raw_image_refs.append(str(dicom["jpg_path_abs"]))
                        elif dicom.get("jpg_path"):
                            raw_image_refs.append(str(dicom["jpg_path"]))
        evidences.append(
            _build_evidence(
                turn_id=turn_id,
                source_type="static_env_field",
                bundle_side=bundle_side,
                field_paths=[field_path],
                raw_text=raw_text,
                raw_structured=raw_structured if isinstance(raw_structured, dict) else {"value": raw_structured} if raw_structured else {},
                raw_image_refs=raw_image_refs,
            )
        )
    return evidences


def canonicalize_turn_input(turn_id: str, input_obj: dict[str, Any] | Any) -> list[CanonicalEvidence]:
    payload = input_obj if isinstance(input_obj, dict) else {"observation": input_obj}
    observation = payload.get("observation", payload)
    info = payload.get("info") or {}
    evidences: list[CanonicalEvidence] = []

    if isinstance(observation, dict) and isinstance(observation.get("question"), str):
        evidences.append(
            _build_evidence(
                turn_id=turn_id,
                source_type="patient_reply",
                bundle_side="ehr",
                field_paths=[],
                raw_text=observation["question"],
            )
        )

    if isinstance(observation, dict) and isinstance(observation.get("tool_outputs"), dict):
        tool_calls = info.get("response") or []
        tool_by_id: dict[str, str] = {}
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                tool_by_id[str(tc.get("id", ""))] = str(fn.get("name", ""))

        for tool_call_id, tool_output in observation["tool_outputs"].items():
            tool_name = tool_by_id.get(tool_call_id, "")
            parsed: Any = tool_output
            raw_text = tool_output if isinstance(tool_output, str) else _to_text(tool_output)
            raw_structured: dict[str, Any] = {}
            raw_image_refs: list[str] = []
            field_paths: list[str] = []
            source_type = "retrieve_output"

            if isinstance(tool_output, str):
                try:
                    parsed = json.loads(tool_output)
                except Exception:
                    parsed = tool_output

            if tool_name == "retrieve":
                source_type = "retrieve_output"
            elif tool_name == "request_exam":
                source_type = "request_exam_output"
                if isinstance(parsed, dict):
                    section = parsed.get("section")
                    if section:
                        field_paths = [f"ehr.{section}"]
            elif tool_name == "cxr":
                source_type = "cxr_output"
                field_paths = ["ehr.CXR"]
            elif tool_name == "cxr_grounding":
                source_type = "cxr_grounding_output"
                field_paths = ["ehr.CXR"]

            if isinstance(parsed, dict):
                raw_structured = parsed
                if tool_name == "cxr":
                    for image in parsed.get("images", []) or []:
                        if isinstance(image, dict) and image.get("jpg_path"):
                            raw_image_refs.append(str(image["jpg_path"]))
                if tool_name == "cxr_grounding" and parsed.get("image_path"):
                    raw_image_refs.append(str(parsed["image_path"]))

            evidences.append(
                _build_evidence(
                    turn_id=turn_id,
                    source_type=source_type,
                    bundle_side="ehr",
                    field_paths=field_paths,
                    raw_text=raw_text,
                    raw_structured=raw_structured,
                    raw_image_refs=raw_image_refs,
                )
            )

    return evidences
