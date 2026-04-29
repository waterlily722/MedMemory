from __future__ import annotations

import json
import re
import uuid
from typing import Any

from ..schemas import CanonicalEvidence, MedEnvCaseBundle
from .bench_adapter import nested_get, unwrap_osce_examination


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

    osce = unwrap_osce_examination(bundle.ehr)

    evidences: list[CanonicalEvidence] = []
    turn_id = "static"
    static_specs = [
        ("ehr", "OSCE_Examination.Meta", osce.get("Meta"), ""),
        ("ehr", "OSCE_Examination.Objective_for_Doctor", None, osce.get("Objective_for_Doctor", "")),
        ("ehr", "OSCE_Examination.Patient_Actor.Demographics", nested_get(osce, ["Patient_Actor", "Demographics"], {}), ""),
        ("ehr", "OSCE_Examination.Patient_Actor.History.HPI", None, nested_get(osce, ["Patient_Actor", "History", "HPI"], "")),
        ("ehr", "OSCE_Examination.Patient_Actor.History.Past_Medical_History", None, nested_get(osce, ["Patient_Actor", "History", "Past_Medical_History"], "")),
        ("ehr", "OSCE_Examination.Patient_Actor.History.Social_History", None, nested_get(osce, ["Patient_Actor", "History", "Social_History"], "")),
        ("ehr", "OSCE_Examination.Patient_Actor.Symptoms", nested_get(osce, ["Patient_Actor", "Symptoms"], {}), ""),
        ("ehr", "OSCE_Examination.Physical_Examination_Findings.vitals", nested_get(osce, ["Physical_Examination_Findings", "vitals"], {}), ""),
        ("ehr", "OSCE_Examination.Physical_Examination_Findings.general_exam", None, nested_get(osce, ["Physical_Examination_Findings", "general_exam"], "")),
        ("ehr", "OSCE_Examination.Test_Results.Labs", nested_get(osce, ["Test_Results", "Labs"], {}), ""),
        ("ehr", "OSCE_Examination.Test_Results.Microbiology", nested_get(osce, ["Test_Results", "Microbiology"], {}), ""),
        ("ehr", "OSCE_Examination.Test_Results.Imaging", nested_get(osce, ["Test_Results", "Imaging"], {}), ""),
        ("ehr", "OSCE_Examination.Test_Results.CXR", nested_get(osce, ["Test_Results", "CXR"], {}), ""),
        ("ehr", "OSCE_Examination.Test_Results.Hosp", nested_get(osce, ["Test_Results", "Hosp"], {}), ""),
        ("ehr", "OSCE_Examination.Correct_Diagnosis", None, _to_text(osce.get("Correct_Diagnosis", {}))),
        ("ehr", "OSCE_Examination.Principal_Diagnosis", None, _to_text(osce.get("Principal_Diagnosis", {}))),
    ]

    for bundle_side, field_path, raw_structured, raw_text in static_specs:
        if not raw_text and not raw_structured:
            continue
        raw_image_refs: list[str] = []
        if field_path == "OSCE_Examination.Test_Results.CXR" and isinstance(raw_structured, dict):
            studies = raw_structured.get("studies", [])
            for study in studies if isinstance(studies, list) else []:
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

    knowledge_block = nested_get(osce, ["knowledge", "principal_diagnosis", "matched_knowledge"], [])
    if isinstance(knowledge_block, list):
        for idx, item in enumerate(knowledge_block[:12]):
            if not isinstance(item, dict):
                continue
            knowledge_text = "\n".join(
                str(part)
                for part in [
                    item.get("introduction", ""),
                    item.get("signs_and_symptoms", ""),
                    item.get("diagnosis", ""),
                ]
                if part
            )
            evidences.append(
                _build_evidence(
                    turn_id=turn_id,
                    source_type="knowledge_item",
                    bundle_side="ehr",
                    field_paths=[f"OSCE_Examination.knowledge.principal_diagnosis.matched_knowledge[{idx}]"],
                    raw_text=knowledge_text,
                    raw_structured=item,
                    raw_image_refs=[],
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
                if fn.get("name") and tc.get("id"):
                    tool_by_id[str(tc.get("id"))] = str(fn.get("name"))
        for tool_id, output in (observation.get("tool_outputs") or {}).items():
            tool_name = tool_by_id.get(str(tool_id), "")
            evidences.append(
                _build_evidence(
                    turn_id=turn_id,
                    source_type=tool_name or "tool",
                    bundle_side="ehr",
                    field_paths=[f"tool_outputs.{tool_id}"],
                    raw_text=_to_text(output)[:800],
                    raw_structured={"tool": tool_name, "payload": output},
                )
            )

    if isinstance(info, dict) and info.get("metadata"):
        evidences.append(
            _build_evidence(
                turn_id=turn_id,
                source_type="env_metadata",
                bundle_side="ehr",
                field_paths=["info.metadata"],
                raw_structured=info.get("metadata"),
            )
        )

    return evidences