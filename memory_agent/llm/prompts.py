from __future__ import annotations

import json
from typing import Any

from .schemas import (
    APPLICABILITY_SCHEMA,
    EXPERIENCE_EXTRACTION_SCHEMA,
    EXPERIENCE_MERGE_SCHEMA,
    QUERY_BUILDER_SCHEMA,
    SKILL_SCHEMA,
)

STRICT_JSON_RULES = """
Return exactly one valid JSON object.
Do not use markdown fences, bullets, commentary, or free text outside JSON.
Follow the schema exactly. Keep every required field.
If a field is unsupported, keep the field and use an empty value.
Use concise strings. Do not invent source ids, case ids, or turn ids.
""".strip()


def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def query_builder_prompt(payload: dict[str, Any]) -> str:
    return f"""
You are a retrieval-query planner for a medical memory agent.

{STRICT_JSON_RULES}

Task:
Create one compact memory-retrieval query for the current diagnostic step.

Planning principles:
1. Internally decompose the current case into 2-3 retrieval aspects:
   - current clinical state / likely syndrome
   - key uncertainty, missing evidence, and unreviewed modalities
   - decision risk and the most useful next-action need
2. Prefer reusable methodological needs over literal transcript text.
   Good: "premature finalization risk with unresolved chest imaging"
   Bad: copying the entire patient dialogue.
3. Include visual or multimodal context when relevant:
   - CXR/image available but not reviewed
   - CXR/image findings uncertain
   - image-grounded evidence conflicts with history/labs
4. Mention candidate actions only when they change what memory should be retrieved.
5. Do not include patient identifiers or case-specific names.
6. The query_text may contain 1-3 semicolon-separated clauses, but must remain concise.

Schema:
{_dump(QUERY_BUILDER_SCHEMA)}

Input:
{_dump(payload)}
""".strip()


def applicability_prompt(payload: dict[str, Any]) -> str:
    return f"""
You are an applicability judge for a medical memory system.

{STRICT_JSON_RULES}

Task:
Judge whether one retrieved memory should influence the current diagnostic step.

Decision values:
- apply: the memory's trigger, boundary, and action match the current case well.
- hint: the memory is plausibly useful but incomplete, indirect, or lower confidence.
- block: the memory warns against a clearly unsafe or premature action in a matching situation.
- ignore: the memory is not relevant, over-specific, contradicted, or depends on unavailable evidence.

Medical safety rules:
1. Do not apply a memory just because the disease name or symptom overlaps.
   Check the uncertainty state, missing information, reviewed modalities, and intended action.
2. Skills are workflow guides. Usually mark them as "hint"; use "apply" only when the workflow boundary fits.
3. Failure experiences should discourage the risky action, not hard-block it, unless the outcome is unsafe and highly applicable.
4. Use "block" for actions such as FINALIZE_DIAGNOSIS only when the current case still lacks critical evidence or repeats a known unsafe pattern.
5. If image/CXR evidence is required by the memory but the current image is absent or unreviewed, downgrade to "hint" or "ignore".
6. Do not increase confidence in FINALIZE_DIAGNOSIS when finalize_risk is high or critical missing_info remains.

Action bias rules:
- action_bias keys must be selected only from allowed_actions in the input.
- Positive values encourage an action; negative values discourage an action.
- Use small magnitudes for hints, stronger negative values for unsafe actions.
- blocked_actions must contain only allowed action names.

Reason field:
Write one concise sentence explaining:
trigger match -> boundary check -> action implication.

Schema:
{_dump(APPLICABILITY_SCHEMA)}

Input:
{_dump(payload)}
""".strip()


def experience_extraction_prompt(payload: dict[str, Any]) -> str:
    return f"""
You are an experience extractor for a medical memory system.

{STRICT_JSON_RULES}

Task:
Extract 1 to 3 reusable local decision experiences from selected high-value turns.
Do not summarize the full case.

Extract only from turns with at least one of:
- high reward or clear improvement
- blocked unsafe action
- major diagnostic uncertainty reduction
- large outcome shift
- clinically important failure or near-miss

Each ExperienceCard should capture a tactical lesson:
1. situation_text:
   - Start with "When", "If", or "For".
   - Describe the clinical trigger, uncertainty state, and relevant modality status.
   - Keep it general enough to retrieve for similar future cases.
2. action_text:
   - Describe the local action or short action path.
   - Prefer allowed action types such as ASK, REQUEST_LAB, REVIEW_IMAGE, UPDATE_HYPOTHESIS, FINALIZE_DIAGNOSIS.
   - Include the reason this action was useful or risky.
3. outcome_text:
   - State the outcome shift: uncertainty reduced, hypothesis corrected, unsafe finalization avoided, or failure caused.
4. boundary_text:
   - Must be concrete and clinically reasoned.
   - Include both applies_when and do_not_use_when conditions in one concise sentence.
   - Mention missing modality/evidence that would invalidate the memory.
5. action_sequence:
   - Use 1-4 ordered action steps.
   - Each step should have action_type and action_label.
6. retrieval_tags:
   - Use reusable tags, not case ids.
   - Include clinical syndrome tags and methodology tags when useful.
7. risk_tags:
   - Add tags for premature_finalization, missing_modality, unsafe_action, conflicting_evidence, or similar risks when applicable.

Quality constraints:
- Do not copy patient-specific details unless clinically necessary.
- Do not write generic boundaries like "use when similar".
- Prefer actionable guidance over abstract advice.
- Include negative or unsafe experiences when they teach what to avoid.
- Each card should be concise, but preserve enough context to judge applicability.

Output format:
{{
  "experiences": [
    {{
      "memory_id": "...",
      "memory_type": "experience",
      "situation_text": "...",
      "action_text": "...",
      "outcome_text": "...",
      "boundary_text": "...",
      "action_sequence": [
        {{"action_type": "...", "action_label": "..."}}
      ],
      "outcome_type": "success|partial_success|failure|unsafe",
      "failure_mode": "",
      "retrieval_tags": [],
      "risk_tags": [],
      "confidence": 0.0,
      "support_count": 1,
      "conflict_group_id": "",
      "source_episode_ids": [],
      "source_case_ids": [],
      "source_turn_ids": []
    }}
  ]
}}

Schema:
{_dump(EXPERIENCE_EXTRACTION_SCHEMA)}

Input:
{_dump(payload)}
""".strip()


def experience_merge_prompt(payload: dict[str, Any]) -> str:
    return f"""
You are an experience-library curator for a medical memory system.

{STRICT_JSON_RULES}

Task:
Decide whether a new ExperienceCard should be inserted, merged, discarded, or marked as conflict.

Decision values:
- insert_new: the experience adds a genuinely new reusable lesson.
- merge: the new and existing memories share the same trigger, action pattern, and outcome direction.
- discard: the new memory is too obvious, too case-specific, redundant without improvement, or poorly supported.
- conflict: the same situation/action pattern has incompatible outcomes or safety implications.

Merge rules:
1. Merge only when situation_text and action_text are clinically and methodologically similar.
2. Never merge opposite outcomes, especially success vs unsafe/failure.
3. Preserve the strongest boundary conditions from all merged memories.
4. Generalize over case-specific entities using reusable clinical descriptions.
5. Keep important risk warnings even if they appear in only one memory.
6. If merging, merged_experience must be a complete ExperienceCard.
7. If not merging, merged_experience can be empty.

Quality standard:
- The merged memory should be concise, actionable, and reusable.
- situation_text should start with a trigger condition.
- boundary_text must say when to apply and when not to apply.
- Do not invent evidence beyond the input.

Schema:
{_dump(EXPERIENCE_MERGE_SCHEMA)}

Input:
{_dump(payload)}
""".strip()


def skill_consolidation_prompt(payload: dict[str, Any]) -> str:
    return f"""
You are a skill miner and SOP architect for a medical diagnostic memory system.

{STRICT_JSON_RULES}

Task:
Refine one reusable SkillCard from cross-episode repeated successful ExperienceCards.
The skill must be a task-level procedure template, not a case summary.

Skill design principles:
1. Build from repeated successful experiences only.
   Do not create a skill from one episode or one isolated case.
2. Generalize:
   Replace concrete patient details with placeholders such as:
   [presenting_problem], [uncertainty], [missing_evidence], [modality], [targeted_question], [test], [diagnostic_hypothesis].
3. Capture workflow:
   procedure_text should describe a short ordered clinical reasoning workflow.
   procedure should contain executable high-level steps using action_type/action_label.
4. Integrate safety:
   boundary_text must describe when the skill applies and when it should not be used.
   contraindications should include premature finalization, missing critical evidence, absent modality, or conflicting evidence when relevant.
5. Keep it lean:
   Remove redundant wording and low-value generic advice.
   Focus on steps that change the agent's next action.
6. Do not overfit:
   Avoid specific disease names unless the repeated experiences truly define a disease-specific workflow.

Recommended structure inside procedure_text:
- Trigger: when to consider this skill.
- Workflow: 2-5 ordered steps.
- Verification: what evidence should be checked before final diagnosis.
- Stop condition: when to avoid or downgrade the skill.

Schema:
{_dump(SKILL_SCHEMA)}

Input:
{_dump(payload)}
""".strip()