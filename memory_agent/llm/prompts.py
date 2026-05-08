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
Use concise strings.
Do not invent source ids, case ids, turn ids, diagnoses, test results, or image findings.
""".strip()


def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def query_builder_prompt(payload: dict[str, Any]) -> str:
    return f"""
You are a clinical memory-retrieval intent planner for a doctor agent.

{STRICT_JSON_RULES}

Task:
Create one compact query_text for retrieving useful memories at the current diagnostic step.

The input contains compact case_memory, not the full transcript:
- current_turn is the newly exposed information for this turn.
- recent_key_evidence and recent_negative_evidence are prior compact state.
- Build the query from current_turn plus the current case state.
- Do not reconstruct or repeat the whole interaction history.

The query should represent what an experienced doctor would want to remember right now,
not merely the suspected diagnosis.

Think clinically:
1. What is the current patient problem representation?
2. What uncertainty is still unresolved?
3. What evidence has already been checked?
4. What evidence is still missing or unreviewed?
5. What is the most valuable next action?
6. What diagnostic mistake should be avoided?

The query should retrieve memories about:
- useful next-step decisions
- targeted questions
- useful tests or image review
- hypothesis updating
- missed evidence
- premature diagnostic closure
- unsafe or low-value actions to avoid

Do not:
- copy the full transcript
- write a generic disease query
- ask only "what is the diagnosis?"
- overfit to patient-specific details
- include patient identifiers

Good query examples:
- "acute chest pain with unresolved imaging evidence; need memory about reviewing available CXR before final diagnosis; avoid premature closure"
- "fever and cough with uncertain pneumonia versus viral illness; missing oxygen status and imaging review; need targeted evidence gathering"
- "abdominal pain with broad differential and incomplete labs; need memory about narrowing diagnosis before finalization"
- "neurologic complaint with unclear focal deficits; need targeted neuro history and exam before imaging decision"

Bad query examples:
- "patient has chest pain"
- "retrieve similar case"
- "diagnose pneumonia"
- copying the whole conversation

Rules:
- query_text should contain 2-4 short clinical clauses.
- Include image, CXR, lab, or multimodal status when clinically relevant.
- Mention FINALIZE_DIAGNOSIS only if the current issue is whether finalization is safe.
- Prefer reusable reasoning needs over literal case details.

Schema:
{_dump(QUERY_BUILDER_SCHEMA)}

Input:
{_dump(payload)}
""".strip()


def applicability_prompt(payload: dict[str, Any]) -> str:
    return f"""
You are a clinical memory applicability judge for a doctor agent.

{STRICT_JSON_RULES}

Task:
Select whether one retrieved memory is reusable for the doctor's current next action.

A memory is useful only if it matches the CURRENT DECISION POINT.
Do not apply a memory merely because the symptom, disease label, or organ system overlaps.

Compare the retrieved memory with the current case using:
1. clinical trigger
2. uncertainty state
3. evidence already reviewed
4. missing or unreviewed evidence
5. intended next action
6. safety boundary

Decision values:
- apply:
  The memory is reusable now: it closely matches the current trigger,
  uncertainty state, boundary, and action timing.
  It should be included as active guidance.

- ignore:
  The memory is not reusable now: it is irrelevant, too generic, too case-specific,
  only partially matched, contradicted by current evidence, or depends on unavailable evidence.

Clinical reasoning rules:
1. Never apply a memory only because diagnoses overlap.
   The action timing and uncertainty state must also match.
2. The memory should answer:
   "What should the doctor notice, ask, review, update, or avoid right now?"
3. Skills are workflow reminders, not final answers.
   Apply a skill only when its trigger and boundary match strongly.
4. Experiences are local decision lessons.
   Apply them only when the current decision point is similar.
5. Negative or unsafe memories can be reusable only when they warn against the
   same risky action in the same decision context. Otherwise ignore them.
6. If the memory requires image, CXR, lab, or exam evidence that has not been obtained or reviewed,
   ignore it.
7. Do not support FINALIZE_DIAGNOSIS if:
   - key evidence is still missing
   - the differential remains broad
   - available image/lab results are unreviewed
   - the memory itself warns against premature closure
8. Memory should not override current patient evidence.
   It is a clinical reminder, not ground truth.

Action bias rules:
- For ignore, action_bias must be empty and blocked_actions must be empty.
- For apply, action_bias may contain only the specific allowed action(s) this memory supports or warns against.
- Positive values encourage an action; negative values discourage an action.
- Keep magnitudes small and conservative.
- blocked_actions must stay empty. Hard safety blocking is handled outside this memory selector.
- Do not invent action names.

Reason field:
Write one concise clinical sentence:
matched cue -> boundary check -> implication for the next action.

Good reason examples:
- "The memory matches unresolved chest imaging before final diagnosis, so it is reusable to encourage REVIEW_IMAGE and discourage premature FINALIZE_DIAGNOSIS."
- "The symptom overlap is present, but the memory depends on CXR findings not yet available, so it is not reusable now."
- "The memory concerns a later post-lab decision point, while this case still needs initial history, so it should be ignored."

Schema:
{_dump(APPLICABILITY_SCHEMA)}

Input:
{_dump(payload)}
""".strip()


def experience_extraction_prompt(payload: dict[str, Any]) -> str:
    return f"""
You are a clinical experience extractor for a medical memory system.

{STRICT_JSON_RULES}

Task:
Extract 1 to 3 reusable ExperienceCards from selected high-value turns.

The goal is NOT to summarize the full case.
The goal is to capture what a real doctor should remember from this interaction
when facing a similar future diagnostic decision point.

Extract an experience only if it teaches one of:
- a useful next-step decision
- a targeted question that reduced uncertainty
- a test, image review, or exam step that changed reasoning
- a hypothesis update that corrected the trajectory
- a premature or unsafe action that should be avoided
- a missed clue that caused delay, wrong direction, or near-miss
- a short reasoning pattern that improved diagnosis

Do not extract:
- generic medical knowledge
- obvious facts
- full case summaries
- disease descriptions
- memories that only say "consider diagnosis X"
- memories without a reusable action lesson

Each ExperienceCard should answer:
"When facing what situation, what should the doctor do or avoid, why did it matter,
and when should this memory not be used?"

Simplified ExperienceCard fields:

1. memory_id:
   Use the provided id if available.
   Do not invent source ids.
   If the schema expects an empty value, use "".

2. memory_type:
   Always use "experience".

3. situation_text:
   Describe the clinical trigger and decision context.
   It should sound like a doctor remembering a scenario.
   Include:
   - problem representation
   - current uncertainty
   - relevant evidence status
   - decision pressure
   Start with "When", "If", or "For".
   Do not include patient identifiers.
   Do not make it too disease-specific unless the lesson truly depends on that disease.

4. action_text:
   Describe the local action or short action path.
   Focus on what the doctor should do next:
   - ask a targeted question
   - request a focused test
   - review an available image
   - perform or request a focused exam
   - compare conflicting evidence
   - update or broaden the differential
   - delay final diagnosis until missing evidence is resolved
   Include why the action was useful or risky.

5. outcome_text:
   Describe why the action mattered.
   Examples:
   - reduced uncertainty
   - corrected a misleading hypothesis
   - revealed missing evidence
   - prevented premature finalization
   - improved diagnostic direction
   - caused failure because evidence was skipped

6. boundary_text:
   This is one of the most important fields.
   Write concrete apply and do-not-use conditions in one concise sentence.
   Must include:
   - when the memory applies
   - when it should not be used
   - what missing evidence, modality, or contradiction would invalidate it
   Avoid vague text like "use in similar cases".

7. action_sequence:
   Use 1-4 ordered steps.
   Each step should be a dict with:
   - action_type
   - action_label

   Prefer action_type values such as:
   ASK, REQUEST_LAB, REVIEW_IMAGE, REQUEST_IMAGING, PHYSICAL_EXAM,
   UPDATE_HYPOTHESIS, BROADEN_DIFFERENTIAL, DELAY_FINALIZATION, FINALIZE_DIAGNOSIS.

   Only use FINALIZE_DIAGNOSIS if the experience teaches when finalization became safe.

8. outcome_type:
   Use:
   - success: the action clearly improved the diagnostic trajectory
   - partial_success: the action helped but did not fully resolve the case
   - failure: the action or omission led to wrong direction or missed opportunity
   - unsafe: the action risked patient safety or premature closure

9. tags:
   Use reusable clinical and reasoning tags.
   Include syndrome tags, action tags, and risk tags when useful.

   Good tag examples:
   - chest_pain
   - dyspnea
   - fever
   - abdominal_pain
   - neurologic_deficit
   - targeted_history
   - image_review_needed
   - missing_lab
   - broad_differential
   - premature_closure
   - diagnostic_anchoring
   - conflicting_evidence
   - unsafe_finalization

   Do not use case ids or patient identifiers as tags.

10. confidence:
   Estimate how reliable and reusable this experience is.
   Higher confidence requires:
   - the action clearly changed the diagnostic trajectory
   - the boundary is clear
   - the lesson is specific and reusable
   Lower confidence when:
   - evidence is incomplete
   - causal value is uncertain
   - the lesson is broad or weak

11. support_count:
   Use 1 for a newly extracted single-turn or single-episode experience unless input states otherwise.

12. source:
   Preserve provided provenance only.
   Use a compact dict, for example:
   {{
     "case_ids": [],
     "episode_ids": [],
     "turn_ids": []
   }}
   Do not invent ids.

Quality requirements:
- Make the experience actionable.
- Preserve clinical uncertainty.
- Prefer decision memories over diagnosis memories.
- Include negative or unsafe experiences when they teach what to avoid.
- Keep the memory reusable across future cases.
- Do not invent evidence, image findings, labs, or outcomes.
- Do not include long case details that will hurt retrieval.

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
      "tags": [],
      "confidence": 0.0,
      "support_count": 1,
      "source": {{
        "case_ids": [],
        "episode_ids": [],
        "turn_ids": []
      }}
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
You are a clinical memory-library curator.

{STRICT_JSON_RULES}

Task:
Decide whether a new ExperienceCard should be inserted, merged, discarded, or marked as conflict.

Your goal is to keep the memory library small, clinically useful, and retrieval-friendly.

Do not keep many near-duplicate memories.
Do not merge memories that teach different actions, risks, or boundaries.
Do not preserve fields just because they exist; preserve only clinically useful information.

Decision values:
- insert_new:
  The new experience teaches a distinct reusable clinical decision lesson.

- merge:
  The new experience and an existing memory share the same clinical trigger,
  uncertainty state, action pattern, outcome direction, and boundary.

- discard:
  The new experience is too generic, too case-specific, redundant without adding value,
  weakly supported, not actionable, or merely restates common medical knowledge.

- conflict:
  The same situation/action pattern has incompatible outcomes or safety implications.
  Example: one memory says finalization was safe after a cue, another says that same cue caused premature closure.

Merge only if all are mostly true:
1. Same decision point:
   The doctor is facing the same kind of uncertainty or next-step pressure.
2. Same action lesson:
   The memory recommends or warns against the same kind of action.
3. Same outcome direction:
   Both are success/partial_success or both are failure/unsafe.
4. Compatible boundary:
   The apply/do-not-use conditions do not contradict each other.
5. Similar retrieval purpose:
   A future doctor would want to retrieve them for the same reason.

Do NOT merge if:
- one memory is about asking targeted history and another is about reviewing imaging
- one warns against finalization and another supports finalization
- one requires CXR/lab/evidence review and the other does not
- disease labels overlap but the decision point differs
- the new memory adds a distinct safety warning
- the memories would bias the doctor toward different next actions

When merging:
- Create a cleaner, more general, more useful ExperienceCard.
- Keep the strongest clinical boundary.
- Keep important safety tags.
- Remove patient-specific details.
- Preserve source ids if provided.
- Increase support_count when appropriate.
- Do not invent clinical evidence.
- Prefer shorter wording if it preserves the clinical lesson.

Quality standard for merged_experience:
- situation_text should begin with a clinical trigger.
- action_text should tell the doctor what to do or avoid.
- outcome_text should explain why the memory matters.
- boundary_text should clearly say when to use and when not to use.
- tags should support future retrieval.
- source should preserve provenance compactly.
- The merged memory should be shorter and better than the originals.

Simplified merged ExperienceCard format:
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
  "tags": [],
  "confidence": 0.0,
  "support_count": 1,
  "source": {{
    "case_ids": [],
    "episode_ids": [],
    "turn_ids": []
  }}
}}

Schema:
{_dump(EXPERIENCE_MERGE_SCHEMA)}

Input:
{_dump(payload)}
""".strip()


def skill_consolidation_prompt(payload: dict[str, Any]) -> str:
    return f"""
You are a clinical skill miner and diagnostic workflow architect.

{STRICT_JSON_RULES}

Task:
Refine one reusable SkillCard from repeated useful ExperienceCards.

A skill is not a case summary.
A skill is not a disease definition.
A skill is a reusable diagnostic workflow that a doctor can adapt at a similar future decision point.

Create a skill only when the provided experiences show a repeated pattern:
- similar type of clinical uncertainty
- similar useful action sequence
- similar safety boundary
- repeated improvement in diagnostic reasoning
- enough support across experiences or cases

Do not create a skill from one isolated experience unless the input explicitly says to do so.
Do not create a skill that only says "consider disease X".
Do not encode textbook knowledge unless it appears as a repeated decision workflow in the experiences.

Simplified SkillCard fields:

1. memory_id:
   Use the provided id if available.
   Do not invent source ids.
   If the schema expects an empty value, use "".

2. memory_type:
   Always use "skill".

3. skill_name:
   Use a short workflow-style name.

   Good:
   - "Review available imaging before final diagnosis in respiratory complaints"
   - "Use targeted history to narrow broad abdominal pain differential"
   - "Delay closure when key negative evidence conflicts with the leading diagnosis"
   - "Resolve missing modality evidence before committing to diagnosis"

   Bad:
   - "Pneumonia diagnosis"
   - "Chest pain"
   - "Ask questions"
   - "Consider differential diagnosis"

4. situation_text:
   Describe when a doctor should consider using this skill.
   Include:
   - clinical presentation type
   - uncertainty state
   - evidence status
   - decision pressure
   Avoid overfitting to one disease or patient.

5. goal_text:
   State what the skill helps achieve.
   Examples:
   - reduce diagnostic uncertainty before finalization
   - identify missing evidence
   - avoid premature closure
   - reconcile conflicting modalities
   - choose the next high-value question or test
   - decide whether final diagnosis is safe

6. procedure_text:
   Write a concise clinical workflow.
   Recommended structure:
   - Trigger: when the skill is activated
   - Step 1: represent the current problem
   - Step 2: identify unresolved uncertainty
   - Step 3: choose the highest-value next evidence/action
   - Step 4: update hypothesis or delay finalization
   - Stop condition: when not to continue or when finalization becomes safer

7. procedure:
   Use 2-5 executable high-level steps.
   Each step should be a dict with:
   - action_type
   - action_label

   Prefer action_type values such as:
   ASK, REQUEST_LAB, REVIEW_IMAGE, REQUEST_IMAGING, PHYSICAL_EXAM,
   UPDATE_HYPOTHESIS, BROADEN_DIFFERENTIAL, DELAY_FINALIZATION, FINALIZE_DIAGNOSIS.

   The steps should be general enough for future cases.

8. boundary_text:
   This is critical.
   Include:
   - when the skill applies
   - when the skill should not be used
   - what evidence or modality requirements must be met
   - when the doctor should downgrade the skill to a weak hint

   Avoid vague boundaries like "use in similar cases".

9. tags:
   Use reusable clinical and reasoning tags.
   Include syndrome tags, workflow tags, and safety tags when useful.

   Good tag examples:
   - respiratory_complaint
   - chest_pain
   - abdominal_pain
   - neurologic_deficit
   - image_review_needed
   - missing_evidence
   - targeted_history
   - broad_differential
   - conflicting_evidence
   - premature_closure
   - finalization_safety

10. confidence:
   Reflect how reusable the skill is.
   Higher confidence requires:
   - repeated support
   - clear boundary
   - consistent positive outcomes
   - low unsafe signal
   Lower confidence when:
   - support is limited
   - boundaries are unclear
   - experiences are heterogeneous

11. support_count:
   Use the number of supporting experiences if provided.
   Otherwise use a conservative value.

12. source:
   Preserve compact provenance only.
   Example:
   {{
     "experience_ids": [],
     "case_ids": []
   }}
   Do not invent ids.

Skill quality requirements:
- The skill should help the doctor decide the next action, not directly guess the diagnosis.
- The skill should preserve uncertainty instead of forcing closure.
- The skill should be adaptable to new cases.
- The skill should be concise enough to inject into an MLLM prompt.
- The skill should not override current patient evidence.
- The skill should have a clear stop condition or non-applicability boundary.

Output format:
{{
  "memory_id": "...",
  "memory_type": "skill",
  "skill_name": "...",
  "situation_text": "...",
  "goal_text": "...",
  "procedure_text": "...",
  "procedure": [
    {{"action_type": "...", "action_label": "..."}}
  ],
  "boundary_text": "...",
  "tags": [],
  "confidence": 0.0,
  "support_count": 1,
  "source": {{
    "experience_ids": [],
    "case_ids": []
  }}
}}

Schema:
{_dump(SKILL_SCHEMA)}

Input:
{_dump(payload)}
""".strip()
