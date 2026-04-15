from __future__ import annotations

import json
import uuid
from typing import Any

from rllm.agents.agent import Action, Trajectory
from rllm.agents.med_agent import MedicalAgent

from .canonicalizer import canonicalize_static_case, canonicalize_turn_input
from .case_memory import init_case_memory, update_case_memory
from .controller import apply_controller
from .decision import decide_action
from .distiller import distill_episode
from .feedback import build_episode_feedback, build_turn_feedback
from .memory_manager import update_memory
from .planner import plan_intent
from .retriever import retrieve_all
from .schemas import ActionDecision, CaseMemory, ExecutionResult, MedEnvCaseBundle


TOOL_TO_ACTION_TYPE = {
    "ask_patient": "ask",
    "retrieve": "retrieve",
    "request_exam": "request_exam",
    "cxr": "cxr",
    "cxr_grounding": "cxr_grounding",
    "diagnosis": "finalize",
}

ACTION_TO_TOOL = {
    "ask": "ask_patient",
    "retrieve": "retrieve",
    "request_exam": "request_exam",
    "cxr": "cxr",
    "cxr_grounding": "cxr_grounding",
    "finalize": "diagnosis",
}


class MemoryWrappedMedicalAgent(MedicalAgent):
    def __init__(self, *args, memory_root: str | None = None, **kwargs):
        self.memory_root = memory_root
        self.case_bundle: MedEnvCaseBundle | None = None
        self.case_memory: CaseMemory | None = None
        self.turn_feedback_list: list[dict[str, Any]] = []
        self.turn_records: list[dict[str, Any]] = []
        self.current_intent_plan = None
        self.current_retrieval_result = None
        self.current_applicability_result = None
        self.current_action_decision = None
        self.pending_turn_context: dict[str, Any] | None = None
        self.static_case_loaded = False
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self.case_bundle = None
        self.case_memory = None
        self.turn_feedback_list = []
        self.turn_records = []
        self.current_intent_plan = None
        self.current_retrieval_result = None
        self.current_applicability_result = None
        self.current_action_decision = None
        self.pending_turn_context = None
        self.static_case_loaded = False

    def _ensure_case_bundle(self, observation: Any) -> None:
        if self.case_bundle is not None or not isinstance(observation, dict):
            return
        bundle_dict = observation.get("medenv_case_bundle")
        if bundle_dict:
            self.case_bundle = MedEnvCaseBundle.from_dict(bundle_dict)
            return
        context = observation.get("context") or {}
        self.case_bundle = MedEnvCaseBundle(
            case_id=str(observation.get("case_id", "")),
            ehr=context.get("ehr") or {},
        )

    def _clone_case_memory(self) -> CaseMemory | None:
        if self.case_memory is None:
            return None
        return CaseMemory.from_dict(self.case_memory.to_dict())

    def _observation_turn_id(self) -> str:
        if self.case_memory is None:
            return "turn_0"
        turn_index = int((self.case_memory.derived_state.get("interaction_state") or {}).get("turn_index", 0))
        return f"turn_{turn_index}"

    def _build_execution_result(self, observation: Any, info: dict[str, Any], executed_action: dict[str, Any] | None) -> ExecutionResult | None:
        if not executed_action:
            return None
        response_type = "finalize_result"
        raw_payload: dict[str, Any] = {}
        if isinstance(observation, dict) and "question" in observation:
            response_type = "patient_reply"
            raw_payload = {"question": observation.get("question", "")}
        elif isinstance(observation, dict) and "tool_outputs" in observation:
            response_type = f"{executed_action.get('action_type', 'tool')}_result"
            raw_payload = observation.get("tool_outputs", {})
        elif info.get("metadata"):
            raw_payload = info

        status = "success"
        if isinstance(observation, dict) and observation.get("tool_outputs") == {}:
            status = "no_effect"
        return ExecutionResult(
            turn_id=self._observation_turn_id(),
            executed_action=executed_action,
            env_response={
                "response_type": response_type,
                "raw_payload": raw_payload,
                "raw_field_refs": [],
            },
            execution_status=status,
            source_field_refs=executed_action.get("source_field_refs", []),
        )

    def _build_memory_brief(self) -> str:
        if self.case_memory is None or self.current_action_decision is None:
            return ""
        derived = self.case_memory.derived_state
        chosen = self.current_action_decision.chosen_action
        blocked = [
            assessment.action_id
            for assessment in (self.current_applicability_result.action_assessments if self.current_applicability_result else [])
            if assessment.decision == "block"
        ]
        top_intents = ", ".join(f"{item.intent_type}:{item.score:.2f}" for item in (self.current_intent_plan.ranked_intents[:3] if self.current_intent_plan else []))
        return (
            "Memory wrapper summary for the next doctor decision:\n"
            f"- Chief complaint: {(self.case_memory.raw_snapshot.get('history') or {}).get('chief_complaint', '')}\n"
            f"- Confirmed facts: {derived.get('confirmed_facts', [])[:4]}\n"
            f"- Missing critical slots: {derived.get('missing_critical_slots', [])[:3]}\n"
            f"- Active differential: {derived.get('tentative_differential', [])[:3]}\n"
            f"- Ranked intents: {top_intents}\n"
            f"- Retrieved memories: exp={len((self.current_retrieval_result.experience_hits if self.current_retrieval_result else []))}, "
            f"skill={len((self.current_retrieval_result.skill_hits if self.current_retrieval_result else []))}, "
            f"guardrail={len((self.current_retrieval_result.guardrail_hits if self.current_retrieval_result else []))}\n"
            f"- Recommended action: {chosen.get('action_type')} -> {chosen.get('action_text')}\n"
            f"- Blocked candidates: {blocked}\n"
            "Respond with exactly one valid tool call. Avoid blocked or high-risk actions."
        )

    def _action_dict_to_env_action(self, action_dict: dict[str, Any]) -> list[dict[str, Any]] | str:
        action_type = action_dict.get("action_type", "ask")
        tool_name = ACTION_TO_TOOL.get(action_type)
        if not tool_name:
            question = action_dict.get("action_text") or "Can you tell me more about your symptoms?"
            return [
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {
                        "name": "ask_patient",
                        "arguments": json.dumps({"question": question}, ensure_ascii=False),
                    },
                }
            ]

        args = dict(action_dict.get("action_args") or {})
        if action_type == "ask" and "question" not in args:
            args["question"] = action_dict.get("action_text", "")
        if action_type == "finalize" and "final_response" not in args:
            args["final_response"] = action_dict.get("action_text", "")

        return [
            {
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
        ]

    def _env_action_to_action_dict(self, env_action: Any) -> dict[str, Any]:
        if isinstance(env_action, list) and env_action:
            call = env_action[-1]
            fn = (call.get("function") or {}) if isinstance(call, dict) else {}
            tool_name = fn.get("name", "")
            action_type = TOOL_TO_ACTION_TYPE.get(tool_name, tool_name)
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            action_text = ""
            if action_type == "ask":
                action_text = str(args.get("question", ""))
            elif action_type == "retrieve":
                action_text = str(args.get("query", ""))
            elif action_type == "request_exam":
                action_text = str(args.get("exam_type", ""))
            elif action_type == "finalize":
                action_text = str(args.get("final_response", ""))
            else:
                action_text = json.dumps(args, ensure_ascii=False)
            return {
                "action_id": tool_name or action_type,
                "action_type": action_type,
                "action_text": action_text,
                "action_args": args,
                "source_field_refs": [],
            }
        if isinstance(env_action, str):
            return {
                "action_id": "free_text",
                "action_type": "ask",
                "action_text": env_action,
                "action_args": {"question": env_action},
                "source_field_refs": [],
            }
        return {
            "action_id": "unknown",
            "action_type": "ask",
            "action_text": "Can you tell me more about your symptoms?",
            "action_args": {"question": "Can you tell me more about your symptoms?"},
            "source_field_refs": [],
        }

    def _is_blocked_action(self, action_dict: dict[str, Any]) -> bool:
        if not self.current_applicability_result:
            return False
        action_type = action_dict.get("action_type", "")
        for candidate in (self.current_intent_plan.action_candidates if self.current_intent_plan else []):
            if candidate.action_type != action_type:
                continue
            for assessment in self.current_applicability_result.action_assessments:
                if assessment.action_id == candidate.action_id and assessment.decision == "block":
                    return True
        if action_type == "finalize" and self.case_memory is not None:
            return (self.case_memory.derived_state.get("safety_state") or {}).get("premature_finalize_risk") == "high"
        return False

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        super().update_from_env(observation, reward, done, info, **kwargs)
        self._ensure_case_bundle(observation)

        executed_action = None
        execution_result = None
        case_before = None
        action_decision = None
        if self.pending_turn_context:
            case_before = self.pending_turn_context.get("case_before")
            action_decision = self.pending_turn_context.get("action_decision")
            executed_action = (action_decision.chosen_action if action_decision else None)
            execution_result = self._build_execution_result(observation, info, executed_action)

        if self.case_bundle is not None and self.case_memory is None:
            self.case_memory = init_case_memory(self.case_bundle, no_cxr="cxr" not in getattr(self.tools, "tools", []))

        evidence_list = canonicalize_turn_input(self._observation_turn_id(), {"observation": observation, "info": info})
        if self.case_bundle is not None and self.case_memory is not None and not self.static_case_loaded:
            static_evidence = canonicalize_static_case(self.case_bundle)
            self.case_memory = update_case_memory(self.case_memory, static_evidence, executed_action=None, execution_result=None)
            self.static_case_loaded = True

        if self.case_memory is not None:
            self.case_memory = update_case_memory(
                self.case_memory,
                evidence_list,
                executed_action=executed_action,
                execution_result=execution_result,
            )

        if case_before is not None and action_decision is not None and execution_result is not None and self.case_memory is not None:
            turn_feedback = build_turn_feedback(case_before, action_decision, execution_result, self.case_memory)
            self.turn_feedback_list.append(turn_feedback.to_dict())
            self.turn_records.append(
                {
                    "case_before": case_before.to_dict(),
                    "intent_plan": self.current_intent_plan.to_dict() if self.current_intent_plan else {},
                    "retrieval_result": self.current_retrieval_result.to_dict() if self.current_retrieval_result else {},
                    "applicability_result": self.current_applicability_result.to_dict() if self.current_applicability_result else {},
                    "action_decision": action_decision.to_dict(),
                    "execution_result": execution_result.to_dict(),
                    "case_after": self.case_memory.to_dict(),
                    "turn_feedback": turn_feedback.to_dict(),
                    "source_field_refs": action_decision.source_field_refs,
                }
            )
            self.pending_turn_context = None

        if not done and self.case_memory is not None:
            self.current_intent_plan = plan_intent(self.case_memory)
            self.current_retrieval_result = retrieve_all(self.current_intent_plan.query_signature, root_dir=self.memory_root)
            self.current_applicability_result = apply_controller(self.case_memory, self.current_intent_plan, self.current_retrieval_result)
            self.current_action_decision = decide_action(self.current_intent_plan, self.current_applicability_result)
            self.messages.append({"role": "system", "content": self._build_memory_brief()})

    def update_from_model(self, response: str, **kwargs) -> Action:
        action = super().update_from_model(response, **kwargs)
        parsed_action = action.action
        actual_action = parsed_action
        actual_decision = self.current_action_decision

        if self.current_action_decision is not None:
            parsed_action_dict = self._env_action_to_action_dict(parsed_action)
            if self._is_blocked_action(parsed_action_dict) or not isinstance(parsed_action, list):
                actual_action = self._action_dict_to_env_action(self.current_action_decision.chosen_action)
                parsed_action_dict = dict(self.current_action_decision.chosen_action)
            else:
                actual_decision = ActionDecision(
                    turn_id=self.current_action_decision.turn_id,
                    chosen_action=parsed_action_dict,
                    candidate_rankings=self.current_action_decision.candidate_rankings,
                    final_rationale=self.current_action_decision.final_rationale,
                    source_field_refs=self.current_action_decision.source_field_refs,
                )

            if self.trajectory.steps:
                self.trajectory.steps[-1].action = actual_action
                self.trajectory.steps[-1].info["memory_agent"] = {
                    "recommended_action": self.current_action_decision.to_dict(),
                    "applied_action": actual_decision.to_dict() if actual_decision else {},
                }

            self.pending_turn_context = {
                "case_before": self._clone_case_memory(),
                "action_decision": actual_decision,
            }

        return Action(action=actual_action)

    def finalize_episode(self, trajectory: Trajectory) -> dict[str, Any]:
        if self.case_bundle is None:
            return {}
        trajectory.info.setdefault("memory_agent", {})
        trajectory.info["memory_agent"]["turn_feedbacks"] = self.turn_feedback_list
        trajectory.info["memory_agent"]["turn_records"] = self.turn_records
        trajectory.info["memory_agent"]["final_case_memory"] = self.case_memory.to_dict() if self.case_memory else {}

        episode_feedback = build_episode_feedback(self.case_bundle, trajectory)
        distilled = distill_episode(trajectory, self.turn_feedback_list, episode_feedback)
        memory_update_plan = update_memory(distilled, root_dir=self.memory_root)
        trajectory.info["memory_agent"]["episode_feedback"] = episode_feedback.to_dict()
        trajectory.info["memory_agent"]["distilled_episode"] = distilled.to_dict()
        trajectory.info["memory_agent"]["memory_update_plan"] = memory_update_plan.to_dict()
        return trajectory.info["memory_agent"]
