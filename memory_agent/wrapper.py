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
from .memory_store import KnowledgeMemoryStore
from .memory_manager import update_memory
from .planner import plan_intent
from .retriever import DEFAULT_MEMORY_ROOT, retrieve_all
from .schemas import ActionDecision, CaseMemory, ExecutionResult, MedEnvCaseBundle
from .utils.bench_adapter import knowledge_items_from_payload, unwrap_osce_examination

TOOL_TO_ACTION_TYPE = {
    "ask_patient": "ASK",
    "retrieve": "REVIEW_HISTORY",
    "request_exam": "REQUEST_EXAM",
    "cxr": "REVIEW_IMAGE",
    "cxr_grounding": "REVIEW_IMAGE",
    "diagnosis": "FINALIZE_DIAGNOSIS",
}

ACTION_TO_TOOL = {
    "ASK": "ask_patient",
    "REVIEW_HISTORY": "retrieve",
    "REQUEST_EXAM": "request_exam",
    "REQUEST_LAB": "request_exam",
    "REVIEW_IMAGE": "cxr",
    "FINALIZE_DIAGNOSIS": "diagnosis",
    "DEFER_FINALIZE": "ask_patient",
    "UPDATE_HYPOTHESIS": "retrieve",
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
        self.knowledge_seeded = False
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
        self.knowledge_seeded = False

    def _ensure_case_bundle(self, observation: Any) -> None:
        if self.case_bundle is not None or not isinstance(observation, dict):
            return
        bundle_dict = observation.get("medenv_case_bundle")
        if bundle_dict:
            self.case_bundle = MedEnvCaseBundle.from_dict(bundle_dict)
            return
        context = observation.get("context") or {}
        self.case_bundle = MedEnvCaseBundle(case_id=str(observation.get("case_id", "")), ehr=context.get("ehr") or {})

    def _seed_knowledge_memory(self) -> None:
        if self.case_bundle is None or self.knowledge_seeded:
            return
        knowledge_root = self.memory_root or DEFAULT_MEMORY_ROOT
        knowledge_store = KnowledgeMemoryStore(knowledge_root)
        osce = unwrap_osce_examination(self.case_bundle.ehr)
        for item in knowledge_items_from_payload(osce, case_id=self.case_bundle.case_id):
            knowledge_store.upsert(item)
        self.knowledge_seeded = True

    def _clone_case_memory(self) -> CaseMemory | None:
        if self.case_memory is None:
            return None
        return CaseMemory.from_dict(self.case_memory.to_dict())

    def _observation_turn_id(self) -> str:
        if self.case_memory is None:
            return "turn_0"
        return f"turn_{self.case_memory.turn_id}"

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
            turn_id=self.case_memory.turn_id if self.case_memory else 0,
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
        chosen = self.current_action_decision.chosen_action
        top_hypo = [h.name for h in self.case_memory.active_hypotheses[:3]]
        blocked = [
            a.action_id
            for a in (self.current_applicability_result.action_assessments if self.current_applicability_result else [])
            if a.decision == "block"
        ]
        return (
            "Memory wrapper summary for the next doctor decision:\n"
            f"- Problem summary: {self.case_memory.problem_summary}\n"
            f"- Missing critical info: {self.case_memory.missing_info[:4]}\n"
            f"- Active hypotheses: {top_hypo}\n"
            f"- Finalize risk: {self.case_memory.finalize_risk}\n"
            f"- Retrieved memories: exp={len((self.current_retrieval_result.experience_hits if self.current_retrieval_result else []))}, "
            f"skill={len((self.current_retrieval_result.skill_hits if self.current_retrieval_result else []))}, "
            f"knowledge={len((self.current_retrieval_result.knowledge_hits if self.current_retrieval_result else []))}\n"
            f"- Recommended action: {chosen.get('action_type')} -> {chosen.get('action_content')}\n"
            f"- Blocked candidates: {blocked}\n"
            "Respond with exactly one valid tool call. Avoid blocked or high-risk actions."
        )

    def _action_dict_to_env_action(self, action_dict: dict[str, Any]) -> list[dict[str, Any]]:
        action_type = action_dict.get("action_type", "ASK")
        tool_name = ACTION_TO_TOOL.get(action_type, "ask_patient")

        args: dict[str, Any] = {}
        if action_type == "ASK":
            args["question"] = action_dict.get("action_content", "Can you tell me more about your symptoms?")
        elif action_type in {"REQUEST_EXAM", "REQUEST_LAB"}:
            args["exam_type"] = action_dict.get("action_content", "targeted exam")
        elif action_type == "REVIEW_HISTORY":
            args["query"] = action_dict.get("action_content", self.case_memory.problem_summary if self.case_memory else "clinical guidance")
        elif action_type == "FINALIZE_DIAGNOSIS":
            args["final_response"] = action_dict.get("action_content", "The final diagnosis is: \\boxed{Undetermined}.")
        elif action_type == "REVIEW_IMAGE":
            args = {}
        else:
            args["question"] = "Please provide additional symptoms."

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
            action_type = TOOL_TO_ACTION_TYPE.get(tool_name, "ASK")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            if action_type == "ASK":
                content = str(args.get("question", ""))
            elif action_type in {"REQUEST_EXAM", "REQUEST_LAB"}:
                content = str(args.get("exam_type", ""))
            elif action_type == "REVIEW_HISTORY":
                content = str(args.get("query", ""))
            elif action_type == "FINALIZE_DIAGNOSIS":
                content = str(args.get("final_response", ""))
            else:
                content = json.dumps(args, ensure_ascii=False)

            return {
                "action_id": tool_name or action_type,
                "action_type": action_type,
                "action_label": tool_name or action_type.lower(),
                "action_content": content,
                "action_args": args,
                "source_field_refs": [],
            }

        return {
            "action_id": "fallback_ask",
            "action_type": "ASK",
            "action_label": "ask_onset",
            "action_content": "Can you tell me more about your symptoms?",
            "action_args": {"question": "Can you tell me more about your symptoms?"},
            "source_field_refs": [],
        }

    def _is_blocked_action(self, action_dict: dict[str, Any]) -> bool:
        if not self.current_applicability_result or not self.current_intent_plan:
            return False
        action_type = action_dict.get("action_type", "")
        for candidate in self.current_intent_plan.action_candidates:
            if candidate.action_type != action_type:
                continue
            for assessment in self.current_applicability_result.action_assessments:
                if assessment.action_id == candidate.action_id and assessment.decision == "block":
                    return True
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
            executed_action = action_decision.chosen_action if action_decision else None
            execution_result = self._build_execution_result(observation, info, executed_action)

        if self.case_bundle is not None and self.case_memory is None:
            self._seed_knowledge_memory()
            self.case_memory = init_case_memory(self.case_bundle, no_cxr="cxr" not in getattr(self.tools, "tools", []))

        evidence_list = canonicalize_turn_input(self._observation_turn_id(), {"observation": observation, "info": info})
        if self.case_bundle is not None and self.case_memory is not None and not self.static_case_loaded:
            static_evidence = canonicalize_static_case(self.case_bundle)
            self.case_memory = update_case_memory(self.case_memory, static_evidence)
            self.static_case_loaded = True

        if self.case_memory is not None:
            self.case_memory = update_case_memory(self.case_memory, evidence_list)

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
            self.current_retrieval_result = retrieve_all(self.current_intent_plan.memory_query, root_dir=self.memory_root)
            self.current_applicability_result = apply_controller(self.case_memory, self.current_intent_plan, self.current_retrieval_result)
            self.current_action_decision = decide_action(self.current_intent_plan, self.current_applicability_result, self.case_memory)
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
