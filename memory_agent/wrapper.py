from __future__ import annotations

import copy
import json
import re
import uuid
from pathlib import Path
from typing import Any

from rllm.agents.agent import Action, Trajectory
from rllm.agents.med_agent import MedicalAgent

from .llm import LLMClient
from .offline.episode_distiller import distill_from_trajectory
from .offline.memory_writer import write_memory_from_distilled_episode
from .online import (
    append_memory_trace,
    apply_applicability_control,
    build_memory_guidance,
    build_memory_query,
    build_trace_payload,
    guidance_to_text,
    init_case_state,
    retrieve_multi_memory,
    update_case_state,
)
from .schemas import ApplicabilityResult, CaseState, DistilledEpisode, EpisodeFeedback, MemoryGuidance, MemoryQuery, MemoryRetrievalResult
from .utils.action_vocab import action_label, candidate_actions, tool_name
from .utils.config import MEMORY_RUNTIME_DEFAULTS
from .utils.medenv_adapter import extract_gold_diagnosis

ACTION_TO_TOOL = {
    "ASK": "ask_patient",
    "REVIEW_HISTORY": "retrieve",
    "REQUEST_EXAM": "request_exam",
    "REQUEST_LAB": "request_exam",
    "REVIEW_IMAGE": "cxr",
    "UPDATE_HYPOTHESIS": "retrieve",
    "DEFER_FINALIZE": "ask_patient",
    "FINALIZE_DIAGNOSIS": "diagnosis",
}

TOOL_TO_ACTION = {value: key for key, value in ACTION_TO_TOOL.items()}


def _boxed_diagnosis(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"\\box(?:ed)?\{(.+?)\}", text)
    return match.group(1).strip() if match else text.strip()


class MemoryWrappedMedicalAgent(MedicalAgent):
    def __init__(
        self,
        *args,
        memory_root: str | None = None,
        enable_memory: bool = False,
        query_builder_mode: str = "rule",
        applicability_mode: str = "rule",
        experience_extraction_mode: str = "rule",
        experience_merge_mode: str = "rule",
        memory_top_k: int = 5,
        log_memory_trace: bool = False,
        disable_memory: bool = False,
        disable_experience_memory: bool = False,
        disable_skill_memory: bool = False,
        disable_knowledge_memory: bool = False,
        memory_llm_model: str = "",
        memory_llm_base_url: str = "",
        memory_llm_api_key: str = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.memory_root = memory_root
        self.enable_memory = bool(enable_memory) and not bool(disable_memory)
        self.query_builder_mode = query_builder_mode or MEMORY_RUNTIME_DEFAULTS["query_builder_mode"]
        self.applicability_mode = applicability_mode or MEMORY_RUNTIME_DEFAULTS["applicability_mode"]
        self.experience_extraction_mode = experience_extraction_mode or MEMORY_RUNTIME_DEFAULTS["experience_extraction_mode"]
        self.experience_merge_mode = experience_merge_mode or MEMORY_RUNTIME_DEFAULTS["experience_merge_mode"]
        self.memory_top_k = int(memory_top_k)
        self.log_memory_trace = bool(log_memory_trace)
        self.disable_experience_memory = bool(disable_experience_memory)
        self.disable_skill_memory = bool(disable_skill_memory)
        self.disable_knowledge_memory = bool(disable_knowledge_memory)
        self.llm_client = LLMClient(model=memory_llm_model, base_url=memory_llm_base_url, api_key=memory_llm_api_key)
        self.case_state: CaseState | None = None
        self.current_memory_query: MemoryQuery | None = None
        self.current_retrieval_result: MemoryRetrievalResult | None = None
        self.current_applicability_result: ApplicabilityResult | None = None
        self.current_memory_guidance: MemoryGuidance | None = None
        self.pending_case_state: CaseState | None = None
        self.turn_records: list[dict[str, Any]] = []
        self.case_id: str = ""
        self.trace_root = Path(__file__).resolve().parent.parent / "logs" / "memory_trace"

    def reset(self):
        super().reset()
        self.case_state = None
        self.current_memory_query = None
        self.current_retrieval_result = None
        self.current_applicability_result = None
        self.current_memory_guidance = None
        self.pending_case_state = None
        self.turn_records = []
        self.case_id = ""

    def _extract_case_bundle(self, observation: Any) -> dict[str, Any] | None:
        if not isinstance(observation, dict):
            return None
        bundle = observation.get("medenv_case_bundle")
        if isinstance(bundle, dict):
            return bundle
        case_id = observation.get("case_id") or observation.get("task_id") or ""
        context = observation.get("context")
        if isinstance(context, dict) and (case_id or context.get("ehr")):
            return {"case_id": case_id, "ehr": context.get("ehr") or context}
        if case_id:
            return {"case_id": case_id, "ehr": observation.get("ehr") or {}}
        return None

    def _ensure_case_state(self, observation: Any) -> None:
        if self.case_state is not None:
            return
        bundle = self._extract_case_bundle(observation)
        if bundle is not None:
            self.case_state = init_case_state(bundle, no_cxr="cxr" not in getattr(self.tools, "tools", []))
            self.case_id = self.case_state.case_id
        else:
            self.case_state = CaseState(case_id=str(getattr(self, "case_id", "") or ""))
            self.case_id = self.case_state.case_id

    def _prune_existing_guidance_messages(self) -> None:
        cleaned: list[dict[str, Any]] = []
        for message in self.messages:
            if message.get("role") == "system" and str(message.get("content", "")).startswith("Memory guidance:"):
                continue
            cleaned.append(message)
        self.messages = cleaned

    def _inject_guidance(self, guidance: MemoryGuidance) -> None:
        self._prune_existing_guidance_messages()
        self.messages.append({"role": "system", "content": guidance_to_text(guidance)})

    def _build_episode_feedback(self, trajectory: Trajectory) -> EpisodeFeedback:
        final_diagnosis = ""
        if self.turn_records:
            last_action = self.turn_records[-1].get("selected_action") or {}
            if isinstance(last_action, dict) and last_action.get("action_type") == "FINALIZE_DIAGNOSIS":
                final_diagnosis = _boxed_diagnosis(str(last_action.get("action_content", "")))
        return EpisodeFeedback(
            episode_id=getattr(trajectory, "uid", ""),
            case_id=self.case_id,
            total_reward=float(getattr(trajectory, "reward", 0.0)),
            success=float(getattr(trajectory, "reward", 0.0)) >= 0.5,
            final_diagnosis=final_diagnosis,
            gold_diagnosis=extract_gold_diagnosis(getattr(trajectory, "task", {}) or {}),
            trajectory_metrics={"turn_count": len(self.turn_records)},
            notes="memory_v1",
        )

    def _action_dict_from_tool_call(self, action: Any) -> dict[str, Any]:
        if isinstance(action, list) and action:
            call = action[0] if isinstance(action[0], dict) else {}
            function = call.get("function") if isinstance(call, dict) else {}
            function = function if isinstance(function, dict) else {}
            name = str(function.get("name", ""))
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    arguments = {}
            action_type = TOOL_TO_ACTION.get(name, "ASK")
            return {
                "action_id": name or action_type.lower(),
                "action_type": action_type,
                "action_label": name or action_label(action_type),
                "action_content": str((arguments or {}).get("final_response") or (arguments or {}).get("question") or (arguments or {}).get("query") or (arguments or {}).get("exam_type") or ""),
                "action_args": arguments,
            }
        if isinstance(action, str):
            return {
                "action_id": "dialogue",
                "action_type": "ASK",
                "action_label": "ask_patient",
                "action_content": action,
                "action_args": {"question": action},
            }
        return {
            "action_id": "fallback_ask",
            "action_type": "ASK",
            "action_label": "ask_patient",
            "action_content": "Can you tell me more about your symptoms?",
            "action_args": {"question": "Can you tell me more about your symptoms?"},
        }

    def _tool_calls_from_action_dict(self, action_dict: dict[str, Any]) -> list[dict[str, Any]]:
        action_type = str(action_dict.get("action_type", "ASK"))
        tool = ACTION_TO_TOOL.get(action_type, "ask_patient")
        if action_type == "ASK":
            arguments = {"question": str(action_dict.get("action_content", "Can you tell me more about your symptoms?"))}
        elif action_type in {"REQUEST_EXAM", "REQUEST_LAB"}:
            arguments = {"exam_type": str(action_dict.get("action_content", "targeted exam"))}
        elif action_type == "REVIEW_HISTORY":
            arguments = {"query": str(action_dict.get("action_content", self.case_state.problem_summary if self.case_state else "clinical guidance"))}
        elif action_type == "FINALIZE_DIAGNOSIS":
            arguments = {"final_response": str(action_dict.get("action_content", "The final diagnosis is: \\boxed{Undetermined}."))}
        else:
            arguments = {"question": str(action_dict.get("action_content", "Can you tell me more about your symptoms?"))}
        return [
            {
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {"name": tool, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
        ]

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        super().update_from_env(observation, reward, done, info, **kwargs)
        if not self.enable_memory:
            return
        self._ensure_case_state(observation)
        if self.case_state is None:
            return
        self.case_state = update_case_state(self.case_state, observation)
        candidate_action_list = candidate_actions(self.case_state)
        self.current_memory_query = build_memory_query(self.case_state, candidate_action_list, mode=self.query_builder_mode, llm_client=self.llm_client)
        self.current_retrieval_result = retrieve_multi_memory(
            self.current_memory_query,
            root_dir=self.memory_root,
            disable_experience_memory=self.disable_experience_memory,
            disable_skill_memory=self.disable_skill_memory,
            disable_knowledge_memory=self.disable_knowledge_memory,
        )
        self.current_applicability_result = apply_applicability_control(
            self.case_state,
            self.current_memory_query,
            self.current_retrieval_result,
            candidate_action_list,
            mode=self.applicability_mode,
            llm_client=self.llm_client,
        )
        self.current_memory_guidance = build_memory_guidance(self.current_applicability_result)
        self._inject_guidance(self.current_memory_guidance)
        self.pending_case_state = CaseState.from_dict(self.case_state.to_dict())

        if self.log_memory_trace:
            append_memory_trace(
                self.trace_root,
                self.case_id or self.case_state.case_id or "unknown_case",
                build_trace_payload(
                    self.case_state,
                    self.current_memory_query,
                    self.current_retrieval_result,
                    self.current_applicability_result,
                    self.current_memory_guidance,
                    {},
                ),
            )

    def update_from_model(self, response: str, **kwargs) -> Action:
        action = super().update_from_model(response, **kwargs)
        if not self.enable_memory:
            return action
        selected_action = self._action_dict_from_tool_call(action.action)
        selected_action_blocked = False
        actual_action = action.action

        if self.current_memory_guidance and selected_action.get("action_type") in self.current_memory_guidance.blocked_actions:
            selected_action_blocked = True
            replacement = {
                "action_type": self.current_memory_guidance.recommended_actions[0] if self.current_memory_guidance.recommended_actions else "ASK",
                "action_content": "Please provide more detail before finalizing.",
            }
            actual_action = self._tool_calls_from_action_dict(replacement)
            selected_action = self._action_dict_from_tool_call(actual_action)
        if self.trajectory.steps:
            self.trajectory.steps[-1].action = actual_action

        record = {
            "turn_id": self.case_state.turn_id if self.case_state else len(self.turn_records),
            "case_state_before": self.pending_case_state.to_dict() if self.pending_case_state else {},
            "memory_query": self.current_memory_query.to_dict() if self.current_memory_query else {},
            "retrieval_result": self.current_retrieval_result.to_dict() if self.current_retrieval_result else {},
            "applicability_result": self.current_applicability_result.to_dict() if self.current_applicability_result else {},
            "memory_guidance": self.current_memory_guidance.to_dict() if self.current_memory_guidance else {},
            "selected_action": selected_action,
            "selected_action_blocked": selected_action_blocked,
            "blocked_actions": list(self.current_memory_guidance.blocked_actions if self.current_memory_guidance else []),
            "env_observation": self.current_observation if isinstance(self.current_observation, dict) else {},
            "env_info": self.trajectory.steps[-1].info if self.trajectory.steps else {},
            "reward": self.trajectory.steps[-1].reward if self.trajectory.steps else 0.0,
            "done": self.trajectory.steps[-1].done if self.trajectory.steps else False,
        }
        self.turn_records.append(record)
        if self.trajectory.steps:
            self.trajectory.steps[-1].info.setdefault("memory_agent", {})
            self.trajectory.steps[-1].info["memory_agent"] = record
        return Action(action=actual_action)

    def finalize_episode(self, trajectory: Trajectory) -> dict[str, Any]:
        trajectory.info.setdefault("memory_agent", {})
        trajectory.info["memory_agent"]["turn_records"] = self.turn_records
        if not self.enable_memory:
            trajectory.info["memory_agent"]["disabled"] = True
            return trajectory.info["memory_agent"]

        episode_feedback = self._build_episode_feedback(trajectory)
        distilled = distill_from_trajectory(trajectory, episode_feedback)
        write_result = write_memory_from_distilled_episode(
            distilled,
            root_dir=self.memory_root,
            experience_extraction_mode=self.experience_extraction_mode,
            experience_merge_mode=self.experience_merge_mode,
            llm_client=self.llm_client,
        )
        trajectory.info["memory_agent"]["episode_feedback"] = episode_feedback.to_dict()
        trajectory.info["memory_agent"]["distilled_episode"] = distilled.to_dict()
        trajectory.info["memory_agent"]["memory_write_result"] = write_result
        return trajectory.info["memory_agent"]
