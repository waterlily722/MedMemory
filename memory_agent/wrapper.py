from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

try:
    from rllm.agents import ToolAgent as _BaseAgent
except Exception:  # pragma: no cover - used in lightweight unit tests
    class _BaseAgent:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.tools = kwargs.get("tools", [])
            self.system_prompt = kwargs.get("system_prompt", "")
            self.parser_name = kwargs.get("parser_name", "")

        def update_from_env(self, *args: Any, **kwargs: Any) -> Any:
            return None

        def update_from_model(self, *args: Any, **kwargs: Any) -> Any:
            return None


from .llm import LLMClient
from .offline import distill_from_trajectory, write_memory_from_distilled_episode
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
from .schemas import (
    ApplicabilityResult,
    CaseState,
    EpisodeFeedback,
    MemoryGuidance,
    MemoryQuery,
    MemoryRetrievalResult,
    TurnRecord,
)
from .utils.config import MEMORY_ROOT_DIRNAME


ACTION_MAP = {
    "ask_patient": "ASK",
    "request_exam": "REQUEST_LAB",
    "retrieve": "REQUEST_LAB",
    "cxr": "REVIEW_IMAGE",
    "cxr_grounding": "REVIEW_IMAGE",
    "diagnosis": "FINALIZE_DIAGNOSIS",
}

DEFAULT_ACTIONS = [
    "ASK",
    "REQUEST_LAB",
    "REVIEW_IMAGE",
    "UPDATE_HYPOTHESIS",
    "FINALIZE_DIAGNOSIS",
]


class MemoryWrappedMedicalAgent(_BaseAgent):
    """
    Memory wrapper for MedEnv doctor agent.

    Online path:
      observation
        -> update CaseState
        -> build MemoryQuery
        -> retrieve Experience/Skill/Knowledge
        -> apply Applicability Controller
        -> build MemoryGuidance
        -> inject guidance into base agent context

    Offline path:
      done episode
        -> DistilledEpisode
        -> LLM experience extraction
        -> merge/write ExperienceStore

    This class does not implement an LLM action policy. The base doctor agent
    remains responsible for final action generation.
    """

    def __init__(
        self,
        *args: Any,
        query_builder_mode: str = "rule",
        applicability_mode: str = "rule",
        experience_extraction_mode: str = "llm",
        experience_merge_mode: str = "rule",
        memory_top_k: int = 5,
        log_memory_trace: bool = False,
        disable_memory: bool = False,
        disable_experience_memory: bool = False,
        disable_skill_memory: bool = False,
        disable_knowledge_memory: bool = False,
        memory_root: str | None = None,
        memory_llm_model: str = "",
        memory_llm_base_url: str = "",
        memory_llm_api_key: str = "",
        enforce_memory_blocks: bool = False,
        no_cxr: bool = False,
        **kwargs: Any,
    ) -> None:
        self.memory_root = memory_root or MEMORY_ROOT_DIRNAME
        self.trace_root = str(Path(self.memory_root) / "trace")

        self.query_builder_mode = query_builder_mode
        self.applicability_mode = applicability_mode
        self.experience_extraction_mode = experience_extraction_mode
        self.experience_merge_mode = experience_merge_mode

        self.memory_top_k = memory_top_k
        self.log_memory_trace = log_memory_trace
        self.disable_memory = disable_memory
        self.disable_experience_memory = disable_experience_memory
        self.disable_skill_memory = disable_skill_memory
        self.disable_knowledge_memory = disable_knowledge_memory
        self.enforce_memory_blocks = enforce_memory_blocks
        self.no_cxr = no_cxr

        self.memory_llm = LLMClient(
            model=memory_llm_model,
            base_url=memory_llm_base_url,
            api_key=memory_llm_api_key,
        )

        self.episode_id = f"episode_{uuid.uuid4().hex[:10]}"
        self.case_state: CaseState | None = None

        self.latest_query: MemoryQuery | None = None
        self.latest_retrieval: MemoryRetrievalResult | None = None
        self.latest_applicability: ApplicabilityResult | None = None
        self.latest_guidance: MemoryGuidance | None = None

        self.pending_selected_action: dict[str, Any] = {}
        self.pending_selected_action_blocked: bool = False
        self.turn_records: list[TurnRecord] = []
        self.episode_finalized = False

        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            # Some unit-test environments use a minimal fallback base class.
            super().__init__()

            self.tools = kwargs.get("tools", [])
            self.system_prompt = kwargs.get("system_prompt", "")
            self.parser_name = kwargs.get("parser_name", "")

    # ---------------------------------------------------------------------
    # Base agent bridge
    # ---------------------------------------------------------------------

    def _call_base(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        method = getattr(super(MemoryWrappedMedicalAgent, self), method_name, None)
        if method is None:
            return None

        try:
            return method(*args, **kwargs)
        except TypeError:
            if args:
                try:
                    return method(args[0])
                except TypeError:
                    return None
            return None

    # ---------------------------------------------------------------------
    # Online path
    # ---------------------------------------------------------------------

    def update_from_env(
        self,
        observation: Any = None,
        reward: float = 0.0,
        done: bool = False,
        info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Called when env returns an observation.

        This method:
          1. finalizes the previous turn record if a pending action exists;
          2. updates current CaseState;
          3. builds memory guidance for the next model call;
          4. injects memory guidance into base agent observation.
        """
        if observation is None and "obs" in kwargs:
            observation = kwargs["obs"]

        info = info or kwargs.get("info") or {}

        self._finalize_pending_turn(
            env_observation=observation,
            env_info=info,
            reward=reward,
            done=done,
        )

        if done:
            self._finalize_episode_if_needed(reward=reward, info=info)
            return self._call_base(
                "update_from_env",
                observation,
                reward=reward,
                done=done,
                info=info,
                **kwargs,
            )

        if self.case_state is None:
            bundle = self._case_bundle_from(observation, info)
            self.case_state = init_case_state(bundle, no_cxr=self.no_cxr)
        else:
            self.case_state = update_case_state(self.case_state, observation)

        if not self.disable_memory:
            self._run_memory_pipeline()

        enriched_observation = self._inject_guidance(observation)

        if self.log_memory_trace and self._has_memory_snapshot():
            payload = build_trace_payload(
                case_state=self.case_state,
                memory_query=self.latest_query,
                retrieval_result=self.latest_retrieval,
                applicability_result=self.latest_applicability,
                memory_guidance=self.latest_guidance,
                selected_action={},
            )
            append_memory_trace(self.trace_root, payload)

        return self._call_base(
            "update_from_env",
            enriched_observation,
            reward=reward,
            done=done,
            info=info,
            **kwargs,
        )

    def update_from_model(
        self,
        model_output: Any = None,
        **kwargs: Any,
    ) -> Any:
        """
        Called after base model produces an action.

        We record the selected action for offline extraction. For dict actions,
        optional enforcement can annotate or replace blocked actions.
        """
        if model_output is None and "output" in kwargs:
            model_output = kwargs["output"]

        parsed_action = self._parse_selected_action(model_output)
        action_type = str(parsed_action.get("action_type") or "").upper()

        blocked = self._is_action_blocked(action_type)
        self.pending_selected_action = parsed_action
        self.pending_selected_action_blocked = blocked

        rewritten_output = model_output
        if blocked and self.enforce_memory_blocks:
            rewritten_output = self._rewrite_blocked_output(model_output, action_type)

        base_result = self._call_base("update_from_model", rewritten_output, **kwargs)
        return base_result if base_result is not None else rewritten_output

    def _run_memory_pipeline(self) -> None:
        if self.case_state is None:
            return

        candidate_actions = self._candidate_actions()

        self.latest_query = build_memory_query(
            case_state=self.case_state,
            candidate_actions=candidate_actions,
            mode=self.query_builder_mode,
            llm_client=self.memory_llm,
        )

        self.latest_retrieval = retrieve_multi_memory(
            memory_query=self.latest_query,
            root_dir=self.memory_root,
            disable_experience_memory=self.disable_experience_memory,
            disable_skill_memory=self.disable_skill_memory,
            disable_knowledge_memory=self.disable_knowledge_memory,
        )

        self.latest_applicability = apply_applicability_control(
            case_state=self.case_state,
            memory_query=self.latest_query,
            retrieval_result=self.latest_retrieval,
            mode=self.applicability_mode,
            llm_client=self.memory_llm,
        )

        self.latest_guidance = build_memory_guidance(self.latest_applicability)

    def _inject_guidance(self, observation: Any) -> Any:
        if self.disable_memory or self.latest_guidance is None:
            return observation

        guidance_text = guidance_to_text(self.latest_guidance)
        if not guidance_text:
            return observation

        if isinstance(observation, dict):
            enriched = dict(observation)
            enriched["memory_guidance"] = guidance_text
            enriched["memory_guidance_structured"] = self.latest_guidance.to_dict()
            return enriched

        if isinstance(observation, str):
            return (
                observation
                + "\n\n[Memory Guidance]\n"
                + guidance_text
            )

        return observation

    def _candidate_actions(self) -> list[str]:
        tools = getattr(self, "tools", None) or []
        actions: list[str] = []

        for tool in tools:
            name = str(tool)
            action = ACTION_MAP.get(name)
            if action:
                actions.append(action)

        if not actions:
            actions = list(DEFAULT_ACTIONS)

        if "UPDATE_HYPOTHESIS" not in actions:
            actions.append("UPDATE_HYPOTHESIS")

        return list(dict.fromkeys(actions))

    # ---------------------------------------------------------------------
    # Turn records
    # ---------------------------------------------------------------------

    def _finalize_pending_turn(
        self,
        env_observation: Any,
        env_info: dict[str, Any],
        reward: float,
        done: bool,
    ) -> None:
        if not self.pending_selected_action:
            return

        if self.case_state is None:
            return

        record = TurnRecord(
            episode_id=self.episode_id,
            case_id=self.case_state.case_id,
            turn_id=self.case_state.turn_id,
            case_state=self.case_state.to_dict(),
            memory_query=self.latest_query.to_dict() if self.latest_query else {},
            retrieval_result=self.latest_retrieval.to_dict() if self.latest_retrieval else {},
            applicability_result=self.latest_applicability.to_dict() if self.latest_applicability else {},
            memory_guidance=self.latest_guidance.to_dict() if self.latest_guidance else {},
            selected_action=dict(self.pending_selected_action),
            env_observation=self._safe_payload(env_observation),
            env_info=self._safe_payload(env_info),
            reward=float(reward or 0.0),
            done=bool(done),
        )

        self.turn_records.append(record)

        self.pending_selected_action = {}
        self.pending_selected_action_blocked = False

    # ---------------------------------------------------------------------
    # Offline write
    # ---------------------------------------------------------------------

    def _finalize_episode_if_needed(
        self,
        reward: float,
        info: dict[str, Any],
    ) -> None:
        if self.episode_finalized:
            return

        if self.case_state is None:
            return

        feedback = EpisodeFeedback(
            episode_id=self.episode_id,
            case_id=self.case_state.case_id,
            success=bool(info.get("success", False)),
            total_reward=float(info.get("total_reward", reward or 0.0) or 0.0),
            final_diagnosis=str(
                info.get("final_diagnosis")
                or info.get("diagnosis")
                or ""
            ),
            gold_diagnosis=str(
                info.get("gold_diagnosis")
                or info.get("target_diagnosis")
                or info.get("answer")
                or ""
            ),
            summary=str(info.get("summary") or ""),
        )

        trajectory = {
            "info": {
                "memory_agent": {
                    "turn_records": [
                        record.to_dict() for record in self.turn_records
                    ]
                }
            }
        }

        distilled = distill_from_trajectory(trajectory, feedback)

        write_memory_from_distilled_episode(
            distilled,
            root_dir=self.memory_root,
            experience_extraction_mode=self.experience_extraction_mode,
            experience_merge_mode=self.experience_merge_mode,
            llm_client=self.memory_llm,
        )

        self.episode_finalized = True

    # ---------------------------------------------------------------------
    # Parsing / safety
    # ---------------------------------------------------------------------

    def _parse_selected_action(self, model_output: Any) -> dict[str, Any]:
        if isinstance(model_output, dict):
            action_type = (
                model_output.get("action_type")
                or model_output.get("tool")
                or model_output.get("name")
                or model_output.get("action")
                or ""
            )
            action_label = (
                model_output.get("action_label")
                or model_output.get("argument")
                or model_output.get("args")
                or model_output.get("content")
                or action_type
                or ""
            )

            return {
                "action_type": self._normalize_action_type(str(action_type)),
                "action_label": str(action_label),
                "raw": model_output,
                "blocked_by_memory": False,
            }

        text = str(model_output or "")
        return {
            "action_type": self._infer_action_type_from_text(text),
            "action_label": text[:240],
            "raw": text,
            "blocked_by_memory": False,
        }

    def _normalize_action_type(self, action: str) -> str:
        action = action.strip()
        if not action:
            return ""

        upper = action.upper()
        if upper in DEFAULT_ACTIONS:
            return upper

        mapped = ACTION_MAP.get(action.lower())
        if mapped:
            return mapped

        return upper

    def _infer_action_type_from_text(self, text: str) -> str:
        lowered = text.lower()

        if "diagnosis" in lowered or "final" in lowered:
            return "FINALIZE_DIAGNOSIS"
        if "cxr" in lowered or "x-ray" in lowered or "image" in lowered:
            return "REVIEW_IMAGE"
        if "lab" in lowered or "test" in lowered or "exam" in lowered:
            return "REQUEST_LAB"
        if "ask" in lowered or "patient" in lowered or "question" in lowered:
            return "ASK"

        return "UPDATE_HYPOTHESIS"

    def _is_action_blocked(self, action_type: str) -> bool:
        if not action_type or self.latest_guidance is None:
            return False
        return action_type in set(self.latest_guidance.blocked_actions)

    def _rewrite_blocked_output(self, model_output: Any, action_type: str) -> Any:
        warning = {
            "blocked_by_memory": True,
            "blocked_action_type": action_type,
            "replacement_action_type": "UPDATE_HYPOTHESIS",
            "reason": (
                self.latest_guidance.why_not_finalize
                if self.latest_guidance
                else "blocked by memory guidance"
            ),
        }

        if isinstance(model_output, dict):
            rewritten = dict(model_output)
            rewritten.update(warning)
            rewritten["action_type"] = "UPDATE_HYPOTHESIS"
            return rewritten

        return {
            "action_type": "UPDATE_HYPOTHESIS",
            "action_label": "reconsider plan due to memory safety block",
            "raw_blocked_output": str(model_output),
            **warning,
        }

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _case_bundle_from(self, observation: Any, info: dict[str, Any]) -> Any:
        return (
            info.get("case")
            or info.get("task")
            or info.get("case_bundle")
            or observation
            or {}
        )

    def _safe_payload(self, value: Any, max_chars: int = 4000) -> Any:
        if isinstance(value, dict):
            return {
                str(key): self._safe_payload(item, max_chars=max_chars)
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [
                self._safe_payload(item, max_chars=max_chars)
                for item in value[:50]
            ]

        text = str(value)
        if len(text) > max_chars:
            return text[:max_chars] + "...[truncated]"
        return value

    def _has_memory_snapshot(self) -> bool:
        return (
            self.case_state is not None
            and self.latest_query is not None
            and self.latest_retrieval is not None
            and self.latest_applicability is not None
            and self.latest_guidance is not None
        )

    def reset_memory(self) -> None:
        self.episode_id = f"episode_{uuid.uuid4().hex[:10]}"
        self.case_state = None

        self.latest_query = None
        self.latest_retrieval = None
        self.latest_applicability = None
        self.latest_guidance = None

        self.pending_selected_action = {}
        self.pending_selected_action_blocked = False
        self.turn_records = []
        self.episode_finalized = False