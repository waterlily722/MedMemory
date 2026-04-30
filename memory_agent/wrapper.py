from __future__ import annotations

import copy
import json
import logging
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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

from .llm import EmbeddingClient, LLMClient
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
    MemoryRetrievalResult,    OutcomeType,    TurnRecord,
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
    Memory wrapper for a MedEnv doctor agent.

    Online path:
        observation -> update CaseState -> build MemoryQuery -> retrieve memory
        -> applicability control -> MemoryGuidance -> inject into base agent.

    Offline path:
        done episode -> DistilledEpisode -> LLM experience extraction
        -> merge/write ExperienceStore.

    The base doctor agent remains responsible for action generation unless
    enforce_memory_blocks=True, in which case blocked actions are rewritten.
    """

    # ------------------------------------------------------------------
    # CXR multi-modal support (adapted from MedicalAgent)
    # ------------------------------------------------------------------
    @staticmethod
    def _cxr_tool_output_to_multimodal(
        tool_output_str: str,
    ) -> tuple[str | None, list[dict] | None]:
        """
        Parse CXR tool JSON: if it contains base64 images, return
        (tool_summary, multimodal_content_list). Otherwise (None, None).
        """
        try:
            data = json.loads(tool_output_str)
        except (json.JSONDecodeError, TypeError):
            return None, None
        if not isinstance(data, dict):
            return None, None
        images = data.get("images") or []
        if not isinstance(images, list) or not images:
            return None, None
        has_b64 = any(
            isinstance(img, dict) and img.get("image_base64") for img in images
        )
        if not has_b64:
            return None, None

        summary = data.get("summary") or ""
        view_list: list[str] = []
        content_parts: list[dict] = [{"type": "text", "text": summary}]

        for img in images:
            if not isinstance(img, dict):
                continue
            b64 = img.get("image_base64")
            vp = img.get("view_position", "unknown")
            view_list.append(vp)
            if b64:
                mime = (img.get("mime") or "image/jpeg").strip().lower()
                url = f"data:{mime};base64,{b64}" if "image/" in mime else f"data:image/jpeg;base64,{b64}"
                content_parts.append({"type": "image_url", "image_url": {"url": url}})

        tool_summary = (
            summary
            or f"CXR returned {len(images)} image(s). Views: {', '.join(view_list)}. Images attached below."
        )
        return tool_summary, content_parts

    def _format_observation_as_messages(self, obs: Any) -> list[dict]:
        """Format environment observation with CXR multi-modal support."""
        messages = []
        if isinstance(obs, dict):
            if "question" in obs:
                messages.append({"role": "user", "content": obs["question"]})
            elif "tool_outputs" in obs:
                for tool_call_id, tool_output_str in obs["tool_outputs"].items():
                    tool_summary, user_parts = self._cxr_tool_output_to_multimodal(
                        tool_output_str
                    )
                    if tool_summary is not None and user_parts is not None:
                        messages.append(
                            {
                                "role": "tool",
                                "content": tool_summary,
                                "tool_call_id": tool_call_id,
                            }
                        )
                        messages.append({"role": "user", "content": user_parts})
                    else:
                        messages.append(
                            {
                                "role": "tool",
                                "content": tool_output_str,
                                "tool_call_id": tool_call_id,
                            }
                        )
            else:
                messages.append(
                    {"role": "user", "content": json.dumps(obs, ensure_ascii=False)}
                )
        elif isinstance(obs, str):
            messages.append({"role": "user", "content": obs})
        elif obs is not None:
            messages.append({"role": "user", "content": str(obs)})
        return messages

    def __init__(
        self,
        *args: Any,
        case_update_mode: str = "llm",
        query_builder_mode: str = "rule",
        applicability_mode: str = "rule",
        experience_extraction_mode: str = "llm",
        experience_merge_mode: str = "rule",
        memory_top_k: int = 5,
        log_memory_trace: bool = False,
        disable_memory: bool = False,
        enable_memory: bool | None = None,
        disable_experience_memory: bool = False,
        disable_skill_memory: bool = False,
        disable_knowledge_memory: bool = False,
        memory_root: str | None = None,
        memory_llm_model: str = "",
        memory_llm_base_url: str = "",
        memory_llm_api_key: str = "",
        memory_embedding_model: str = "",
        memory_embedding_base_url: str = "",
        memory_embedding_api_key: str = "",
        enforce_memory_blocks: bool = False,
        no_cxr: bool = False,
        **kwargs: Any,
    ) -> None:
        # Support both enable_memory (from run script) and disable_memory (internal)
        if enable_memory is not None:
            disable_memory = not enable_memory
        self.memory_root = memory_root or MEMORY_ROOT_DIRNAME
        self.trace_root = str(Path(self.memory_root) / "trace")
        self.case_update_mode = case_update_mode
        self.query_builder_mode = query_builder_mode
        self.applicability_mode = applicability_mode
        self.experience_extraction_mode = experience_extraction_mode
        self.experience_merge_mode = experience_merge_mode
        # memory_top_k is reserved for future top-k filtering at the wrapper
        # level. Currently per-memory-type top-k is configured in RETRIEVAL_CONFIG.
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
        self.memory_embedding = EmbeddingClient(
            model=memory_embedding_model,
            base_url=memory_embedding_base_url,
            api_key=memory_embedding_api_key,
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
        # Cached original task bundle (set during first update_from_env call)
        self._case_bundle: dict[str, Any] = {}

        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            # Some unit-test environments use a minimal fallback base class.
            super().__init__()
            self.tools = kwargs.get("tools", [])
            self.system_prompt = kwargs.get("system_prompt", "")
            self.parser_name = kwargs.get("parser_name", "")

    # ------------------------------------------------------------------
    # Base agent bridge
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Online path
    # ------------------------------------------------------------------
    def update_from_env(
        self,
        observation: Any = None,
        reward: float = 0.0,
        done: bool = False,
        info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
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
            self._case_bundle = bundle if isinstance(bundle, dict) else {}
            self.case_state = init_case_state(bundle, no_cxr=self.no_cxr)
        else:
            self.case_state = update_case_state(
                self.case_state,
                observation,
                mode=self.case_update_mode,
                llm_client=self.memory_llm,
            )

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

    def update_from_model(self, model_output: Any = None, **kwargs: Any) -> Any:
        if model_output is None and "output" in kwargs:
            model_output = kwargs["output"]

        parsed_action = self._parse_selected_action(model_output)
        action_type = str(parsed_action.get("action_type") or "").upper()
        blocked = self._is_action_blocked(action_type)
        parsed_action["blocked_by_memory"] = bool(blocked)

        rewritten_output = model_output
        if blocked and self.enforce_memory_blocks:
            rewritten_output = self._rewrite_blocked_output(model_output, action_type)
            # pending_selected_action records the *actually executed* action
            parsed_action = self._parse_selected_action(rewritten_output)
            parsed_action["original_action"] = action_type
            parsed_action["blocked_by_memory"] = True

        self.pending_selected_action = parsed_action
        self.pending_selected_action_blocked = blocked

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
            embedding_client=self.memory_embedding if self.memory_embedding.available() else None,
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
            return observation + "\n\n[Memory Guidance]\n" + guidance_text
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

    # ------------------------------------------------------------------
    # Turn records
    # ------------------------------------------------------------------
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
            case_state=self.case_state,
            memory_query=self.latest_query,
            retrieval_result=self.latest_retrieval,
            applicability_result=self.latest_applicability,
            memory_guidance=self.latest_guidance,
            selected_action=dict(self.pending_selected_action),
            env_observation=self._safe_payload(env_observation),
            env_info=self._safe_payload(env_info),
            reward=float(reward or 0.0),
            done=bool(done),
        )
        self.turn_records.append(record)
        self.pending_selected_action = {}
        self.pending_selected_action_blocked = False

    # ------------------------------------------------------------------
    # Offline write
    # ------------------------------------------------------------------
    def _finalize_episode_if_needed(self, reward: float, info: dict[str, Any]) -> None:
        if self.episode_finalized:
            return
        if self.case_state is None:
            logger.warning(
                "Episode %s done but case_state is None — skipping offline write",
                self.episode_id,
            )
            return

        # ── Extract fields from env info dict ─────────────────────────
        # Env finalize returns: {"is_correct": ..., "metadata": {...}, "response": ...}
        is_correct = bool(info.get("is_correct", False))

        # Total reward from trajectory (sum of step rewards) or fallback
        total_reward = float(
            sum(record.reward for record in self.turn_records) or reward or 0.0
        )

        # Final diagnosis: extract from last turn record's selected_action
        final_diagnosis = ""
        if self.turn_records:
            last_action = self.turn_records[-1].selected_action or {}
            final_diagnosis = str(
                last_action.get("action_label")
                or last_action.get("raw")
                or ""
            )

        # Gold diagnosis: from cached bundle (medenv_case_bundle -> ehr -> Correct_Diagnosis)
        gold_diagnosis = ""
        bundle_ehr = (
            self._case_bundle.get("ehr")
            or (self._case_bundle.get("medenv_case_bundle") or {}).get("ehr")
            or {}
        )
        if isinstance(bundle_ehr, dict):
            gold_diagnosis = str(
                bundle_ehr.get("Correct_Diagnosis")
                or bundle_ehr.get("Principal_Diagnosis")
                or ""
            )
        # Also check task-level ground_truth
        if not gold_diagnosis:
            gold_diagnosis = str(
                info.get("gold_diagnosis")
                or self._case_bundle.get("ground_truth")
                or self._case_bundle.get("gold_diagnosis")
                or ""
            )

        summary = str(
            info.get("summary")
            or (info.get("metadata") or {}).get("result_reward_reason")
            or ""
        )

        feedback = EpisodeFeedback(
            episode_id=self.episode_id,
            case_id=self.case_state.case_id,
            success=is_correct,
            total_reward=total_reward,
            final_diagnosis=final_diagnosis,
            gold_diagnosis=gold_diagnosis,
            summary=summary,
        )
        trajectory = {
            "info": {
                "memory_agent": {
                    "turn_records": [record.to_dict() for record in self.turn_records]
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

    # ------------------------------------------------------------------
    # Parsing / safety
    # ------------------------------------------------------------------
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
            rewritten = copy.deepcopy(model_output)
            rewritten.update(warning)
            rewritten["action_type"] = "UPDATE_HYPOTHESIS"
            return rewritten
        return {
            "action_type": "UPDATE_HYPOTHESIS",
            "action_label": "reconsider plan due to memory safety block",
            "raw_blocked_output": str(model_output),
            **warning,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _case_bundle_from(self, observation: Any, info: dict[str, Any]) -> Any:
        return info.get("case") or info.get("task") or info.get("case_bundle") or observation or {}

    def _safe_payload(self, value: Any, max_chars: int = 4000) -> Any:
        if isinstance(value, dict):
            return {
                str(key): self._safe_payload(item, max_chars=max_chars)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._safe_payload(item, max_chars=max_chars) for item in value[:50]]
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
