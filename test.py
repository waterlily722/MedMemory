#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test MedicalDialogueEnv ask_patient tool-call path using a real case json.

Default case:
  /data/xuxiang/mimic-iv/virtual_env/data/json_case_new-2/osce_22276256.json

Usage:
  python test_medgym_env_main.py --mode env
  python test_medgym_env_main.py --mode direct

Notes:
- ask_patient tool 的 patient-agent（vLLM）配置依赖环境变量：
  RLLM_PATIENT_BASE_URL / RLLM_PATIENT_MODEL / RLLM_PATIENT_API_KEY
- EHR/KB 可以直接从 case json 注入，不依赖 RLLM_CASE_DIR（但你也可以设置）
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from rllm.tools.multi_tool import MultiTool
from rllm.environments.medgym.medgym_env import MedicalDialogueEnv


DEFAULT_CASE_JSON = "/data/xuxiang/mimic-iv/virtual_env/data/json_case_new-2/osce_22276256.json"


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def derive_case_id(case_json_path: str, obj: Any) -> str:
    # 1) case json 里有就用
    if isinstance(obj, dict) and isinstance(obj.get("case_id"), str) and obj["case_id"].strip():
        return obj["case_id"].strip()
    # 2) 否则用文件名（不含后缀）
    return Path(case_json_path).stem


def extract_ehr_and_kb(obj: Any) -> tuple[Dict[str, Any], Any]:
    """
    兼容几种常见结构：
    - { "ehr": {...}, "knowledge": ... }
    - { "ehr": {...}, "knowbase": ... }
    - 直接就是 EHR dict（含 Basic_Information/Symptoms/...）
    """
    if not isinstance(obj, dict):
        return {}, None

    # case wrapper
    if isinstance(obj.get("ehr"), dict):
        ehr = obj["ehr"]
        kb = obj.get("knowledge", obj.get("knowbase", None))
        return ehr, kb

    # maybe raw ehr dict
    # 如果包含固定 section keys，则认为本身就是 ehr
    maybe_keys = {"Basic_Information", "Symptoms", "Physical_Examination", "Test_Results", "Medications", "Treatment"}
    if any(k in obj for k in maybe_keys):
        return obj, None

    return {}, None


def get_chief_complaint(ehr: Dict[str, Any]) -> str:
    """
    取 Basic_Information.Chief_Complaint，缺失就返回空串。
    """
    bi = ehr.get("Basic_Information", {})
    if isinstance(bi, dict):
        cc = bi.get("Chief_Complaint", "")
        return str(cc).strip()
    return ""


def build_task_from_case(case_json_path: str, knowbase_json: Optional[str] = None) -> Dict[str, Any]:
    obj = load_json(case_json_path)
    case_id = derive_case_id(case_json_path, obj)
    ehr, kb = extract_ehr_and_kb(obj)

    # 如果外部传了 knowbase_json，就用它覆盖/补充（可选）
    if knowbase_json:
        try:
            kb = load_json(knowbase_json)
        except Exception:
            pass

    chief = get_chief_complaint(ehr)
    init_question = f"I have been feeling unwell recently. {chief}".strip() if chief else ""

    task = {
        "case_id": case_id,
        "context": {
            "ehr": ehr or {},
            "knowbase": kb,
            "dialogue": [],
        },
        "question": init_question,
    }
    return task


def make_ask_tool_call(question: str, call_id: str) -> List[Dict[str, Any]]:

    return [
        {
            "id": call_id,
            "type": "function",
            "function": {
                "name": "ask_patient",
                "arguments": {"question": question},
            },
        }
    ]


def run_env_mode(task: Dict[str, Any], max_steps: int, questions: List[str]):
    env = MedicalDialogueEnv(
        task=task,
        tools=["ask_patient"],
        reward_fn=None, 
        max_steps=max_steps,
        ask_tool_name="ask_patient",
        parallel_tool_calls=False,
        max_tool_workers=4,
        timeout=60,
        start_method="spawn",
    )

    try:
        obs, info = env.reset()
        print("=== RESET ===")
        print("case_id:", task.get("case_id"))
        print("initial task.question:", repr(task.get("question", "")))
        print("obs type:", type(obs), "obs keys:", list(obs.keys()) if isinstance(obs, dict) else None)
        print("info:", info)

        print("\n=== STEPS (env.step(tool_calls)) ===")
        for i, q in enumerate(questions, 1):
            action = make_ask_tool_call(q, call_id=f"ask_{i}")
            next_obs, reward, done, step_info = env.step(action)

            print(f"\n--- Turn {i} ---")
            print("Doctor asks:", q)
            print("next_obs:", next_obs)  # 期望 {"question": reply}
            print("reward:", reward, "done:", done)

            md = step_info.get("metadata", {})
            print("metadata.turn_type:", md.get("turn_type"))
            print("metadata.ask_tool:", md.get("ask_tool"))
            print("metadata.context_keys:", md.get("context_keys"))  # 这里能看到 tool_context 里是否包含 ehr/knowbase/dialogue

            if done:
                break

    finally:
        env.close()


def run_direct_tool_mode(task: Dict[str, Any], questions: List[str]):
    """
    直接调用 MultiTool -> ask_patient，更直观看到 tool 返回的 context.dialogue 是否累积
    """
    tool_runner = MultiTool(tools=["ask_patient"])

    cid = str(task.get("case_id", ""))
    ctx = dict(task.get("context", {}) or {})
    if not isinstance(ctx.get("dialogue"), list):
        ctx["dialogue"] = []

    print("=== DIRECT TOOL CALLS (MultiTool -> ask_patient) ===")
    print("case_id:", cid)
    print("context keys:", sorted(list(ctx.keys())))

    for i, q in enumerate(questions, 1):
        out = tool_runner(tool_name="ask_patient", question=q, case_id=cid, context=ctx)

        print(f"\n--- Turn {i} ---")
        print("Doctor asks:", q)

        if hasattr(out, "output") and isinstance(out.output, dict):
            ans = str(out.output.get("answer", ""))
            new_ctx = out.output.get("context", None)
            dbg = out.output.get("debug", None)

            print("answer:", ans)

            if isinstance(new_ctx, dict):
                ctx = new_ctx
                dlg = ctx.get("dialogue", [])
                print("dialogue_len:", len(dlg))
                if dlg:
                    print("dialogue_tail(last2):")
                    print(json.dumps(dlg[-2:], ensure_ascii=False, indent=2))
            else:
                print("tool returned no context dict; raw type:", type(new_ctx))

            if isinstance(dbg, dict):
                print("debug.answer_source:", dbg.get("answer_source"))
                print("debug.route:", dbg.get("route"))
                print("debug.ehr_answerable:", dbg.get("ehr_answerable"))
                print("debug.kb_answerable:", dbg.get("kb_answerable"))
        else:
            # fallback：打印工具的字符串输出
            print(out.to_string())


def parse_questions(arg: Optional[str]) -> List[str]:
    if not arg:
        return [
            "What is bothering you the most right now?",
            "How long has this been going on?",
            "Do you have any nausea, vomiting, fever, or chest pain?",
            "Are you taking any medications currently?",
            "Do you know anything about your lab tests (e.g., calcium, PTH)?",
        ]
    # 支持传 JSON list 或者用 | 分隔
    s = arg.strip()
    if s.startswith("["):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(x) for x in arr]
        except Exception:
            pass
    return [x.strip() for x in s.split("|") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_json", default=DEFAULT_CASE_JSON, help="Path to a case json (osce_xxx.json).")
    parser.add_argument("--knowbase_json", default="", help="Optional external knowbase json; if set, overrides/uses KB from file.")
    parser.add_argument("--mode", choices=["env", "direct"], default="env", help="env: test MedicalDialogueEnv; direct: test MultiTool directly.")
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--questions", default="", help="Questions: JSON list string or 'q1|q2|q3'.")

    args = parser.parse_args()

    case_json = args.case_json
    if not os.path.exists(case_json):
        raise FileNotFoundError(case_json)

    kb_path = args.knowbase_json.strip() or None
    task = build_task_from_case(case_json, knowbase_json=kb_path)

    questions = parse_questions(args.questions)

    print("=== ENV VARS (patient-agent) ===")
    print("RLLM_PATIENT_BASE_URL =", os.getenv("RLLM_PATIENT_BASE_URL"))
    print("RLLM_PATIENT_MODEL    =", os.getenv("RLLM_PATIENT_MODEL"))
    print("RLLM_PATIENT_API_KEY  =", os.getenv("RLLM_PATIENT_API_KEY"))
    print("RLLM_CASE_DIR         =", os.getenv("RLLM_CASE_DIR"))
    print("RLLM_KNOWBASE_JSON    =", os.getenv("RLLM_KNOWBASE_JSON"))

    if args.mode == "env":
        run_env_mode(task=task, max_steps=args.max_steps, questions=questions)
    else:
        run_direct_tool_mode(task=task, questions=questions)


if __name__ == "__main__":
    main()
