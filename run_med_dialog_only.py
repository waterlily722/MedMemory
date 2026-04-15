#!/usr/bin/env python3
"""
纯多轮对话问诊测试脚本（无工具调用）

适用于没有 agentic 能力的模型：医生模型直接输出问题文本，不调用 ask_patient 等工具。
通过 patient simulator 模拟患者回答，实现多轮对话。

流程：
  1. 加载 case（EHR + knowledge）
  2. 医生模型根据患者主诉和对话历史，输出下一个问题（纯文本）
  3. Patient simulator 根据 EHR/KB 生成患者回答（复用 dialog_tool.patient_answer）
  4. 可选：Examiner 模型在达到最大轮次时提示医生给出最终诊断
  5. 循环直到医生输出诊断或达到 max_turns

环境变量（与 dialog_tool 一致）：
  RLLM_PATIENT_BASE_URL  - Patient simulator API（如 http://127.0.0.1:30001）
  RLLM_PATIENT_MODEL     - Patient 模型名
  RLLM_DOCTOR_BASE_URL   - 医生模型 API（默认同 base_url）
  RLLM_EXAMINER_BASE_URL - Examiner 模型 API（可选，不设则不用 examiner）

用法示例:
  cd examples/MedGym
  python run_med_dialog_only.py \\
    --case_dir /data/xuxiang/mimic-iv/osce_data/bench \\
    --doctor_model doctor_agent \\
    --patient_model patient_agent \\
    --max_cases 2 --max_turns 10 \\
    --output outputs/dialog_results.json

  # 启用 LLM-as-judge 判断诊断一致性
  python run_med_dialog_only.py ... --judge_model Qwen/Qwen3-8B --judge_base_url http://localhost:30002/v1
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

# 确保能 import rllm
_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir.parents[1]))  # rllm repo root

from rllm.tools.med_tool.dialog_tool import (
    OpenAICompatChatModel,
    patient_answer,
    strip_think,
    load_json_file,
)
from rllm.rewards.med_diagnosis_reward import _call_judge_api

# 医生纯对话 system prompt（无工具）
DOCTOR_DIALOG_SYSTEM_PROMPT = """You are a physician conducting a multi-turn clinical interview to reach a diagnosis.

You do NOT have access to tools. You must ask the patient questions directly by outputting your question as plain text.

RULES:
- Output exactly ONE question per turn. No tool calls, no JSON, no extra formatting.
- Ask clear, focused questions (e.g., "When did the chest pain start?", "Do you have any fever?")
- Use the dialogue history to avoid repeating questions and to choose the next most informative step.
- When you are ready to give your final diagnosis, output exactly one line in this format:
  The final diagnosis is: \\boxed{<diagnosis>}.

GOAL:
- Collect evidence through questioning and then submit your final diagnosis in the format above."""

DOCTOR_DIALOG_SYSTEM_PROMPT_wo_IMG = """You are a physician conducting a multi-turn clinical interview to reach a diagnosis.

You do NOT have access to tools. You must ask the patient questions directly by outputting your question as plain text.

RULES:
- Output exactly ONE question per turn. No tool calls, no JSON, no extra formatting.
- Ask clear, focused questions.
- Use the dialogue history to avoid repeating questions.
- When you are ready to give your final diagnosis, output exactly one line:
  The final diagnosis is: \\boxed{<diagnosis>}.

GOAL:
- Collect evidence through questioning and then submit your final diagnosis."""

# Examiner 提示医生给出诊断
EXAMINER_PROMPT_TEMPLATE = """The clinical interview has reached the maximum number of turns.

Based on the dialogue below, output your final diagnosis in exactly this format:
The final diagnosis is: \\boxed{<diagnosis>}.

Dialogue:
{dialogue_text}

Your final diagnosis:"""


def _normalize_openai_base_url(u: str) -> str:
    u = (u or "").strip().rstrip("/")
    if not u:
        return u
    if u.endswith("/v1/chat/completions"):
        return u[: -len("/chat/completions")]
    if u.endswith("/chat/completions"):
        return u[: -len("/chat/completions")]
    if u.endswith("/v1"):
        return u
    return u + "/v1"


def _check_containment(pred: str, gold: str) -> bool:
    """包含判断：gold 是否在 pred 中（归一化后）"""
    if not pred or not gold:
        return False
    pred_n = re.sub(r"[^\w\s]", " ", pred.lower())
    gold_n = re.sub(r"[^\w\s]", " ", gold.lower())
    pred_n = " ".join(pred_n.split())
    gold_n = " ".join(gold_n.split())
    if gold_n in pred_n:
        return True
    gold_tokens = set(gold_n.split())
    pred_tokens = set(pred_n.split())
    overlap = len(gold_tokens & pred_tokens) / len(gold_tokens) if gold_tokens else 0
    return overlap >= 0.5


def extract_boxed_diagnosis(text: str) -> str | None:
    """从 'The final diagnosis is: \\boxed{xxx}.' 中提取 xxx"""
    if not text:
        return None
    text = text.split("</think>")[-1].strip()
    m = re.search(
        r"The\s+final\s+diagnosis\s+is\s*:\s*\\box(?:ed)?\{(.+?)\}\s*\.?\s*",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    m = re.search(r"\\box(?:ed)?\{(.+?)\}", text)
    if m:
        return m.group(1).strip()
    return None


def _dialogue_to_text(dialogue: list[dict]) -> str:
    parts = []
    for item in dialogue:
        role = item.get("role", "")
        content = item.get("content", "")
        parts.append(f"{role.upper()}: {content}")
    return "\n\n".join(parts)


# bench 目录结构常量（与 /data/xuxiang/mimic-iv/osce_data/bench 一致）
# bench/
#   with_img/ed_hosp/{hadm_id}/ehr_{hadm_id}.json
#   without_img/hosp_only/{hadm_id}/ehr_{hadm_id}.json
SUBDIR_WITH_CXR = "with_img/ed_hosp"
SUBDIR_NO_CXR = "without_img/hosp_only"


def load_cases_from_bench(
    bench_root: str,
    subdir: str = "without_img/hosp_only",
    max_cases: int = -1,
) -> list[dict]:
    """
    直接从 bench 目录结构加载 cases。
    bench_root: 如 /data/xuxiang/mimic-iv/osce_data/bench
    subdir: with_img/ed_hosp 或 without_img/hosp_only
    """
    bench_root = os.path.abspath(os.path.expanduser(bench_root))
    search_dir = os.path.join(bench_root, subdir)
    pattern = os.path.join(search_dir, "*", "ehr_*.json")
    paths = sorted(glob.glob(pattern))
    if max_cases > 0:
        paths = paths[:max_cases]

    tasks = []
    for ehr_path in paths:
        case_dir = os.path.dirname(ehr_path)
        case_id = os.path.splitext(os.path.basename(ehr_path))[0].replace("ehr_", "")

        try:
            obj = load_json_file(ehr_path)
        except Exception:
            continue

        if not isinstance(obj, dict):
            continue

        if "ehr" in obj and isinstance(obj["ehr"], dict):
            ehr = obj["ehr"].copy()
            knowledge = obj.get("knowledge") or []
        else:
            ehr = obj if isinstance(obj, dict) else {}
            knowledge = []

        gold = (ehr.get("Final_Result") or "") if isinstance(ehr, dict) else ""
        info = ehr.get("Patient_info") or {}
        history = ehr.get("History") or {}
        chief = history.get("Chief_Complaint", "")

        tasks.append({
            "case_id": case_id,
            "context": {
                "ehr": ehr,
                "knowbase": knowledge,
                "case_dir": case_dir,
            },
            "question": (
                f"I am a {info.get('age', '')}-year-old {info.get('gender', '')} patient. "
                f"I haven't been feeling well recently. My main issue is: {chief}."
            ),
            "ground_truth": gold,
        })
    return tasks


def run_single_case(
    case: dict,
    doctor_model: OpenAICompatChatModel,
    patient_model: OpenAICompatChatModel | None,
    examiner_model: OpenAICompatChatModel | None,
    max_turns: int = 15,
    patient_max_tokens: int = 512,
    doctor_max_tokens: int = 512,
    use_examiner_at_end: bool = True,
    verbose: bool = True,
) -> dict:
    """
    对单个 case 运行多轮对话。

    Returns:
        {
            "case_id": str,
            "dialogue": [{"role": "doctor"|"patient", "content": str}, ...],
            "final_diagnosis": str | None,
            "ground_truth": str,
            "termination_reason": str,
        }
    """
    case_id = case.get("case_id", "unknown")
    context = case.get("context", {})
    ehr = context.get("ehr") or case.get("ehr", {})
    knowledge = context.get("knowledge") or context.get("knowbase") or case.get("knowledge")
    ground_truth = case.get("ground_truth") or (ehr.get("Final_Result") if isinstance(ehr, dict) else "")

    if not isinstance(ehr, dict):
        ehr = {}

    # 构建初始患者信息
    info = ehr.get("Patient_info") or {}
    history = ehr.get("History") or {}
    chief = history.get("Chief_Complaint", "")
    question = case.get("question", "")
    if not question:
        question = (
            f"I am a {info.get('age', '')}-year-old {info.get('gender', '')} patient. "
            f"I haven't been feeling well recently. My main issue is: {chief}."
        )

    dialogue: list[dict] = [
        {"role": "patient", "content": question},
    ]

    # 构建医生首轮 prompt
    def build_doctor_messages() -> list[dict]:
        msgs = [
            {"role": "system", "content": DOCTOR_DIALOG_SYSTEM_PROMPT_wo_IMG},
            {"role": "user", "content": f"Patient's opening statement:\n{question}\n\nYour first question to the patient:"},
        ]
        # 后续轮次：加入完整对话历史
        if len(dialogue) > 1:
            dlg_text = _dialogue_to_text(dialogue)
            msgs = [
                {"role": "system", "content": DOCTOR_DIALOG_SYSTEM_PROMPT_wo_IMG},
                {"role": "user", "content": f"Dialogue so far:\n{dlg_text}\n\nYour next question (or final diagnosis):"},
            ]
        return msgs

    termination_reason = "MAX_TURNS"
    final_diagnosis = None

    for turn in range(max_turns):
        # 医生生成
        doctor_msgs = build_doctor_messages()
        doctor_out = doctor_model.chat(
            doctor_msgs,
            temperature=0.6,
            max_tokens=doctor_max_tokens,
        )
        doctor_out = strip_think(doctor_out).strip()

        if not doctor_out:
            doctor_out = "Could you tell me more about your symptoms?"

        dialogue.append({"role": "doctor", "content": doctor_out})

        # 检查是否已给出诊断
        final_diagnosis = extract_boxed_diagnosis(doctor_out)
        if final_diagnosis is not None:
            termination_reason = "DOCTOR_DIAGNOSIS"
            if verbose:
                print(f"  [Turn {turn + 1}] Doctor gave diagnosis: {final_diagnosis}")
            break

        # 从医生输出中提取问题（用于 patient_answer）
        doctor_question = doctor_out
        if "\n" in doctor_question:
            doctor_question = doctor_question.split("\n")[0].strip()
        if not doctor_question:
            doctor_question = doctor_out

        # Patient simulator 回答
        try:
            patient_ans, _ = patient_answer(
                llm=patient_model,
                doctor_question=doctor_question,
                ehr=ehr,
                knowledge_obj=knowledge,
                max_answer_tokens=patient_max_tokens,
            )
        except Exception as e:
            patient_ans = f"I'm not sure. ({type(e).__name__}: {e})"

        patient_ans = strip_think(patient_ans).strip() or "I'm not sure. Could you ask more specifically?"
        dialogue.append({"role": "patient", "content": patient_ans})

        if verbose:
            print(f"  [Turn {turn + 1}] Doctor: {doctor_question[:80]}...")
            print(f"           Patient: {patient_ans[:80]}...")

    # 若未给出诊断且启用 examiner，用 examiner 提示医生给出诊断
    if final_diagnosis is None and use_examiner_at_end and examiner_model is not None:
        dlg_text = _dialogue_to_text(dialogue)
        examiner_prompt = EXAMINER_PROMPT_TEMPLATE.format(dialogue_text=dlg_text)
        examiner_msgs = [
            {"role": "system", "content": "You are a physician. Output your final diagnosis in the format: The final diagnosis is: \\boxed{<diagnosis>}."},
            {"role": "user", "content": examiner_prompt},
        ]
        try:
            examiner_out = examiner_model.chat(examiner_msgs, temperature=0.0, max_tokens=256)
            final_diagnosis = extract_boxed_diagnosis(examiner_out)
            if final_diagnosis:
                termination_reason = "EXAMINER_PROMPTED"
                dialogue.append({"role": "doctor", "content": examiner_out.strip()})
        except Exception as e:
            if verbose:
                print(f"  Examiner failed: {e}")

    res = {
        "case_id": case_id,
        "dialogue": dialogue,
        "final_diagnosis": final_diagnosis,
        "ground_truth": str(ground_truth).strip() if ground_truth else "",
        "termination_reason": termination_reason,
    }
    return res


def evaluate_result(
    res: dict,
    judge_model_name: str = "",
    judge_base_url: str = "",
    judge_api_key: str = "None",
) -> dict:
    """
    对单个 case 结果做判断：若启用 judge 则用 LLM 判断诊断一致性，否则用包含规则。
    写入 judge_consistent, is_correct, judge_raw_response 等字段。
    """
    pred = res.get("final_diagnosis") or ""
    gold = res.get("ground_truth") or ""

    if not gold:
        res["judge_consistent"] = False
        res["is_correct"] = False
        res["evaluate_reason"] = "no_ground_truth"
        return res

    if judge_model_name and judge_base_url:
        is_consistent, raw_judge = _call_judge_api(
            pred, gold, judge_model_name, judge_base_url, judge_api_key
        )
        res["judge_consistent"] = is_consistent
        res["is_correct"] = is_consistent
        res["judge_raw_response"] = raw_judge if raw_judge else ""
        res["evaluate_mode"] = "judge"
    else:
        is_correct = _check_containment(pred, gold)
        res["judge_consistent"] = is_correct  # 兼容字段
        res["is_correct"] = is_correct
        res["evaluate_mode"] = "containment"

    res["model_output_excerpt"] = pred[:300] if pred else ""
    return res


def main():
    parser = argparse.ArgumentParser(description="纯多轮对话问诊测试（无工具调用）")
    parser.add_argument("--case_dir", required=True, help="Bench 根目录，如 osce_data/bench")
    parser.add_argument("--max_cases", type=int, default=3)
    parser.add_argument("--max_turns", type=int, default=15)
    parser.add_argument("--no_cxr", action="store_true", help="使用 without_img/hosp_only")
    parser.add_argument("--base_url", default="http://localhost:30000/v1")
    parser.add_argument("--api_key", default="None")
    parser.add_argument("--doctor_model", required=True, help="医生模型名（--served-model-name）")
    parser.add_argument("--patient_model", default="", help="Patient simulator 模型名，空则用默认回答")
    parser.add_argument("--examiner_model", default="", help="Examiner 模型名，空则不用")
    parser.add_argument("--patient_base_url", default="", help="Patient API URL，默认同 base_url")
    parser.add_argument("--examiner_base_url", default="", help="Examiner API URL，默认同 base_url")
    parser.add_argument("--no_examiner", action="store_true", help="不在结束时用 examiner 提示诊断")
    parser.add_argument("--output", default="", help="结果保存路径（JSON）")
    parser.add_argument("--verbose", action="store_true", default=True)
    # LLM-as-judge：用 LLM 判断诊断与真实是否一致（不设则用包含规则）
    parser.add_argument("--judge_model", default="", help="Judge 模型名，如 Qwen/Qwen3-8B。设则用 LLM 判断一致性")
    parser.add_argument("--judge_base_url", default="", help="Judge API URL，默认同 base_url")
    parser.add_argument("--judge_api_key", default="", help="Judge API key，默认同 api_key")
    args = parser.parse_args()

    # 加载 cases：基于 bench 目录结构直接读取
    # bench_root/case_dir 如 /data/xuxiang/mimic-iv/osce_data/bench
    bench_root = os.path.abspath(os.path.expanduser(args.case_dir))
    subdir = SUBDIR_NO_CXR if args.no_cxr else SUBDIR_WITH_CXR
    tasks = load_cases_from_bench(
        bench_root=bench_root,
        subdir=subdir,
        max_cases=args.max_cases,
    )
    if not tasks:
        print(f"Error: No cases found under {bench_root}/{subdir}")
        print("Expected structure: bench_root/{subdir}/{hadm_id}/ehr_{hadm_id}.json")
        return []

    base_url = _normalize_openai_base_url(args.base_url)
    patient_base = _normalize_openai_base_url(args.patient_base_url or args.base_url)
    examiner_base = _normalize_openai_base_url(args.examiner_base_url or args.base_url)
    judge_base = _normalize_openai_base_url(args.judge_base_url or args.base_url)
    judge_key = args.judge_api_key if args.judge_api_key else args.api_key

    doctor_model = OpenAICompatChatModel(
        base_url=base_url,
        api_key=args.api_key,
        model=args.doctor_model,
        timeout=60,
    )

    patient_model = None
    if args.patient_model:
        patient_model = OpenAICompatChatModel(
            base_url=patient_base,
            api_key=args.api_key,
            model=args.patient_model,
            timeout=60,
        )

    examiner_model = None
    if args.examiner_model and not args.no_examiner:
        examiner_model = OpenAICompatChatModel(
            base_url=examiner_base,
            api_key=args.api_key,
            model=args.examiner_model,
            timeout=60,
        )

    results = []
    for i, task in enumerate(tasks):
        print(f"\n=== Case {i + 1}/{len(tasks)}: {task.get('case_id', '?')} ===")
        res = run_single_case(
            case=task,
            doctor_model=doctor_model,
            patient_model=patient_model,
            examiner_model=examiner_model,
            max_turns=args.max_turns,
            use_examiner_at_end=not args.no_examiner,
            verbose=args.verbose,
        )
        res = evaluate_result(
            res,
            judge_model_name=args.judge_model,
            judge_base_url=judge_base,
            judge_api_key=judge_key,
        )
        results.append(res)
        print(f"  Final: {res.get('final_diagnosis', 'N/A')}")
        print(f"  Gold:  {res.get('ground_truth', '')}")
        print(f"  Judge: {'✓' if res.get('is_correct') else '✗'} ({res.get('evaluate_mode', '?')})")

    # 汇总准确率
    n_correct = sum(1 for r in results if r.get("is_correct"))
    n_total = len(results)
    acc = 100 * n_correct / n_total if n_total else 0
    print(f"\n=== Summary: {n_correct}/{n_total} correct ({acc:.1f}%) ===")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved to {out_path}")

    return results


if __name__ == "__main__":
    main()


"""
python run_med_dialog_only.py \
  --case_dir /data/xuxiang/mimic-iv/osce_data/bench \
  --base_url http://localhost:30000/v1 \
  --patient_base_url http://localhost:30001/v1 \
  --examiner_base_url http://localhost:30000/v1 \
  --doctor_model doctor_agent \
  --patient_model patient_agent \
  --examiner_model doctor_agent \
  --max_cases 2 --output outputs/dialog_results.json
"""