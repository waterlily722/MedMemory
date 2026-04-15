#!/usr/bin/env python3
"""
测试 doctor（视觉模型）在收到 CXR 图像后的反应。

用法:
  # 使用真实 case 的 CXR 图像（从 case_dir 选一个含 CXR 的 case）
  python test_doctor_vision.py --base_url http://localhost:30000/v1 --model doctor_agent \\
    --case_dir /path/to/showcase_cases --case_id <case_id>

  # 使用单张本地图片
  python test_doctor_vision.py --base_url http://localhost:30000/v1 --model doctor_agent \\
    --image_path /path/to/test.jpg

  # 不传图像，仅测 API 连通与文本回复（会提示无图）
  python test_doctor_vision.py --base_url http://localhost:30000/v1 --model doctor_agent
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys

# 保证能 import rllm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from rllm.agents.system_prompts import DOCTOR_SYSTEM_PROMPT
from rllm.tools.multi_tool import MultiTool
from rllm.tools.tool_base import ToolOutput


def _image_to_data_url(path: str, mime: str = "image/jpeg") -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def get_tools_prompt() -> str:
    """与 MedicalAgent 一致的 tool schema 文本，用于 system message。"""
    tools = MultiTool(tools=["cxr", "finish"])
    return "\n# Tools\n```json\n" + json.dumps(tools.json, indent=2, ensure_ascii=False) + "\n```"


def build_messages_with_real_cxr(
    case_dir: str,
    case_id: str,
    patient_question: str,
) -> tuple[list[dict], str]:
    """
    从真实 case 调用 CXR tool 得到带图输出，并构建发给 API 的 messages。
    返回 (messages, tool_output_str_for_log)。
    """
    from rllm.tools.med_tool import CXRTool
    from rllm.agents.med_agent import _cxr_tool_output_to_multimodal

    case_path = os.path.join(case_dir, f"{case_id}.json")
    if not os.path.isfile(case_path):
        raise FileNotFoundError(f"Case not found: {case_path}")
    with open(case_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    ehr = obj.get("ehr") or obj
    if not isinstance(ehr, dict) or not (ehr.get("CXR")):
        raise ValueError(f"Case {case_id} has no CXR section or invalid ehr.")

    tool = CXRTool(case_dir=case_dir)
    out = tool.forward(case_id=case_id, context={"ehr": ehr}, study_index=-1, max_images=2)
    if out.error:
        raise RuntimeError(f"CXR tool error: {out.error}")

    tool_output_str = json.dumps(out.output, ensure_ascii=False)
    tool_summary, user_content_parts = _cxr_tool_output_to_multimodal(tool_output_str)
    if tool_summary is None or user_content_parts is None:
        raise RuntimeError("CXR output had no images; cannot build multimodal message.")

    system_content = DOCTOR_SYSTEM_PROMPT + get_tools_prompt()
    tool_call_id = "call_cxr_test"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": patient_question},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": "cxr", "arguments": json.dumps({"study_index": -1})},
                }
            ],
        },
        {"role": "tool", "content": tool_summary, "tool_call_id": tool_call_id},
        {"role": "user", "content": user_content_parts},
    ]
    return messages, tool_output_str[:200] + "..."


def build_messages_with_image_path(
    image_path: str,
    patient_question: str,
    view_name: str = "PA",
) -> list[dict]:
    """使用单张本地图片构建 messages（无真实 CXR 报告，仅测看图反应）。"""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    url = _image_to_data_url(image_path)
    summary = f"CXR image returned: 1 image — view: {view_name}. (Test image for doctor vision.)"
    user_content = [
        {"type": "text", "text": summary},
        {"type": "image_url", "image_url": {"url": url}},
    ]

    system_content = DOCTOR_SYSTEM_PROMPT + get_tools_prompt()
    tool_call_id = "call_cxr_test"
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": patient_question},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": "cxr", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": summary, "tool_call_id": tool_call_id},
        {"role": "user", "content": user_content},
    ]
    return messages


def build_messages_no_image(patient_question: str) -> list[dict]:
    """无图：仅 tool 文本摘要，用于检查 API 是否正常。"""
    summary = "CXR returned 0 images (no image in this test). Please respond with your conclusion."
    system_content = DOCTOR_SYSTEM_PROMPT + get_tools_prompt()
    tool_call_id = "call_cxr_test"
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": patient_question},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": tool_call_id, "type": "function", "function": {"name": "cxr", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "content": summary, "tool_call_id": tool_call_id},
        {"role": "user", "content": summary},
    ]
    return messages


async def call_doctor(base_url: str, model: str, api_key: str, messages: list[dict], max_tokens: int = 2048):
    import openai
    client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key or "None")
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3,
    )
    msg = resp.choices[0].message
    content = (msg.content or "").strip()
    tool_calls = getattr(msg, "tool_calls", None) or []
    return content, tool_calls, resp.usage


def main():
    parser = argparse.ArgumentParser(description="Test doctor (vision) reaction after receiving CXR images.")
    parser.add_argument("--base_url", default="http://localhost:30000/v1", help="OpenAI-compatible API base URL.")
    parser.add_argument("--model", default="doctor_agent", help="Model name (e.g. served-model-name).")
    parser.add_argument("--api_key", default="None")
    parser.add_argument("--case_dir", default=None, help="Directory of case JSONs; use with --case_id for real CXR.")
    parser.add_argument("--case_id", default=None, help="Case ID (filename without .json) for real CXR.")
    parser.add_argument("--image_path", default=None, help="Single image path (e.g. test CXR jpg).")
    parser.add_argument("--patient_question", default=None, help="Patient intro; default built from case or fixed text.")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--no_image", action="store_true", help="Do not send any image; only text (for API check).")
    args = parser.parse_args()

    patient_question = args.patient_question or (
        "Briefly describe this CXR and write a simple report."
    )

    if args.no_image:
        messages = build_messages_no_image(patient_question)
        print("[test_doctor_vision] Mode: no image (API check only).")
    elif args.case_dir and args.case_id:
        messages, _ = build_messages_with_real_cxr(args.case_dir, args.case_id, patient_question)
        print(f"[test_doctor_vision] Mode: real CXR from case {args.case_id}.")
    elif args.image_path:
        messages = build_messages_with_image_path(args.image_path, patient_question)
        print(f"[test_doctor_vision] Mode: single image {args.image_path}.")
    else:
        print("No image source: use --case_dir + --case_id, or --image_path, or --no_image for text-only test.")
        messages = build_messages_no_image(patient_question)
        print("[test_doctor_vision] Fallback: no image.")

    print("\n--- Patient (user) ---\n" + patient_question)
    print("\n--- Doctor response (after seeing CXR / tool result) ---\n")

    content, tool_calls, usage = asyncio.run(
        call_doctor(args.base_url, args.model, args.api_key, messages, args.max_tokens)
    )
    print(content)
    if tool_calls:
        print("\n[Tool calls]", json.dumps([t.get("function") for t in tool_calls], ensure_ascii=False, indent=2))
    if usage:
        print(f"\n[Usage] prompt_tokens={getattr(usage, 'prompt_tokens', None)} completion_tokens={getattr(usage, 'completion_tokens', None)}")


if __name__ == "__main__":
    main()


"""
python examples/MedGym/test_doctor_vision.py \
  --base_url http://localhost:30000/v1 \
  --model doctor_agent \
  --case_dir /data/xuxiang/mimic-iv/virtual_env/data/showcase_cases \
  --case_id osce_23053402
"""