# /home/xuxiang/virtual_env/code/rllm/examples/MedGym/test_doctor_dialog.py
import argparse
import asyncio
import os

from transformers import AutoTokenizer

# from rllm.agents import ToolAgent
from rllm.agents.system_prompts import DOCTOR_SYSTEM_PROMPT, DOCTOR_SYSTEM_PROMPT_wo_IMG
from memory_agent.wrapper import MemoryWrappedMedicalAgent

from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.medgym.medgym_env import MedicalDialogueEnv
# from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.utils.diagnose_acc import evaluate_doctor_results
# from prepare_med_data import prepare_med_data
from rllm.rewards.reward_fn import search_reward_fn
from rllm.rewards.med_diagnosis_reward import med_diagnosis_reward  # 结果+过程奖励：诊断包含 + ask_patient 置信度

from prepare_med_data_bench import prepare_med_data, SUBDIR_WITH_CXR, SUBDIR_NO_CXR


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True, help="Doctor served model name (e.g., --served-model-name doctor_agent).")
    parser.add_argument("--tokenizer_path", required=True, help="Local tokenizer/model path for AutoTokenizer.")
    parser.add_argument("--base_url", default="http://localhost:30000/v1")
    parser.add_argument("--api_key", default="None")
    parser.add_argument("--case_dir", required=True, help="Bench 根目录，例如 osce_data/bench")
    parser.add_argument("--max_cases", type=int, default=10)
    parser.add_argument("--repeat_k", type=int, default=1)
    parser.add_argument("--no_cxr", action="store_true", help="使用无 CXR 数据（bench/without_img/hosp_only）；默认使用有 CXR（bench/with_img/ed_hosp）")
    parser.add_argument("--n_parallel_agents", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--max_response_length", type=int, default=8192)
    parser.add_argument("--parser_name", default="qwen")
    # Judge 模式：用 Qwen3-8B 等模型判断诊断是否与真实一致（不填则用“包含”规则）
    parser.add_argument("--judge_model", default="", help="Judge model name (e.g. Qwen/Qwen3-8B). If set, reward uses LLM judge instead of containment.")
    parser.add_argument("--judge_base_url", default="", help="Judge API base URL (default: same as --base_url).")
    parser.add_argument("--judge_api_key", default="", help="Judge API key (default: same as --api_key).")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    if "RETRIEVAL_SERVER_URL" not in os.environ:
        os.environ["RETRIEVAL_SERVER_URL"] = "http://127.0.0.1:8000"

    subdir = SUBDIR_NO_CXR if args.no_cxr else SUBDIR_WITH_CXR
    tasks, cases, _ = prepare_med_data(
        args.case_dir,
        max_cases=args.max_cases,
        repeat_k=args.repeat_k,
        subdir=subdir,
    )

    # Judge 模式：把 judge 配置注入到每个 task，reward 里会读 task_info["judge_model_name"] 等
    if args.judge_model:
        judge_base = (args.judge_base_url or args.base_url).strip()
        judge_key = args.judge_api_key if args.judge_api_key else args.api_key
        for t in tasks:
            t["judge_model_name"] = args.judge_model
            t["judge_base_url"] = judge_base
            t["judge_api_key"] = judge_key

    # tasks, cases, _ = prepare_med_data(
    #     case_dir=args.case_dir,
    #     max_cases=args.max_cases,
    #     repeat_k=args.repeat_k,
    # )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    if args.no_cxr:
        agent_args = {
            "tools": ["ask_patient", "diagnosis", "retrieve", "request_exam"],
            "parser_name": args.parser_name,
            "system_prompt": DOCTOR_SYSTEM_PROMPT_wo_IMG,
        }

        env_args = {
            "tools": ["ask_patient", "diagnosis", "retrieve", "request_exam"],
            "reward_fn": med_diagnosis_reward,
            "max_steps": 15,
        }

    else:
        agent_args = {
            "tools": ["ask_patient", "diagnosis", "retrieve", "cxr", "request_exam", "cxr_grounding"],
            "parser_name": args.parser_name,
            "system_prompt": DOCTOR_SYSTEM_PROMPT,
        }

        env_args = {
            "tools": ["ask_patient", "diagnosis", "retrieve", "cxr", "request_exam", "cxr_grounding"],
            "reward_fn": med_diagnosis_reward,
            "max_steps": 15,
            "context_injected_tool_names": ["cxr", "request_exam", "cxr_grounding"],
        }

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "model": args.model,
    }

    engine = AgentExecutionEngine(
        agent_class=MemoryWrappedMedicalAgent,
        agent_args=agent_args,
        env_class=MedicalDialogueEnv,
        env_args=env_args,
        engine_name="openai",
        rollout_engine_args={"base_url": args.base_url, "api_key": args.api_key},
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        max_response_length=args.max_response_length,
        max_prompt_length=args.max_prompt_length,
        n_parallel_agents=args.n_parallel_agents,
    )

    results = asyncio.run(engine.execute_tasks(tasks))
    evaluate_doctor_results(results, tasks, print_examples=3)


if __name__ == "__main__":
    main()
