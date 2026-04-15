"""
MedGym Doctor Agent 训练脚本（PPO）。
使用 prepare_med_data 从 case_dir 构建任务并注册为数据集，再用 AgentTrainer 训练。
需通过 train_med_agent.sh 或从仓库根目录以模块方式运行，例如:
  cd /path/to/rllm && python3 -m examples.MedGym.train_med_agent medgym.case_dir=/path/to/cases ...
"""
from __future__ import annotations

import os

import hydra
from omegaconf import OmegaConf

from rllm.agents.med_agent import MedicalAgent
from rllm.agents.system_prompts import DOCTOR_SYSTEM_PROMPT
from rllm.data import Dataset, DatasetRegistry
from rllm.environments.medgym.medgym_env import MedicalDialogueEnv
from rllm.trainer.agent_trainer import AgentTrainer

from .prepare_med_data import prepare_med_data
from .prepare_med_data_bench import (
    prepare_med_data as prepare_med_data_bench_fn,
    SUBDIR_WITH_CXR,
    SUBDIR_NO_CXR,
)


def _get_medgym_config(config) -> dict:
    """从 config 中读取 medgym 相关配置。"""
    if hasattr(config, "medgym") and config.medgym is not None:
        m = OmegaConf.to_container(config.medgym, resolve=True) or {}
    else:
        m = {}
    use_bench = m.get("use_bench", False)
    if isinstance(use_bench, str):
        use_bench = use_bench.lower() in ("true", "1", "yes")
    with_cxr = m.get("with_cxr", True)
    if isinstance(with_cxr, str):
        with_cxr = with_cxr.lower() not in ("false", "0", "no")

    return {
        "case_dir": m.get("case_dir") or os.getenv("MEDGYM_CASE_DIR", ""),
        "max_cases": int(m.get("max_cases", 100)),
        "repeat_k": int(m.get("repeat_k", 1)),
        "val_ratio": float(m.get("val_ratio", 0.1)),
        "use_bench": use_bench,
        "with_cxr": with_cxr,
        "judge_model_name": (m.get("judge_model_name") or os.getenv("MEDGYM_JUDGE_MODEL", "") or "").strip(),
        "judge_base_url": (m.get("judge_base_url") or os.getenv("MEDGYM_JUDGE_BASE_URL", "") or "").strip(),
        "judge_api_key": (m.get("judge_api_key") or os.getenv("MEDGYM_JUDGE_API_KEY", "") or "").strip(),
    }


def _build_and_register_medgym_data(config) -> tuple[Dataset | None, Dataset | None]:
    """用 prepare_med_data 构建任务并注册为 medgym train/val，返回 Dataset。"""
    medgym_cfg = _get_medgym_config(config)
    case_dir = medgym_cfg["case_dir"]
    if not case_dir or not os.path.isdir(case_dir):
        print(f"[train_med_agent] medgym.case_dir 未设置或目录不存在: {case_dir}")
        return None, None

    if medgym_cfg.get("use_bench"):
        subdir = SUBDIR_NO_CXR if not medgym_cfg.get("with_cxr", True) else SUBDIR_WITH_CXR
        print(f"[train_med_agent] 使用 bench 数据: subdir={subdir} (with_cxr={medgym_cfg.get('with_cxr', True)})")
        tasks, _, _ = prepare_med_data_bench_fn(
            case_dir=case_dir,
            max_cases=medgym_cfg["max_cases"],
            repeat_k=medgym_cfg["repeat_k"],
            subdir=subdir,
        )
    else:
        tasks, _, _ = prepare_med_data(
            case_dir=case_dir,
            max_cases=medgym_cfg["max_cases"],
            repeat_k=medgym_cfg["repeat_k"],
        )
    if not tasks:
        print("[train_med_agent] 未生成任何任务，请检查 case_dir 与 max_cases。")
        return None, None

    val_ratio = medgym_cfg["val_ratio"]
    n_val = max(1, int(len(tasks) * val_ratio))
    n_train = len(tasks) - n_val
    train_tasks = tasks[:n_train]
    val_tasks = tasks[n_train:]

    # Judge 奖励：若配置了 judge_model_name + judge_base_url，注入到每个 task
    judge_model_name = medgym_cfg.get("judge_model_name") or ""
    judge_base_url = medgym_cfg.get("judge_base_url") or ""
    judge_api_key = medgym_cfg.get("judge_api_key") or ""
    if judge_model_name and judge_base_url:
        for t in train_tasks + val_tasks:
            t["judge_model_name"] = judge_model_name
            t["judge_base_url"] = judge_base_url
            if judge_api_key:
                t["judge_api_key"] = judge_api_key
        print(f"[train_med_agent] 已启用 Judge 奖励: model={judge_model_name}, base_url={judge_base_url}")

    # verl 要求 train dataloader 至少 1 个 batch，即 train 样本数 >= data.train_batch_size
    train_batch_size = int(getattr(getattr(config, "data", None), "train_batch_size", 64))
    if len(train_tasks) < train_batch_size:
        n_repeat = (train_batch_size + len(train_tasks) - 1) // len(train_tasks)
        train_tasks = (train_tasks * n_repeat)[:train_batch_size]
        print(f"[train_med_agent] 训练样本数 < train_batch_size({train_batch_size})，已重复至 {len(train_tasks)} 条以满足至少 1 个 batch。")

    DatasetRegistry.register_dataset("medgym", train_tasks, split="train")
    DatasetRegistry.register_dataset("medgym", val_tasks, split="val")

    train_dataset = DatasetRegistry.load_dataset("medgym", "train")
    val_dataset = DatasetRegistry.load_dataset("medgym", "val")
    print(f"[train_med_agent] 注册并加载 medgym: train={len(train_tasks)}, val={len(val_tasks)}")
    return train_dataset, val_dataset


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y))

    agent_args = {
        "tools": ["ask_patient", "diagnosis", "retrieve", "cxr"],
        "parser_name": "qwen",
        "system_prompt": DOCTOR_SYSTEM_PROMPT,
    }
    env_args = {
        "tools": ["ask_patient", "diagnosis", "retrieve", "cxr"],
        "reward_fn": "rllm.rewards.med_diagnosis_reward:med_diagnosis_reward",
        "max_steps": 15,
        "ask_tool_name": "ask_patient",
    }

    train_dataset, val_dataset = _build_and_register_medgym_data(config)
    if train_dataset is None:
        raise ValueError("未生成训练数据，请设置 medgym.case_dir 并保证目录下存在 case 的 json 文件。")

    if hasattr(config, "rllm") and config.rllm is not None:
        if getattr(config.rllm.agent, "max_steps", None) is not None:
            env_args["max_steps"] = config.rllm.agent.max_steps

    trainer = AgentTrainer(
        agent_class=MedicalAgent,
        env_class=MedicalDialogueEnv,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        agent_args=agent_args,
        env_args=env_args,
    )
    trainer.train()


if __name__ == "__main__":
    main()
