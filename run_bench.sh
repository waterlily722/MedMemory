#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# MedMemory — 一键跑 benchmark（无 CXR 模式，5 个 case）
# 使用前确保三个 vLLM 服务已启动（见 README）
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

cd /oral_llm/xiweidai/med_env/code/rllm
export PYTHONPATH="/oral_llm/xiweidai/med_env/code/rllm"
export RLLM_PATIENT_BASE_URL="${RLLM_PATIENT_BASE_URL:-http://127.0.0.1:30001/v1}"
export RLLM_PATIENT_MODEL="${RLLM_PATIENT_MODEL:-patient_agent}"
export RLLM_PATIENT_API_KEY="${RLLM_PATIENT_API_KEY:-None}"

NO_CXR="${1:-true}"  # 默认无 CXR

CXR_FLAG=""
MODE="with_img"
SUFFIX="ed_hosp"
if [ "$NO_CXR" = "true" ]; then
    CXR_FLAG="--no_cxr"
    MODE="without_img"
    SUFFIX="hosp_only"
fi

echo "============================================"
echo "  MedMemory Benchmark"
echo "  Mode: $MODE/$SUFFIX"
echo "  Case: /oral_llm/xiweidai/med_env/bench"
echo "============================================"

python examples/MedGym/run_med_with_tool.py \
    --model doctor_agent \
    --tokenizer_path /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
    --base_url http://127.0.0.1:30000/v1 \
    --case_dir /oral_llm/xiweidai/med_env/bench \
    --max_cases 5 \
    $CXR_FLAG \
    --parser_name qwen \
    --judge_model judge_agent \
    --judge_base_url http://127.0.0.1:30002/v1 \
    --enable_memory
