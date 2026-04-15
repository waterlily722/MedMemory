#!/usr/bin/env bash
# Doctor=30000, Patient=30001, Judge=30002（仅当 USE_DEDICATED_JUDGE=1 时启动）
#
# 8 卡 GPU 推荐用法：
#   1. 先启动检索服务（另开终端）：bash examples/search/retrieval/launch_server.sh  # GPU 0
#   2. 再运行本脚本：bash examples/MedGym/run_med_with_tool.sh
#      Doctor 用 2 卡 tp（GPU 1,2），Patient=3，Judge=4
#
set -euo pipefail

DOCTOR_PORT=30000
PATIENT_PORT=30001
JUDGE_PORT=30002
RETRIEVAL_PORT=8003

DOCTOR_MODEL_PATH="/data/xuxiang/mimic-iv/models/Qwen3-VL-8B-Instruct"
PATIENT_MODEL_PATH="/data/xuxiang/mimic-iv/models/Qwen3-8B"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/data/xuxiang/mimic-iv/models/Qwen3-8B}"
DATA_DIR="/data/xuxiang/mimic-iv/osce_data"

DOCTOR_SERVED_NAME="doctor_agent"
PATIENT_SERVED_NAME="patient_agent"
JUDGE_SERVED_NAME="judge_agent"

export PYTHONPATH="/data/xuxiang/mimic-iv/virtual_env/code/rllm"
export TOKENIZERS_PARALLELISM="true"

export RLLM_PATIENT_BASE_URL="http://127.0.0.1:${PATIENT_PORT}/v1"
export RLLM_PATIENT_MODEL="${PATIENT_SERVED_NAME}"
export RLLM_PATIENT_API_KEY="None"
export RLLM_CASE_DIR="/data/xuxiang/mimic-iv/virtual_env/data"
export RETRIEVAL_SERVER_URL="http://127.0.0.1:${RETRIEVAL_PORT}"


RUN_SCRIPT="examples/MedGym/run_med_with_tool.py"

# 是否单独起一个 API 端口给 judge 模型（1=在 JUDGE_PORT 上起 vLLM，0=用 patient 端口当 judge）
# USE_DEDICATED_JUDGE="${USE_DEDICATED_JUDGE:-0}"
USE_DEDICATED_JUDGE=1

# GPU 分配（8 卡机器）：
#   GPU 0：检索服务 launch_server.sh 占用（~800 MiB），需先单独启动
#   Doctor：2 卡 tp（GPU 1,2）或 4 卡 tp（GPU 1,2,3,4），VL 模型显存大
#   Patient/Judge：自动排在 Doctor 之后，避免 GPU 重叠导致 "Free memory... less than desired"
DOCTOR_GPUS="${DOCTOR_GPUS:-1,2}"
DOCTOR_TP_SIZE="${DOCTOR_TP_SIZE:-2}"
# 从 DOCTOR_GPUS 取最大编号，Patient/Judge 用后续卡（如 Doctor 用 1,2,3,4 则 Patient=5, Judge=6）
DOCTOR_LAST_GPU=$(echo "${DOCTOR_GPUS}" | tr ',' '\n' | sort -n | tail -1)
PATIENT_GPU="${PATIENT_GPU:-$((DOCTOR_LAST_GPU + 1))}"
JUDGE_GPU="${JUDGE_GPU:-$((DOCTOR_LAST_GPU + 2))}"

echo "[INFO] GPU layout: retrieval=0, Doctor=${DOCTOR_GPUS} (tp=${DOCTOR_TP_SIZE}), Patient=${PATIENT_GPU}, Judge=${JUDGE_GPU}"

cleanup() {
  echo "[INFO] Cleaning up vLLM servers..."
  if [[ -n "${DOCTOR_PID:-}" ]]; then kill "${DOCTOR_PID}" 2>/dev/null || true; fi
  if [[ -n "${PATIENT_PID:-}" ]]; then kill "${PATIENT_PID}" 2>/dev/null || true; fi
  if [[ -n "${JUDGE_PID:-}" ]]; then kill "${JUDGE_PID}" 2>/dev/null || true; fi
}
trap cleanup EXIT

echo "[INFO] Starting doctor vLLM on :${DOCTOR_PORT} (GPU ${DOCTOR_GPUS}, tp=${DOCTOR_TP_SIZE}) ..."
CUDA_VISIBLE_DEVICES=${DOCTOR_GPUS} vllm serve "${DOCTOR_MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${DOCTOR_PORT}" \
  --served-model-name "${DOCTOR_SERVED_NAME}" \
  --tensor-parallel-size "${DOCTOR_TP_SIZE}" \
  --max-model-len 32768 \
  >/tmp/vllm_doctor_${DOCTOR_PORT}.log 2>&1 &
DOCTOR_PID=$!

echo "[INFO] Starting patient vLLM on :${PATIENT_PORT} (GPU ${PATIENT_GPU}) ..."
CUDA_VISIBLE_DEVICES=${PATIENT_GPU} vllm serve "${PATIENT_MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PATIENT_PORT}" \
  --served-model-name "${PATIENT_SERVED_NAME}" \
  --max-model-len 32768 \
  >/tmp/vllm_patient_${PATIENT_PORT}.log 2>&1 &
PATIENT_PID=$!

if [[ "${USE_DEDICATED_JUDGE}" == "1" ]]; then
  echo "[INFO] Starting judge vLLM on :${JUDGE_PORT} (GPU ${JUDGE_GPU}) ..."
  CUDA_VISIBLE_DEVICES=${JUDGE_GPU} python -m vllm.entrypoints.openai.api_server \
    --model "${JUDGE_MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${JUDGE_PORT}" \
    --served-model-name "${JUDGE_SERVED_NAME}" \
    --max-model-len 32768 \
    >/tmp/vllm_judge_${JUDGE_PORT}.log 2>&1 &
  JUDGE_PID=$!
fi

echo "[INFO] Waiting for vLLM servers to be ready..."
WAIT_PORTS="${DOCTOR_PORT} ${PATIENT_PORT}"
if [[ "${USE_DEDICATED_JUDGE}" == "1" ]]; then
  WAIT_PORTS="${WAIT_PORTS} ${JUDGE_PORT}"
fi
for PORT in ${WAIT_PORTS}; do
  for i in {1..120}; do
    if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "[INFO] vLLM on :${PORT} is ready."
      break
    fi
    sleep 1
    if [[ "$i" -eq 120 ]]; then
      echo "[ERROR] vLLM on :${PORT} not ready. Check logs:"
      echo "  /tmp/vllm_doctor_${DOCTOR_PORT}.log"
      echo "  /tmp/vllm_patient_${PATIENT_PORT}.log"
      [[ "${USE_DEDICATED_JUDGE}" == "1" ]] && echo "  /tmp/vllm_judge_${JUDGE_PORT}.log"
      exit 1
    fi
  done
done

# Judge 模式：用 LLM 判断诊断是否与真实一致。
# - USE_DEDICATED_JUDGE=1：在 JUDGE_PORT 单独起 judge 服务，并自动传 --judge_model/--judge_base_url
# - 或手动指定：JUDGE_MODEL=patient_agent ./run_med_with_tool.sh（复用 patient 端口）
# - 不启用则奖励用“诊断包含真实结果”的规则
# 1=有 CXR（with_img/ed_hosp，默认），0=无 CXR（without_img/hosp_only）。用法：WITH_CXR=0 ./run_med_with_tool.sh
WITH_CXR="${WITH_CXR:-1}"
DATA_ARGS=()
if [[ "${WITH_CXR}" == "0" ]]; then
  DATA_ARGS=(--no_cxr)
  echo "[INFO] Using data: no CXR (without_img/hosp_only)"
else
  echo "[INFO] Using data: with CXR (with_img/ed_hosp)"
fi

JUDGE_ARGS=()
if [[ "${USE_DEDICATED_JUDGE}" == "1" ]]; then
  JUDGE_MODEL="${JUDGE_SERVED_NAME}"
  JUDGE_BASE_URL="http://localhost:${JUDGE_PORT}/v1"
  JUDGE_ARGS=(--judge_model "${JUDGE_MODEL}" --judge_base_url "${JUDGE_BASE_URL}")
elif [[ -n "${JUDGE_MODEL:-}" ]]; then
  JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://localhost:${PATIENT_PORT}/v1}"
  JUDGE_ARGS=(--judge_model "${JUDGE_MODEL}" --judge_base_url "${JUDGE_BASE_URL}")
fi

echo "[INFO] Running MedGym run_med_with_tool.py ..."
python "${RUN_SCRIPT}" \
  --model "${DOCTOR_SERVED_NAME}" \
  --tokenizer_path "${DOCTOR_MODEL_PATH}" \
  --base_url "http://localhost:${DOCTOR_PORT}/v1" \
  --api_key "None" \
  --case_dir "${DATA_DIR}/bench" \
  --max_cases 0 \
  --repeat_k 1 \
  --n_parallel_agents 1 \
  --temperature 0.6 \
  --top_p 0.95 \
  --max_prompt_length 8192 \
  --max_response_length 16384 \
  --parser_name qwen \
  "${DATA_ARGS[@]}" \
  "${JUDGE_ARGS[@]}"

echo "[INFO] Done."
