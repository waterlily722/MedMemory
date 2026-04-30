# 1. 激活环境
conda activate scientist

# 2. 安装 rllm 依赖（首次只需一次）
cd /oral_llm/xiweidai/med_env/code/rllm
pip install -e .                     # 安装 rllm 包（editable mode）
pip install -r examples/MedGym/requirements.txt  # MedGym 额外依赖

# 3. 设置 PYTHONPATH
export PYTHONPATH="/oral_llm/xiweidai/med_env/code/rllm"

# 4. 启动三个 vLLM 服务（按 README 三个终端）
# 终端 A — Doctor (port 30000)
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
  --model /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --served-model-name doctor_agent \
  --host 0.0.0.0 --port 30000 --tensor-parallel-size 4

# 终端 B — Patient (port 30001)
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
  --model /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --served-model-name patient_agent \
  --host 0.0.0.0 --port 30001 --tensor-parallel-size 4

# 终端 C — Judge (port 30002)
CUDA_VISIBLE_DEVICES=4,5 python -m vllm.entrypoints.openai.api_server \
  --model /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --served-model-name judge_agent \
  --host 0.0.0.0 --port 30002

# 5. 设置 Patient 服务环境变量
export RLLM_PATIENT_BASE_URL="http://127.0.0.1:30001/v1"
export RLLM_PATIENT_MODEL="patient_agent"
export RLLM_PATIENT_API_KEY="None"

# 6. 跑测试（无 CXR 模式）
python examples/MedGym/run_med_with_tool.py \
  --model doctor_agent \
  --tokenizer_path /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --base_url http://127.0.0.1:30000/v1 \
  --case_dir /oral_llm/xiweidai/med_env/bench \
  --max_cases 5 \
  --no_cxr \
  --parser_name qwen \
  --judge_model judge_agent \
  --judge_base_url http://127.0.0.1:30002/v1 \
  --enable_memory \
  --memory_root memory_data