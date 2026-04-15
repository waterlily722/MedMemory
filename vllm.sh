# export HF_ENDPOINT=https://hf-mirror.com
# huggingface-cli download ZJU-AI4H/Hulu-Med-14B \
#   --repo-type model \
#   --local-dir /data/xuxiang/mimic-iv/virtual_env/models/Hulu-Med-14B \
#   --local-dir-use-symlinks False \
#   --resume-download

huggingface-cli download ZJU-AI4H/Hulu-Med-7B \
  --repo-type model \
  --local-dir /data/xuxiang/mimic-iv/virtual_env/models/Hulu-Med-7B \
  --local-dir-use-symlinks False \
  --resume-download
  
export CUDA_VISIBLE_DEVICES=1,2
python -m vllm.entrypoints.openai.api_server \
  --model /data/xuxiang/mimic-iv/virtual_env/models/Hulu-Med-7B \
  --served-model-name doctor_agent \
  --host 0.0.0.0 \
  --port 30000 \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768

python -m vllm.entrypoints.openai.api_server \
  --model /data/xuxiang/mimic-iv/virtual_env/models/Hulu-Med-8B \
  --served-model-name doctor_agent \
  --host 0.0.0.0 \
  --port 30000 \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code

# Qwen3-VL: 视觉模型显存占用大，必须加 --max-model-len；2 卡建议 16384，4 卡可用 32768
# 若报 WorkerProc initialization failed，先加 --max-model-len 或设 VLLM_USE_V1=0 用旧引擎看具体错误
export CUDA_VISIBLE_DEVICES=2,3
python -m vllm.entrypoints.openai.api_server \
  --model /data/xuxiang/mimic-iv/models/Qwen3-VL-8B-Instruct \
  --served-model-name doctor_agent \
  --host 0.0.0.0 \
  --port 30000 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code

python -m vllm.entrypoints.openai.api_server \
  --model /data/xuxiang/mimic-iv/models/Qwen3-8B \
  --served-model-name doctor_agent \
  --host 0.0.0.0 \
  --port 30000 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code

export CUDA_VISIBLE_DEVICES=4,5,6,7 
python -m vllm.entrypoints.openai.api_server \
  --model /data/xuxiang/mimic-iv/models/Qwen3-8B \
  --served-model-name patient_agent \
  --host 0.0.0.0 \
  --port 30001 \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.8

export CUDA_VISIBLE_DEVICES=1
python -m vllm.entrypoints.openai.api_server \
  --model /data/xuxiang/mimic-iv/models/Qwen3-8B \
  --served-model-name judge_agent \
  --host 0.0.0.0 \
  --port 30002 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code
