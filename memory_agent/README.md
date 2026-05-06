# MedMemory — Medical Memory Agent

医学诊断任务的记忆增强系统。基于 rllm 框架，支持在线记忆检索与离线经验提取。

## 目录结构

```
memory_agent/
├── schemas/           # 数据模型（dataclass + SerializableMixin）
│   ├── case_state.py      # 当前病例状态
│   ├── experience_card.py # 经验记忆卡
│   ├── skill_card.py      # 技能记忆卡
│   ├── knowledge_item.py  # 知识条目
│   ├── memory_query.py    # 检索查询
│   ├── retrieval.py       # 检索结果
│   ├── applicability.py   # 适用性评估
│   ├── guidance.py        # 记忆引导
│   ├── turn_record.py     # 单步记录
│   └── episode.py         # 回合反馈
├── online/            # 在线推理 pipeline
│   ├── case_updater.py           # 病例状态更新
│   ├── query_builder.py          # 检索查询构建
│   ├── retriever.py              # 多源记忆检索（支持 embedding）
│   ├── applicability_controller.py # 记忆适用性判断
│   └── memory_guidance.py        # 引导生成
├── offline/           # 离线经验提取 pipeline
│   ├── episode_distiller.py      # 轨迹蒸馏
│   ├── experience_extractor.py   # 经验提取（LLM）
│   ├── experience_merger.py      # 经验合并
│   ├── memory_writer.py          # 记忆写入
│   └── skill_consolidator.py     # 技能归纳
├── memory_store/      # JSONL 持久化
├── llm/               # LLM / Embedding 客户端
├── utils/             # 工具函数（tokenizer, cosine similarity）
├── tests/             # 单元测试（76 tests）
└── wrapper.py         # MemoryWrappedMedicalAgent 主入口
```

## 环境准备

```bash
# 1. 激活 conda 环境
conda activate vllm_env

# 2. 安装 rllm（首次）
cd /oral_llm/xiweidai/med_env/code/rllm
pip install -e .

# 3. 验证导入
cd /oral_llm/xiweidai/med_env/code/rllm
PYTHONPATH="/oral_llm/xiweidai/med_env/code/rllm/examples/MedGym" \
  python -c "from memory_agent import MemoryWrappedMedicalAgent; print('memory_agent OK')"

# 4. （可选）MedGym 环境依赖 — 跑完整 benchmark 时需要
#    包含 decord / moviepy / opencv / scikit-learn 等，编译较慢
pip install -r /oral_llm/xiweidai/med_env/code/rllm/examples/MedGym/requirements.txt
```

## 运行测试

```bash
# 单元测试（93 个，无需 LLM/GPU）
cd /oral_llm/xiweidai/med_env/code/rllm/examples/MedGym
PYTHONPATH="/oral_llm/xiweidai/med_env/code/rllm" python -m unittest discover -s tests -v

# memory_agent 内部测试
PYTHONPATH="/oral_llm/xiweidai/med_env/code/rllm" python -c "
import sys; sys.path.insert(0, 'memory_agent/tests')
for m in ['test_schemas','test_scoring','test_parser','test_store','test_integration','test_e2e_pipeline']:
    __import__(m); print(f'{m}: OK')
print('All imports OK')
"
```

## 部署测试（需 GPU）

### 终端 A — Doctor (port 30000)

```bash
conda activate vllm_env
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m vllm.entrypoints.openai.api_server \
  --model /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --served-model-name doctor_agent \
  --host 0.0.0.0 --port 30000 \
  --dtype bfloat16 --max-model-len 32768 \
  --tensor-parallel-size 4 --gpu-memory-utilization 0.9 \
  --trust-remote-code
```

cd ~

export CUDA_VISIBLE_DEVICES=0,1,2,3

vllm serve /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --served-model-name doctor_agent \
  --host 0.0.0.0 \
  --port 30000 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code

### 终端 B — Patient (port 30001)

```bash
conda activate vllm_env
export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m vllm.entrypoints.openai.api_server \
  --model /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --served-model-name patient_agent \
  --host 0.0.0.0 --port 30001 \
  --dtype bfloat16 \
  --tensor-parallel-size 4 --gpu-memory-utilization 0.8
```

### 终端 C — Judge (port 30002)

```bash
conda activate vllm_env
export CUDA_VISIBLE_DEVICES=1
python -m vllm.entrypoints.openai.api_server \
  --model /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --served-model-name judge_agent \
  --host 0.0.0.0 --port 30002 \
  --dtype bfloat16 --max-model-len 32768 \
  --gpu-memory-utilization 0.9 --trust-remote-code
```

### 验证服务就绪

```bash
curl -s http://127.0.0.1:30000/v1/models | head
curl -s http://127.0.0.1:30001/v1/models | head
curl -s http://127.0.0.1:30002/v1/models | head
```

### 运行评测

```bash
cd /oral_llm/xiweidai/med_env/code/rllm
export PYTHONPATH="/oral_llm/xiweidai/med_env/code/rllm"
export RLLM_PATIENT_BASE_URL="http://127.0.0.1:30001/v1"
export RLLM_PATIENT_MODEL="patient_agent"
export RLLM_PATIENT_API_KEY="None"

# 无 CXR 模式（默认 5 个 case）
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

# 有 CXR 模式（去掉 --no_cxr 即可）
python examples/MedGym/run_med_with_tool.py \
  --model doctor_agent \
  --tokenizer_path /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --base_url http://127.0.0.1:30000/v1 \
  --case_dir /oral_llm/xiweidai/med_env/bench \
  --max_cases 5 \
  --parser_name qwen \
  --judge_model judge_agent \
  --judge_base_url http://127.0.0.1:30002/v1 \
  --enable_memory \
  --memory_root memory_data
```

## 记忆系统参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable_memory` | off | 启用记忆系统 |
| `--memory_root` | `memory_data` | 记忆存储根目录 |
| `--query_builder_mode` | `rule` | 检索查询构建模式 (`rule` / `llm`) |
| `--applicability_mode` | `rule` | 适用性判断模式 (`rule` / `llm` / `hybrid`) |
| `--experience_extraction_mode` | `llm` | 经验提取模式 (`rule` / `llm`) |
| `--experience_merge_mode` | `rule` | 经验合并模式 (`rule` / `llm`) |
| `--log_memory_trace` | off | 记录每步记忆 trace 到文件 |
| `--disable_experience_memory` | off | 禁用经验记忆 |
| `--disable_skill_memory` | off | 禁用技能记忆 |
| `--disable_knowledge_memory` | off | 禁用知识记忆 |
| `--memory_llm_model` | — | 记忆系统专用 LLM 模型名 |
| `--memory_llm_base_url` | — | 记忆系统专用 LLM 地址 (默认同主 LLM) |
