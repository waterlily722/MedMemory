# MedGym 测试说明（OSCE bench + 手动 vLLM）

本说明对应 **先手动拉起 Qwen3-8B 的 OpenAI 兼容 API**，再跑 `run_med_with_tool.py`。**bench 数据根目录**为：

`/oral_llm/xiweidai/med_env/bench`

- **有 CXR**：`bench/with_img/ed_hosp/<case_id>/`
- **无 CXR**：`bench/without_img/hosp_only/<case_id>/`

---

## 环境

进入 rllm 仓库根目录并激活虚拟环境（启动 vLLM 与运行下方测试脚本前均可先执行）：

```bash
cd /oral_llm/xiweidai/med_env/code/rllm
source .venv/bin/activate
```

---

## 1. 启动 vLLM（三个终端，或 tmux/screen 分屏）

以下命令摘自 `vllm.sh`（约 49–79 行），模型路径为 `/oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct`。请按机器实际 GPU 情况修改 `CUDA_VISIBLE_DEVICES`；三个服务必须同时存活。建议已按上文「环境」进入同一 `venv`。

### 终端 A：Doctor（端口 30000，TP=4）

`vllm.sh` 中该段未写 `CUDA_VISIBLE_DEVICES`，请自行在运行前 `export`，避免与其它实例抢卡，例如：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3   # 按需修改
python -m vllm.entrypoints.openai.api_server \
  --model /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --served-model-name doctor_agent \
  --host 0.0.0.0 \
  --port 30000 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code
```

### 终端 B：Patient（端口 30001，TP=4）

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7   # 按需修改，勿与 A/C 重叠
python -m vllm.entrypoints.openai.api_server \
  --model /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --served-model-name patient_agent \
  --host 0.0.0.0 \
  --port 30001 \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.8
```

（与 `vllm.sh` 一致，未加 `--trust-remote-code`。若 Patient 启动失败，可尝试补上 `--trust-remote-code` 或 `--max-model-len 32768`。）

### 终端 C：Judge（端口 30002，单卡）

```bash
export CUDA_VISIBLE_DEVICES=1       # 按需修改，勿与 A/B 冲突
python -m vllm.entrypoints.openai.api_server \
  --model /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --served-model-name judge_agent \
  --host 0.0.0.0 \
  --port 30002 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code
```

**注意**：示例中的 GPU 编号（如 Doctor 用 0–3、Patient 用 4–7、Judge 用 1）在物理上会冲突，实际部署时请重新划分，保证任意时刻每张卡只被一个 vLLM 进程使用。

---

## 2. 检查服务是否就绪

```bash
curl -s http://127.0.0.1:30000/v1/models | head
curl -s http://127.0.0.1:30001/v1/models | head
curl -s http://127.0.0.1:30002/v1/models | head
```

---

## 3. 运行测试（`run_med_with_tool.py`）

先完成「环境」中的 `cd` 与 `source .venv/bin/activate`，再执行。将 `case_dir` 指到本机的 bench 根目录（见上）。

**提示：** Agent / 环境实际暴露的 **工具列表**（`agent_args` 与 `env_args` 中的 `tools`）、`system_prompt`、`max_steps`、有 CXR 时的 `context_injected_tool_names` 等，均在 [`run_med_with_tool.py`](./run_med_with_tool.py) **约第 75–100 行** 按 `--no_cxr` 分支写死，而非命令行参数。简要对应关系：

| 模式 | `tools`（与代码中顺序一致） |
|------|------------------------------|
| 无 CXR（`--no_cxr`） | `ask_patient`, `diagnosis`, `retrieve`, `request_exam` |
| 有 CXR（默认） | `ask_patient`, `diagnosis`, `retrieve`, `cxr`, `request_exam`, `cxr_grounding` |

修改可用工具时，请直接编辑该文件对应分支。

```bash
cd /oral_llm/xiweidai/med_env/code/rllm
source .venv/bin/activate
export PYTHONPATH="/oral_llm/xiweidai/med_env/code/rllm"

# 有 CXR（默认，对应 with_img/ed_hosp）
python examples/MedGym/run_med_with_tool.py \
  --model doctor_agent \
  --tokenizer_path /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --base_url http://127.0.0.1:30000/v1 \
  --api_key None \
  --case_dir /oral_llm/xiweidai/med_env/bench \
  --parser_name qwen \
  --judge_model judge_agent \
  --judge_base_url http://127.0.0.1:30002/v1

# 无 CXR（对应 without_img/hosp_only）
python examples/MedGym/run_med_with_tool.py \
  --model doctor_agent \
  --tokenizer_path /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --base_url http://127.0.0.1:30000/v1 \
  --api_key None \
  --case_dir /oral_llm/xiweidai/med_env/bench \
  --no_cxr \
  --parser_name qwen \
  --judge_model judge_agent \
  --judge_base_url http://127.0.0.1:30002/v1
```

若脚本或环境通过 `RLLM_PATIENT_BASE_URL` 连接 Patient 服务，请设为：

```bash
export RLLM_PATIENT_BASE_URL="http://127.0.0.1:30001/v1"
export RLLM_PATIENT_MODEL="patient_agent"
export RLLM_PATIENT_API_KEY="None"
```

（与 `run_med_with_tool.sh` 中一致。）

若使用 `retrieve` 等工具，需另起检索服务，并设置 `RETRIEVAL_SERVER_URL`（默认脚本里常为 `http://127.0.0.1:8003`，与 `run_med_with_tool.py` 默认 `8000` 不同，请按实际端口统一）。

---

## 4. 可选：一键脚本代替手动 vLLM

若希望由脚本自动起 Doctor/Patient/Judge 的 vLLM，可直接使用 `run_med_with_tool.sh`（会改 Doctor 为 VL 模型等，与本文「纯 Qwen3-8B 三端口」不完全相同）。使用前请阅读该脚本内的 `DATA_DIR`、`DOCTOR_MODEL_PATH` 等变量。
