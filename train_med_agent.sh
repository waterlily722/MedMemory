set -x

export CUDA_VISIBLE_DEVICES=1,2
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
cd "${RLLM_DIR}"
export PYTHONPATH="${RLLM_DIR}:${PYTHONPATH:-}"

# MedGym 数据目录。用 bench 时填 bench 根目录（如 DATA_DIR/bench），否则填普通 case 目录
CASE_DIR="/data/xuxiang/mimic-iv/virtual_env/data/showcase_cases"
# 使用 bench 数据时：1=有 CXR（with_img/ed_hosp），0=无 CXR（without_img/hosp_only）
USE_BENCH="${USE_BENCH:-0}"
WITH_CXR="${WITH_CXR:-1}"
MODEL_PATH="/data/xuxiang/mimic-iv/models/Qwen3-VL-8B-Instruct"

python3 -m examples.MedGym.train_med_agent \
    algorithm.adv_estimator=rloo \
    data.train_batch_size=64 \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='medgym-agent' \
    trainer.experiment_name='medgym-ppo' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=2\
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    rllm.agent.max_steps=15 \
    trainer.total_epochs=100 \
    medgym.case_dir="${CASE_DIR}" \
    medgym.max_cases=100 \
    medgym.val_ratio=0.1 \
    medgym.repeat_k=1 \
    medgym.use_bench="${USE_BENCH}" \
    medgym.with_cxr="${WITH_CXR}"
# 使用 Judge 奖励时取消下面三行注释，并在上一行 medgym.repeat_k=1 末尾加 \ ；训练时需能访问 Judge API
#    medgym.judge_model_name=judge_agent \
#    medgym.judge_base_url="http://localhost:30002/v1" \
#    medgym.judge_api_key=""
