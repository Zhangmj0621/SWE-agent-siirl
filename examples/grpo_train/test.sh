#!/usr/bin/env bash
# ===================================================================================
# ===                       USER CONFIGURATION SECTION                            ===
# ===================================================================================

export SIIRL_DIR="${SIIRL_DIR:-{siirl-agentic-dir}}"
export PYTHONPATH="$SIIRL_DIR:$PYTHONPATH"

#  --- Experiment and Model Definition ---
export DATASET=deepscaler
export ALG=grpo
export MODEL_NAME=qwen3_30b_a3b

# --- Path Definitions ---
export TRAIN_DATA_PATH=/inspire/hdd/project/qianghuaxuexi/public/datasets/deepscaler/train.parquet
export TEST_DATA_PATH=/inspire/hdd/project/qianghuaxuexi/public/datasets/deepscaler/test.parquet
export MODEL_PATH=/inspire/hdd/project/qianghuaxuexi/public/models/Qwen3-30B-A3B

export WANDB_BASE_URL=https://wandb1.sii.edu.cn/
export WANDB_API_KEY=local-6a4cc4c8b917355ce21530f9c9be52014cc55ee2

# Base output paths
export BASE_CKPT_PATH=ckpts
export BASE_TENSORBOARD_PATH=tensorboard

# --- Key Training Hyperparameters ---
export TRAIN_BATCH_SIZE=512
export PPO_MINI_BATCH_SIZE=256
export PPO_MICRO_BATCH_SIZE_PER_GPU=8
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=4096
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.7

export ROLLOUT_TP=4                    # Tensor parallelism for rollout
export ROLLOUT_N=8                     # Number of samples per prompt
export SAVE_FREQ=3000
export TEST_FREQ=10
export TOTAL_EPOCHS=30
export MAX_CKPT_KEEP=5

# --- GPU Resource Allocation (Separated Mode) ---
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
export NNODES=${PET_NNODES:-1}
export NODE_RANK=${PET_NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}
export ACTOR_GPUS=8                   
export ROLLOUT_GPUS=8               

# --- Actor Parallelism Configuration ---
# TP (Tensor Parallel): Model sharding across GPUs within a group
# PP (Pipeline Parallel): Model layer sharding across pipeline stages
# CP (Context Parallel): Sequence parallelism for long context
# DP (Data Parallel): Automatically computed as ACTOR_GPUS / (TP * PP * CP)
export ACTOR_TP=4                      # Actor tensor parallelism (default: 1)
export ACTOR_PP=1                      # Actor pipeline parallelism (default: 1)
export ACTOR_CP=1                      # Actor context parallelism (default: 1)
export ACTOR_EP=8

# --- Output Paths and Experiment Naming ---
timestamp=$(date +"%Y%m%d_%H%M%S")
export CKPT_PATH=${BASE_CKPT_PATH}/${MODEL_NAME}_${ALG}_${DATASET}_${NNODES}node_${ACTOR_GPUS}actor_${ROLLOUT_GPUS}rollout
export PROJECT_NAME=zp_${DATASET}_${ALG}_router_replay
# export EXPERIMENT_NAME=${MODEL_NAME}_wo_router_replay_baseline
export EXPERIMENT_NAME=${MODEL_NAME}_r2
export TENSORBOARD_DIR=${BASE_TENSORBOARD_PATH}/${MODEL_NAME}_${ALG}_${DATASET}_tensorboard_$timestamp

# --- Define the Training Command and its Arguments ---
TRAINING_CMD=(
    python3 -m siirl.async_train
    # === Algorithm Settings ===
    algorithm.adv_estimator=$ALG
    algorithm.kl_ctrl.kl_coef=0.001
    # === Data Settings ===
    data.train_files=$TRAIN_DATA_PATH
    data.val_files=$TEST_DATA_PATH
    data.train_batch_size=$TRAIN_BATCH_SIZE
    data.max_prompt_length=$MAX_PROMPT_LENGTH
    data.max_response_length=$MAX_RESPONSE_LENGTH
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
    # === Actor/Ref Model Settings ===
    actor_ref.model.path=$MODEL_PATH
    actor_ref.model.trust_remote_code=True
    # === Actor Training Settings ===
    actor_ref.actor.optim.lr=1e-6
    actor_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE
    actor_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU
    actor_ref.actor.use_kl_loss=True
    actor_ref.actor.clip_ratio=0.2
    actor_ref.actor.kl_loss_coef=0.01
    actor_ref.actor.kl_loss_type=low_var_kl
    actor_ref.actor.enable_routing_replay=True
    actor_ref.actor.megatron.param_offload=True
    actor_ref.actor.megatron.optimizer_offload=True
    # Actor parallelism is configured via trainer.* below
    # === Reference Model Settings ===
    actor_ref.ref.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU
    actor_ref.ref.megatron.param_offload=True
    actor_ref.actor.megatron.use_mbridge=True
    # === Rollout Settings (SGLang) ===
    rollout.name=sglang
    rollout.tensor_model_parallel_size=$ROLLOUT_TP
    rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION
    rollout.n=$ROLLOUT_N
    rollout.trust_remote_code=True
    # === Trainer Settings ===
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE
    trainer.nnodes=$NNODES
    trainer.actor_gpus=$ACTOR_GPUS
    trainer.rollout_gpus=$ROLLOUT_GPUS
    trainer.colocate=False
    trainer.total_epochs=$TOTAL_EPOCHS
    trainer.save_freq=$SAVE_FREQ
    trainer.test_freq=$TEST_FREQ
    trainer.max_actor_ckpt_to_keep=$MAX_CKPT_KEEP
    trainer.default_local_dir=$CKPT_PATH
    trainer.project_name=$PROJECT_NAME
    trainer.experiment_name=$EXPERIMENT_NAME
    # trainer.logger="['console','tensorboard']"
    trainer.logger="['wandb']"
    # trainer.logger="['console']"
    trainer.resume_mode=auto
    trainer.val_before_train=True
    # === Parallel Config ===
    trainer.tensor_model_parallel_size=$ACTOR_TP
    trainer.expert_model_parallel_size=$ACTOR_EP
    trainer.pipeline_model_parallel_size=$ACTOR_PP
    trainer.context_parallel_size=$ACTOR_CP
)