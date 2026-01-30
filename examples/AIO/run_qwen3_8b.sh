#!/usr/bin/env bash
# ===================================================================================
# ===                       USER CONFIGURATION SECTION                            ===
# ===================================================================================
# Single machine 8 GPUs: 2 GPUs for Actor (training), 6 GPUs for Rollout (inference)


export SIIRL_DIR="${SIIRL_DIR:-{siirl-agentic-dir}}"
export PYTHONPATH="$SIIRL_DIR:$PYTHONPATH"

# --- Experiment and Model Definition ---
export DATASET=deepscaler
export ALG=grpo
export MODEL_NAME=qwen3-8b

# --- Path Definitions ---
# Modify these paths according to your environment
export HOME_DIR=${HOME_DIR:-{your-home-dir}}
export TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-$HOME_DIR/data/datasets/$DATASET/train.parquet}
export TEST_DATA_PATH=${TEST_DATA_PATH:-$HOME_DIR/data/datasets/$DATASET/test.parquet}
export MODEL_PATH=${MODEL_PATH:-$HOME_DIR/data/models/Qwen3-8B}

# Base output paths
export BASE_CKPT_PATH=ckpts
export BASE_TENSORBOARD_PATH=tensorboard

# --- Key Training Hyperparameters ---
export TRAIN_BATCH_SIZE=256
export PPO_MINI_BATCH_SIZE=64
export PPO_MICRO_BATCH_SIZE_PER_GPU=8
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=4096
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.5
export ROLLOUT_TP=2                    # Tensor parallelism for rollout
export ROLLOUT_N=8                     # Number of samples per prompt
export SAVE_FREQ=30
export TEST_FREQ=10
export TOTAL_EPOCHS=30
export MAX_CKPT_KEEP=5

# --- GPU Resource Allocation (Separated Mode) ---
export N_GPUS_PER_NODE=8
export ACTOR_GPUS=2                    # 2 GPUs for training (Actor/Ref)
export ROLLOUT_GPUS=6                  # 6 GPUs for inference (SGLang)
export NNODES=1

# --- Output Paths and Experiment Naming ---
timestamp=$(date +"%Y%m%d_%H%M%S")
export CKPT_PATH=${BASE_CKPT_PATH}/${MODEL_NAME}_${ALG}_${DATASET}_${NNODES}node_${ACTOR_GPUS}actor_${ROLLOUT_GPUS}rollout
export PROJECT_NAME=siirl_${DATASET}_${ALG}
export EXPERIMENT_NAME=siirl_${MODEL_NAME}_${ALG}_${DATASET}_experiment
export TENSORBOARD_DIR=${BASE_TENSORBOARD_PATH}/${MODEL_NAME}_${ALG}_${DATASET}_tensorboard_$timestamp

# --- Define the Training Command and its Arguments ---
# Parameter structure follows SiiRLArguments:
#   - data: DataArguments
#   - actor_ref: ActorRefArguments (model, actor, ref, algo)
#   - rollout: RolloutArguments
#   - algorithm: AlgorithmArguments
#   - trainer: TrainingArguments
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
    actor_ref.actor.megatron.param_offload=True
    actor_ref.actor.megatron.optimizer_offload=False
    # === Reference Model Settings ===
    actor_ref.ref.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU
    actor_ref.ref.megatron.param_offload=True
    # === Rollout Settings (SGLang) ===
    rollout.name=sglang
    rollout.tensor_model_parallel_size=$ROLLOUT_TP
    rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION
    rollout.n=$ROLLOUT_N
    rollout.trust_remote_code=True
    # === MultiTurn Env Settings ===
    rollout.multiturn.env_type='tool_env'
    rollout.multiturn.env_path='examples/AIO/config/tools_config_search.yaml'
    rollout.multiturn.max_env_turns=2
    rollout.multiturn.max_assistant_turns=2
    rollout.multiturn.max_env_response_length=512
    rollout.max_model_len = 10 * 1024
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
    trainer.logger="['console','tensorboard']"
    trainer.resume_mode=auto
    trainer.val_before_train=True
)

# ===================================================================================
# ===                  MAIN EXECUTION LOGIC & INFRASTRUCTURE                      ===
# ===================================================================================

set -e
set -o pipefail
set -x

# --- Start Ray Cluster (Single Node) ---
start_ray_cluster() {
    local ray_start_opts=(
        --num-gpus "$N_GPUS_PER_NODE"
        --object-store-memory 50000000000
        --memory 50000000000
    )

    echo "INFO: Starting Ray in single-node mode with $N_GPUS_PER_NODE GPUs..."
    ray start --head "${ray_start_opts[@]}"
}

# --- Main Execution Function ---
main() {
    echo "============================================================"
    echo "SiiRL Async Training - Separated Mode"
    echo "============================================================"
    echo "Model: $MODEL_NAME"
    echo "Dataset: $DATASET"
    echo "Algorithm: $ALG"
    echo "GPU Allocation: $ACTOR_GPUS Actor + $ROLLOUT_GPUS Rollout"
    echo "Rollout TP: $ROLLOUT_TP (${ROLLOUT_GPUS}/${ROLLOUT_TP} = $((ROLLOUT_GPUS / ROLLOUT_TP)) engines)"
    echo "============================================================"

    # Stop any existing Ray cluster
    ray stop --force 2>/dev/null || true

    # Environment settings
    export TOKENIZERS_PARALLELISM=true
    export NCCL_DEBUG=WARN

    # Start Ray cluster
    start_ray_cluster

    echo "INFO: Starting training..."
    eval "${TRAINING_CMD[@]}" "$@"

    echo "INFO: Training finished."
    ray stop --force 2>/dev/null || true
}

# --- Script Entrypoint ---
main "$@"
