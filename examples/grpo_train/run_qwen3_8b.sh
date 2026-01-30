#!/usr/bin/env bash
# ===================================================================================
# ===                       USER CONFIGURATION SECTION                            ===
# ===================================================================================
# Single machine 8 GPUs: 2 GPUs for Actor (training), 6 GPUs for Rollout (inference)


export SIIRL_DIR="${SIIRL_DIR:-{siirl-agentic-dir}}"
export PYTHONPATH="$SIIRL_DIR:$PYTHONPATH"

#  --- Experiment and Model Definition ---
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
export TRAIN_BATCH_SIZE=512
export PPO_MINI_BATCH_SIZE=256
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
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
export NNODES=${PET_NNODES:-1}
export NODE_RANK=${PET_NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}
export ACTOR_GPUS=4                    # 2 GPUs for training (Actor/Ref)
export ROLLOUT_GPUS=4                  # 6 GPUs for inference (SGLang)

# --- Actor Parallelism Configuration ---
# TP (Tensor Parallel): Model sharding across GPUs within a group
# PP (Pipeline Parallel): Model layer sharding across pipeline stages
# CP (Context Parallel): Sequence parallelism for long context
# DP (Data Parallel): Automatically computed as ACTOR_GPUS / (TP * PP * CP)
export ACTOR_TP=4                      # Actor tensor parallelism (default: 1)
export ACTOR_PP=1                      # Actor pipeline parallelism (default: 1)
export ACTOR_CP=1                      # Actor context parallelism (default: 1)
# With ACTOR_GPUS=2, TP=1, PP=1, CP=1 -> DP=2 (2 data parallel trainers)

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
    trainer.logger="['console','tensorboard']"
    trainer.resume_mode=auto
    trainer.val_before_train=True
    # === Parallel Config ===
    trainer.tensor_model_parallel_size=$ACTOR_TP
    trainer.pipeline_model_parallel_size=$ACTOR_PP
    trainer.context_parallel_size=$ACTOR_CP
)

# ===================================================================================
# ===                          EXECUTION LOGIC                                    ===
# ===================================================================================

# --- Boilerplate Setup ---
set -e
set -o pipefail
set -x

# --- Infrastructure & Boilerplate Functions ---
start_ray_cluster() {
    local RAY_HEAD_WAIT_TIMEOUT=600
    export RAY_RAYLET_NODE_MANAGER_CONFIG_NIC_NAME=${INTERFACE_NAME}
    export RAY_GCS_SERVER_CONFIG_NIC_NAME=${INTERFACE_NAME}
    export RAY_RUNTIME_ENV_AGENT_CREATION_TIMEOUT_S=1200
    export RAY_GCS_RPC_CLIENT_CONNECT_TIMEOUT_S=120

    local ray_start_common_opts=(
        --num-gpus "$N_GPUS_PER_NODE"
        --object-store-memory 100000000000
        --memory 100000000000
    )

    if [ "$NNODES" -gt 1 ]; then
        if [ "$NODE_RANK" = "0" ]; then
            echo "INFO: Starting Ray head node on $(hostname)..."
            export RAY_ADDRESS="$RAY_MASTER_ADDR:$RAY_MASTER_PORT"
            ray start --head --port="$RAY_MASTER_PORT" --dashboard-port="$RAY_DASHBOARD_PORT" "${ray_start_common_opts[@]}" --system-config='{"gcs_server_request_timeout_seconds": 60, "gcs_rpc_server_reconnect_timeout_s": 60}'
            local start_time=$(date +%s)
            while ! ray health-check --address "$RAY_ADDRESS" &>/dev/null; do
                if [ "$(( $(date +%s) - start_time ))" -ge "$RAY_HEAD_WAIT_TIMEOUT" ]; then echo "ERROR: Timed out waiting for head node. Exiting." >&2; ray stop --force; exit 1; fi
                echo "Head node not healthy yet. Retrying in 5s..."
                sleep 5
            done
            echo "INFO: Head node is healthy."
        else
            local head_node_address="$MASTER_ADDR:$RAY_MASTER_PORT"
            echo "INFO: Worker node $(hostname) waiting for head at $head_node_address..."
            local start_time=$(date +%s)
            while ! ray health-check --address "$head_node_address" &>/dev/null; do
                if [ "$(( $(date +%s) - start_time ))" -ge "$RAY_HEAD_WAIT_TIMEOUT" ]; then echo "ERROR: Timed out waiting for head. Exiting." >&2; exit 1; fi
                echo "Head not healthy yet. Retrying in 5s..."
                sleep 5
            done
            echo "INFO: Head is healthy. Worker starting..."
            ray start --address="$head_node_address" "${ray_start_common_opts[@]}"
        fi
    else
        echo "INFO: Starting Ray in single-node mode..."
        ray start --head "${ray_start_common_opts[@]}"
    fi
}

# --- Main Execution Function ---
main() {
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    ray stop --force 2>/dev/null || true

    export VLLM_USE_V1=1
    export GLOO_SOCKET_TIMEOUT=600
    export GLOO_TCP_TIMEOUT=600
    export GLOO_LOG_LEVEL=DEBUG
    export RAY_MASTER_PORT=${RAY_MASTER_PORT:-6379}
    export RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
    export RAY_MASTER_ADDR=$MASTER_ADDR

    start_ray_cluster

    if [ "$NNODES" -gt 1 ] && [ "$NODE_RANK" = "0" ]; then
        echo "Waiting for all $NNODES nodes to join..."
        local TIMEOUT=600; local start_time=$(date +%s)
        while true; do
            if [ "$(( $(date +%s) - start_time ))" -ge "$TIMEOUT" ]; then echo "Error: Timeout waiting for nodes." >&2; exit 1; fi
            # Use Python API to check node count (avoids dashboard dependency)
            local ready_nodes=$(python3 -c "
import ray
try:
    ray.init(address='auto', ignore_reinit_error=True)
    nodes = ray.nodes()
    alive = len([n for n in nodes if n['Alive']])
    print(alive)
except:
    print(0)
" 2>/dev/null)
            ready_nodes=${ready_nodes:-0}
            if [ "$ready_nodes" -ge "$NNODES" ]; then break; fi
            echo "Waiting... ($ready_nodes / $NNODES nodes ready)"
            sleep 5
        done
        echo "All $NNODES nodes have joined."
    fi

    if [ "$NODE_RANK" = "0" ]; then
        echo "INFO [RANK 0]: Starting main training command."
        eval "${TRAINING_CMD[@]}" "$@"
        echo "INFO [RANK 0]: Training finished."
        sleep 30; ray stop --force >/dev/null 2>&1
    elif [ "$NNODES" -gt 1 ]; then
        local head_node_address="$MASTER_ADDR:$RAY_MASTER_PORT"
        echo "INFO [RANK $NODE_RANK]: Worker active. Monitoring head node at $head_node_address."
        while ray health-check --address "$head_node_address" &>/dev/null; do sleep 15; done
        echo "INFO [RANK $NODE_RANK]: Head node down. Exiting."
    fi

    echo "INFO: Script finished on rank $NODE_RANK."
}

# --- Script Entrypoint ---
main "$@"
