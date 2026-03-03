#!/usr/bin/env bash
# --- Experiment and Model Definition ---
export DATASET=deepscaler
export ALG=ppo
export MODEL_NAME=qwen3-1.7b

# --- Path Definitions ---
export HOME_DIR=${HOME_DIR:-{your-home-dir}}
export TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-$HOME_DIR/data/datasets/$DATASET/train.parquet}
export TEST_DATA_PATH=${TEST_DATA_PATH:-$HOME_DIR/data/datasets/$DATASET/test.parquet}
export MODEL_PATH=${MODEL_PATH:-$HOME_DIR/data/models/Qwen3-1.7B}

# --- Output ---
export BASE_CKPT_PATH=ckpts
export BASE_TENSORBOARD_PATH=tensorboard

# --- Hyperparameters ---
export TRAIN_BATCH_SIZE=512
export PPO_MINI_BATCH_SIZE=256
export PPO_MICRO_BATCH_SIZE_PER_GPU=8
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=4096
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.7
export ROLLOUT_TP=2
export ROLLOUT_N=1
export SAVE_FREQ=30
export TEST_FREQ=10
export TOTAL_EPOCHS=30
export MAX_CKPT_KEEP=5

# --- Cluster ---
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
export NNODES=${PET_NNODES:-1}
export NODE_RANK=${PET_NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}

# --- Megatron parallel config ---
export ACTOR_TP=2
export ACTOR_PP=1
export ACTOR_CP=1

# --- Names ---
timestamp=$(date +"%Y%m%d_%H%M%S")
export CKPT_PATH=${BASE_CKPT_PATH}/${MODEL_NAME}_${ALG}_${DATASET}_${NNODES}node_${ACTOR_GPUS}gpu_colocate
export PROJECT_NAME=siirl_agentic_${DATASET}_${ALG}
export EXPERIMENT_NAME=siirl_${MODEL_NAME}_${ALG}_${DATASET}_colocate
export TENSORBOARD_DIR=${BASE_TENSORBOARD_PATH}/${MODEL_NAME}_${ALG}_${DATASET}_tensorboard_$timestamp

export WANDB_BASE_URL=${WANDB_BASE_URL:-https://xxx}
export WANDB_API_KEY=${WANDB_API_KEY:-}

TRAINING_CMD=(
    python3 -m siirl.async_train
    # Algorithm
    actor_ref.algorithm.adv_estimator=$ALG
    actor_ref.algorithm.gamma=1.0
    actor_ref.algorithm.lam=1.0
    # Data
    data.train_files=$TRAIN_DATA_PATH
    data.val_files=$TEST_DATA_PATH
    data.train_batch_size=$TRAIN_BATCH_SIZE
    data.max_prompt_length=$MAX_PROMPT_LENGTH
    data.max_response_length=$MAX_RESPONSE_LENGTH
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
    # Model
    actor_ref.model.path=$MODEL_PATH
    actor_ref.model.trust_remote_code=True
    # Actor
    actor_ref.actor.optim.lr=1e-6
    actor_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE
    actor_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU
    actor_ref.actor.use_dynamic_batch=True
    actor_ref.actor.max_tokens_per_gpu=16384
    actor_ref.actor.use_workload_balance=True
    actor_ref.actor.denominator_scope=local
    actor_ref.actor.use_kl_loss=True
    actor_ref.actor.clip_ratio=0.2
    actor_ref.actor.kl_loss_coef=0.01
    actor_ref.actor.kl_loss_type=low_var_kl
    actor_ref.actor.megatron.param_offload=True
    actor_ref.actor.megatron.optimizer_offload=False
    actor_ref.actor.megatron.use_mbridge=True
    # Critic (PPO)
    critic.model.path=$MODEL_PATH
    critic.model.trust_remote_code=True
    critic.optim.lr=5e-6
    critic.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE
    critic.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU
    critic.ppo_epochs=1
    critic.cliprange_value=0.5
    critic.megatron.param_offload=True
    critic.megatron.optimizer_offload=False
    # Ref
    actor_ref.ref.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU
    actor_ref.ref.megatron.param_offload=True
    # Rollout
    rollout.name=sglang
    rollout.tensor_model_parallel_size=$ROLLOUT_TP
    rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION
    rollout.n=$ROLLOUT_N
    rollout.trust_remote_code=True
    # Trainer
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE
    trainer.nnodes=$NNODES
    trainer.colocate=True
    trainer.validate_reuse_train_gpus=False
    trainer.total_epochs=$TOTAL_EPOCHS
    trainer.save_freq=$SAVE_FREQ
    trainer.test_freq=$TEST_FREQ
    trainer.max_actor_ckpt_to_keep=$MAX_CKPT_KEEP
    trainer.max_critic_ckpt_to_keep=$MAX_CKPT_KEEP
    trainer.default_local_dir=$CKPT_PATH
    trainer.project_name=$PROJECT_NAME
    trainer.experiment_name=$EXPERIMENT_NAME
    trainer.logger="['console','tensorboard','wandb']"
    trainer.resume_mode=auto
    trainer.val_before_train=True
    # Parallel
    trainer.tensor_model_parallel_size=$ACTOR_TP
    trainer.pipeline_model_parallel_size=$ACTOR_PP
    trainer.context_parallel_size=$ACTOR_CP
)

set -e
set -o pipefail
set -x

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
            ray start --head --port="$RAY_MASTER_PORT" --dashboard-port="$RAY_DASHBOARD_PORT" \
                "${ray_start_common_opts[@]}" \
                --system-config='{"gcs_server_request_timeout_seconds": 60, "gcs_rpc_server_reconnect_timeout_s": 60}'
            local start_time
            start_time=$(date +%s)
            while ! ray health-check --address "$RAY_ADDRESS" &>/dev/null; do
                if [ "$(( $(date +%s) - start_time ))" -ge "$RAY_HEAD_WAIT_TIMEOUT" ]; then
                    echo "ERROR: Timed out waiting for head node. Exiting." >&2
                    ray stop --force
                    exit 1
                fi
                echo "Head node not healthy yet. Retrying in 5s..."
                sleep 5
            done
            echo "INFO: Head node is healthy."
        else
            local head_node_address="$MASTER_ADDR:$RAY_MASTER_PORT"
            echo "INFO: Worker node $(hostname) waiting for head at $head_node_address..."
            local start_time
            start_time=$(date +%s)
            while ! ray health-check --address "$head_node_address" &>/dev/null; do
                if [ "$(( $(date +%s) - start_time ))" -ge "$RAY_HEAD_WAIT_TIMEOUT" ]; then
                    echo "ERROR: Timed out waiting for head. Exiting." >&2
                    exit 1
                fi
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

main() {
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
        local TIMEOUT=600
        local start_time
        start_time=$(date +%s)
        while true; do
            if [ "$(( $(date +%s) - start_time ))" -ge "$TIMEOUT" ]; then
                echo "Error: Timeout waiting for nodes." >&2
                exit 1
            fi
            local ready_nodes
            ready_nodes=$(python3 -c "
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
            if [ "$ready_nodes" -ge "$NNODES" ]; then
                break
            fi
            echo "Waiting... ($ready_nodes / $NNODES nodes ready)"
            sleep 5
        done
        echo "All $NNODES nodes have joined."
    fi

    if [ "$NODE_RANK" = "0" ]; then
        echo "INFO [RANK 0]: Starting main training command."
        eval "${TRAINING_CMD[@]}" "$@"
        echo "INFO [RANK 0]: Training finished."
        sleep 30
        ray stop --force >/dev/null 2>&1
    elif [ "$NNODES" -gt 1 ]; then
        local head_node_address="$MASTER_ADDR:$RAY_MASTER_PORT"
        echo "INFO [RANK $NODE_RANK]: Worker active. Monitoring head node at $head_node_address."
        while ray health-check --address "$head_node_address" &>/dev/null; do
            sleep 15
        done
        echo "INFO [RANK $NODE_RANK]: Head node down. Exiting."
    fi

    echo "INFO: Script finished on rank $NODE_RANK."
}

main "$@"
