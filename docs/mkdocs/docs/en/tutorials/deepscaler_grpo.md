# Tutorial: GRPO Training on DeepScaleR

> Reproduce a complete GRPO training run on DeepScaleR math reasoning from scratch.

This tutorial walks you through preparing data, configuring hyperparameters, launching training, and monitoring results for Group Relative Policy Optimization (GRPO) on the DeepScaleR math dataset using siirl-agentic. This matches the example scripts in `examples/grpo_train/`.


## Prerequisites

!!! tip "Key Takeaway"
    GRPO is a critic-free RL algorithm that computes advantages by comparing responses within a group, making it simpler to set up than PPO -- no value model required.

| Requirement  | Minimum      | Recommended  |
| ------------ | ------------ | ------------ |
| Python       | 3.10+        | 3.12         |
| GPUs         | 4x A100 80GB | 8x A100 80GB |
| VRAM per GPU | 80 GB        | 80 GB        |
| RAM          | 256 GB       | 512 GB       |
| Disk         | 50 GB free   | 100 GB free  |

Install siirl-agentic and its dependencies:

```bash
pip install -e "siirl-agentic/[gpu]"
pip install datasets pyarrow ray[default] sglang
```


## Step 1: Prepare the Dataset

!!! tip "Key Takeaway"
    The framework reads Parquet files with specific columns. The `data_source` field determines which reward function is dispatched at `default_compute_score` (`__init__.py`). DeepScaleR uses the `math` reward module which extracts answers from `\boxed{}`.

Download DeepScaleR from HuggingFace and convert it into the expected Parquet schema:

```python title="prepare_deepscaler.py"
from datasets import load_dataset
import pandas as pd

ds = load_dataset("agentica-org/DeepScaleR-Preview-Dataset")

records = []
for split_name in ["train", "test"]:
    if split_name not in ds:
        continue
    for row in ds[split_name]:
        records.append({
            "split": split_name,
            "prompt": [
                {
                    "role": "user",
                    "content": row["problem"]
                        + "\nPlease reason step by step, and put your final answer within \\boxed{}."
                }
            ],
            # data_source drives reward dispatch in default_compute_score
            # "agentica-org/DeepScaleR-Preview-Dataset" maps to math.compute_score
            "data_source": "agentica-org/DeepScaleR-Preview-Dataset",
            # ground truth passed to math.compute_score for \boxed{} comparison
            "reward_model": {"ground_truth": row["answer"]},
        })

df = pd.DataFrame(records)
df[df["split"] == "train"].drop(columns=["split"]).to_parquet("deepscaler_train.parquet")
df[df["split"] == "test"].drop(columns=["split"]).to_parquet("deepscaler_test.parquet")
print(f"Train: {len(df[df['split']=='train'])} samples, Test: {len(df[df['split']=='test'])} samples")
```

The reward function in `math.py` (`compute_score`, line 17) extracts the last `\boxed{...}` from the model output via `last_boxed_only_string`, then compares it against the ground truth using `is_equiv` (symbolic math equivalence). A match returns `1.0`, otherwise `0.0`.


## Step 2: Configure Training

!!! tip "Key Takeaway"
    siirl-agentic uses Hydra-style flat key overrides on the CLI. The parameter hierarchy is defined by `SiiRLArguments` (`training_args.py:136`), which nests `DataArguments`, `ActorRefArguments`, `RolloutArguments`, and `TrainingArguments`.

Below is a complete configuration adapted from `examples/grpo_train/run_qwen3_8b_colocate.sh`. We show two deployment modes:

=== "Colocate Mode (8 GPUs shared)"

    All 8 GPUs run both training and rollout, time-sliced.

    | Parameter                            | Value  | Notes                                 |
    | ------------------------------------ | ------ | ------------------------------------- |
    | `trainer.colocate`                   | `True` | GPUs shared between actor and rollout |
    | `trainer.n_gpus_per_node`            | `8`    | Total GPUs on the node                |
    | `rollout.gpu_memory_utilization`     | `0.6`  | Lower to leave room for training      |
    | `rollout.tensor_model_parallel_size` | `2`    | Rollout TP degree                     |
    | `trainer.tensor_model_parallel_size` | `2`    | Actor TP degree                       |

=== "Separated Mode (4+4 GPUs)"

    4 GPUs dedicated to training, 4 GPUs dedicated to rollout.

    | Parameter                            | Value   | Notes                       |
    | ------------------------------------ | ------- | --------------------------- |
    | `trainer.colocate`                   | `False` | Dedicated GPU pools         |
    | `trainer.actor_gpus`                 | `4`     | GPUs for Actor/Ref          |
    | `trainer.rollout_gpus`               | `4`     | GPUs for SGLang rollout     |
    | `rollout.gpu_memory_utilization`     | `0.7`   | Higher since no contention  |
    | `trainer.tensor_model_parallel_size` | `4`     | Actor TP matches actor_gpus |

### Core Hyperparameters

These values match `examples/grpo_train/run_qwen3_8b_colocate.sh`:

| Parameter Path                                 | Default      | DeepScaleR Value | Description                                       |
| ---------------------------------------------- | ------------ | ---------------- | ------------------------------------------------- |
| `data.train_batch_size`                        | `1024`       | `512`            | Prompts per training step                         |
| `data.max_prompt_length`                       | `512`        | `2048`           | Max prompt tokens (DeepScaleR has longer prompts) |
| `data.max_response_length`                     | `512`        | `4096`           | Max response tokens for chain-of-thought          |
| `data.shuffle`                                 | `True`       | `False`          | Disabled in example script                        |
| `rollout.n`                                    | `1`          | `8`              | GRPO group size -- responses generated per prompt |
| `actor_ref.actor.optim.lr`                     | `1e-6`       | `1e-6`           | Learning rate                                     |
| `actor_ref.actor.clip_ratio`                   | `0.2`        | `0.2`            | PPO clipping ratio                                |
| `actor_ref.actor.use_kl_loss`                  | `False`      | `True`           | Enable KL regularization                          |
| `actor_ref.actor.kl_loss_coef`                 | `0.001`      | `0.01`           | KL loss weight                                    |
| `actor_ref.actor.kl_loss_type`                 | `low_var_kl` | `low_var_kl`     | Low-variance KL estimator                         |
| `actor_ref.actor.ppo_mini_batch_size`          | `256`        | `256`            | Mini-batch size for actor update                  |
| `actor_ref.actor.ppo_micro_batch_size_per_gpu` | `None`       | `8`              | Micro-batch per GPU                               |
| `actor_ref.actor.use_dynamic_batch`            | `False`      | `True`           | Token-based dynamic batching                      |
| `actor_ref.actor.max_tokens_per_gpu`           | `4096`       | `16384`          | Max tokens per GPU in dynamic mode                |
| `actor_ref.actor.use_workload_balance`         | `True`       | `True`           | FLOPs-based workload balancing across GPUs        |
| `actor_ref.actor.denominator_scope`            | `local`      | `local`          | Loss denominator scope                            |
| `algorithm.adv_estimator`                      | `grpo`       | `grpo`           | GRPO advantage estimator                          |
| `algorithm.gamma`                              | `1.0`        | `1.0`            | Discount factor (1.0 for outcome reward)          |
| `algorithm.norm_adv_by_std_in_grpo`            | `True`       | `True`           | Normalize by group std (Dr.GRPO sets False)       |
| `trainer.total_epochs`                         | `30`         | `30`             | Total training epochs                             |
| `trainer.save_freq`                            | `-1`         | `30`             | Checkpoint every N steps                          |
| `trainer.test_freq`                            | `-1`         | `10`             | Validate every N steps                            |

### GRPO Advantage Computation

The GRPO advantage is computed in `compute_grpo_outcome_advantage` (`advantage.py:149`). For each prompt group (identified by `uid`), the function:

1. Sums token-level rewards into a scalar score per response
2. Computes the group mean and standard deviation
3. Normalizes: `advantage = (score - mean) / (std + epsilon)` when `norm_adv_by_std_in_grpo=True`
4. Broadcasts the scalar advantage to all response tokens via `response_mask`


## Step 3: Launch Training

!!! tip "Key Takeaway"
    The entry point is `python3 -m siirl.async_train`. Ray manages distributed actors for training and rollout. Always start a Ray cluster before launching.

### Single Node, Colocate Mode (8 GPUs)

```bash title="run_deepscaler_grpo_colocate.sh"
#!/usr/bin/env bash
set -euo pipefail

# --- Paths ---
export MODEL_PATH=/path/to/Qwen3-8B
export TRAIN_DATA_PATH=./deepscaler_train.parquet
export TEST_DATA_PATH=./deepscaler_test.parquet

# --- Start Ray ---
ray stop --force 2>/dev/null || true
ray start --head --num-gpus 8 \
    --object-store-memory 100000000000 \
    --memory 100000000000

# --- Launch Training ---
python3 -m siirl.async_train \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    data.train_files=$TRAIN_DATA_PATH \
    data.val_files=$TEST_DATA_PATH \
    data.train_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_ref.model.path=$MODEL_PATH \
    actor_ref.model.trust_remote_code=True \
    actor_ref.actor.optim.lr=1e-6 \
    actor_ref.actor.ppo_mini_batch_size=256 \
    actor_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_ref.actor.use_dynamic_batch=True \
    actor_ref.actor.max_tokens_per_gpu=16384 \
    actor_ref.actor.use_workload_balance=True \
    actor_ref.actor.denominator_scope=local \
    actor_ref.actor.use_kl_loss=True \
    actor_ref.actor.clip_ratio=0.2 \
    actor_ref.actor.kl_loss_coef=0.01 \
    actor_ref.actor.kl_loss_type=low_var_kl \
    actor_ref.actor.megatron.param_offload=True \
    actor_ref.actor.megatron.optimizer_offload=False \
    actor_ref.actor.megatron.use_mbridge=True \
    actor_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_ref.ref.megatron.param_offload=True \
    rollout.name=sglang \
    rollout.tensor_model_parallel_size=2 \
    rollout.gpu_memory_utilization=0.6 \
    rollout.n=8 \
    rollout.trust_remote_code=True \
    rollout.max_num_seqs=512 \
    rollout.train_server_concurrency=96 \
    rollout.validate_chunk_size=1024 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.colocate=True \
    trainer.validate_reuse_train_gpus=False \
    trainer.total_epochs=30 \
    trainer.save_freq=30 \
    trainer.test_freq=10 \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.default_local_dir=ckpts/deepscaler_grpo \
    trainer.project_name=siirl_agentic_deepscaler_grpo \
    trainer.experiment_name=deepscaler_grpo_colocate \
    trainer.logger="['console','tensorboard','wandb']" \
    trainer.resume_mode=auto \
    trainer.val_before_train=True \
    trainer.tensor_model_parallel_size=2 \
    trainer.pipeline_model_parallel_size=1 \
    trainer.context_parallel_size=1

ray stop --force
```

### Multi-Node (2 Nodes, 16 GPUs Total)

On each node, start Ray and then launch training only on rank 0. Refer to the full multi-node Ray setup in `examples/grpo_train/run_qwen3_8b_colocate.sh` for the cluster formation logic with health-check loops.


## Step 4: Monitor Training

!!! tip "Key Takeaway"
    The three most important early signals are: reward/mean trending upward, KL divergence staying controlled, and policy loss decreasing.

### Key Metrics

| Metric              | What to Watch               | Healthy Signal                                 |
| ------------------- | --------------------------- | ---------------------------------------------- |
| `reward/mean`       | Average reward across batch | Steady upward trend from ~0.05-0.2 to 0.4+    |
| `reward/std`        | Reward variance             | Decreases as policy improves                   |
| `kl_divergence`     | KL from reference policy    | Stays below 5-10; spikes indicate instability  |
| `actor/policy_loss` | PPO clipped loss            | Gradually decreasing                           |
| `actor/entropy`     | Action entropy              | Slow decrease (too fast = mode collapse)       |
| `critic/value_loss` | N/A for GRPO                | Not present -- GRPO has no critic              |
| `val/reward_mean`   | Validation accuracy         | Best indicator of generalization               |

### Expected Training Curve

For a Qwen3-8B model on DeepScaleR with GRPO (`rollout.n=8`):

1. **Steps 0--50**: reward/mean around 0.05--0.15, policy is mostly random
2. **Steps 50--200**: reward/mean climbs to 0.2--0.4, model learns `\boxed{}` formatting
3. **Steps 200--500**: reward/mean plateaus around 0.4--0.6
4. **Steps 500+**: Marginal gains, watch for KL divergence creep

### Signs of Problems

- **reward/mean flat at 0**: Reward function not matching model output format -- check that model outputs `\boxed{answer}`
- **KL divergence exploding**: Learning rate too high, or `kl_loss_coef` too low
- **NaN in loss**: Reduce learning rate, check for data corruption in Parquet


## Step 5: Evaluate

After training completes, checkpoints are saved to `trainer.default_local_dir`. To evaluate:

```bash
# Run validation only with a trained checkpoint
python3 -m siirl.async_train \
    actor_ref.model.path=ckpts/deepscaler_grpo/actor/step_XXX \
    data.val_files=./deepscaler_test.parquet \
    trainer.val_only=True \
    rollout.n=1 \
    rollout.temperature=0.0 \
    # ... (same other config as training)
```

Set `trainer.val_only=True` to skip training and only run validation. Use greedy decoding (`rollout.temperature=0.0`, `rollout.n=1`) for deterministic evaluation.


## Expected Results

!!! warning
    These are approximate ranges from published literature and internal experiments. Your results depend on hyperparameters, random seed, hardware, and base model quality. DeepScaleR is harder than GSM8K due to competition-level math problems.

| Model      | GPUs    | Approx. Time | DeepScaleR Accuracy (test) |
| ---------- | ------- | ------------ | -------------------------- |
| Qwen2.5-7B | 8x A100 | 6--12 hours  | 50--65%                    |
| Qwen3-8B   | 8x A100 | 6--12 hours  | 55--70%                    |
| LLaMA-3-8B | 8x A100 | 6--12 hours  | 45--60%                    |

Accuracy ranges assume 30 epochs with `rollout.n=8` and `lr=1e-6`. Base model zero-shot performance sets the floor.


## Troubleshooting This Tutorial

| Symptom                                                                       | Cause                                                                                                           | Fix                                                                                                                     |
| ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `NotImplementedError: Reward function is not implemented for data_source=...` | `data_source` in Parquet does not match any key in `default_compute_score` (`__init__.py:44`)                   | Set `data_source` to exactly `"agentica-org/DeepScaleR-Preview-Dataset"`                                                |
| OOM on rollout GPUs                                                           | `gpu_memory_utilization` too high or `max_response_length` too large                                            | Lower `rollout.gpu_memory_utilization` to 0.5, reduce `data.max_response_length`                                        |
| `truncation='error'` raises exception                                         | Prompts exceed `max_prompt_length`                                                                              | Increase `data.max_prompt_length` or set `data.truncation='right'`                                                      |
| Reward stays at 0.0 for all steps                                             | Model not producing `\boxed{answer}` format                                                                     | Add formatting instruction in prompt (e.g., "put your final answer within \\boxed{}") or check `math.py` extraction     |
| Ray cluster fails to start                                                    | Port conflict or insufficient shared memory                                                                     | Run `ray stop --force`, set `--object-store-memory` appropriately                                                       |
| Training hangs at weight sync                                                 | NCCL timeout between training and rollout GPUs                                                                  | Increase `GLOO_SOCKET_TIMEOUT=600` and `GLOO_TCP_TIMEOUT=600`                                                           |
