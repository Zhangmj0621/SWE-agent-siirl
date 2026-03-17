# Quickstart

*Run your first GRPO training job on a single node with 8 GPUs.*

## Prerequisites

-   siirl-agentic installed (see [Installation](installation.md))
-   8 GPUs available (4 for training, 4 for rollout)
-   A model (e.g., Qwen3-8B) downloaded locally
-   Training data in Parquet format

!!! tip "Preparing Parquet Data"
    If you have data in JSON/JSONL format, convert it to Parquet using the `datasets` library:
    ```python
    from datasets import load_dataset
    ds = load_dataset("json", data_files="train.jsonl")
    ds["train"].to_parquet("train.parquet")
    ```
    Each row should have at least a `prompt` field (string or list of chat messages).

## Step 1: Prepare Your Environment

``` bash
# Set paths
export HOME_DIR=/path/to/your/home
export MODEL_PATH=$HOME_DIR/data/models/Qwen3-8B
export TRAIN_DATA_PATH=$HOME_DIR/data/datasets/deepscaler/train.parquet
export TEST_DATA_PATH=$HOME_DIR/data/datasets/deepscaler/test.parquet
```

## Step 2: Run GRPO Training (Separated Mode)

``` bash
cd siirl-agentic
bash examples/grpo_train/run_qwen3_8b_separated.sh
```

This script configures:

- **4 GPUs for Actor** (training with TP=4)
- **4 GPUs for Rollout** (SGLang inference with TP=2)
- **GRPO algorithm** with batch_size=512, n=8 samples per prompt
- **Qwen3-8B model** with max_prompt=2048, max_response=4096

The script passes configuration via OmegaConf CLI arguments. You can customize any parameter by editing the script or appending overrides:

``` bash
bash examples/grpo_train/run_qwen3_8b_separated.sh \
    data.train_batch_size=256 \
    rollout.temperature=0.8
```

**Expected output:**

```
INFO  | Ray is initialized. Time cost: 150.23 ms
INFO  | MainRunner started. Beginning workflow setup...
INFO  | Initializing DataCoordinator...
SUCCESS | DataCoordinator initialized
SUCCESS | RolloutManager initialized. Router at: http://10.0.0.1:30000
SUCCESS | TrainerGroup initialized with 4 trainers
INFO  | Starting async training loop...
```

!!! note
    The exact numbers (batches/epoch, total steps) depend on your dataset size and `train_batch_size` setting.

## Step 3: Run PPO Training

``` bash
bash examples/ppo_train/run_qwen3_8b_separated.sh
```

PPO adds a Critic model on top of GRPO. The script adjusts resource allocation accordingly.

## Step 4: Colocated Mode (Share GPUs)

For smaller setups or maximum GPU utilization:

``` bash
bash examples/grpo_train/run_qwen3_8b_colocate.sh
```

In colocated mode, training and rollout share all 8 GPUs. The framework automatically manages weight offloading.

## Key Parameters to Tune

| Parameter                                     | What it controls             | Recommendation               |
| --------------------------------------------- | ---------------------------- | ---------------------------- |
| `data.train_batch_size`                       | Samples per training step    | Start with 512               |
| `rollout.n`                                   | Samples per prompt           | 8 for GRPO, 1 for PPO        |
| `data.max_response_length`                    | Max generation length        | Match your task needs        |
| `trainer.actor_gpus` / `trainer.rollout_gpus` | GPU split                    | Even split for most cases    |
| `rollout.tensor_model_parallel_size`          | Inference tensor parallelism | 2 for 8B models, 4+ for 70B+ |

!!! tip "Batch Size First-Run Tip"
    Start with a smaller batch size (`data.train_batch_size=128`) for your first run to confirm everything works before scaling up. Large batch sizes require proportionally more GPU memory for activations. If you see OOM errors, halve the batch size first before adjusting other settings.

## Monitoring

Training logs are written to stdout and optionally to WandB. Configure via CLI overrides:

``` bash
# Example: enable WandB logging
python -m siirl.async_train \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=siirl_examples \
    trainer.experiment_name=my_first_run
```

## What success looks like

After completing this guide, you should see:

- Ray initializes successfully: `INFO | Ray is initialized. Time cost: ...`
- All components come up: `SUCCESS | RolloutManager initialized`, `SUCCESS | TrainerGroup initialized`
- Training loop starts: `INFO | Starting async training loop...`
- `reward/mean` metric appears in logs and trends upward over steps
- No CUDA OOM errors or Ray actor crashes
- Checkpoint saved to the configured output directory at the end of each epoch

**Sample log output from a healthy training run:**

```
INFO  | Ray is initialized. Time cost: 150.23 ms
SUCCESS | DataCoordinator initialized
SUCCESS | RolloutManager initialized. Router at: http://10.0.0.1:30000
SUCCESS | TrainerGroup initialized with 4 trainers
INFO  | Starting async training loop...
INFO  | [Step 1] rollout started, batch_size=512
INFO  | [Step 1] rollout done. reward/mean=0.12, reward/std=0.08
INFO  | [Step 1] training started
INFO  | [Step 1] training done. loss=1.42, grad_norm=0.87
INFO  | [Step 10] rollout done. reward/mean=0.31, reward/std=0.11
INFO  | [Step 10] training done. loss=1.18, grad_norm=0.74
INFO  | Checkpoint saved to: /path/to/output/step_10/
```

The key signal is `reward/mean` increasing from step 1 to step 10. The exact starting value depends on your task and model — what matters is the upward trend.

## If something goes wrong

| Symptom                                 | Likely cause                                     | Fix                                                                                  |
| --------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------ |
| `CUDA out of memory`                    | `rollout.n` or `train_batch_size` too large      | Reduce `rollout.n` first (e.g., from `8` to `4`), then halve `data.train_batch_size` |
| `SGLang error` or SGLang fails to start | Insufficient GPU memory for the inference engine | Reduce `rollout.gpu_memory_utilization` (e.g., from `0.85` to `0.7`)                 |
| `Ray actor crashed`                     | Worker OOM or NCCL timeout                       | Check `dmesg` for OOM kills; increase `RAY_memory_monitor_refresh_ms=0`              |
| Training hangs after rollout            | NCCL deadlock during weight sync                 | Set `NCCL_TIMEOUT=1800` and restart                                                  |
| `reward/mean` stays flat at 0           | Reward function returning 0 for all outputs      | Verify your reward function logic with a small test batch first                      |

For more detailed troubleshooting steps, see [Troubleshooting](../reference/troubleshooting.md).

## Next steps

- [GRPO Training](../guides/grpo_training.md) — Deep dive into group-sampling configuration and GRPO-specific tuning knobs
- [First Agentic Training Job](first_agentic_training_job.md) — Add tool interaction to your training and enable multi-turn rollouts
- [Configuration System](../guides/configuration_system.md) — Understand how to override any parameter from the CLI using Hydra
