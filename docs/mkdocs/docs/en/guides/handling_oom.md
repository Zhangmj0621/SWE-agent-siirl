# Handling Out-of-Memory (OOM)

*Estimate memory requirements and resolve OOM errors across training and rollout components.*

## Memory Estimation Formula

!!! tip "Key Insight"
    Rule of thumb: an N-billion parameter model in BF16 needs ~2N GB for weights alone. Training adds 4--8x overhead for optimizer states and gradients. Always budget for the rollout engine's KV cache separately.

### Per-Component Memory

| Component                        | Formula                         | Example (7B, BF16)   |
| -------------------------------- | ------------------------------- | -------------------- |
| Actor weights                    | 2 x N_params GB                 | ~14 GB               |
| Optimizer states (Adam)          | 8 x N_params GB                 | ~56 GB               |
| Gradients                        | 2 x N_params GB                 | ~14 GB               |
| Activations                      | Varies by batch/seq length      | ~2--8 GB             |
| Reference model weights          | 2 x N_params GB                 | ~14 GB               |
| Rollout engine (SGLang KV cache) | `gpu_memory_utilization` x VRAM | ~40 GB (0.5 x 80 GB) |

!!! note
    With `use_distributed_optimizer=True` (default in `MegatronArguments`, model_args.py:22), optimizer states are sharded across data-parallel ranks, reducing per-GPU optimizer memory by `1/dp_size`.

### Separated vs Colocated Memory Layout

In **separated mode** (`trainer.colocate=False`, default), training and rollout use different GPU sets:

| GPU Set       | Components                                        | Example (7B, 2 train GPUs + 6 rollout GPUs) |
| ------------- | ------------------------------------------------- | ------------------------------------------- |
| Training GPUs | Actor + Ref + Optimizer + Gradients + Activations | ~50 GB/GPU (with distributed optimizer)     |
| Rollout GPUs  | SGLang engine + KV cache                          | ~40 GB/GPU (`gpu_memory_utilization=0.5`)   |

Configuration:

```yaml
trainer:
  colocate: false        # Default
  actor_gpus: 2          # Default: 2
  rollout_gpus: 6        # Default: 6
```

In **colocated mode** (`trainer.colocate=True`), training and rollout share the same GPUs with time-multiplexing:

| Phase          | Active Components                                 | Memory                   |
| -------------- | ------------------------------------------------- | ------------------------ |
| Rollout phase  | SGLang engine (weights offloaded during training) | KV cache + model weights |
| Training phase | Actor + Ref + Optimizer (rollout offloaded)       | Full training memory     |

Configuration:

```yaml
trainer:
  colocate: true
  # actor_gpus and rollout_gpus are ignored in colocate mode
```

!!! warning
    Colocate mode automatically enforces several safety guards (in `_apply_colocate_guards`, async_train.py:42):

    - Forces `param_offload=True` on actor, ref, and critic megatron configs
    - Clamps `rollout.gpu_memory_utilization` to 0.45 maximum
    - Disables `validate_reuse_train_gpus`

## Three-Level Offloading

`MegatronArguments` (model_args.py:21) provides three levels of CPU offloading, each progressively reducing GPU memory at the cost of training speed:

| Level | Parameter           | Default | What It Offloads        | GPU Memory Saved (7B) |
| ----- | ------------------- | ------- | ----------------------- | --------------------- |
| 1     | `param_offload`     | `False` | Model parameters to CPU | ~14 GB                |
| 2     | `grad_offload`      | `False` | Gradients to CPU        | ~14 GB                |
| 3     | `optimizer_offload` | `False` | Optimizer states to CPU | ~56 GB                |

### Configuration Example

=== "No Offloading (default)"

    ```yaml
    actor_ref:
      actor:
        megatron:
          param_offload: false
          grad_offload: false
          optimizer_offload: false
      ref:
        megatron:
          param_offload: false
    ```

    GPU memory per training GPU: ~100 GB (7B model)

=== "Parameter Offload Only"

    ```yaml
    actor_ref:
      actor:
        megatron:
          param_offload: true
          grad_offload: false
          optimizer_offload: false
      ref:
        megatron:
          param_offload: true
    ```

    GPU memory per training GPU: ~72 GB (7B model)

=== "Full Offload"

    ```yaml
    actor_ref:
      actor:
        megatron:
          param_offload: true
          grad_offload: true
          optimizer_offload: true
      ref:
        megatron:
          param_offload: true
    ```

    GPU memory per training GPU: ~16 GB (7B model, activations + workspace only)

!!! note
    When `param_offload=True`, the training loop in `Trainer.train_step` (trainer.py:716) automatically loads actor weights to GPU before compute and offloads after, coordinated with weight sync in `_sync_rollout_workers` (trainer.py:489).

## Quick Fixes

| OOM Location       | First Try                                                | Second Try                                                | Last Resort                                                         |
| ------------------ | -------------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------- |
| Trainer init       | Reduce `ppo_mini_batch_size` (default: `256`)            | Enable `actor.megatron.param_offload`                     | Switch to `colocate: true`                                          |
| Rollout init       | Reduce `rollout.gpu_memory_utilization` (default: `0.5`) | Reduce `rollout.max_num_batched_tokens` (default: `8192`) | Add more rollout GPUs via `trainer.rollout_gpus`                    |
| Training step      | Reduce `actor.ppo_micro_batch_size_per_gpu`              | Enable `actor.megatron.grad_offload`                      | Enable `actor.megatron.optimizer_offload`                           |
| Multi-turn rollout | Reduce `data.max_response_length` (default: `512`)       | Reduce `rollout.multiturn.max_env_turns` (default: `1`)   | Reduce `rollout.multiturn.max_env_response_length` (default: `256`) |
| Reference model    | Enable `ref.megatron.param_offload`                      | Reduce `ref.ppo_micro_batch_size_per_gpu`                 | N/A                                                                 |
| Critic model (PPO) | Enable `critic.megatron.param_offload`                   | Reduce `critic.ppo_micro_batch_size_per_gpu`              | Reduce `critic.ppo_mini_batch_size` (default: `256`)                |

## Dynamic Batching for Variable Sequences

!!! tip "Key Insight"
    When sequence lengths vary significantly (common in agentic multi-turn tasks), dynamic batching prevents OOM by capping total tokens per GPU instead of using a fixed sample count.

`ActorArguments` (model_args.py:99) provides dynamic batching controls:

| Parameter              | Default   | Description                                                                                                 |
| ---------------------- | --------- | ----------------------------------------------------------------------------------------------------------- |
| `use_dynamic_batch`    | `False`   | Enable token-based batching instead of fixed batch size                                                     |
| `max_tokens_per_gpu`   | `4096`    | Maximum tokens per GPU when dynamic batching is enabled                                                     |
| `use_workload_balance` | `True`    | Use FLOPs-based balancing (accounts for quadratic attention cost); otherwise uses sequence-length balancing |
| `denominator_scope`    | `"local"` | Loss denominator scope: `"local"` (per-GPU) or `"dp_global"` (across all DP ranks)                          |
| `loss_scale_factor`    | `None`    | Optional fixed denominator for `seq-mean-token-sum-norm` loss mode                                          |

### When to Use Dynamic Batching

Use dynamic batching when:

- Sequence lengths vary by more than 3x across samples in a batch
- Multi-turn agentic trajectories produce unpredictable response lengths
- You see sporadic OOM on some training steps but not others

### Configuration Example

```yaml
actor_ref:
  actor:
    use_dynamic_batch: true
    max_tokens_per_gpu: 4096
    use_workload_balance: true
    denominator_scope: "local"
```

### Fixed Batch vs Dynamic Batch

| Aspect                | Fixed Batch                                         | Dynamic Batch                        |
| --------------------- | --------------------------------------------------- | ------------------------------------ |
| Batch size control    | `ppo_micro_batch_size_per_gpu` (sample count)       | `max_tokens_per_gpu` (token count)   |
| Memory predictability | Peak memory determined by longest sequence in batch | Peak memory bounded by token budget  |
| OOM risk              | High with variable-length sequences                 | Low, consistent memory usage         |
| Throughput            | Higher when sequences are uniform                   | Higher when sequences vary in length |

## Memory Profiling Tools

siirl-agentic includes built-in memory profiling via `MemoryProfiler` (memory_profiler.py:424):

```bash
# Enable per-step peak memory logging
export SIIRL_MEMORY_STEP_PROFILE=1

# Enable full memory snapshot export (step 0 only, rank 0)
export SIIRL_MEMORY_PROFILE=1

# Enable detailed memory debug logging
export SIIRL_MEMORY_DEBUG=1
```

### Analyzing Memory Snapshots

When `SIIRL_MEMORY_PROFILE=1`, a snapshot is exported after the first training step. The exported file can be analyzed with the built-in tool:

```bash
# Analyze the snapshot
python -m siirl.utils.logger.memory_profiler memory_snapshot_rank0_step0.dat
```

The snapshot can also be visualized at [pytorch.org/memory_viz](https://pytorch.org/memory_viz).

### Programmatic Memory Inspection

```python
from siirl.utils.logger.memory_profiler import (
    get_memory_stats,
    log_memory,
    memory_trace,
    GPUMemoryLogger,
)

# Quick stats
stats = get_memory_stats()
print(f"Allocated: {stats['allocated_gb']:.2f} GB")

# Context manager for tracing a code block
with memory_trace("my_computation"):
    result = model(input_ids)

# Decorator for function-level profiling
@GPUMemoryLogger(role="actor")
def update_actor(batch):
    ...
```

## Common Mistakes

| Symptom                | Cause                                                | Fix                                                                                             |
| ---------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| OOM only on some steps | Variable sequence lengths with fixed batch           | Enable `use_dynamic_batch=True` with `max_tokens_per_gpu=4096`                                  |
| OOM during weight sync | Both actor and rollout weights on GPU simultaneously | Enable `param_offload=True` (auto-enabled in colocate mode)                                     |
| OOM in colocate mode   | `gpu_memory_utilization` too high                    | Reduce to 0.45 or below (auto-clamped by `_apply_colocate_guards`)                              |
| Gradual memory growth  | CUDA memory fragmentation                            | Set `rollout.free_cache_engine=True` (default) and periodically call `torch.cuda.empty_cache()` |
| OOM during validation  | Validation batch larger than training batch          | Set `data.val_batch_size` explicitly instead of using entire validation set                     |
| OOM with PPO critic    | Three models loaded simultaneously                   | Enable `param_offload` on all three: actor, ref, and critic                                     |
