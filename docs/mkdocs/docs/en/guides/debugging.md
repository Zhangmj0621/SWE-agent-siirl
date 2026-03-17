# Debugging Guide

*Systematic approaches to diagnosing training issues in siirl-agentic.*

## Ray Actor Debugging

!!! tip "Key Insight"
    All siirl-agentic workers are Ray actors. The fastest way to understand what went wrong is `ray status` plus the Ray Dashboard. Always check `/tmp/ray/session_latest/logs/` for actor-level logs before diving into code.

### Cluster Status

Use `ray status` to verify resource availability and actor placement:

```bash
# Check cluster health
ray status

# Ray Dashboard (default port 8265)
# Open http://<head-node-ip>:8265 in a browser
```

### Actor Log Locations

Ray stores per-actor logs under the session directory:

| Log Type               | Path                                                                |
| ---------------------- | ------------------------------------------------------------------- |
| Worker stdout/stderr   | `/tmp/ray/session_latest/logs/worker-*.out` / `.err`                |
| Ray system logs        | `/tmp/ray/session_latest/logs/raylet.out`                           |
| GCS server logs        | `/tmp/ray/session_latest/logs/gcs_server.out`                       |
| siirl application logs | `./siirl_logs/siirl_*.log` (configurable via `SIIRL_LOG_DIRECTORY`) |

### Useful Environment Variables

```bash
# Disable Ray log deduplication (see every log line from every worker)
export RAY_DEDUP_LOGS=0

# Set loguru log level (DEBUG shows concurrency, batch sync, memory details)
export LOGURU_LEVEL=DEBUG

# Enable verbose validation logs on rollout workers
export SIIRL_VERBOSE_VALIDATE_LOGS=1
```

### Inspecting Actor State

The `MainRunner` (async_train.py:69) orchestrates all components. Key actors to inspect:

| Actor                       | Class                                      | Purpose                           |
| --------------------------- | ------------------------------------------ | --------------------------------- |
| `MainRunner`                | `MainRunner` (async_train.py:69)           | Orchestration, lifecycle          |
| `trainer_rank{N}_bundle{M}` | `Trainer` (trainer.py:103)                 | Training on each GPU              |
| `RolloutManager`            | `RolloutManager` (rollout_manager.py)      | Rollout coordination              |
| `TaskCoordinator`           | `TaskCoordinator` (task_coordinator.py:64) | Stop signals, failure propagation |
| `MetricWorker`              | `MetricWorker` (metrics.py)                | Distributed metric aggregation    |

Use the Ray Dashboard "Actors" tab to see each actor's status, resource usage, and recent logs.

## Multi-Turn Trajectory Inspection

!!! tip "Key Insight"
    For multi-turn agentic workflows, most training bugs come from incorrect `response_mask` or malformed trajectories. Always dump and inspect a few samples before trusting reward or loss numbers.

### Dumping Generated Trajectories

`NaiveExecutor` (naive_executor.py:37) manages the trajectory generation pipeline. To inspect trajectories, enable debug logging and save samples:

```python
# In your custom flow or debugging script:
import torch

# Inside NaiveExecutor.generate(), after _post_process:
# Uncomment the debug block at naive_executor.py:414-417
# samples = await asyncio.gather(*tasks)
# batch = Samples2Dict(samples=samples)
# torch.save(batch, f"save_dict/{os.environ.get('RANK')}_batch.pt")
```

Then load and inspect:

```python
import torch

batch = torch.load("save_dict/0_batch.pt")
print("Keys:", list(batch.keys()))
print("input_ids shape:", batch["input_ids"].shape)
print("response_mask shape:", batch["response_mask"].shape)
print("attention_mask shape:", batch["attention_mask"].shape)
```

### Verifying loss_mask Correctness

For multi-turn trajectories, the `response_mask` determines which tokens contribute to the loss. Expected patterns:

| Scenario               | Prompt Region | Assistant Turn 1 | Tool Response | Assistant Turn 2          |
| ---------------------- | ------------- | ---------------- | ------------- | ------------------------- |
| Standard multi-turn    | All zeros     | All ones         | All zeros     | All ones                  |
| `train_on_prompt=True` | All ones      | All ones         | All zeros     | All ones                  |
| `mask_history=True`    | All zeros     | All zeros        | All zeros     | All ones (last turn only) |

```python
# Verify mask pattern
mask = batch["response_mask"][0]
print("Non-zero positions:", mask.nonzero().squeeze().tolist())
print("Sum (trainable tokens):", mask.sum().item())
```

### Token-Level Log Prob Inspection

The `rollout_log_prob` field in each sample stores per-token log probabilities from the rollout engine. Compare with training-time log probs to detect off-policy drift:

```python
import numpy as np

# rollout_log_prob is padded to max_response_length in _post_process
rollout_lp = batch["rollout_log_prob"][0]
print("Log prob range:", rollout_lp.min(), "to", rollout_lp.max())
print("Mean log prob (non-pad):", rollout_lp[batch["response_mask"][0] > 0].mean())
```

## Reward Computation Debugging

!!! tip "Key Insight"
    Reward issues are the single most common cause of training failure. Always test your reward function in isolation before running a full training job.

### Reward Function Dispatch

The reward dispatch logic lives in `default_compute_score` (reward_score/\_\_init\_\_.py:17). It routes based on the `data_source` field:

| `data_source` Value                      | Reward Module                          |
| ---------------------------------------- | -------------------------------------- |
| `openai/gsm8k`                           | `reward_score/gsm8k.py`                |
| `lighteval/MATH`, `AIME2024`, `AIME2025` | `reward_score/math.py`                 |
| `math_dapo`, `aime*`                     | `reward_score/math_dapo.py`            |
| `hiyouga/geometry3k`                     | `reward_score/geo3k.py`                |
| `mm_eureka`                              | `reward_score/mm_eureka.py`            |
| `searchR1_*`                             | `reward_score/search_r1_like_qa_em.py` |

For custom reward functions, use the `custom_reward_function.path` config field. The function is loaded in `NaiveExecutor.__init__` (naive_executor.py:79).

### Testing Reward Functions in Isolation

```python
from siirl.utils.reward_score import default_compute_score

# Test with known input/output
score = default_compute_score(
    data_source="openai/gsm8k",
    solution_str="The answer is 42.",
    ground_truth="42",
)
print(f"Score: {score}")  # Expected: 1.0
```

### Common Reward Issues

| Symptom               | Cause                                          | Fix                                                           |
| --------------------- | ---------------------------------------------- | ------------------------------------------------------------- |
| Reward always 0       | `data_source` not matching any dispatch branch | Check `data.reward_fn_key` and dataset's `data_source` column |
| Reward is NaN         | Division by zero in reward function            | Add NaN guard in custom reward                                |
| Wrong reward scale    | Reward returns bool instead of float           | Ensure `float(res)` conversion                                |
| `NotImplementedError` | Unknown `data_source` value                    | Add dispatch branch or use `custom_reward_function.path`      |

## Weight Sync Debugging

!!! tip "Key Insight"
    Weight sync failures typically manifest as stale rollout behavior (reward plateaus) or NCCL timeouts. The key parameter is `param_sync_rpc_timeout_s` (default: `120`).

### ParamSyncDistributed vs ParamSyncColocated

The sync strategy is selected automatically based on `trainer.colocate`:

| Mode                         | Class                                         | Mechanism                                                                 |
| ---------------------------- | --------------------------------------------- | ------------------------------------------------------------------------- |
| Separated (`colocate=False`) | `ParamSyncDistributed` (update_weight.py:145) | NCCL process group between trainer and rollout GPUs                       |
| Colocated (`colocate=True`)  | `ParamSyncColocated` (update_weight.py:452)   | CUDA IPC handles via `FlattenedTensorBucket`, topology-aware lane routing |

### Verifying Weight Sync

The `weight_version` counter increments after each successful sync. Check it on both sides:

```python
# On trainer side (from logs)
# Look for: "[ParamSyncDistributed] sync_total_ms=... sync_bucket_count=..."

# On rollout side
# Look for: "engine._weight_version" in rollout worker logs
```

### Timeout Diagnosis

=== "Separated Mode"

    ```yaml
    trainer:
      param_sync_rpc_timeout_s: 120  # Default: 120s
    ```

    If you see `NCCL timeout` or `ProcessGroup timeout`:

    1. Check network connectivity between trainer and rollout nodes
    2. Set `NCCL_DEBUG=INFO` for detailed NCCL diagnostics
    3. Increase `param_sync_rpc_timeout_s` for large models

=== "Colocated Mode"

    ```yaml
    trainer:
      colocate_timeout_s: 60              # Default: 60s, Ray-level timeout
      colocate_flattened_fail_fast: true   # Default: true, no fallback on bucket sync failure
    ```

    If colocated sync fails:

    1. Check GPU memory: both trainer and rollout share the same GPU
    2. Reduce `rollout.gpu_memory_utilization` (clamped to 0.45 in colocate mode)
    3. Enable `actor.megatron.param_offload` (auto-forced in colocate mode)

## Training Loss Debugging

!!! tip "Key Insight"
    Loss not decreasing after 50+ steps is almost always a data or reward issue, not a hyperparameter issue. Check reward distribution first.

### Loss Not Decreasing Checklist

1. **Verify rewards are non-zero**: Check `data/reward_mean` metric in wandb or console
2. **Verify response_mask is correct**: Non-zero mask count should match generated token count
3. **Check advantage normalization**: `norm_adv_by_std_in_grpo=True` (default) can cause zero advantages if all rewards are identical
4. **Check learning rate**: Default `lr=1e-6` may be too low for some tasks
5. **Check gradient clipping**: Default `clip_grad=1.0` in `OptimizerArguments`

### KL Divergence Explosion

KL divergence measures drift from the reference model. If KL grows rapidly:

```yaml
actor_ref:
  actor:
    use_kl_loss: true            # Default: false
    kl_loss_coef: 0.001          # Default: 0.001, increase to constrain drift
    kl_loss_type: "low_var_kl"   # Default: "low_var_kl"
    clip_ratio: 0.2              # Default: 0.2, tighter clipping reduces drift
```

### Gradient Norm Anomalies

Enable memory profiling to track gradient norms per step:

```bash
export SIIRL_MEMORY_STEP_PROFILE=1
```

The `MemoryProfiler` (memory_profiler.py:424) logs peak memory each step when enabled. For gradient-specific debugging, use the `memory_trace` context manager:

```python
from siirl.utils.logger.memory_profiler import memory_trace

with memory_trace("gradient_computation"):
    loss.backward()
```

### loss_agg_mode Effects

The `loss_agg_mode` parameter in `ActorArguments` (model_args.py:132) controls how per-token losses are aggregated:

| Mode                   | Behavior                                             | When to Use                                    |
| ---------------------- | ---------------------------------------------------- | ---------------------------------------------- |
| `token-mean` (default) | Mean over all valid tokens                           | Standard training                              |
| `seq-mean-token-sum`   | Sum tokens per sequence, then mean across sequences  | When sequence lengths vary widely              |
| `seq-mean-token-mean`  | Mean tokens per sequence, then mean across sequences | Equal weight per sequence regardless of length |

## Common Error Messages

| Error                                  | Location                                    | Cause                              | Fix                                                                             |
| -------------------------------------- | ------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------- |
| `NCCL timeout`                         | `TrainerGroup.init_actors`                  | Network issue between nodes        | Set `NCCL_DEBUG=INFO`, check firewall rules, verify `MASTER_ADDR`/`MASTER_PORT` |
| `CUDA out of memory`                   | `Trainer.train_step`                        | Batch too large for GPU            | See [Handling OOM](handling_oom.md)                                             |
| `Ray actor died unexpectedly`          | Any actor                                   | OOM kill or unhandled exception    | Check `/tmp/ray/session_latest/logs/` and system `dmesg` for OOM kills          |
| `Missing TRAIN_MASTER_PORT`            | `TrainerGroup._resolve_master_endpoint`     | Multi-node without port config     | Set `TRAIN_MASTER_PORT` environment variable                                    |
| `Param sync is unhealthy`              | `ParamSyncDistributed.update_weights_mixed` | Previous sync group setup failed   | Restart training; check NCCL connectivity                                       |
| `Trainer distributed layout mismatch`  | `TrainerGroup._validate_distributed_setup`  | Resource allocation inconsistency  | Verify `trainer.actor_gpus` and `trainer.nnodes` match cluster                  |
| `NotImplementedError: Reward function` | `reward_score/__init__.py`                  | Unknown `data_source` value        | Add custom reward via `custom_reward_function.path`                             |
| `Training failed: [source] reason`     | `MainRunner._wait_for_completion`           | Any component reported failure     | Check `TaskCoordinator` events via logs for root cause                          |
| `bootstrap-phase offload failure`      | `MainRunner.run` (colocate mode)            | Rollout offload failed during init | Reduce `rollout.gpu_memory_utilization`, check GPU memory                       |
| `Timeout waiting for batch`            | `Trainer.train` loop                        | Rollout producing data too slowly  | Increase `trainer.async_factor` (default: `1`), add rollout GPUs                |

## Environment Variables Reference

| Variable                      | Default       | Purpose                                                |
| ----------------------------- | ------------- | ------------------------------------------------------ |
| `SIIRL_LOG_DIRECTORY`         | `siirl_logs`  | Application log output directory                       |
| `SIIRL_MEMORY_DEBUG`          | `0`           | Enable detailed memory logging (`1` to enable)         |
| `SIIRL_MEMORY_PROFILE`        | `0`           | Export memory snapshots for first step (`1` to enable) |
| `SIIRL_MEMORY_STEP_PROFILE`   | `0`           | Log peak memory per step (`1` to enable)               |
| `SIIRL_VERBOSE_VALIDATE_LOGS` | `0`           | Verbose validation timing logs (`1` to enable)         |
| `LOGURU_LEVEL`                | `INFO`        | Log level for loguru (`DEBUG` for maximum detail)      |
| `RAY_DEDUP_LOGS`              | (Ray default) | Set to `0` to disable Ray log deduplication            |
| `NCCL_DEBUG`                  | `WARN`        | NCCL debug level (`INFO` for connection diagnostics)   |
| `NCCL_CUMEM_ENABLE`           | `0`           | NCCL CUDA managed memory                               |
