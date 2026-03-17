# Performance Profiling

*Identify bottlenecks in the async training pipeline with built-in timing metrics, Ray tracing, and GPU profiling tools.*

## Built-in Timing Metrics

!!! tip "Key Takeaway"
    Every `Sample` records a `timing_info` dictionary with five fields: `rollout_start_at`, `rollout_end_at`, `rollout_duration`, `generation_duration`, and `reward_duration`. These are automatically aggregated across the batch and logged to WandB as `perf/delta_time/*` metrics.

The `Sample` dataclass (`siirl/data_coordinator/sample.py:21`) carries per-sample timing information populated during rollout:

| Field                 | Type    | Description                                              |
| --------------------- | ------- | -------------------------------------------------------- |
| `rollout_start_at`    | `float` | Unix timestamp when rollout began for this sample        |
| `rollout_end_at`      | `float` | Unix timestamp when rollout completed                    |
| `rollout_duration`    | `float` | Total wall-clock time for the entire rollout (seconds)   |
| `generation_duration` | `float` | Time spent in LLM inference / token generation (seconds) |
| `reward_duration`     | `float` | Time spent computing rewards (seconds)                   |

After a training batch is assembled, `extract_rollout_timing_metrics()` (`siirl/utils/metrics/metric_utils.py:322`) aggregates these per-sample values into batch-level metrics:

| Logged Metric                           | Meaning                                                  |
| --------------------------------------- | -------------------------------------------------------- |
| `perf/delta_time/rollout_per_sample`    | Average rollout duration across all samples in the batch |
| `perf/delta_time/generation_per_sample` | Average generation duration                              |
| `perf/delta_time/reward_per_sample`     | Average reward computation duration                      |

Additionally, `compute_timing_metrics()` (`siirl/utils/metrics/metric_utils.py:179`) computes per-token timing for training stages:

| Logged Metric                      | Meaning                                   |
| ---------------------------------- | ----------------------------------------- |
| `perf/delta_time/gen`              | Generation time (seconds)                 |
| `perf/delta_time/ref`              | Reference model log-prob computation time |
| `perf/delta_time/update_actor`     | Actor update time                         |
| `perf/delta_time/update_critic`    | Critic update time (PPO only)             |
| `timing_per_token_ms/gen`          | Generation time per response token (ms)   |
| `timing_per_token_ms/update_actor` | Actor update time per total token (ms)    |

Throughput metrics from `compute_throughput_metrics()` (`siirl/utils/metrics/metric_utils.py:229`):

| Logged Metric           | Meaning                                         |
| ----------------------- | ----------------------------------------------- |
| `perf/total_num_tokens` | Total tokens processed in the batch             |
| `perf/time_per_step`    | Wall-clock time for one training step (seconds) |

### Reading Timing Metrics in WandB

All `perf/delta_time/*` metrics are plotted automatically. To diagnose the pipeline:

1. **Compare `generation_per_sample` vs `time_per_step`**: If generation is much larger, rollout is the bottleneck.
2. **Check `reward_per_sample`**: If reward is a significant fraction of `rollout_per_sample`, your reward function needs optimisation (e.g., batching, caching).
3. **Watch `timing_per_token_ms/update_actor`**: This should remain stable; spikes indicate memory pressure or communication stalls.

## Ray Timeline

!!! tip "Key Takeaway"
    Ray's built-in timeline tool captures actor scheduling, task execution, and idle gaps across the cluster. Export it in Chrome trace format and inspect it in `chrome://tracing`.

```bash
# Capture a 30-second timeline from a running Ray cluster
ray timeline --format chrome-trace --output timeline.json
```

Open `timeline.json` in `chrome://tracing` (or [Perfetto UI](https://ui.perfetto.dev/)). Look for:

- **Long gaps between actor calls**: Indicates the `DataCoordinator` is waiting for data. Increase `trainer.async_factor` to overlap more rollout batches.
- **Uneven actor scheduling**: One rollout worker finishing much later than others suggests data skew. Check prompt length distribution.
- **Frequent `param_sync` bars**: Weight synchronisation is blocking rollout. Consider colocated mode (`trainer.colocate: true`) to eliminate network transfer.

## PyTorch Profiler

!!! tip "Key Takeaway"
    Use `torch.profiler` to capture CUDA kernel-level traces for training steps. Profile rank 0 only to avoid flooding storage.

```python
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=2, warmup=2, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(training_loop):
        train_step(batch)
        prof.step()
```

Key areas to inspect in the resulting trace:

- **NCCL AllReduce / ReduceScatter**: Communication overhead in distributed training. Should be < 30% of step time.
- **CUDA kernel gaps**: Idle time between kernels indicates CPU bottleneck (Python overhead, data preprocessing).
- **Memory allocations**: Large spikes suggest dynamic tensor creation inside the training loop.

Only profile on rank 0 to avoid I/O contention:

```python
import torch.distributed as dist

if dist.get_rank() == 0:
    # attach profiler
    ...
```

## GPU Utilization Monitoring

!!! tip "Key Takeaway"
    Target > 80% GPU SM utilization during training steps and > 60% during rollout. Persistent low utilization typically points to data starvation or excessive synchronisation.

### Real-time monitoring with nvidia-smi

```bash
# Sample every 1 second, show SM utilization and memory
nvidia-smi dmon -s u -d 1
```

| Column              | Healthy Range  | Red Flag                                   |
| ------------------- | -------------- | ------------------------------------------ |
| SM % (training GPU) | > 80%          | < 50% sustained — waiting for data or comm |
| SM % (rollout GPU)  | > 60%          | < 30% sustained — batch too small          |
| FB Used (MB)        | < 95% of total | > 95% — OOM imminent, enable offloading    |

### DCGM integration

For cluster-wide monitoring, use NVIDIA DCGM:

```bash
# Start DCGM and enable profiling metrics
dcgmi profile --pause
dcgmi profile --resume
dcgmi dmon -e 1001,1002,1003,1004 -d 1000
```

Metric IDs: `1001` = SM active, `1002` = SM occupancy, `1003` = tensor active, `1004` = DRAM active.

## Identifying Common Bottlenecks

| Symptom                      | Profiling Indicator                                   | Likely Cause                         | Fix                                                                               |
| ---------------------------- | ----------------------------------------------------- | ------------------------------------ | --------------------------------------------------------------------------------- |
| Low GPU util during training | Long idle gaps in Ray timeline between training steps | Waiting for rollout data             | Increase `trainer.async_factor` (default: `1`)                                    |
| Low GPU util during rollout  | SGLang server shows idle periods                      | Request batch too small              | Increase `rollout.train_server_concurrency` (default: `256`)                      |
| High communication time      | NCCL kernels > 30% of step in PyTorch profiler        | Tensor/pipeline parallelism too high | Reduce `trainer.tensor_model_parallel_size`                                       |
| Reward computation slow      | `reward_per_sample` >> `generation_per_sample`        | Expensive reward function            | Batch reward calls; use async reward; cache results                               |
| Weight sync stalls           | `param_sync` bars dominate Ray timeline               | Network bandwidth saturated          | Switch to colocated mode; tune `trainer.param_sync_buffer_size` (default: 512 MB) |
| OOM during training          | FB usage at 100%, then crash                          | Micro-batch too large                | Reduce `actor.ppo_micro_batch_size_per_gpu`; enable `megatron.param_offload`      |
| Uneven step times            | `time_per_step` has high variance across steps        | Variable-length sequences            | Enable `actor.use_dynamic_batch: true` with `actor.max_tokens_per_gpu`            |

### Quick Diagnostic Checklist

```bash
# 1. Check if rollout or training is the bottleneck
#    (compare generation_per_sample vs time_per_step in WandB)

# 2. Check GPU utilization on both sides
nvidia-smi dmon -s u -d 1

# 3. Check Ray actor scheduling
ray timeline --format chrome-trace --output timeline.json

# 4. Profile a few training steps on rank 0
#    (see PyTorch Profiler section above)
```
