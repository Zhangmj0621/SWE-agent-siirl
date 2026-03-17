# Best Practices

*Practical guidelines for configuration, monitoring, debugging, and production deployment of siirl-agentic training runs.*

## Configuration Best Practices

| Before (common mistake)               | After (recommended)                          | Why                                                    |
| ------------------------------------- | -------------------------------------------- | ------------------------------------------------------ |
| Setting all parameters at once        | Start with defaults, override 3-5 key params | Fewer variables to debug                               |
| `rollout.temperature=0.0`             | `rollout.temperature=0.7`                    | Zero temp gives no diversity for GRPO                  |
| `trainer.total_epochs=100`            | `trainer.total_epochs=30`, then evaluate     | Saves GPU hours; most gains in first 30                |
| `trainer.colocate=true` for 70B model | Use offload (separated) mode first           | Prove stability before adding colocate overhead        |
| `data.max_response_length=512`        | `data.max_response_length=2048` or higher    | This is characters, not tokens; 512 chars ≈ 128 tokens |

## Configuration Guidelines

### Start Simple, Scale Up

Begin with the smallest viable configuration and scale incrementally:

``` yaml
# Step 1: Verify single-node, small model, GRPO
actor_ref:
  model:
    path: Qwen/Qwen2.5-1.5B-Instruct
  algorithm:
    adv_estimator: grpo        # Simpler than PPO (no critic)

rollout:
  n: 4                         # 4 samples per prompt for group-relative advantage

trainer:
  colocate: false
  actor_gpus: 2
  rollout_gpus: 6
  total_epochs: 1
  save_freq: 10
```

Once this works end-to-end, progressively increase model size, batch size, and number of GPUs.

### Batch Size and Learning Rate

| Model Size | `ppo_micro_batch_size_per_gpu` | Learning Rate | Notes                          |
| ---------- | ------------------------------ | ------------- | ------------------------------ |
| 1.5B–3B    | 4–8                            | 1e-6 to 5e-6  | Can use larger micro-batches   |
| 7B–8B      | 2–4                            | 5e-7 to 2e-6  | Monitor GPU memory             |
| 13B–14B    | 1–2                            | 3e-7 to 1e-6  | May need gradient accumulation |
| 70B+       | 1                              | 1e-7 to 5e-7  | Multi-node required            |

!!! tip "Learning Rate Warmup"
    Always use warmup for the first 5–10% of training steps. Jumping to a high learning rate immediately can destabilize policy updates:
    ```yaml
    actor_ref:
      actor:
        optim:
          lr: 1e-6
          warmup_steps: 50    # Warm up for 50 steps
    ```

### Data Configuration

All production examples consistently use these data settings:

``` yaml
data:
  train_batch_size: 512         # Default: 1024, examples use 512
  shuffle: false                # RL training: deterministic data order
  truncation: error             # Fail on overlong prompts, don't silently truncate
  filter_overlong_prompts: true # Discard prompts exceeding max_prompt_length
  max_prompt_length: 2048
  max_response_length: 4096     # Default: 512, increase for longer tasks
```

!!! warning "Response Length for Multi-Turn"
    For multi-turn agentic tasks, `max_response_length` must cover ALL turns (assistant + tool responses). Monitor `rollout/response_length_mean` — if it approaches `max_response_length`, trajectories are being truncated and you're losing training signal. SWE-style tasks typically need 8192–16384.

### Dynamic Batching

All production examples enable dynamic batching for efficient variable-length sequence packing:

``` yaml
actor_ref:
  actor:
    use_dynamic_batch: true        # Token-based batching, not fixed batch (default: false)
    max_tokens_per_gpu: 16384      # Max tokens per GPU per micro-batch
    use_workload_balance: true     # FLOPs-based load balancing across GPUs (default: true)
```

## Multi-Turn Agentic Training

### Tool Configuration

``` yaml
rollout:
  multiturn:
    max_env_turns: 10              # Max tool interaction rounds (default: 1)
    max_assistant_turns: 20        # Max total assistant responses (default: 1)
    max_parallel_calls: 1          # Sequential by default (default: 1)
    max_env_response_length: 256   # Characters, not tokens (default: 256)
    env_response_truncate_side: middle  # Keep start+end of long outputs (default: middle)
```

**Key rules:**

1. **Start with `max_parallel_calls: 1`** — Parallel tool calls can cause ordering issues. Only increase after verifying correctness. Tool calls beyond this limit are **silently discarded** (not re-queued).
2. **Use `middle` truncation** — For code execution outputs, the beginning (command) and end (result/error) are most informative. Also supports `left` (keep start) and `right` (keep end).
3. **Set reasonable turn limits** — Too many turns waste compute. Monitor `rollout/env_turns_mean` to find the sweet spot. The defaults of 1/1 are for single-turn tasks.

### Reward Function Design

Custom reward functions must follow this signature (loaded via `custom_reward_function.path`):

``` python
def reward_function(data_source: str, solution_str: str, ground_truth: str, **kwargs) -> float:
    """
    Args:
        data_source: Dataset source identifier (from data.reward_fn_key column)
        solution_str: Decoded model output string
        ground_truth: Expected answer from dataset
    Returns:
        Scalar reward value
    """
    # Binary: did the agent solve the task?
    task_reward = 1.0 if is_correct(solution_str, ground_truth) else 0.0
    return task_reward
```

Good reward functions for agentic tasks should:

- **Be binary or sparse** — GRPO works best with clear pass/fail signals (e.g., test suite pass rate)
- **Avoid dense shaping** — Per-step rewards can create reward hacking in multi-turn settings
- **Include format penalties** — Penalize malformed tool calls to teach proper formatting early

### GRPO vs PPO Selection

| Scenario                        | Recommended | Why                                      |
| ------------------------------- | ----------- | ---------------------------------------- |
| Binary rewards (pass/fail)      | GRPO        | No critic needed, simpler training       |
| Dense rewards (per-step scores) | PPO         | Critic enables credit assignment via GAE |
| Quick prototyping               | GRPO        | Fewer hyperparameters to tune            |
| Fine-grained reward shaping     | PPO         | GAE handles temporal credit assignment   |

**Dr.GRPO variant:** Set `algorithm.norm_adv_by_std_in_grpo: false` to skip standard deviation normalization. This prevents advantage magnitude scaling that can hurt training stability with absolute-scale rewards.

## Memory Management

### Preventing OOM

The most common failure mode is CUDA OOM during training. Mitigation strategies in priority order:

1. **Reduce micro-batch size**: `actor_ref.actor.ppo_micro_batch_size_per_gpu: 1`
2. **Enable dynamic batching**: `actor_ref.actor.use_dynamic_batch: true`
3. **Enable parameter offloading**: `actor_ref.actor.megatron.param_offload: true`
4. **Reduce rollout memory**: `rollout.gpu_memory_utilization: 0.5` (default)
5. **Use gradient checkpointing**: Enabled by default in Megatron backend

!!! warning "Colocated Mode Memory"
    In colocated mode, `gpu_memory_utilization` is automatically clamped to 0.45. Parameter offloading (`param_offload: true`) is also forced on. If you still get OOM, reduce `ppo_micro_batch_size_per_gpu` or switch to separated mode.

### Checkpoint Disk Space

``` yaml
trainer:
  save_freq: 10
  max_actor_ckpt_to_keep: 5      # Don't keep 100 (default)
  max_critic_ckpt_to_keep: 5
```

A 7B model checkpoint is ~14GB. With `max_actor_ckpt_to_keep=100` (default), that's 1.4TB. Set this to 3–5 for production runs.

## Monitoring

### Essential Metrics to Watch

| Metric                        | Healthy Range         | Action if Out of Range             |
| ----------------------------- | --------------------- | ---------------------------------- |
| `reward/mean`                 | Trending upward       | Check reward function, increase lr |
| `kl/mean`                     | < 15                  | Reduce lr or add KL penalty        |
| `actor/clip_ratio`            | 0.1–0.3               | Adjust `clip_ratio` hyperparameter |
| `rollout/generation_duration` | Stable or decreasing  | Add rollout GPUs if too high       |
| `rollout/env_duration`        | < generation_duration | Scale AIO if too high              |

### Enable WandB Early

``` yaml
trainer:
  logger: ["console", "wandb"]    # This is actually the default
  project_name: siirl_experiments
  experiment_name: grpo_qwen2.5_7b_swe
```

Always enable WandB from the start. Console-only logging makes it hard to diagnose issues retrospectively. The default `logger` already includes both console and wandb.

### KL Regularization

All production examples use this consistent KL configuration:

``` yaml
actor_ref:
  actor:
    use_kl_loss: true              # Enable KL penalty (default: false)
    kl_loss_coef: 0.01             # KL loss coefficient (default: 0.001)
    kl_loss_type: low_var_kl       # Variance-reduced KL (default: low_var_kl)
  algorithm:
    kl_penalty: kl                 # KL penalty type (default: kl)
```

The `low_var_kl` type computes `exp(kl) - kl - 1`, which is more numerically stable than raw KL divergence and doesn't explode at large KL values.

## Production Deployment

### Pre-Flight Checklist

Before launching a long training run:

- [ ] Run 1 epoch with `save_freq: 1` to verify checkpoint saving works
- [ ] Verify `max_actor_ckpt_to_keep` won't fill disk
- [ ] Test resume: stop and restart from the saved checkpoint
- [ ] Enable WandB logging with a descriptive experiment name
- [ ] Set `val_before_train: true` (default) to get a baseline metric
- [ ] Verify tool environments are accessible (AIO proxy health check)
- [ ] Check Ray cluster: `ray status` shows expected nodes and GPUs

### Reproducibility

``` yaml
trainer:
  seed: 1                     # Training seed (default: 1)

rollout:
  temperature: 0.7            # Fixed temperature (default: 1.0)
  top_p: 0.95                 # Fixed top-p (default: 1.0)

data:
  shuffle: false              # Deterministic data order
```

!!! note
    Full reproducibility in distributed multi-turn training is not guaranteed due to non-deterministic CUDA operations and async scheduling. Seeds control data ordering but not GPU-level computation order.

### Failure Recovery

Always configure automatic checkpoint saving:

``` yaml
trainer:
  save_freq: 10               # Save every 10 steps (default: -1, disabled!)
  resume_mode: auto            # Auto-resume from latest checkpoint (default)
```

With `resume_mode: auto`, a crashed run will automatically resume from the latest checkpoint when restarted. See [Checkpoint & Resume](../guides/checkpoint_resume.md) for details.

## Common Failure Patterns

!!! tip
    These are the most frequent mistakes in production deployments. Each pattern includes symptoms, diagnosis, and fix.

### Pattern 1: Setting async_factor Too High

**Symptom:** Training metrics look good but eval performance plateaus or degrades.

**Diagnosis:** Off-policy staleness — rollout data is too old by the time it is trained on. Check the `data_version` gap in logs between the rollout batch version and the current training version.

**Fix:** Start with `trainer.async_factor=1` (synchronous, the default). Increase to 2 only if GPU utilization during rollout wait is below 50%. Never exceed 3 without also enabling off-policy correction via `trainer.off_policy_step`. With `async_factor=2` and `off_policy_step=1`, the maximum data staleness is 3 versions.

### Pattern 2: Not Monitoring KL Divergence

**Symptom:** Reward increases but model generates repetitive or degenerate outputs.

**Diagnosis:** Policy has diverged too far from the reference model. Check `kl/mean` in WandB — if it exceeds 10–15, the model is likely exploiting reward function artifacts rather than learning the intended behavior.

**Fix:** Enable KL regularization with these parameters:

``` yaml
actor_ref:
  actor:
    use_kl_loss: true           # Default: false
    kl_loss_coef: 0.01          # Default: 0.001
    kl_loss_type: low_var_kl    # Default: low_var_kl
```

Monitor `kl/mean` and adjust `kl_loss_coef` upward if KL continues to grow. The `low_var_kl` type computes `exp(kl) - kl - 1`, which is more stable than raw KL at large values.

### Pattern 3: Ignoring Off-Policy Staleness

**Symptom:** Training loss oscillates, reward is unstable even with high batch sizes.

**Diagnosis:** `trainer.off_policy_step` is set too high, training on data from many versions ago.

**Fix:** Keep `trainer.off_policy_step <= 2` (default: 0, strictly on-policy). Use `trainer.off_policy_strategy=oldest_first` to train on stale data first so it gets discarded sooner. The default strategy is `fifo`. With `async_factor=2` and `off_policy_step=1`, max data staleness is 3 versions — a good balance between GPU utilization and data freshness.

### Pattern 4: Wrong loss_agg_mode for Variable-Length Data

**Symptom:** Model generates only short responses, ignoring long-answer prompts. Or: long sequences dominate gradients and short-sequence tasks regress.

**Diagnosis:** The default `actor_ref.actor.loss_agg_mode=token-mean` averages loss over all tokens in a micro-batch. A sequence with 2000 tokens contributes 10x more gradient than one with 200 tokens, biasing the model toward short outputs (to minimize loss exposure) or letting long sequences dominate updates.

**Fix:** For agentic tasks with variable trajectory lengths, use `actor_ref.actor.loss_agg_mode=seq-mean-token-mean` to give equal weight to each sequence regardless of length. This normalizes the loss per-sequence first, then averages across sequences.

``` yaml
actor_ref:
  actor:
    loss_agg_mode: seq-mean-token-mean    # Default: token-mean
```

### Pattern 5: Tool Environment Timeout Too Short

**Symptom:** Multi-turn training has many failed episodes. `SWERolloutResult.FAILURE` rate is high in logs.

**Diagnosis:** Complex tool operations (Docker build, test suite execution, repository setup) take longer than the default timeout. The `ContainerStartArgs.container_timeout` defaults to `"2h"`, but individual command execution in `ContainerEnv.execute` defaults to 180 seconds — which may be insufficient for heavy operations like running a full test suite.

**Fix:** Increase container and command timeouts as needed:

- `ContainerStartArgs.container_timeout`: overall container lifetime (default: `"2h"`). Increase for long training runs with many turns.
- `ContainerEnv.execute` timeout: per-command timeout (default: 180s). Increase for heavy operations like `pytest` on large repositories.
- `rollout.multiturn.max_env_response_length`: truncation may cut off important output (default: 256 characters). Increase to capture full test output that the agent needs for reasoning.

Also check `ContainerStartArgs.startup_timeout` (default: 180.0s) — if containers are slow to start (pulling images, installing dependencies), increase this to avoid premature timeout during environment initialization.

## Common Pitfalls

| Pitfall                                        | Symptom                                    | Fix                                              |
| ---------------------------------------------- | ------------------------------------------ | ------------------------------------------------ |
| `max_response_length` too small (default: 512) | Truncated trajectories, flat reward        | Increase to 4096+ (8192+ for multi-turn)         |
| `save_freq: -1` (default)                      | No checkpoints saved, lost progress        | Set to 10–30                                     |
| Wrong `tool_format` in `env_kwargs`            | Tool calls not parsed, 0 reward            | Match model's expected format (hermes/gpt-oss)   |
| `rollout.n: 1` with GRPO                       | No group variance for advantage estimation | Use `rollout.n: 4` or higher (examples use 8)    |
| Missing `env_path`                             | `Error: tools_config_file is None`         | Point to tool config YAML                        |
| KL divergence explosion                        | `kl/mean` > 50, policy collapse            | Reduce lr by 2–5x, enable `use_kl_loss`          |
| Colocated + validate_reuse                     | Silently disabled with warning             | Use separated mode for validate_reuse            |
| `max_parallel_calls` too high                  | Tool calls silently discarded              | Extra calls beyond limit are dropped, not queued |

## Next steps

- [Performance Tuning](../guides/performance_tuning.md) — Go deeper on GPU utilization and throughput optimization once the basics are solid
- [Troubleshooting](troubleshooting.md) — Use this reference when the checklist items reveal a problem that needs diagnosing
- [Configuration Reference](config_reference.md) — Look up exact parameter names and defaults when configuring the items on this checklist
