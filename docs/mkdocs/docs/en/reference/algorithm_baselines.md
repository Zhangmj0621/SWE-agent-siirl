# Algorithm Baselines

> Reference performance numbers for siirl-agentic on standard benchmarks.

!!! warning "Approximate Results Only"
    These numbers are **approximate ranges** based on published literature (DeepSeek-R1, Qwen2.5/Qwen3 technical reports) and internal testing. Your results may vary depending on hyperparameters, hardware, data preprocessing, and random seed. We encourage you to report your results to help improve these baselines.

## Math Reasoning

!!! tip "Key Takeaway"
    Math reasoning is the most well-studied RL post-training benchmark. GRPO is the recommended starting point -- it requires no critic model and achieves competitive results with less GPU memory.

| Model        | Algorithm | Dataset    | Metric   | Approximate Range | GPUs       | Notes                                           |
| ------------ | --------- | ---------- | -------- | ----------------- | ---------- | ----------------------------------------------- |
| Qwen3-8B     | GRPO      | DeepScaleR | Accuracy | ~70-80%           | 8xA100-80G | `rollout.n=8`, `actor_ref.actor.optim.lr=1e-6`  |
| Qwen3-8B     | PPO       | DeepScaleR | Accuracy | ~68-78%           | 8xA100-80G | with critic (`critic.optim.lr=5e-6`)            |
| Qwen2.5-7B   | GRPO      | GSM8K      | Accuracy | ~75-82%           | 8xA100-80G | `rollout.n=16`, `actor_ref.actor.optim.lr=1e-6` |
| Qwen2.5-7B   | PPO       | GSM8K      | Accuracy | ~72-80%           | 8xA100-80G | `trainer.critic_warmup=10` recommended          |
| Qwen2.5-1.5B | GRPO      | GSM8K      | Accuracy | ~55-65%           | 4xA100-80G | smaller model, faster iteration                 |
| Qwen2.5-7B   | GRPO      | MATH       | Accuracy | ~45-55%           | 8xA100-80G | longer chains, `data.max_response_length=4096`  |
| Qwen2.5-7B   | PPO       | MATH       | Accuracy | ~42-52%           | 8xA100-80G |                                                 |

## Code Generation

!!! tip "Key Takeaway"
    Code benchmarks benefit from code-specialized base models and sandbox-based execution rewards. Use longer `data.max_response_length` to accommodate code output.

| Model            | Algorithm | Dataset      | Metric    | Approximate Range | GPUs        | Notes                    |
| ---------------- | --------- | ------------ | --------- | ----------------- | ----------- | ------------------------ |
| Qwen2.5-7B       | GRPO      | CodeContests | Pass Rate | ~15-25%           | 16xA100-80G | sandbox execution reward |
| Qwen2.5-7B-Coder | GRPO      | APPS         | Pass Rate | ~30-45%           | 8xA100-80G  | code-specific base model |

## Software Engineering (Multi-Turn)

!!! tip "Key Takeaway"
    SWE benchmarks test the agent's ability to make multi-turn tool calls to fix real bugs. These require the AIO tool infrastructure with `rollout.multiturn.env_type=tool_env` and `rollout.multiturn.max_env_turns > 1`.

| Model    | Algorithm | Dataset        | Metric       | Approximate Range | GPUs       | Notes                      |
| -------- | --------- | -------------- | ------------ | ----------------- | ---------- | -------------------------- |
| Qwen3-8B | GRPO      | SWE-bench Lite | Resolve Rate | ~15-25%           | 8xA100-80G | multi-turn, Docker sandbox |

## Key Hyperparameters per Benchmark

!!! tip "Key Takeaway"
    Start with these recommended settings. The parameter names below correspond exactly to CLI arguments passed to `python3 -m siirl.async_train`.

| Benchmark    | Recommended Algorithm | `rollout.n` | `actor_ref.actor.optim.lr` | `data.max_response_length` | Key Setting                                                                    |
| ------------ | --------------------- | ----------- | -------------------------- | -------------------------- | ------------------------------------------------------------------------------ |
| GSM8K        | GRPO                  | 16          | `1e-6`                     | 2048                       | `actor_ref.algorithm.norm_adv_by_std_in_grpo=True`                             |
| MATH         | GRPO                  | 16          | `5e-7`                     | 4096                       | longer reasoning chains                                                        |
| DeepScaleR   | GRPO                  | 8           | `1e-6`                     | 4096                       | `data.train_batch_size=512`                                                    |
| CodeContests | GRPO                  | 8           | `1e-6`                     | 4096                       | sandbox reward function                                                        |
| SWE-bench    | GRPO                  | 8           | `1e-6`                     | 4096                       | `rollout.multiturn.max_env_turns=2`, `rollout.multiturn.max_assistant_turns=2` |

### GRPO-Specific Settings

```bash
# GRPO advantage estimation (from AlgorithmArguments)
actor_ref.algorithm.adv_estimator=grpo
actor_ref.algorithm.gamma=1.0            # discount factor (default: 1.0)
actor_ref.algorithm.lam=1.0              # GAE lambda (default: 1.0)
actor_ref.algorithm.norm_adv_by_std_in_grpo=True  # standard GRPO normalization

# Actor training (from ActorArguments)
actor_ref.actor.clip_ratio=0.2           # PPO clipping ratio (default: 0.2)
actor_ref.actor.ppo_epochs=1             # gradient updates per batch (default: 1)
actor_ref.actor.use_kl_loss=True         # KL divergence regularization
actor_ref.actor.kl_loss_coef=0.01        # KL loss weight
actor_ref.actor.kl_loss_type=low_var_kl  # low-variance KL estimator
```

### PPO-Specific Settings

```bash
# PPO requires a critic model in addition to GRPO settings
actor_ref.algorithm.adv_estimator=ppo

# Critic configuration (from CriticArguments)
critic.model.path=/path/to/model         # typically same as actor model
critic.optim.lr=5e-6                     # critic lr, usually 5x actor lr
critic.ppo_epochs=1                      # critic update epochs (default: 1)
critic.cliprange_value=0.5               # value function clipping (default: 0.5)
critic.ppo_mini_batch_size=256           # critic mini-batch size
critic.ppo_micro_batch_size_per_gpu=8    # per-GPU micro-batch size
```

### Dr. GRPO Variant

To use the Dr. GRPO variant ([arXiv:2503.20783](https://arxiv.org/abs/2503.20783)) which skips standard deviation normalization:

```bash
actor_ref.algorithm.norm_adv_by_std_in_grpo=False
```

## Reproducing Results

For step-by-step reproduction guides, see:

- [DeepScaleR GRPO Tutorial](../tutorials/deepscaler_grpo.md)
- [SWE Agent Training Tutorial](../tutorials/swe_agent_training.md)

The provided example scripts are the best starting point:

=== "GRPO (Separated Mode)"

    ```bash
    # 8 GPUs: 4 train + 4 rollout
    bash examples/grpo_train/run_qwen3_8b_separated.sh
    ```

=== "GRPO (Colocate Mode)"

    ```bash
    # 8 GPUs shared between train and rollout
    bash examples/grpo_train/run_qwen3_8b_colocate.sh
    ```

=== "PPO (Separated Mode)"

    ```bash
    # 8 GPUs: 4 train + 4 rollout (includes critic)
    bash examples/ppo_train/run_qwen3_8b_separated.sh
    ```

=== "AIO Multi-Turn"

    ```bash
    # 8 GPUs: 2 train + 6 rollout, with tool environment
    bash examples/AIO/run_qwen3_8b.sh
    ```

## Hardware Requirements

!!! tip "Key Takeaway"
    The minimum viable setup is 4 GPUs in separated mode or 4 GPUs in colocate mode. Colocate mode uses `param_offload` to swap between training and rollout on the same GPUs.

| Setup               | GPU Memory | Minimum GPUs            | Recommended | Mode                     |
| ------------------- | ---------- | ----------------------- | ----------- | ------------------------ |
| 8B separated        | 80 GB/GPU  | 4 (2 train + 2 rollout) | 8 (4+4)     | `trainer.colocate=False` |
| 8B colocate         | 80 GB/GPU  | 4                       | 8           | `trainer.colocate=True`  |
| 1.5B-1.7B separated | 40 GB/GPU  | 2 (1+1)                 | 4 (2+2)     | `trainer.colocate=False` |
| 1.5B-1.7B colocate  | 40 GB/GPU  | 2                       | 4           | `trainer.colocate=True`  |

### Memory Optimization Tips

```yaml title="Quick memory reduction configs"
# Level 1: Parameter offload (~30-40% GPU savings)
actor_ref.actor.megatron.param_offload: true
actor_ref.ref.megatron.param_offload: true       # Also offload reference model

# Level 2: Optimizer offload (~40-50% more savings)
actor_ref.actor.megatron.optimizer_offload: true

# Level 3: Dynamic batching (prevents OOM spikes on long sequences)
actor_ref.actor.use_dynamic_batch: true
actor_ref.actor.max_tokens_per_gpu: 16384

# Level 4: Reduce SGLang VRAM reservation
rollout.gpu_memory_utilization: 0.5               # Default 0.9, lower = more headroom
```

## Comparison Notes

!!! tip "Key Takeaway"
    siirl-agentic's core differentiator is native async multi-turn agent training with tool interaction, powered by the AIO (AgentFlow + Infrastructure + Orchestration) architecture.

| Feature          | siirl-agentic                         | veRL   | OpenRLHF  |
| ---------------- | ------------------------------------- | ------ | --------- |
| Async multi-turn | Native                                | Manual | No        |
| Tool interaction | AIO 3-layer                           | Custom | No        |
| Rollout engine   | SGLang                                | vLLM   | vLLM      |
| Training backend | Megatron-LM                           | FSDP   | DeepSpeed |
| Colocate mode    | Yes (offload-based)                   | Yes    | No        |
| Dr. GRPO support | Yes (`norm_adv_by_std_in_grpo=False`) | No     | No        |
| Dynamic batching | Yes (token-based)                     | No     | No        |
