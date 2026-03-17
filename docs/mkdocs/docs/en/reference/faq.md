# FAQ

*Answers to the most frequently asked questions about siirl-agentic.*

## General

!!! tip "Start here if you are evaluating siirl-agentic for the first time."

### How is siirl-agentic different from veRL / OpenRLHF?

| Capability          | siirl-agentic                                                                  | veRL             | OpenRLHF             |
| ------------------- | ------------------------------------------------------------------------------ | ---------------- | -------------------- |
| Multi-turn native   | Yes -- async state machine with `max_env_turns` / `max_assistant_turns`        | Single-turn only | Single-turn only     |
| Tool infrastructure | AIO three-layer scheduling (Proxy -> Scheduler -> Sandbox)                     | N/A              | N/A                  |
| Rollout protocol    | AgentFlow (`preprocess` / `generate` / `reward`) with dynamic method injection | Fixed pipeline   | Fixed pipeline       |
| Inference engine    | SGLang                                                                         | vLLM             | vLLM                 |
| Training backend    | Megatron-LM (distributed checkpoint, TP/PP/DP resharding)                      | FSDP             | DeepSpeed / Megatron |
| Async training      | Yes -- rollout and training overlap via `async_factor`                         | No               | No                   |

In short, siirl-agentic is purpose-built for **agentic RL** where the model interacts with external tools across multiple turns, while veRL and OpenRLHF focus on single-turn RLHF/DPO pipelines.

### What models are supported?

Any Hugging Face model that SGLang can serve, including:

- **Qwen 2 / 2.5 / 3** — `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen3-8B`
- **LLaMA 3 / 3.1** — `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **DeepSeek** — `deepseek-ai/DeepSeek-V2-Lite`, `deepseek-ai/DeepSeek-R1`
- **Mistral / Mixtral** — `mistralai/Mistral-7B-Instruct-v0.3`
- **Any HF checkpoint** — set `actor_ref.model.path` and `trust_remote_code: true`

### Can I use it for single-turn RL (no tool interaction)?

Yes. Set the multiturn parameters to disable multi-turn behavior:

```yaml
rollout:
  multiturn:
    max_env_turns: 1
    max_assistant_turns: 1
    env_type: null            # no tool environment
```

This is functionally equivalent to a standard single-turn GRPO/PPO loop. The framework will generate one response per prompt and compute the reward directly without any tool interaction.

---

## Training

!!! tip "Common questions about GPU allocation, algorithms, and checkpointing."

### How many GPUs do I need?

| Mode                              | Minimum                      | Recommended                  | Notes                                                                          |
| --------------------------------- | ---------------------------- | ---------------------------- | ------------------------------------------------------------------------------ |
| **Separated** (`colocate: false`) | 2 GPUs (1 actor + 1 rollout) | 8 GPUs (2 actor + 6 rollout) | `trainer.actor_gpus` + `trainer.rollout_gpus` must equal total                 |
| **Colocated** (`colocate: true`)  | 1 node (all GPUs shared)     | 1+ nodes                     | Training and rollout share the same GPUs; weights are offloaded between phases |

For a 7B model, 8 GPUs on a single node in separated mode is a typical starting point. For 70B+ models, use `tensor_model_parallel_size: 4` or higher and multiple nodes.

### Can I resume training with different TP/PP settings?

Yes. siirl-agentic uses **Megatron distributed checkpoint** format, which supports transparent resharding across tensor-parallel (TP), pipeline-parallel (PP), and data-parallel (DP) dimensions.

```yaml
trainer:
  resume_mode: auto                           # or set resume_from_path explicitly
  tensor_model_parallel_size: 4               # different from original TP
  pipeline_model_parallel_size: 2             # different from original PP
```

The checkpoint system (`CheckpointArguments`) saves `model`, `optimizer`, and `extra` states by default. On resume, Megatron handles the weight resharding automatically.

### PPO vs GRPO -- which should I use?

| Aspect            | PPO (`adv_estimator: ppo`)       | GRPO (`adv_estimator: grpo`)         |
| ----------------- | -------------------------------- | ------------------------------------ |
| Critic model      | Required (separate value head)   | Not needed                           |
| GPU overhead      | Higher (actor + critic)          | Lower (actor only)                   |
| Stability         | More stable with `critic_warmup` | Simpler but noisier                  |
| Dual-clip support | Yes (`clip_ratio_c`)             | N/A                                  |
| Dr.GRPO variant   | N/A                              | Set `norm_adv_by_std_in_grpo: false` |

For a detailed comparison of the mathematical formulations, see the [Algorithm Theory](../concepts/algorithm_theory.md) page.

---

## Multi-Turn / Agentic

!!! tip "Questions about tool environments, AgentFlow, and multi-turn rollout."

### How do I add a new tool?

1. Subclass `ToolEnv` from `siirl.environment.tool_env.base_tool_env`
2. Define the tool's OpenAI function schema (`OpenAIFunctionToolSchema`)
3. Implement `reset()`, `step()`, and `release()` methods
4. Register the tool in your environment YAML config

For a step-by-step walkthrough, see the [Custom Agent](../guides/custom_agent.md) guide.

### What happens when a tool call times out?

When a tool execution exceeds the timeout or raises an exception, the framework catches the error and returns an `EnvResponse` with an error message:

```python
EnvResponse(
    text="Error when executing tool: <exception message>",
)
```

The rollout continues -- the error text is appended to the conversation as a `tool` role message, and the model sees the error in its next generation turn. This means:

- The rollout is **not** aborted on tool failure
- The model can learn to recover from tool errors
- `env_response.rewards` will be `None` for failed tool calls (no intermediate reward)
- If `max_env_turns` is reached, the rollout terminates regardless

You can control response truncation via `rollout.multiturn.env_response_truncate_side` (`left`, `middle`, or `right`, default: `middle`).

---

## Performance

!!! tip "Tuning tips for throughput and memory."

### Training is slow -- how do I speed it up?

Common bottlenecks and solutions:

| Bottleneck          | Diagnostic                                   | Fix                                                                     |
| ------------------- | -------------------------------------------- | ----------------------------------------------------------------------- |
| Rollout throughput  | WandB `rollout_duration` >> `train_duration` | Increase `rollout_gpus`, raise `train_server_concurrency`               |
| Training throughput | `train_duration` >> `rollout_duration`       | Enable `use_dynamic_batch: true`, tune `max_tokens_per_gpu`             |
| Weight sync         | Large `param_sync_duration` in logs          | Increase `trainer.param_sync_buffer_size` for MoE models                |
| Async overlap       | No overlap visible in timeline               | Set `trainer.async_factor: 2` (rollout starts before training finishes) |

For detailed profiling instructions, see the [Performance Tuning](../guides/performance_tuning.md) and [Profiling](../guides/profiling.md) guides.

### How do I reduce GPU memory usage?

| Technique          | Config                                                                    | Typical savings                 |
| ------------------ | ------------------------------------------------------------------------- | ------------------------------- |
| Parameter offload  | `actor_ref.model.megatron.param_offload: true`                            | 30-40% GPU memory               |
| Gradient offload   | `actor_ref.model.megatron.grad_offload: true`                             | 15-20% GPU memory               |
| Optimizer offload  | `actor_ref.model.megatron.optimizer_offload: true`                        | 40-50% GPU memory               |
| Reduce micro-batch | `actor_ref.actor.ppo_micro_batch_size_per_gpu: 1`                         | Proportional                    |
| Dynamic batching   | `actor_ref.actor.use_dynamic_batch: true` with `max_tokens_per_gpu: 2048` | Prevents OOM spikes             |
| Rollout memory     | `rollout.gpu_memory_utilization: 0.4`                                     | Lower SGLang VRAM reservation   |
| Ref param offload  | `actor_ref.ref.param_offload: true`                                       | Offloads reference model to CPU |

For a complete OOM troubleshooting workflow, see the [Handling OOM](../guides/handling_oom.md) guide.
