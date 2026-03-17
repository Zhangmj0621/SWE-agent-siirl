# 常见问题

*关于 siirl-agentic 的常见问题解答。*

## 概述

!!! tip "如果你是第一次评估 siirl-agentic，请从这里开始。"

### siirl-agentic 与 veRL / OpenRLHF 有什么区别？

| 能力         | siirl-agentic                                           | veRL  | OpenRLHF             |
| ---------- | ------------------------------------------------------- | ----- | -------------------- |
| 原生多轮支持     | 是 -- 异步状态机，支持 `max_env_turns` / `max_assistant_turns`   | 仅单轮   | 仅单轮                  |
| 工具基础设施     | AIO 三层调度（Proxy -> Scheduler -> Sandbox）                 | 无     | 无                    |
| Rollout 协议 | AgentFlow（`preprocess` / `generate` / `reward`）+ 动态方法注入 | 固定流水线 | 固定流水线                |
| 推理引擎       | SGLang                                                  | vLLM  | vLLM                 |
| 训练后端       | Megatron-LM（分布式检查点，TP/PP/DP 重分片）                        | FSDP  | DeepSpeed / Megatron |
| 异步训练       | 是 -- 通过 `async_factor` 实现 rollout 与训练重叠                 | 否     | 否                    |

简言之，siirl-agentic 专为 **Agentic RL** 设计，模型在多轮交互中与外部工具协作；而 veRL 和 OpenRLHF 专注于单轮 RLHF/DPO 流水线。

### 支持哪些模型？

所有 SGLang 能够服务的 Hugging Face 模型均可使用，包括：

- **Qwen 2 / 2.5 / 3** — `Qwen/Qwen2.5-7B-Instruct`、`Qwen/Qwen3-8B`
- **LLaMA 3 / 3.1** — `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **DeepSeek** — `deepseek-ai/DeepSeek-V2-Lite`、`deepseek-ai/DeepSeek-R1`
- **Mistral / Mixtral** — `mistralai/Mistral-7B-Instruct-v0.3`
- **任意 HF 检查点** — 设置 `actor_ref.model.path` 和 `trust_remote_code: true`

### 可以用于单轮 RL（无工具交互）吗？

可以。将多轮参数设置如下即可禁用多轮行为：

```yaml
rollout:
  multiturn:
    max_env_turns: 1
    max_assistant_turns: 1
    env_type: null            # 不使用工具环境
```

这在功能上等同于标准的单轮 GRPO/PPO 循环。框架将为每个 prompt 生成一条回复，并直接计算奖励，不进行任何工具交互。

---

## 训练

!!! tip "关于 GPU 分配、算法和检查点的常见问题。"

### 需要多少 GPU？

| 模式                           | 最低要求                         | 推荐配置                         | 说明                                                   |
| ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------------------------------- |
| **分离模式** (`colocate: false`) | 2 块 GPU（1 actor + 1 rollout） | 8 块 GPU（2 actor + 6 rollout） | `trainer.actor_gpus` + `trainer.rollout_gpus` 必须等于总数 |
| **共置模式** (`colocate: true`)  | 1 个节点（所有 GPU 共享）             | 1+ 个节点                       | 训练和 rollout 共享同一批 GPU；权重在不同阶段进行 offload              |

对于 7B 模型，单节点 8 卡分离模式是典型的起始配置。70B+ 模型请使用 `tensor_model_parallel_size: 4` 或更高，并配合多节点部署。

### 可以用不同的 TP/PP 设置恢复训练吗？

可以。siirl-agentic 使用 **Megatron 分布式检查点** 格式，支持在张量并行（TP）、流水线并行（PP）和数据并行（DP）维度之间透明地重新分片。

```yaml
trainer:
  resume_mode: auto                           # 或显式指定 resume_from_path
  tensor_model_parallel_size: 4               # 与原始 TP 不同
  pipeline_model_parallel_size: 2             # 与原始 PP 不同
```

检查点系统（`CheckpointArguments`）默认保存 `model`、`optimizer` 和 `extra` 状态。恢复时，Megatron 会自动处理权重的重新分片。

### PPO 还是 GRPO -- 应该用哪个？

| 方面           | PPO (`adv_estimator: ppo`) | GRPO (`adv_estimator: grpo`)        |
| ------------ | -------------------------- | ----------------------------------- |
| Critic 模型    | 必需（独立的 value head）         | 不需要                                 |
| GPU 开销       | 较高（actor + critic）         | 较低（仅 actor）                         |
| 稳定性          | 配合 `critic_warmup` 更稳定     | 更简单但噪声更大                            |
| Dual-clip 支持 | 是（`clip_ratio_c`）          | 不适用                                 |
| Dr.GRPO 变体   | 不适用                        | 设置 `norm_adv_by_std_in_grpo: false` |

关于数学公式的详细对比，请参阅[算法理论](../concepts/algorithm_theory.md)页面。

---

## 多轮 / Agentic

!!! tip "关于工具环境、AgentFlow 和多轮 rollout 的问题。"

### 如何添加新工具？

1. 继承 `siirl.environment.tool_env.base_tool_env` 中的 `ToolEnv`
2. 定义工具的 OpenAI 函数 schema（`OpenAIFunctionToolSchema`）
3. 实现 `reset()`、`step()` 和 `release()` 方法
4. 在环境 YAML 配置中注册该工具

详细的操作步骤请参阅[自定义 Agent](../guides/custom_agent.md) 指南。

### 工具调用超时会怎样？

当工具执行超时或抛出异常时，框架会捕获错误并返回一个包含错误信息的 `EnvResponse`：

```python
EnvResponse(
    text="Error when executing tool: <异常信息>",
)
```

Rollout 会继续进行 -- 错误文本作为 `tool` 角色的消息追加到对话中，模型在下一轮生成时会看到该错误。这意味着：

- 工具失败时 rollout **不会**中止
- 模型可以学习从工具错误中恢复
- 失败的工具调用 `env_response.rewards` 为 `None`（无中间奖励）
- 达到 `max_env_turns` 后，无论如何 rollout 都会终止

你可以通过 `rollout.multiturn.env_response_truncate_side` 控制响应截断方式（`left`、`middle` 或 `right`，默认：`middle`）。

---

## 性能

!!! tip "吞吐量和内存的调优建议。"

### 训练很慢 -- 如何加速？

常见瓶颈及解决方案：

| 瓶颈          | 诊断方式                                           | 解决方法                                                 |
| ----------- | ---------------------------------------------- | ---------------------------------------------------- |
| Rollout 吞吐量 | WandB 中 `rollout_duration` >> `train_duration` | 增加 `rollout_gpus`，提高 `train_server_concurrency`      |
| 训练吞吐量       | `train_duration` >> `rollout_duration`         | 启用 `use_dynamic_batch: true`，调优 `max_tokens_per_gpu` |
| 权重同步        | 日志中 `param_sync_duration` 较大                   | 对 MoE 模型增加 `trainer.param_sync_buffer_size`          |
| 异步重叠        | 时间线中无重叠                                        | 设置 `trainer.async_factor: 2`（rollout 在训练结束前开始）       |

详细的性能分析说明，请参阅[性能调优](../guides/performance_tuning.md)和[性能分析](../guides/profiling.md)指南。

### 如何减少 GPU 内存使用？

| 技术             | 配置                                                                      | 典型节省量               |
| -------------- | ----------------------------------------------------------------------- | ------------------- |
| 参数 offload     | `actor_ref.model.megatron.param_offload: true`                          | 30-40% GPU 内存       |
| 梯度 offload     | `actor_ref.model.megatron.grad_offload: true`                           | 15-20% GPU 内存       |
| 优化器 offload    | `actor_ref.model.megatron.optimizer_offload: true`                      | 40-50% GPU 内存       |
| 减小 micro-batch | `actor_ref.actor.ppo_micro_batch_size_per_gpu: 1`                       | 成比例减少               |
| 动态 batching    | `actor_ref.actor.use_dynamic_batch: true` 配合 `max_tokens_per_gpu: 2048` | 防止 OOM 峰值           |
| Rollout 内存     | `rollout.gpu_memory_utilization: 0.4`                                   | 降低 SGLang 显存预留      |
| Ref 参数 offload | `actor_ref.ref.param_offload: true`                                     | 将参考模型 offload 到 CPU |

完整的 OOM 排障流程，请参阅[内存溢出处理](../guides/handling_oom.md)指南。
