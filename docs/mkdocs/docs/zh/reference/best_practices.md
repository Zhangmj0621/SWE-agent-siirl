# 最佳实践

*配置、监控、调试和生产部署 siirl-agentic 训练任务的实用指南。*

## 配置最佳实践

| 错误做法                              | 推荐做法                                | 原因                                 |
| --------------------------------- | ----------------------------------- | ---------------------------------- |
| 一次性设置所有参数                         | 从默认值开始，只覆盖 3-5 个关键参数                | 减少需要调试的变量                          |
| `rollout.temperature=0.0`         | `rollout.temperature=0.7`           | 零温度导致 GRPO 没有多样性                   |
| `trainer.total_epochs=100`        | `trainer.total_epochs=30`，然后评估      | 节省 GPU 计算小时数；大多数提升在前 30 个 epoch    |
| 70B 模型直接用 `trainer.colocate=true` | 先使用 offload（分离）模式                   | 先验证稳定性，再引入 colocate 开销             |
| `data.max_response_length=512`    | `data.max_response_length=2048` 或更高 | 这是字符数而非 token 数；512 字符 ≈ 128 token |

## 配置指南

### 从简单开始，逐步扩展

从最小可行配置开始，逐步增加规模：

``` yaml
# 第一步：验证单节点、小模型、GRPO
actor_ref:
  model:
    path: Qwen/Qwen2.5-1.5B-Instruct
  algorithm:
    adv_estimator: grpo        # 比 PPO 更简单（无 critic）

rollout:
  n: 4                         # 每个 prompt 4 个采样，用于组相对优势估计

trainer:
  colocate: false
  actor_gpus: 2
  rollout_gpus: 6
  total_epochs: 1
  save_freq: 10
```

验证端到端运行正常后，再逐步增加模型大小、batch size 和 GPU 数量。

### Batch Size 与学习率

| 模型大小    | `ppo_micro_batch_size_per_gpu` | 学习率         | 备注                |
| ------- | ------------------------------ | ----------- | ----------------- |
| 1.5B–3B | 4–8                            | 1e-6 ~ 5e-6 | 可使用较大 micro-batch |
| 7B–8B   | 2–4                            | 5e-7 ~ 2e-6 | 注意 GPU 显存         |
| 13B–14B | 1–2                            | 3e-7 ~ 1e-6 | 可能需要梯度累积          |
| 70B+    | 1                              | 1e-7 ~ 5e-7 | 需要多节点             |

!!! tip "学习率 Warmup"
    始终对前 5–10% 的训练步骤使用 warmup。直接跳到高学习率会导致策略更新不稳定：
    ```yaml
    actor_ref:
      actor:
        optim:
          lr: 1e-6
          warmup_steps: 50    # 预热 50 步
    ```

### 数据配置

所有生产示例一致使用以下数据设置：

``` yaml
data:
  train_batch_size: 512         # 默认值：1024，示例使用 512
  shuffle: false                # RL 训练：确定性数据顺序
  truncation: error             # 超长 prompt 报错，不静默截断
  filter_overlong_prompts: true # 丢弃超过 max_prompt_length 的 prompt
  max_prompt_length: 2048
  max_response_length: 4096     # 默认值：512，长任务需要增大
```

!!! warning "多轮任务的响应长度"
    对于多轮 agentic 任务，`max_response_length` 必须覆盖所有轮次（助手 + 工具响应）。监控 `rollout/response_length_mean` — 如果接近 `max_response_length`，说明轨迹被截断，训练信号会丢失。SWE 风格任务通常需要 8192–16384。

### 动态 Batching

所有生产示例都启用动态 batching 以高效处理变长序列：

``` yaml
actor_ref:
  actor:
    use_dynamic_batch: true        # 基于 token 的 batching（默认值：false）
    max_tokens_per_gpu: 16384      # 每 GPU 每 micro-batch 最大 token 数
    use_workload_balance: true     # 基于 FLOPs 的跨 GPU 负载均衡（默认值：true）
```

## 多轮 Agentic 训练

### 工具配置

``` yaml
rollout:
  multiturn:
    max_env_turns: 10              # 最大工具交互轮数（默认值：1）
    max_assistant_turns: 20        # 最大助手响应总数（默认值：1）
    max_parallel_calls: 1          # 默认串行（默认值：1）
    max_env_response_length: 256   # 字符数，非 token 数（默认值：256）
    env_response_truncate_side: middle  # 保留输出首尾（默认值：middle）
```

**关键规则：**

1. **从 `max_parallel_calls: 1` 开始** — 并行工具调用可能导致顺序问题。确认正确性后再增加。超出限制的工具调用会被**静默丢弃**（不会排队等待）。
2. **使用 `middle` 截断** — 对于代码执行输出，开头（命令）和结尾（结果/错误）最有信息量。也支持 `left`（保留开头）和 `right`（保留结尾）。
3. **设置合理的轮数限制** — 过多轮次浪费算力。监控 `rollout/env_turns_mean` 找到最佳值。默认值 1/1 适用于单轮任务。

### 奖励函数设计

自定义奖励函数必须遵循以下签名（通过 `custom_reward_function.path` 加载）：

``` python
def reward_function(data_source: str, solution_str: str, ground_truth: str, **kwargs) -> float:
    """
    Args:
        data_source: 数据集来源标识（来自 data.reward_fn_key 列）
        solution_str: 解码后的模型输出字符串
        ground_truth: 数据集中的预期答案
    Returns:
        标量奖励值
    """
    # 二元：agent 是否解决了任务？
    task_reward = 1.0 if is_correct(solution_str, ground_truth) else 0.0
    return task_reward
```

良好的 agentic 任务奖励函数应该：

- **使用二元或稀疏奖励** — GRPO 配合明确的通过/失败信号效果最好（如测试通过率）
- **避免密集奖励塑造** — 每步奖励在多轮设置中容易导致奖励黑客
- **包含格式惩罚** — 对格式错误的工具调用施加惩罚，尽早教会正确格式

### GRPO vs PPO 选择

| 场景          | 推荐   | 原因                   |
| ----------- | ---- | -------------------- |
| 二元奖励（通过/失败） | GRPO | 无需 critic，训练更简单      |
| 密集奖励（每步评分）  | PPO  | Critic 通过 GAE 支持信用分配 |
| 快速原型验证      | GRPO | 更少的超参数需要调优           |
| 细粒度奖励塑造     | PPO  | GAE 处理时序信用分配         |

**Dr.GRPO 变体：** 设置 `algorithm.norm_adv_by_std_in_grpo: false` 跳过标准差归一化。这可以防止优势量级缩放在绝对尺度奖励下导致训练不稳定。

## 显存管理

### 预防 OOM

最常见的故障模式是训练时 CUDA OOM。按优先级排序的缓解策略：

1. **减小 micro-batch size**：`actor_ref.actor.ppo_micro_batch_size_per_gpu: 1`
2. **启用动态 batching**：`actor_ref.actor.use_dynamic_batch: true`
3. **启用参数卸载**：`actor_ref.actor.megatron.param_offload: true`
4. **减少 rollout 显存**：`rollout.gpu_memory_utilization: 0.5`（默认值）
5. **使用梯度检查点**：Megatron 后端默认启用

!!! warning "Colocated 模式显存"
    在 colocated 模式下，`gpu_memory_utilization` 自动限制为 0.45。参数卸载（`param_offload: true`）也会被强制开启。如果仍然 OOM，减小 `ppo_micro_batch_size_per_gpu` 或切换到 separated 模式。

### 检查点磁盘空间

``` yaml
trainer:
  save_freq: 10
  max_actor_ckpt_to_keep: 5      # 不要保留 100 个（默认值）
  max_critic_ckpt_to_keep: 5
```

7B 模型的检查点约 14GB。`max_actor_ckpt_to_keep=100`（默认）意味着 1.4TB。生产环境设置为 3–5。

## 监控

### 关键指标

| 指标                            | 健康范围                  | 异常时的操作              |
| ----------------------------- | --------------------- | ------------------- |
| `reward/mean`                 | 持续上升                  | 检查奖励函数，增大 lr        |
| `kl/mean`                     | < 15                  | 减小 lr 或添加 KL 惩罚     |
| `actor/clip_ratio`            | 0.1–0.3               | 调整 `clip_ratio` 超参数 |
| `rollout/generation_duration` | 稳定或下降                 | 太高则增加 rollout GPU   |
| `rollout/env_duration`        | < generation_duration | 太高则扩展 AIO           |

### 尽早启用 WandB

``` yaml
trainer:
  logger: ["console", "wandb"]    # 这实际上就是默认值
  project_name: siirl_experiments
  experiment_name: grpo_qwen2.5_7b_swe
```

始终从一开始就启用 WandB。仅使用控制台日志难以回溯诊断问题。默认 `logger` 已包含 console 和 wandb。

### KL 正则化

所有生产示例使用一致的 KL 配置：

``` yaml
actor_ref:
  actor:
    use_kl_loss: true              # 启用 KL 惩罚（默认值：false）
    kl_loss_coef: 0.01             # KL 损失系数（默认值：0.001）
    kl_loss_type: low_var_kl       # 低方差 KL（默认值：low_var_kl）
  algorithm:
    kl_penalty: kl                 # KL 惩罚类型（默认值：kl）
```

`low_var_kl` 类型计算 `exp(kl) - kl - 1`，比原始 KL 散度更数值稳定，在大 KL 值时不会爆炸。

## 生产部署

### 飞行前检查清单

启动长时间训练前：

- [ ] 用 `save_freq: 1` 运行 1 个 epoch，验证检查点保存正常
- [ ] 确认 `max_actor_ckpt_to_keep` 不会填满磁盘
- [ ] 测试恢复：停止并从保存的检查点重新启动
- [ ] 启用 WandB 日志，使用描述性实验名称
- [ ] 设置 `val_before_train: true`（默认值）获取基线指标
- [ ] 验证工具环境可访问（AIO proxy 健康检查）
- [ ] 检查 Ray 集群：`ray status` 显示预期的节点和 GPU

### 可复现性

``` yaml
trainer:
  seed: 1                     # 训练种子（默认值：1）

rollout:
  temperature: 0.7            # 固定温度（默认值：1.0）
  top_p: 0.95                 # 固定 top-p（默认值：1.0）

data:
  shuffle: false              # 确定性数据顺序
```

!!! note
    由于 CUDA 非确定性操作和异步调度，分布式多轮训练无法保证完全可复现。种子控制数据顺序，但不控制 GPU 级别的计算顺序。

### 故障恢复

始终配置自动检查点保存：

``` yaml
trainer:
  save_freq: 10               # 每 10 步保存（默认值：-1，禁用！）
  resume_mode: auto            # 自动从最新检查点恢复（默认值）
```

使用 `resume_mode: auto`，崩溃的运行在重启时会自动从最新检查点恢复。详见 [检查点与恢复](../guides/checkpoint_resume.md)。

## 常见故障模式

!!! tip
    以下是生产部署中最常见的错误。每个模式包含症状、诊断和修复方案。

### 模式 1：async_factor 设置过高

**症状：** 训练指标看起来不错，但评估性能停滞或下降。

**诊断：** Off-policy 数据陈旧——rollout 数据在被训练时已经过期太久。检查日志中 rollout 批次版本与当前训练版本之间的 `data_version` 差距。

**修复：** 从 `trainer.async_factor=1`（同步，默认值）开始。仅当 rollout 等待期间 GPU 利用率低于 50% 时才增加到 2。不要超过 3，除非同时通过 `trainer.off_policy_step` 启用 off-policy 纠正。当 `async_factor=2` 且 `off_policy_step=1` 时，最大数据陈旧度为 3 个版本。

### 模式 2：未监控 KL 散度

**症状：** 奖励持续上升，但模型生成重复或退化的输出。

**诊断：** 策略偏离参考模型过远。检查 WandB 中的 `kl/mean`——如果超过 10–15，模型可能在利用奖励函数的漏洞而非学习预期行为。

**修复：** 启用 KL 正则化：

``` yaml
actor_ref:
  actor:
    use_kl_loss: true           # 默认值：false
    kl_loss_coef: 0.01          # 默认值：0.001
    kl_loss_type: low_var_kl    # 默认值：low_var_kl
```

监控 `kl/mean`，如果 KL 持续增长则调高 `kl_loss_coef`。`low_var_kl` 类型计算 `exp(kl) - kl - 1`，在大 KL 值时比原始 KL 更稳定。

### 模式 3：忽视 Off-Policy 数据陈旧

**症状：** 训练 loss 振荡，即使 batch size 很大奖励也不稳定。

**诊断：** `trainer.off_policy_step` 设置过高，在很多版本之前的数据上训练。

**修复：** 保持 `trainer.off_policy_step <= 2`（默认值：0，严格 on-policy）。使用 `trainer.off_policy_strategy=oldest_first` 优先训练陈旧数据以便更快丢弃。默认策略为 `fifo`。当 `async_factor=2` 且 `off_policy_step=1` 时，最大数据陈旧度为 3 个版本——在 GPU 利用率和数据新鲜度之间取得良好平衡。

### 模式 4：变长数据使用了错误的 loss_agg_mode

**症状：** 模型只生成短回复，忽略需要长回答的 prompt。或者：长序列主导梯度，短序列任务性能退化。

**诊断：** 默认的 `actor_ref.actor.loss_agg_mode=token-mean` 对 micro-batch 中所有 token 平均 loss。一个 2000 token 的序列贡献的梯度是 200 token 序列的 10 倍，导致模型偏向生成短输出（以减少 loss 暴露面）或让长序列主导参数更新。

**修复：** 对于轨迹长度变化大的 agentic 任务，使用 `actor_ref.actor.loss_agg_mode=seq-mean-token-mean`，让每个序列无论长度如何都获得相等的权重。这会先在序列内归一化 loss，再在序列间平均。

``` yaml
actor_ref:
  actor:
    loss_agg_mode: seq-mean-token-mean    # 默认值：token-mean
```

### 模式 5：工具环境超时时间过短

**症状：** 多轮训练中大量 episode 失败。日志中 `SWERolloutResult.FAILURE` 比率很高。

**诊断：** 复杂的工具操作（Docker 构建、测试套件执行、仓库环境搭建）超过了默认超时时间。`ContainerStartArgs.container_timeout` 默认为 `"2h"`，但 `ContainerEnv.execute` 的单命令执行默认超时为 180 秒——对于运行完整测试套件等重操作可能不够。

**修复：** 根据需要增加容器和命令超时时间：

- `ContainerStartArgs.container_timeout`：容器总生命周期（默认：`"2h"`）。多轮长时间训练需要增加。
- `ContainerEnv.execute` 超时：单命令超时（默认：180s）。对大型仓库的 `pytest` 等重操作需要增加。
- `rollout.multiturn.max_env_response_length`：截断可能丢失重要输出（默认：256 字符）。增大以捕获 agent 推理所需的完整测试输出。

同时检查 `ContainerStartArgs.startup_timeout`（默认：180.0s）——如果容器启动慢（拉取镜像、安装依赖），增大此值以避免环境初始化时过早超时。

## 常见陷阱

| 陷阱                                | 症状                                 | 修复                             |
| --------------------------------- | ---------------------------------- | ------------------------------ |
| `max_response_length` 太小（默认值：512） | 截断的轨迹，奖励平坦                         | 增加到 4096+（多轮 8192+）            |
| `save_freq: -1`（默认）               | 无检查点，进度丢失                          | 设置为 10–30                      |
| `env_kwargs.tool_format` 错误       | 工具调用未解析，0 奖励                       | 匹配模型格式（hermes/gpt-oss）         |
| GRPO 使用 `rollout.n: 1`            | 无组内方差计算优势                          | 使用 `rollout.n: 4` 或更高（示例用 8）   |
| 缺少 `env_path`                     | `Error: tools_config_file is None` | 指向工具配置 YAML                    |
| KL 散度爆炸                           | `kl/mean` > 50，策略崩溃                | 将 lr 减小 2–5 倍，启用 `use_kl_loss` |
| Colocated + validate_reuse        | 静默禁用并打印警告                          | 使用 separated 模式                |
| `max_parallel_calls` 过高           | 工具调用被静默丢弃                          | 超出限制的调用会被丢弃，不会排队               |

## 下一步

- [性能调优](../guides/performance_tuning.md) — 在基础配置完善后，深入优化 GPU 利用率和吞吐量
- [常见问题与排障](troubleshooting.md) — 当检查清单中的项目暴露问题时，使用本参考进行诊断
- [配置参考](config_reference.md) — 配置检查清单各项时查找精确的参数名和默认值
