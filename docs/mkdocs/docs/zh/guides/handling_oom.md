# 处理 Out-of-Memory (OOM)

*估算显存需求并解决训练和 rollout 组件的 OOM 错误。*

## 显存估算公式

!!! tip "核心要点"
    经验法则：BF16 精度下，N 十亿参数的模型仅权重就需要约 2N GB。训练会额外增加 4--8 倍开销用于优化器状态和梯度。Rollout 引擎的 KV 缓存需要单独预算。

### 各组件显存用量

| 组件                       | 公式                             | 示例（7B，BF16）         |
| ------------------------ | ------------------------------ | ------------------- |
| Actor 权重                 | 2 x N_params GB                | ~14 GB              |
| 优化器状态（Adam）              | 8 x N_params GB                | ~56 GB              |
| 梯度                       | 2 x N_params GB                | ~14 GB              |
| 激活值                      | 取决于 batch/序列长度                 | ~2--8 GB            |
| Reference 模型权重           | 2 x N_params GB                | ~14 GB              |
| Rollout 引擎（SGLang KV 缓存） | `gpu_memory_utilization` x 总显存 | ~40 GB（0.5 x 80 GB） |

!!! note
    当 `use_distributed_optimizer=True`（`MegatronArguments` 默认值，model_args.py:22）时，优化器状态在数据并行 rank 间分片，每 GPU 优化器显存降低为 `1/dp_size`。

### 分离模式 vs 共置模式显存布局

在**分离模式**（`trainer.colocate=False`，默认）下，训练和 rollout 使用不同的 GPU 集合：

| GPU 集合      | 组件                           | 示例（7B，2 训练 GPU + 6 rollout GPU）          |
| ----------- | ---------------------------- | ---------------------------------------- |
| 训练 GPU      | Actor + Ref + 优化器 + 梯度 + 激活值 | ~50 GB/GPU（使用分布式优化器）                     |
| Rollout GPU | SGLang 引擎 + KV 缓存            | ~40 GB/GPU（`gpu_memory_utilization=0.5`） |

配置：

```yaml
trainer:
  colocate: false        # 默认
  actor_gpus: 2          # 默认: 2
  rollout_gpus: 6        # 默认: 6
```

在**共置模式**（`trainer.colocate=True`）下，训练和 rollout 共享相同 GPU，通过时间分片复用：

| 阶段         | 活跃组件                           | 显存           |
| ---------- | ------------------------------ | ------------ |
| Rollout 阶段 | SGLang 引擎（训练期间权重已卸载）           | KV 缓存 + 模型权重 |
| 训练阶段       | Actor + Ref + 优化器（rollout 已卸载） | 完整训练显存       |

配置：

```yaml
trainer:
  colocate: true
  # 共置模式下 actor_gpus 和 rollout_gpus 被忽略
```

!!! warning
    共置模式自动启用多项安全保护（在 `_apply_colocate_guards` 中，async_train.py:42）：

    - 强制在 actor、ref 和 critic 的 megatron 配置中启用 `param_offload=True`
    - 将 `rollout.gpu_memory_utilization` 限制为最大 0.45
    - 禁用 `validate_reuse_train_gpus`

## 三级卸载策略

`MegatronArguments`（model_args.py:21）提供三级 CPU 卸载，每级递进降低 GPU 显存占用，但会牺牲训练速度：

| 级别  | 参数                  | 默认值     | 卸载内容         | GPU 显存节省（7B） |
| --- | ------------------- | ------- | ------------ | ------------ |
| 1   | `param_offload`     | `False` | 模型参数卸载到 CPU  | ~14 GB       |
| 2   | `grad_offload`      | `False` | 梯度卸载到 CPU    | ~14 GB       |
| 3   | `optimizer_offload` | `False` | 优化器状态卸载到 CPU | ~56 GB       |

### 配置示例

=== "不卸载（默认）"

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

    每训练 GPU 显存占用：~100 GB（7B 模型）

=== "仅参数卸载"

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

    每训练 GPU 显存占用：~72 GB（7B 模型）

=== "全部卸载"

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

    每训练 GPU 显存占用：~16 GB（7B 模型，仅激活值 + 工作空间）

!!! note
    当 `param_offload=True` 时，`Trainer.train_step`（trainer.py:716）中的训练循环会在计算前自动将 actor 权重加载到 GPU，计算后卸载，并与 `_sync_rollout_workers`（trainer.py:489）中的权重同步协调。

## 快速修复

| OOM 位置         | 第一步尝试                                         | 第二步尝试                                          | 最后手段                                                     |
| -------------- | --------------------------------------------- | ---------------------------------------------- | -------------------------------------------------------- |
| Trainer 初始化    | 减小 `ppo_mini_batch_size`（默认：`256`）            | 启用 `actor.megatron.param_offload`              | 切换到 `colocate: true`                                     |
| Rollout 初始化    | 减小 `rollout.gpu_memory_utilization`（默认：`0.5`） | 减小 `rollout.max_num_batched_tokens`（默认：`8192`） | 通过 `trainer.rollout_gpus` 增加 rollout GPU                 |
| 训练步骤           | 减小 `actor.ppo_micro_batch_size_per_gpu`       | 启用 `actor.megatron.grad_offload`               | 启用 `actor.megatron.optimizer_offload`                    |
| 多轮 Rollout     | 减小 `data.max_response_length`（默认：`512`）       | 减小 `rollout.multiturn.max_env_turns`（默认：`1`）   | 减小 `rollout.multiturn.max_env_response_length`（默认：`256`） |
| Reference 模型   | 启用 `ref.megatron.param_offload`               | 减小 `ref.ppo_micro_batch_size_per_gpu`          | 不适用                                                      |
| Critic 模型（PPO） | 启用 `critic.megatron.param_offload`            | 减小 `critic.ppo_micro_batch_size_per_gpu`       | 减小 `critic.ppo_mini_batch_size`（默认：`256`）                |

## 变长序列的动态批处理

!!! tip "核心要点"
    当序列长度差异显著时（多轮 agent 任务中很常见），动态批处理通过限制每 GPU 总 token 数而非固定样本数来防止 OOM。

`ActorArguments`（model_args.py:99）提供动态批处理控制：

| 参数                     | 默认值       | 说明                                                     |
| ---------------------- | --------- | ------------------------------------------------------ |
| `use_dynamic_batch`    | `False`   | 启用基于 token 的批处理（而非固定批大小）                               |
| `max_tokens_per_gpu`   | `4096`    | 动态批处理启用时每 GPU 最大 token 数                               |
| `use_workload_balance` | `True`    | 使用基于 FLOPs 的均衡（考虑 attention 的二次复杂度）；否则使用序列长度均衡         |
| `denominator_scope`    | `"local"` | Loss 分母范围：`"local"`（单 GPU）或 `"dp_global"`（跨所有 DP rank） |
| `loss_scale_factor`    | `None`    | `seq-mean-token-sum-norm` loss 模式的可选固定分母               |

### 何时使用动态批处理

适用场景：

- 批次中样本的序列长度差异超过 3 倍
- 多轮 agent 轨迹产生不可预测的响应长度
- 某些训练步骤偶发 OOM，但其他步骤正常

### 配置示例

```yaml
actor_ref:
  actor:
    use_dynamic_batch: true
    max_tokens_per_gpu: 4096
    use_workload_balance: true
    denominator_scope: "local"
```

### 固定批处理 vs 动态批处理

| 方面     | 固定批处理                               | 动态批处理                         |
| ------ | ----------------------------------- | ----------------------------- |
| 批大小控制  | `ppo_micro_batch_size_per_gpu`（样本数） | `max_tokens_per_gpu`（token 数） |
| 显存可预测性 | 峰值显存取决于批次中最长的序列                     | 峰值显存受 token 预算限制              |
| OOM 风险 | 变长序列时风险高                            | 低，显存使用一致                      |
| 吞吐量    | 序列长度均匀时更高                           | 序列长度差异大时更高                    |

## 显存分析工具

siirl-agentic 内置了显存分析功能，通过 `MemoryProfiler`（memory_profiler.py:424）实现：

```bash
# 启用每步峰值显存日志
export SIIRL_MEMORY_STEP_PROFILE=1

# 启用完整显存快照导出（仅第 0 步，rank 0）
export SIIRL_MEMORY_PROFILE=1

# 启用详细显存调试日志
export SIIRL_MEMORY_DEBUG=1
```

### 分析显存快照

当 `SIIRL_MEMORY_PROFILE=1` 时，第一个训练步骤结束后会导出快照。可以用内置工具分析：

```bash
# 分析快照
python -m siirl.utils.logger.memory_profiler memory_snapshot_rank0_step0.dat
```

快照也可在 [pytorch.org/memory_viz](https://pytorch.org/memory_viz) 上可视化查看。

### 编程式显存检查

```python
from siirl.utils.logger.memory_profiler import (
    get_memory_stats,
    log_memory,
    memory_trace,
    GPUMemoryLogger,
)

# 快速查看统计
stats = get_memory_stats()
print(f"Allocated: {stats['allocated_gb']:.2f} GB")

# 上下文管理器追踪代码块
with memory_trace("my_computation"):
    result = model(input_ids)

# 装饰器用于函数级分析
@GPUMemoryLogger(role="actor")
def update_actor(batch):
    ...
```

## 常见错误

| 症状             | 原因                          | 修复                                                                      |
| -------------- | --------------------------- | ----------------------------------------------------------------------- |
| 仅某些步骤 OOM      | 固定批处理下序列长度不一                | 启用 `use_dynamic_batch=True`，设置 `max_tokens_per_gpu=4096`                |
| 权重同步时 OOM      | Actor 和 rollout 权重同时在 GPU 上 | 启用 `param_offload=True`（共置模式下自动启用）                                      |
| 共置模式下 OOM      | `gpu_memory_utilization` 过高 | 降低到 0.45 或以下（被 `_apply_colocate_guards` 自动限制）                           |
| 显存逐渐增长         | CUDA 显存碎片化                  | 设置 `rollout.free_cache_engine=True`（默认）并定期调用 `torch.cuda.empty_cache()` |
| 验证时 OOM        | 验证批次大于训练批次                  | 显式设置 `data.val_batch_size`，而非使用整个验证集                                    |
| PPO critic OOM | 三个模型同时加载                    | 对 actor、ref 和 critic 三者都启用 `param_offload`                              |
