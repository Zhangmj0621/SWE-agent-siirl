# Routing Replay 实现对比：siirl-agentic vs slime

## 概述

两个项目都实现了 MoE Routing Replay（R2/R3），核心思路一致：在 RL 训练的多次 forward/backward 中保持 routing 决策一致。但在 patch 层级、缓存格式、阶段控制等方面存在本质差异。

## 核心差异

### 1. Patch 层级（最根本的差异）

|  | siirl-agentic | slime |
|---|---|---|
| **Patch 对象** | `TopKRouter.routing()`（外层方法） | `compute_topk()`（内部函数） |
| **跳过的代码** | 整个 `topk_routing_with_score_function` | 仅 topk 选择逻辑 |
| **Score 重算** | **自己实现** softmax/sigmoid/scaling 逻辑 | **Megatron 自己完成**（topk 后的 scoring 逻辑照常执行） |

**siirl-agentic** patch 的是 `TopKRouter.routing()`，在 REPLAY 阶段直接跳过 `topk_routing_with_score_function` 调用，因此必须自己复刻 Megatron 的 score 计算逻辑（softmax/sigmoid + pre/post-softmax + scaling_factor）：

```python
# siirl-agentic: patched_routing 中手动重算 scores
def patched_routing(self_router, logits, **kwargs):
    ...
    # REPLAY 路径：自己重算
    if score_function == "softmax":
        if use_pre_softmax:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(orig_logits)
            scores = scores * top_mask.to(scores.dtype)
        else:
            masked_logits = logits.masked_fill(~top_mask, float('-inf'))
            scores = torch.softmax(masked_logits, dim=-1, dtype=torch.float32).type_as(orig_logits)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits).type_as(orig_logits)
        scores = scores * top_mask.to(scores.dtype)
        score_sum = scores.sum(dim=-1, keepdim=True) + 1e-20
        scores = scores / score_sum
    ...
```

**slime** patch 的是更内层的 `compute_topk()`，只替换 topk 选择步骤，外层的 score 计算由 Megatron 的 `topk_routing_with_score_function` 自然完成：

```python
# slime: 只替换 topk 选择，score 计算由 Megatron 完成
def compute_topk(scores, topk, num_groups=None, group_topk=None):
    if routing_replay_stage == "replay_forward":
        top_indices = ROUTING_REPLAY.pop_forward()
        probs = scores.gather(1, top_indices)  # 直接 gather，外层负责后续处理
    ...
    return probs, top_indices
```

**影响**：siirl-agentic 需要手动维护与 Megatron 的 scoring 逻辑一致性。已经因 post-softmax 模式的 score 重算不匹配导致过严重 bug（ppo_kl≈15，reward 崩塌）。slime 的方式天然规避了这类问题。

### 2. 缓存内容格式

|  | siirl-agentic | slime |
|---|---|---|
| **缓存内容** | `routing_map` `[n_tokens, num_experts]` bool | `top_indices` `[n_tokens, topk]` int64 |
| **内存占用** | `n_tokens × num_experts × 1 byte` | `n_tokens × topk × 8 bytes` |
| **示例 (Qwen3-30B)** | `n_tokens × 128 × 1` = 128 bytes/token | `n_tokens × 8 × 8` = 64 bytes/token |

slime 缓存的是 `top_indices`（和 R2 RECORD 路径一致的格式），体积更小。siirl-agentic 在 `fill_from_rollout` 中先将 expert indices 转换为 routing_map（scatter 操作），存储的是展开后的 bool 矩阵。

### 3. 阶段控制机制

|  | siirl-agentic | slime |
|---|---|---|
| **控制方式** | 单例 `RoutingReplayManager` 的 `_stage` 属性 | 环境变量 `os.environ["ROUTING_REPLAY_STAGE"]` |
| **线程安全** | 有 `threading.Lock` 保护单例创建 | 无（依赖单进程） |
| **Context manager** | `with manager.stage(target):` | 无（手动 set/unset） |

siirl-agentic 的 Manager 模式更 Pythonic、更安全（context manager 保证异常时恢复状态）。slime 的环境变量方式更简单直接，但缺乏异常安全保障。

### 4. Gradient Checkpointing 处理

|  | siirl-agentic | slime |
|---|---|---|
| **处理方式** | 用 `torch.is_grad_enabled()` 区分 checkpointed forward / backward recompute | 不区分（假设不使用 activation checkpointing，或由 Megatron 内部 `router_replay` 处理） |
| **forward/backward 索引** | 分离的 `_forward_idx` / `_backward_idx` | 分离但无 `is_grad_enabled()` 判断 |

siirl-agentic 显式处理了 `recompute_granularity="full"` 下每层 router 被调用两次的问题（checkpointed forward under `no_grad` + backward recompute under `enable_grad`）。slime 的 `replay_backward` 阶段直接调用 `pop_backward()`，没有 `is_grad_enabled()` 分支。

### 5. fill_from_rollout 实现差异

|  | siirl-agentic | slime |
|---|---|---|
| **数据输入** | 统一 `[batch, max_seq_len, moe_dim]` tensor | 按 micro-batch 从 data_iterator 逐个取 |
| **micro-batch 切分** | 自己按 `micro_batch_size` 切分 batch 维度 | 复用 Megatron 的 data_iterator |
| **Padding 策略** | `np.zeros` — 全部路由到 expert 0 | `torch.arange(...) % num_experts` — 分散到不同 expert |
| **Pipeline/VP 支持** | 假设所有注册的 cache 都是 MoE layer | 显式处理 `vp_stage` + `moe_layer_freq`（跳过 dense layer） |
| **存入 cache** | `routing_map` bool `[n_tokens, num_experts]` | `top_indices` int `[n_tokens]`（单层切片） |

#### Padding 策略对比

siirl-agentic：
```python
# padding token 全部路由到 expert 0
padded = np.zeros((max_seq_len, moe_dim), dtype=routing_data.dtype)
```

slime：
```python
# padding token 分散到不同 expert，避免 expert 0 虚假负载集中
pad = torch.arange(pad * num_layers * topk, ...).reshape((pad, num_layers, topk)) % num_experts
```

slime 的策略更合理——避免 padding token 在 load balancing auxiliary loss 中对 expert 0 产生不真实的高负载信号。

#### Pipeline Parallel 支持

slime 显式处理了 virtual pipeline 和混合 dense/MoE 架构：

```python
for vp_stage, model in enumerate(self.model):
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
    for layer_id in range(offset, offset + num_layers_to_build):
        # 跳过 dense layer（moe_layer_freq 控制）
        if isinstance(config.moe_layer_freq, int):
            if layer_id % config.moe_layer_freq != 0:
                continue
```

siirl-agentic 假设所有注册的 cache 就是 MoE layer，依赖 `patched_init` 的注册顺序。在混合 dense/MoE 或 VP 场景下可能需要额外处理。

## 各自优势总结

### siirl-agentic 优势

- **Gradient checkpointing 兼容**：显式处理 activation recomputation 双消费
- **阶段管理更安全**：context manager + try/finally 保证异常时清理
- **独立于 Megatron 版本**：不依赖 Megatron 内部的 `router_replay` 参数

### slime 优势

- **Score 重算天然正确**：不需要复刻 Megatron 的 scoring 逻辑，消除了 score 重算 bug 的可能性
- **缓存更紧凑**：存 `top_indices` 而非展开的 `routing_map`，内存更省
- **Pipeline/VP 支持更完善**：显式处理 `vp_stage`、`moe_layer_freq`、dense layer 跳过
- **Padding 策略更合理**：分散到不同 expert，避免 load balancing 偏差
- **复用 Megatron data_iterator**：micro-batch 切分与 Megatron 训练循环天然一致，不存在 micro-batch 数量不匹配的风险

## 潜在改进方向

如果要将 siirl-agentic 的实现向 slime 对齐，最有价值的改动是：

1. **将 patch 层级下沉到 `compute_topk`**：消除自己维护 score 重算逻辑的负担和风险
2. **缓存 `top_indices` 而非 `routing_map`**：减小内存占用，统一 R2/R3 的缓存格式
3. **改进 padding 策略**：用 `arange % num_experts` 分散 padding token 的路由
4. **添加 `moe_layer_freq` 支持**：处理混合 dense/MoE 架构
