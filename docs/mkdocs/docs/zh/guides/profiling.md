# 性能分析

*使用内置计时指标、Ray 追踪和 GPU 分析工具定位异步训练管线中的瓶颈。*

## 内置计时指标

!!! tip "核心要点"
    每个 `Sample` 都记录一个 `timing_info` 字典，包含五个字段：`rollout_start_at`、`rollout_end_at`、`rollout_duration`、`generation_duration` 和 `reward_duration`。这些指标会在批次级别自动聚合，并作为 `perf/delta_time/*` 指标记录到 WandB。

`Sample` 数据类（`siirl/data_coordinator/sample.py:21`）携带在 rollout 期间填充的逐样本计时信息：

| 字段                    | 类型      | 说明                       |
| --------------------- | ------- | ------------------------ |
| `rollout_start_at`    | `float` | 该样本开始 rollout 的 Unix 时间戳 |
| `rollout_end_at`      | `float` | rollout 完成的 Unix 时间戳     |
| `rollout_duration`    | `float` | 整个 rollout 的挂钟时间（秒）      |
| `generation_duration` | `float` | LLM 推理/token 生成耗时（秒）     |
| `reward_duration`     | `float` | 奖励计算耗时（秒）                |

训练批次组装完成后，`extract_rollout_timing_metrics()`（`siirl/utils/metrics/metric_utils.py:322`）将逐样本值聚合为批次级指标：

| 记录的指标                                   | 含义                    |
| --------------------------------------- | --------------------- |
| `perf/delta_time/rollout_per_sample`    | 批次内所有样本的平均 rollout 时长 |
| `perf/delta_time/generation_per_sample` | 平均生成时长                |
| `perf/delta_time/reward_per_sample`     | 平均奖励计算时长              |

此外，`compute_timing_metrics()`（`siirl/utils/metrics/metric_utils.py:179`）计算训练各阶段的 per-token 计时：

| 记录的指标                              | 含义                         |
| ---------------------------------- | -------------------------- |
| `perf/delta_time/gen`              | 生成时间（秒）                    |
| `perf/delta_time/ref`              | 参考模型 log-prob 计算时间         |
| `perf/delta_time/update_actor`     | Actor 更新时间                 |
| `perf/delta_time/update_critic`    | Critic 更新时间（仅 PPO）         |
| `timing_per_token_ms/gen`          | 每个响应 token 的生成时间（ms）       |
| `timing_per_token_ms/update_actor` | 每个总 token 的 Actor 更新时间（ms） |

吞吐量指标来自 `compute_throughput_metrics()`（`siirl/utils/metrics/metric_utils.py:229`）：

| 记录的指标                   | 含义              |
| ----------------------- | --------------- |
| `perf/total_num_tokens` | 批次中处理的总 token 数 |
| `perf/time_per_step`    | 一个训练步的挂钟时间（秒）   |

### 在 WandB 中查看计时指标

所有 `perf/delta_time/*` 指标会自动绘图。诊断管线的方法：

1. **对比 `generation_per_sample` 与 `time_per_step`**：如果生成时间远大于训练步时间，说明 rollout 是瓶颈。
2. **检查 `reward_per_sample`**：如果奖励耗时占 `rollout_per_sample` 的很大比例，需要优化奖励函数（如批量化、缓存）。
3. **关注 `timing_per_token_ms/update_actor`**：此值应保持稳定；突增表明存在显存压力或通信阻塞。

## Ray 时间线

!!! tip "核心要点"
    Ray 内置的时间线工具捕获集群中 actor 的调度、任务执行和空闲间隙。导出为 Chrome trace 格式，在 `chrome://tracing` 中查看。

```bash
# 从运行中的 Ray 集群捕获 30 秒时间线
ray timeline --format chrome-trace --output timeline.json
```

在 `chrome://tracing`（或 [Perfetto UI](https://ui.perfetto.dev/)）中打开 `timeline.json`。重点关注：

- **actor 调用之间的长间隙**：表示 `DataCoordinator` 在等待数据。增大 `trainer.async_factor` 以重叠更多 rollout 批次。
- **actor 调度不均匀**：某个 rollout worker 完成时间远晚于其他——提示数据倾斜。检查 prompt 长度分布。
- **频繁的 `param_sync` 条**：权重同步阻塞了 rollout。考虑使用共置模式（`trainer.colocate: true`）消除网络传输。

## PyTorch Profiler

!!! tip "核心要点"
    使用 `torch.profiler` 捕获训练步的 CUDA kernel 级 trace。仅在 rank 0 上采集以避免存储溢出。

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

在生成的 trace 中重点检查：

- **NCCL AllReduce / ReduceScatter**：分布式训练中的通信开销。应小于步时间的 30%。
- **CUDA kernel 间隙**：kernel 之间的空闲时间表示 CPU 瓶颈（Python 开销、数据预处理）。
- **内存分配**：大的尖峰表明训练循环内有动态张量创建。

仅在 rank 0 上采集以避免 I/O 争用：

```python
import torch.distributed as dist

if dist.get_rank() == 0:
    # 挂载 profiler
    ...
```

## GPU 利用率监控

!!! tip "核心要点"
    训练步期间目标 GPU SM 利用率 > 80%，rollout 期间 > 60%。持续低利用率通常指向数据饥饿或过度同步。

### 使用 nvidia-smi 实时监控

```bash
# 每 1 秒采样，显示 SM 利用率和显存
nvidia-smi dmon -s u -d 1
```

| 列                 | 健康范围      | 警告信号                        |
| ----------------- | --------- | --------------------------- |
| SM %（训练 GPU）      | > 80%     | < 50% 持续——在等待数据或通信          |
| SM %（rollout GPU） | > 60%     | < 30% 持续——批次太小              |
| FB Used (MB)      | < 总量的 95% | > 95%——即将 OOM，启用 offloading |

### DCGM 集成

集群级监控使用 NVIDIA DCGM：

```bash
# 启动 DCGM 并启用分析指标
dcgmi profile --pause
dcgmi profile --resume
dcgmi dmon -e 1001,1002,1003,1004 -d 1000
```

指标 ID：`1001` = SM 活跃率，`1002` = SM 占用率，`1003` = tensor 活跃率，`1004` = DRAM 活跃率。

## 常见瓶颈定位

| 现象                 | 分析指标                                           | 可能原因           | 解决方案                                                                |
| ------------------ | ---------------------------------------------- | -------------- | ------------------------------------------------------------------- |
| 训练时 GPU 利用率低       | Ray 时间线中训练步之间有长空闲                              | 等待 rollout 数据  | 增大 `trainer.async_factor`（默认：`1`）                                   |
| Rollout 时 GPU 利用率低 | SGLang 服务端有空闲期                                 | 请求批次太小         | 增大 `rollout.train_server_concurrency`（默认：`256`）                     |
| 通信时间占比高            | PyTorch profiler 中 NCCL kernel > 步时间的 30%      | 张量/管线并行度过高     | 减小 `trainer.tensor_model_parallel_size`                             |
| 奖励计算慢              | `reward_per_sample` >> `generation_per_sample` | 奖励函数开销大        | 批量化奖励调用；使用异步奖励；缓存结果                                                 |
| 权重同步阻塞             | Ray 时间线中 `param_sync` 条占主导                     | 网络带宽饱和         | 切换到共置模式；调整 `trainer.param_sync_buffer_size`（默认：512 MB）              |
| 训练时 OOM            | FB 使用率 100% 后崩溃                                | micro-batch 过大 | 减小 `actor.ppo_micro_batch_size_per_gpu`；启用 `megatron.param_offload` |
| 步时间不均匀             | `time_per_step` 在各步间方差大                        | 变长序列           | 启用 `actor.use_dynamic_batch: true` 配合 `actor.max_tokens_per_gpu`    |

### 快速诊断检查清单

```bash
# 1. 判断 rollout 还是训练是瓶颈
#    （在 WandB 中对比 generation_per_sample 与 time_per_step）

# 2. 检查两侧的 GPU 利用率
nvidia-smi dmon -s u -d 1

# 3. 检查 Ray actor 调度
ray timeline --format chrome-trace --output timeline.json

# 4. 在 rank 0 上 profile 几个训练步
#    （参见上方 PyTorch Profiler 章节）
```
