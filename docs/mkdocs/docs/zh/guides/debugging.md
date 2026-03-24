# 调试指南

*系统化诊断 siirl-agentic 训练问题的方法。*

## Ray Actor 调试

!!! tip "核心要点"
    siirl-agentic 的所有 worker 都是 Ray actor。出问题时最快的排查路径是 `ray status` 加 Ray Dashboard。在深入代码之前，务必先检查 `/tmp/ray/session_latest/logs/` 中的 actor 级别日志。

### 集群状态检查

使用 `ray status` 验证资源可用性和 actor 分配：

```bash
# 检查集群健康状态
ray status

# Ray Dashboard（默认端口 8265）
# 在浏览器中打开 http://<head-node-ip>:8265
```

### Actor 日志位置

Ray 将每个 actor 的日志存储在会话目录下：

| 日志类型           | 路径                                                       |
| -------------- | -------------------------------------------------------- |
| Worker 标准输出/错误 | `/tmp/ray/session_latest/logs/worker-*.out` / `.err`     |
| Ray 系统日志       | `/tmp/ray/session_latest/logs/raylet.out`                |
| GCS 服务器日志      | `/tmp/ray/session_latest/logs/gcs_server.out`            |
| siirl 应用日志     | `./siirl_logs/siirl_*.log`（可通过 `SIIRL_LOG_DIRECTORY` 配置） |

### 常用环境变量

```bash
# 禁用 Ray 日志去重（查看每个 worker 的每行日志）
export RAY_DEDUP_LOGS=0

# 设置 loguru 日志级别（DEBUG 显示并发、批次同步、内存详情）
export LOGURU_LEVEL=DEBUG

# 启用 rollout worker 的详细验证日志
export SIIRL_VERBOSE_VALIDATE_LOGS=1
```

### 检查 Actor 状态

`MainRunner`（async_train.py:69）负责编排所有组件。需要关注的核心 actor：

| Actor                       | 类                                          | 用途          |
| --------------------------- | ------------------------------------------ | ----------- |
| `MainRunner`                | `MainRunner` (async_train.py:69)           | 编排、生命周期管理   |
| `trainer_rank{N}_bundle{M}` | `Trainer` (trainer.py:103)                 | 每个 GPU 上的训练 |
| `RolloutManager`            | `RolloutManager` (rollout_manager.py)      | Rollout 协调  |
| `TaskCoordinator`           | `TaskCoordinator` (task_coordinator.py:64) | 停止信号、故障传播   |
| `MetricWorker`              | `MetricWorker` (metrics.py)                | 分布式指标聚合     |

在 Ray Dashboard 的 "Actors" 标签页中可以查看每个 actor 的状态、资源使用和最近的日志。

## 多轮轨迹检查

!!! tip "核心要点"
    对于多轮 agent 工作流，大多数训练 bug 源于不正确的 `response_mask` 或格式异常的轨迹。在信任 reward 或 loss 数值之前，务必先导出并检查几个样本。

### 导出生成的轨迹

`NaiveExecutor`（naive_executor.py:37）管理轨迹生成流水线。要检查轨迹，可以启用调试日志并保存样本：

```python
# 在自定义 flow 或调试脚本中：
import torch

# 在 NaiveExecutor.generate() 中，_post_process 之后：
# 取消注释 naive_executor.py:414-417 的调试代码块
# samples = await asyncio.gather(*tasks)
# batch = Samples2Dict(samples=samples)
# torch.save(batch, f"save_dict/{os.environ.get('RANK')}_batch.pt")
```

然后加载并检查：

```python
import torch

batch = torch.load("save_dict/0_batch.pt")
print("Keys:", list(batch.keys()))
print("input_ids shape:", batch["input_ids"].shape)
print("response_mask shape:", batch["response_mask"].shape)
print("attention_mask shape:", batch["attention_mask"].shape)
```

### 验证 loss_mask 正确性

对于多轮轨迹，`response_mask` 决定哪些 token 参与 loss 计算。预期模式如下：

| 场景                     | Prompt 区域 | Assistant 第1轮 | 工具响应 | Assistant 第2轮 |
| ---------------------- | --------- | ------------- | ---- | ------------- |
| 标准多轮                   | 全零        | 全一            | 全零   | 全一            |
| `train_on_prompt=True` | 全一        | 全一            | 全零   | 全一            |
| `mask_history=True`    | 全零        | 全零            | 全零   | 全一（仅最后一轮）     |

```python
# 验证 mask 模式
mask = batch["response_mask"][0]
print("Non-zero positions:", mask.nonzero().squeeze().tolist())
print("Sum (trainable tokens):", mask.sum().item())
```

### Token 级 Log Prob 检查

每个样本的 `rollout_log_prob` 字段存储了 rollout 引擎的逐 token 对数概率。与训练时的 log prob 对比可以检测 off-policy 漂移：

```python
import numpy as np

# rollout_log_prob 在 _post_process 中被填充到 max_response_length
rollout_lp = batch["rollout_log_prob"][0]
print("Log prob range:", rollout_lp.min(), "to", rollout_lp.max())
print("Mean log prob (non-pad):", rollout_lp[batch["response_mask"][0] > 0].mean())
```

## Reward 计算调试

!!! tip "核心要点"
    Reward 问题是训练失败最常见的原因。在运行完整训练作业之前，务必先单独测试你的 reward 函数。

### Reward 函数分发

Reward 分发逻辑在 `default_compute_score`（reward_score/\_\_init\_\_.py:17）中。它根据 `data_source` 字段路由：

| `data_source` 值                        | Reward 模块                              |
| -------------------------------------- | -------------------------------------- |
| `openai/gsm8k`                         | `reward_score/gsm8k.py`                |
| `lighteval/MATH`、`AIME2024`、`AIME2025` | `reward_score/math.py`                 |
| `math_dapo`、`aime*`                    | `reward_score/math_dapo.py`            |
| `hiyouga/geometry3k`                   | `reward_score/geo3k.py`                |
| `mm_eureka`                            | `reward_score/mm_eureka.py`            |
| `searchR1_*`                           | `reward_score/search_r1_like_qa_em.py` |

自定义 reward 函数通过 `custom_reward_function.path` 配置项加载。加载发生在 `NaiveExecutor.__init__`（naive_executor.py:79）中。

### 单独测试 Reward 函数

```python
from siirl.utils.reward_score import default_compute_score

# 用已知输入/输出测试
score = default_compute_score(
    data_source="openai/gsm8k",
    solution_str="The answer is 42.",
    ground_truth="42",
)
print(f"Score: {score}")  # 预期: 1.0
```

### 常见 Reward 问题

| 症状                    | 原因                      | 修复                                            |
| --------------------- | ----------------------- | --------------------------------------------- |
| Reward 始终为 0          | `data_source` 未匹配任何分发分支 | 检查 `data.reward_fn_key` 和数据集的 `data_source` 列 |
| Reward 为 NaN          | Reward 函数中除零            | 在自定义 reward 中添加 NaN 防护                        |
| Reward 尺度错误           | Reward 返回 bool 而非 float | 确保 `float(res)` 转换                            |
| `NotImplementedError` | 未知 `data_source` 值      | 添加分发分支或使用 `custom_reward_function.path`       |

## 权重同步调试

!!! tip "核心要点"
    权重同步失败通常表现为 rollout 行为停滞（reward 平台期）或 NCCL 超时。关键参数是 `param_sync_rpc_timeout_s`（默认：`120`）。

### ParamSyncDistributed vs ParamSyncColocated

同步策略根据 `trainer.colocate` 自动选择：

| 模式                     | 类                                             | 机制                                                     |
| ---------------------- | --------------------------------------------- | ------------------------------------------------------ |
| 分离模式（`colocate=False`） | `ParamSyncDistributed` (update_weight.py:145) | Trainer 和 rollout GPU 之间的 NCCL 进程组                     |
| 共置模式（`colocate=True`）  | `ParamSyncColocated` (update_weight.py:452)   | 通过 `FlattenedTensorBucket` 的 CUDA IPC 句柄，拓扑感知的 lane 路由 |

### 验证权重同步

`weight_version` 计数器在每次成功同步后递增。在两端检查：

```python
# Trainer 端（从日志中查看）
# 查找: "[ParamSyncDistributed] sync_total_ms=... sync_bucket_count=..."

# Rollout 端
# 查找: "engine._weight_version" 在 rollout worker 日志中
```

### 超时诊断

=== "分离模式"

    ```yaml
    trainer:
      param_sync_rpc_timeout_s: 120  # 默认: 120s
    ```

    如果出现 `NCCL timeout` 或 `ProcessGroup timeout`：

    1. 检查 trainer 和 rollout 节点之间的网络连通性
    2. 设置 `NCCL_DEBUG=INFO` 获取详细的 NCCL 诊断信息
    3. 对大模型增加 `param_sync_rpc_timeout_s`

=== "共置模式"

    ```yaml
    trainer:
      colocate_timeout_s: 60              # 默认: 60s，Ray 级别超时
      colocate_flattened_fail_fast: true   # 默认: true，bucket 同步失败时不回退
    ```

    共置模式同步失败时：

    1. 检查 GPU 显存：trainer 和 rollout 共享同一 GPU
    2. 降低 `rollout.gpu_memory_utilization`（共置模式下自动限制为 0.45）
    3. 启用 `actor.megatron.param_offload`（共置模式下自动强制启用）

## 训练 Loss 调试

!!! tip "核心要点"
    训练 50+ 步后 loss 不下降，几乎都是数据或 reward 问题，而非超参数问题。先检查 reward 分布。

### Loss 不下降检查清单

1. **验证 reward 非零**：检查 wandb 或控制台中的 `data/reward_mean` 指标
2. **验证 response_mask 正确**：非零 mask 计数应与生成的 token 数匹配
3. **检查优势函数归一化**：`norm_adv_by_std_in_grpo=True`（默认）在所有 reward 相同时会导致零优势
4. **检查学习率**：默认 `lr=1e-6` 对某些任务可能太低
5. **检查梯度裁剪**：`OptimizerArguments` 中默认 `clip_grad=1.0`

### KL 散度爆炸

KL 散度衡量相对参考模型的漂移。如果 KL 快速增长：

```yaml
actor_ref:
  actor:
    use_kl_loss: true            # 默认: false
    kl_loss_coef: 0.001          # 默认: 0.001，增大以约束漂移
    kl_loss_type: "low_var_kl"   # 默认: "low_var_kl"
    clip_ratio: 0.2              # 默认: 0.2，更紧的裁剪减少漂移
```

### 梯度范数异常

启用内存分析以跟踪每步梯度范数：

```bash
export SIIRL_MEMORY_STEP_PROFILE=1
```

`MemoryProfiler`（memory_profiler.py:424）启用后会记录每步的峰值显存。要针对梯度做调试，可使用 `memory_trace` 上下文管理器：

```python
from siirl.utils.logger.memory_profiler import memory_trace

with memory_trace("gradient_computation"):
    loss.backward()
```

### loss_agg_mode 的影响

`ActorArguments` 中的 `loss_agg_mode` 参数（model_args.py:132）控制逐 token loss 的聚合方式：

| 模式                    | 行为                       | 适用场景            |
| --------------------- | ------------------------ | --------------- |
| `token-mean`（默认）      | 对所有有效 token 取均值          | 标准训练            |
| `seq-mean-token-sum`  | 每个序列先对 token 求和，再跨序列取均值  | 序列长度差异大时        |
| `seq-mean-token-mean` | 每个序列先对 token 取均值，再跨序列取均值 | 无论序列长度，每个序列权重相同 |

## 常见错误信息

| 错误                                     | 位置                                          | 原因                      | 修复                                                            |
| -------------------------------------- | ------------------------------------------- | ----------------------- | ------------------------------------------------------------- |
| `NCCL timeout`                         | `TrainerGroup.init_actors`                  | 节点间网络问题                 | 设置 `NCCL_DEBUG=INFO`，检查防火墙规则，验证 `MASTER_ADDR`/`MASTER_PORT`   |
| `CUDA out of memory`                   | `Trainer.train_step`                        | 批次太大                    | 参见 [OOM 处理指南](handling_oom.md)                                |
| `Ray actor died unexpectedly`          | 任意 actor                                    | OOM kill 或未处理异常         | 检查 `/tmp/ray/session_latest/logs/` 和系统 `dmesg` 中的 OOM kill 记录 |
| `Missing TRAIN_MASTER_PORT`            | `TrainerGroup._resolve_master_endpoint`     | 多节点未配置端口                | 设置 `TRAIN_MASTER_PORT` 环境变量                                   |
| `Param sync is unhealthy`              | `ParamSyncDistributed.update_weights_mixed` | 之前的同步组建立失败              | 重启训练；检查 NCCL 连通性                                              |
| `Trainer distributed layout mismatch`  | `TrainerGroup._validate_distributed_setup`  | 资源分配不一致                 | 验证 `trainer.actor_gpus` 和 `trainer.nnodes` 与集群匹配              |
| `NotImplementedError: Reward function` | `reward_score/__init__.py`                  | 未知 `data_source` 值      | 通过 `custom_reward_function.path` 添加自定义 reward                 |
| `Training failed: [source] reason`     | `MainRunner._wait_for_completion`           | 某组件报告失败                 | 检查日志中 `TaskCoordinator` 事件以定位根因                               |
| `bootstrap-phase offload failure`      | `MainRunner.run`（共置模式）                      | 初始化时 rollout offload 失败 | 降低 `rollout.gpu_memory_utilization`，检查 GPU 显存                 |
| `Timeout waiting for batch`            | `Trainer.train` 循环                          | Rollout 产生数据过慢          | 增加 `trainer.async_factor`（默认：`1`），增加 rollout GPU              |

## 环境变量参考

| 变量                            | 默认值          | 用途                          |
| ----------------------------- | ------------ | --------------------------- |
| `SIIRL_LOG_DIRECTORY`         | `siirl_logs` | 应用日志输出目录                    |
| `SIIRL_MEMORY_DEBUG`          | `0`          | 启用详细内存日志（设为 `1` 启用）         |
| `SIIRL_MEMORY_PROFILE`        | `0`          | 导出第一步的内存快照（设为 `1` 启用）       |
| `SIIRL_MEMORY_STEP_PROFILE`   | `0`          | 每步记录峰值显存（设为 `1` 启用）         |
| `SIIRL_VERBOSE_VALIDATE_LOGS` | `0`          | 详细验证计时日志（设为 `1` 启用）         |
| `LOGURU_LEVEL`                | `INFO`       | loguru 日志级别（`DEBUG` 获取最大详情） |
| `RAY_DEDUP_LOGS`              | （Ray 默认值）    | 设为 `0` 禁用 Ray 日志去重          |
| `NCCL_DEBUG`                  | `WARN`       | NCCL 调试级别（`INFO` 用于连接诊断）    |
| `NCCL_CUMEM_ENABLE`           | `0`          | NCCL CUDA 托管内存              |
