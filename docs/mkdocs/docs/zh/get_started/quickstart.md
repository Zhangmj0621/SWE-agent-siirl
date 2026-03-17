# 快速开始

*在单节点 8 GPU 上运行第一个 GRPO 训练任务。*

## 前提条件

-   siirl-agentic 已安装（参见 [安装指南](installation.md)）
-   8 块 GPU 可用（4 块用于训练，4 块用于 rollout）
-   已下载模型（如 Qwen3-8B）到本地
-   Parquet 格式的训练数据

!!! tip "准备 Parquet 数据"
    如果你的数据是 JSON/JSONL 格式，可以使用 `datasets` 库转换为 Parquet：
    ```python
    from datasets import load_dataset
    ds = load_dataset("json", data_files="train.jsonl")
    ds["train"].to_parquet("train.parquet")
    ```
    每行数据至少需要一个 `prompt` 字段（字符串或聊天消息列表）。

## 第一步：准备环境

``` bash
# 设置路径
export HOME_DIR=/path/to/your/home
export MODEL_PATH=$HOME_DIR/data/models/Qwen3-8B
export TRAIN_DATA_PATH=$HOME_DIR/data/datasets/deepscaler/train.parquet
export TEST_DATA_PATH=$HOME_DIR/data/datasets/deepscaler/test.parquet
```

## 第二步：运行 GRPO 训练（Separated 模式）

``` bash
cd siirl-agentic
bash examples/grpo_train/run_qwen3_8b_separated.sh
```

该脚本配置了：

- **4 块 GPU 用于 Actor**（TP=4 训练）
- **4 块 GPU 用于 Rollout**（SGLang 推理，TP=2）
- **GRPO 算法**，batch_size=512，每个 prompt 采样 n=8 个响应
- **Qwen3-8B 模型**，max_prompt=2048，max_response=4096

脚本通过 OmegaConf CLI 参数传递配置。你可以通过编辑脚本或追加覆盖参数来自定义任何配置：

``` bash
bash examples/grpo_train/run_qwen3_8b_separated.sh \
    data.train_batch_size=256 \
    rollout.temperature=0.8
```

**预期输出：**

```
INFO  | Ray is initialized. Time cost: 150.23 ms
INFO  | MainRunner started. Beginning workflow setup...
INFO  | Initializing DataCoordinator...
SUCCESS | DataCoordinator initialized
SUCCESS | RolloutManager initialized. Router at: http://10.0.0.1:30000
SUCCESS | TrainerGroup initialized with 4 trainers
INFO  | Starting async training loop...
```

!!! note
    具体数字（批次/epoch、总步数）取决于你的数据集大小和 `train_batch_size` 设置。

## 第三步：运行 PPO 训练

``` bash
bash examples/ppo_train/run_qwen3_8b_separated.sh
```

PPO 在 GRPO 的基础上增加了一个 Critic 模型。脚本会相应地调整资源分配。

## 第四步：Colocated 模式（共享 GPU）

适用于小规模配置或追求最大 GPU 利用率的场景：

``` bash
bash examples/grpo_train/run_qwen3_8b_colocate.sh
```

在 colocated 模式下，训练和 rollout 共享全部 8 块 GPU。框架自动管理权重卸载。

## 关键调优参数

| 参数                                            | 控制内容           | 推荐值                  |
| --------------------------------------------- | -------------- | -------------------- |
| `data.train_batch_size`                       | 每个训练步的样本数      | 从 512 开始             |
| `rollout.n`                                   | 每个 prompt 的采样数 | GRPO 用 8，PPO 用 1     |
| `data.max_response_length`                    | 最大生成长度         | 根据任务需求设定             |
| `trainer.actor_gpus` / `trainer.rollout_gpus` | GPU 分配         | 大多数情况下均分             |
| `rollout.tensor_model_parallel_size`          | 推理张量并行度        | 8B 模型用 2，70B+ 模型用 4+ |

!!! tip "首次运行的批量大小建议"
    首次运行时建议使用较小的批量（`data.train_batch_size=128`），确认一切正常后再扩大规模。大批量需要按比例更多的 GPU 显存用于激活值。如果出现 OOM 错误，优先将批量大小减半，再调整其他设置。

## 监控

训练日志输出到 stdout，可选输出到 WandB。通过 CLI 覆盖参数配置：

``` bash
# 示例：启用 WandB 日志
python -m siirl.async_train \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=siirl_examples \
    trainer.experiment_name=my_first_run
```

## 运行成功的标志

完成本指南后，你应该看到：

- Ray 初始化成功：`INFO | Ray is initialized. Time cost: ...`
- 所有组件启动完成：`SUCCESS | RolloutManager initialized`、`SUCCESS | TrainerGroup initialized`
- 训练循环开始：`INFO | Starting async training loop...`
- 日志中出现 `reward/mean` 指标，并随步数逐渐上升
- 无 CUDA OOM 错误或 Ray actor 崩溃
- 每个 epoch 结束时，checkpoint 已保存到配置的输出目录

**健康训练运行的日志示例：**

```
INFO  | Ray is initialized. Time cost: 150.23 ms
SUCCESS | DataCoordinator initialized
SUCCESS | RolloutManager initialized. Router at: http://10.0.0.1:30000
SUCCESS | TrainerGroup initialized with 4 trainers
INFO  | Starting async training loop...
INFO  | [Step 1] rollout started, batch_size=512
INFO  | [Step 1] rollout done. reward/mean=0.12, reward/std=0.08
INFO  | [Step 1] training started
INFO  | [Step 1] training done. loss=1.42, grad_norm=0.87
INFO  | [Step 10] rollout done. reward/mean=0.31, reward/std=0.11
INFO  | [Step 10] training done. loss=1.18, grad_norm=0.74
INFO  | Checkpoint saved to: /path/to/output/step_10/
```

关键信号是 `reward/mean` 从第 1 步到第 10 步持续上升。起始值取决于任务和模型——重要的是上升趋势。

## 出现问题时的排查方法

| 现象                           | 可能原因                                | 解决方法                                                          |
| ---------------------------- | ----------------------------------- | ------------------------------------------------------------- |
| `CUDA out of memory`         | `rollout.n` 或 `train_batch_size` 过大 | 先减小 `rollout.n`（如从 `8` 改为 `4`），再将 `data.train_batch_size` 减半  |
| `SGLang error` 或 SGLang 启动失败 | GPU 显存不足以运行推理引擎                     | 减小 `rollout.gpu_memory_utilization`（如从 `0.85` 改为 `0.7`）       |
| `Ray actor crashed`          | Worker OOM 或 NCCL 超时                | 用 `dmesg` 检查 OOM kill 记录；设置 `RAY_memory_monitor_refresh_ms=0` |
| rollout 完成后训练挂起              | 权重同步时 NCCL 死锁                       | 设置 `NCCL_TIMEOUT=1800` 后重启                                    |
| `reward/mean` 始终为 0          | 奖励函数对所有输出返回 0                       | 先用小批量数据测试奖励函数逻辑                                               |

更详细的排查步骤请参阅[故障排除](../reference/troubleshooting.md)。

## 下一步

- [GRPO 训练](../guides/grpo_training.md) — 深入了解分组采样配置和 GRPO 专属调优参数
- [首个 Agentic 训练任务](first_agentic_training_job.md) — 为训练添加工具交互，启用多轮 rollout
- [配置系统](../guides/configuration_system.md) — 了解如何通过 CLI 使用 Hydra 覆盖任意参数
