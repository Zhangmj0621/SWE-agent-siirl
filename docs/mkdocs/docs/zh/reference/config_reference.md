# 配置参考

*所有 `SiiRLArguments` 参数的完整参考，含默认值与中文说明。*

## 必填参数

以下 15 个参数在每次训练中都需要设置，其余参数可从默认值开始：

| 参数                                      | 默认值     | 说明                                         |
| --------------------------------------- | ------- | ------------------------------------------ |
| `actor_ref.model.path`                  | —       | HuggingFace 模型路径或本地目录（必填）                  |
| `actor_ref.algorithm.adv_estimator`     | `grpo`  | 算法：`grpo`（无 critic）或 `ppo`（有 critic）       |
| `rollout.n`                             | 1       | 每个 prompt 生成的响应数；GRPO 需 ≥ 4 以获得组内方差        |
| `rollout.temperature`                   | 1.0     | 采样温度；推荐 0.7，0.0 会消除 GRPO 多样性               |
| `rollout.multiturn.max_env_turns`       | 1       | 每条轨迹的最大工具交互轮数（agentic 任务需 > 1）             |
| `rollout.multiturn.max_assistant_turns` | 1       | 每条轨迹的最大助手响应总数                              |
| `trainer.total_epochs`                  | 30      | 训练 epoch 数；大多数提升在前 30 个 epoch 内体现          |
| `trainer.nnodes`                        | 1       | 训练节点数                                      |
| `trainer.n_gpus_per_node`               | 8       | 每个节点的 GPU 数                                |
| `trainer.actor_gpus`                    | 2       | 训练预留 GPU 数（Actor + Ref + 可选 Critic）        |
| `trainer.rollout_gpus`                  | 6       | SGLang rollout 引擎预留 GPU 数                  |
| `trainer.colocate`                      | `false` | 训练与 rollout 是否共享 GPU（先用 offload 模式验证）      |
| `trainer.learning_rate`                 | 1e-6    | warmup 后的峰值学习率                             |
| `data.max_response_length`              | 512     | 响应最大长度，单位为**字符**（非 token）；英文目标 token 数乘以 4 |
| `trainer.save_freq`                     | -1      | 检查点保存频率（步数）；默认 -1 禁用保存——必须设置此参数            |

## 概述

siirl-agentic 的所有配置由 `SiiRLArguments` 数据类表示，包含六个顶层命名空间：

``` yaml
data:                    # DataArguments — 数据集路径、批次大小、分词
actor_ref:               # ActorRefArguments — Actor、Ref、算法、检查点
rollout:                 # RolloutArguments — 推理引擎、采样、多轮
critic:                  # CriticArguments — Critic 模型（仅 PPO）
trainer:                 # TrainingArguments — 训练循环与资源
custom_reward_function:  # CustomRewardArguments — 自定义奖励配置
```

使用方法参见 [配置系统](../guides/configuration_system.md)。

## 关键参数速查

### 数据参数（`data:`）

  参数                    默认值       说明
  ----------------------- ------------ ---------------------------------------
  `train_files`           GSM8K 示例   训练数据集路径（Parquet 格式）
  `val_files`             GSM8K 示例   验证数据集路径
  `train_batch_size`      1024         每步样本数
  `max_prompt_length`     512          prompt 最大 token 长度
  `max_response_length`   512          响应最大 token 长度（多轮应设 4096+）
  `mask_history`          false        只在最后一轮计算 loss
  `train_on_prompt`       false        prompt token 是否参与 loss

### 训练参数（`trainer:`）

  -------------------------------------------------------------------------
  参数                          默认值     说明
  ----------------------------- ---------- --------------------------------
  `total_epochs`                30         训练总 epoch 数

  `actor_gpus`                  2          训练用 GPU 数

  `rollout_gpus`                6          Rollout 用 GPU 数

  `colocate`                    false      共享训练和 rollout GPU

  `async_factor`                1          Rollout 批次预缓冲数量

  `off_policy_step`             0          Off-policy 版本容忍窗口

  `save_freq`                   -1         检查点保存频率（-1=禁用）

  `test_freq`                   -1         验证频率（-1=禁用）

  `resume_mode`                 \"auto\"   auto / disable / resume_path

  `validate_reuse_train_gpus`   false      验证时复用训练 GPU
  -------------------------------------------------------------------------

### Rollout 参数（`rollout:`）

  -----------------------------------------------------------------------------------
  参数                           默认值       说明
  ------------------------------ ------------ ---------------------------------------
  `name`                         \"sglang\"   推理引擎（当前只支持 sglang）

  `temperature`                  1.0          采样温度

  `n`                            1            每个 prompt 的响应数（GRPO 通常为 8）

  `gpu_memory_utilization`       0.5          SGLang GPU 内存占用比例

  `tensor_model_parallel_size`   1            推理 TP 并行度

  `train_server_concurrency`     256          每引擎最大并发请求数

  `flow_function`                \"naive\"    rollout flow 实现

  `max_model_len`                null         最大序列长度（null=从模型推断）
  -----------------------------------------------------------------------------------

### 多轮参数（`rollout.multiturn:`）

  --------------------------------------------------------------------------------
  参数                           默认值       说明
  ------------------------------ ------------ ------------------------------------
  `env_type`                     null         环境类型：tool_env / vla_env

  `max_env_turns`                1            最大环境交互轮数

  `max_assistant_turns`          1            最大模型生成轮数

  `max_parallel_calls`           1            每个样本的并发工具调用数

  `max_env_response_length`      256          最大环境响应 token 数

  `env_response_truncate_side`   \"middle\"   截断方向：left / middle / right
  --------------------------------------------------------------------------------

### Actor/算法参数（`actor_ref.algorithm:`）

  --------------------------------------------------------------------------
  参数                        默认值     说明
  --------------------------- ---------- -----------------------------------
  `adv_estimator`             \"grpo\"   优势估计方法：grpo / ppo

  `norm_adv_by_std_in_grpo`   true       GRPO 是否用组内标准差归一化优势

  `gamma`                     1.0        折扣因子（PPO GAE）

  `lam`                       1.0        GAE lambda

  `kl_penalty`                \"kl\"     KL 惩罚类型
  --------------------------------------------------------------------------

## 完整参数表

完整的 702 个参数表格（含所有默认值和详细说明）请参阅英文版：

-   [Configuration Reference (English)](../../en/reference/config_reference.md)

或直接查看源代码：

-   `siirl/params/training_args.py` — TrainingArguments、SiiRLArguments
-   `siirl/params/model_args.py` — ModelArguments、ActorArguments、RolloutArguments 等
-   `siirl/params/data_args.py` — DataArguments

## 下一步

- [配置系统](../guides/configuration_system.md) — 了解 Hydra 如何组合这些参数，以及如何从 CLI 覆盖它们
- [最佳实践](best_practices.md) — 了解哪些参数最关键以及常见训练场景的推荐值
