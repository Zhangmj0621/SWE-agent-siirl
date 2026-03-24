# 算法基线

> siirl-agentic 在标准基准测试上的参考性能数据。

!!! warning "仅为近似结果"
    以下数据为**近似范围**，来源于已发表文献（DeepSeek-R1、Qwen2.5/Qwen3 技术报告）和内部测试。实际结果会因超参数、硬件、数据预处理和随机种子而异。我们鼓励您提交您的结果，帮助完善这些基线。

## 数学推理

!!! tip "核心要点"
    数学推理是研究最充分的 RL 后训练基准。推荐从 GRPO 入手——它不需要 critic 模型，GPU 显存占用更低，且能达到有竞争力的效果。

| 模型           | 算法   | 数据集        | 指标  | 近似范围    | GPU        | 备注                                              |
| ------------ | ---- | ---------- | --- | ------- | ---------- | ----------------------------------------------- |
| Qwen3-8B     | GRPO | DeepScaleR | 准确率 | ~70-80% | 8xA100-80G | `rollout.n=8`, `actor_ref.actor.optim.lr=1e-6`  |
| Qwen3-8B     | PPO  | DeepScaleR | 准确率 | ~68-78% | 8xA100-80G | 需要 critic（`critic.optim.lr=5e-6`）               |
| Qwen2.5-7B   | GRPO | GSM8K      | 准确率 | ~75-82% | 8xA100-80G | `rollout.n=16`, `actor_ref.actor.optim.lr=1e-6` |
| Qwen2.5-7B   | PPO  | GSM8K      | 准确率 | ~72-80% | 8xA100-80G | 建议 `trainer.critic_warmup=10`                   |
| Qwen2.5-1.5B | GRPO | GSM8K      | 准确率 | ~55-65% | 4xA100-80G | 小模型，迭代更快                                        |
| Qwen2.5-7B   | GRPO | MATH       | 准确率 | ~45-55% | 8xA100-80G | 更长推理链，`data.max_response_length=4096`           |
| Qwen2.5-7B   | PPO  | MATH       | 准确率 | ~42-52% | 8xA100-80G |                                                 |

## 代码生成

!!! tip "核心要点"
    代码基准测试受益于代码专用的基础模型和基于沙箱执行的奖励。使用更长的 `data.max_response_length` 以容纳代码输出。

| 模型               | 算法   | 数据集          | 指标  | 近似范围    | GPU         | 备注       |
| ---------------- | ---- | ------------ | --- | ------- | ----------- | -------- |
| Qwen2.5-7B       | GRPO | CodeContests | 通过率 | ~15-25% | 16xA100-80G | 沙箱执行奖励   |
| Qwen2.5-7B-Coder | GRPO | APPS         | 通过率 | ~30-45% | 8xA100-80G  | 代码专用基础模型 |

## 软件工程（多轮对话）

!!! tip "核心要点"
    SWE 基准测试考察 agent 通过多轮工具调用修复真实 bug 的能力。需要 AIO 工具基础设施，配置 `rollout.multiturn.env_type=tool_env` 且 `rollout.multiturn.max_env_turns > 1`。

| 模型       | 算法   | 数据集            | 指标  | 近似范围    | GPU        | 备注             |
| -------- | ---- | -------------- | --- | ------- | ---------- | -------------- |
| Qwen3-8B | GRPO | SWE-bench Lite | 解决率 | ~15-25% | 8xA100-80G | 多轮对话，Docker 沙箱 |

## 各基准推荐超参数

!!! tip "核心要点"
    建议从以下配置开始。参数名与 `python3 -m siirl.async_train` 的 CLI 参数完全对应。

| 基准           | 推荐算法 | `rollout.n` | `actor_ref.actor.optim.lr` | `data.max_response_length` | 关键设置                                                                           |
| ------------ | ---- | ----------- | -------------------------- | -------------------------- | ------------------------------------------------------------------------------ |
| GSM8K        | GRPO | 16          | `1e-6`                     | 2048                       | `actor_ref.algorithm.norm_adv_by_std_in_grpo=True`                             |
| MATH         | GRPO | 16          | `5e-7`                     | 4096                       | 更长推理链                                                                          |
| DeepScaleR   | GRPO | 8           | `1e-6`                     | 4096                       | `data.train_batch_size=512`                                                    |
| CodeContests | GRPO | 8           | `1e-6`                     | 4096                       | 沙箱奖励函数                                                                         |
| SWE-bench    | GRPO | 8           | `1e-6`                     | 4096                       | `rollout.multiturn.max_env_turns=2`, `rollout.multiturn.max_assistant_turns=2` |

### GRPO 专用设置

```bash
# GRPO 优势估计（来自 AlgorithmArguments）
actor_ref.algorithm.adv_estimator=grpo
actor_ref.algorithm.gamma=1.0            # 折扣因子（默认: 1.0）
actor_ref.algorithm.lam=1.0              # GAE lambda（默认: 1.0）
actor_ref.algorithm.norm_adv_by_std_in_grpo=True  # 标准 GRPO 归一化

# Actor 训练（来自 ActorArguments）
actor_ref.actor.clip_ratio=0.2           # PPO 裁剪比率（默认: 0.2）
actor_ref.actor.ppo_epochs=1             # 每批次梯度更新次数（默认: 1）
actor_ref.actor.use_kl_loss=True         # KL 散度正则化
actor_ref.actor.kl_loss_coef=0.01        # KL 损失权重
actor_ref.actor.kl_loss_type=low_var_kl  # 低方差 KL 估计器
```

### PPO 专用设置

```bash
# PPO 在 GRPO 设置基础上额外需要 critic 模型
actor_ref.algorithm.adv_estimator=ppo

# Critic 配置（来自 CriticArguments）
critic.model.path=/path/to/model         # 通常与 actor 使用同一模型
critic.optim.lr=5e-6                     # critic 学习率，通常为 actor 的 5 倍
critic.ppo_epochs=1                      # critic 更新轮数（默认: 1）
critic.cliprange_value=0.5               # 值函数裁剪范围（默认: 0.5）
critic.ppo_mini_batch_size=256           # critic mini-batch 大小
critic.ppo_micro_batch_size_per_gpu=8    # 每 GPU micro-batch 大小
```

### Dr. GRPO 变体

使用 Dr. GRPO 变体（[arXiv:2503.20783](https://arxiv.org/abs/2503.20783)），跳过标准差归一化：

```bash
actor_ref.algorithm.norm_adv_by_std_in_grpo=False
```

## 复现结果

详细的逐步复现指南请参阅：

- [DeepScaleR GRPO 教程](../tutorials/deepscaler_grpo.md)
- [SWE Agent 训练教程](../tutorials/swe_agent_training.md)

提供的示例脚本是最佳起点：

=== "GRPO（分离模式）"

    ```bash
    # 8 GPU: 4 训练 + 4 推理
    bash examples/grpo_train/run_qwen3_8b_separated.sh
    ```

=== "GRPO（共置模式）"

    ```bash
    # 8 GPU 在训练和推理之间共享
    bash examples/grpo_train/run_qwen3_8b_colocate.sh
    ```

=== "PPO（分离模式）"

    ```bash
    # 8 GPU: 4 训练 + 4 推理（包含 critic）
    bash examples/ppo_train/run_qwen3_8b_separated.sh
    ```

=== "AIO 多轮对话"

    ```bash
    # 8 GPU: 2 训练 + 6 推理，带工具环境
    bash examples/AIO/run_qwen3_8b.sh
    ```

## 硬件需求

!!! tip "核心要点"
    最低可用配置为分离模式 4 GPU 或共置模式 4 GPU。共置模式使用 `param_offload` 在相同 GPU 上交替进行训练和推理。

| 配置             | GPU 显存    | 最少 GPU         | 推荐 GPU | 模式                       |
| -------------- | --------- | -------------- | ------ | ------------------------ |
| 8B 分离模式        | 80 GB/GPU | 4（2 训练 + 2 推理） | 8（4+4） | `trainer.colocate=False` |
| 8B 共置模式        | 80 GB/GPU | 4              | 8      | `trainer.colocate=True`  |
| 1.5B-1.7B 分离模式 | 40 GB/GPU | 2（1+1）         | 4（2+2） | `trainer.colocate=False` |
| 1.5B-1.7B 共置模式 | 40 GB/GPU | 2              | 4      | `trainer.colocate=True`  |

### 显存优化技巧

```yaml title="快速显存优化配置"
# 第 1 级：参数卸载（节省约 30-40% GPU 显存）
actor_ref.actor.megatron.param_offload: true
actor_ref.ref.megatron.param_offload: true       # 同时卸载参考模型

# 第 2 级：优化器卸载（额外节省约 40-50%）
actor_ref.actor.megatron.optimizer_offload: true

# 第 3 级：动态批处理（防止长序列 OOM）
actor_ref.actor.use_dynamic_batch: true
actor_ref.actor.max_tokens_per_gpu: 16384

# 第 4 级：降低 SGLang 显存预留
rollout.gpu_memory_utilization: 0.5               # 默认 0.9，越低越安全
```

## 对比说明

!!! tip "核心要点"
    siirl-agentic 的核心差异化优势是原生异步多轮 agent 训练与工具交互能力，由 AIO（AgentFlow + Infrastructure + Orchestration）架构驱动。

| 特性          | siirl-agentic                       | veRL  | OpenRLHF  |
| ----------- | ----------------------------------- | ----- | --------- |
| 异步多轮        | 原生支持                                | 需手动实现 | 不支持       |
| 工具交互        | AIO 三层架构                            | 自定义   | 不支持       |
| 推理引擎        | SGLang                              | vLLM  | vLLM      |
| 训练后端        | Megatron-LM                         | FSDP  | DeepSpeed |
| 共置模式        | 支持（基于卸载）                            | 支持    | 不支持       |
| Dr. GRPO 支持 | 支持（`norm_adv_by_std_in_grpo=False`） | 不支持   | 不支持       |
| 动态批处理       | 支持（基于 token）                        | 不支持   | 不支持       |
