GRPO 训练
=========

   **适合谁：** 使用 siirl-agentic 运行 GRPO（组相对策略优化）的用户。

   **你将获得：** GRPO 专属配置、分组采样机制说明与调优建议。

概述
----

GRPO（Group Relative Policy Optimization）是一种无 Critic 的 RL 算法，通过比较**同一 prompt 的一组响应**的奖励来估计优势值。这省去了独立的 Critic 模型，降低了 GPU 内存占用并简化了训练流程。

GRPO 专属配置
-------------

.. code:: yaml

   actor_ref:
     algorithm:
       adv_estimator: grpo               # 使用 GRPO 优势估计
       norm_adv_by_std_in_grpo: true     # 用组内标准差归一化优势

     actor:
       ppo_mini_batch_size: 256
       ppo_micro_batch_size_per_gpu: 8
       clip_ratio: 0.2
       ppo_epochs: 1
       optim:
         lr: 1e-6

   rollout:
     n: 8                                # 每个 prompt 生成 8 个响应
     temperature: 1.0                    # 采样温度
     do_sample: true

   data:
     train_batch_size: 512               # 每步 prompt 数（总样本 = 512 × 8 = 4096）

最小可运行示例
--------------

.. code:: bash

   export MODEL_PATH=/path/to/Qwen3-8B
   export TRAIN_DATA_PATH=/path/to/train.parquet
   export TEST_DATA_PATH=/path/to/test.parquet

   bash examples/grpo_train/run_qwen3_8b_separated.sh

GRPO 工作原理
-------------

.. mermaid::

   graph TB
   A[Prompt] --> B[生成 N=8 个响应]
   B --> C[响应 1: reward=0.8]
   B --> D[响应 2: reward=0.2]
   B --> E[响应 3: reward=1.0]
   B --> F[...]
   B --> G[响应 8: reward=0.5]
   C --> H[组均值=0.6, 标准差=0.3]
   D --> H
   E --> H
   F --> H
   G --> H
   H --> I[优势 = reward - 均值 / 标准差]
   I --> J[PPO 风格裁剪 loss]


对每个 prompt，GRPO 执行：

1. 生成 ``n`` 个响应（默认 8 个）
2. 计算每个响应的奖励
3. 计算组相对优势：``adv_i = (reward_i - mean) / std``
4. 使用 PPO 风格裁剪目标更新策略

关键参数
--------

+---------------------------------------+----------------------------------------------------+-------------------+
| 参数                                  | 影响                                               | 推荐范围          |
+=======================================+====================================================+===================+
| ``rollout.n``                         | 组大小（越大估计越准，计算开销越大）               | 4 到 16           |
+---------------------------------------+----------------------------------------------------+-------------------+
| ``rollout.temperature``               | 响应多样性                                         | 0.8 到 1.2        |
+---------------------------------------+----------------------------------------------------+-------------------+
| ``algorithm.norm_adv_by_std_in_grpo`` | 优势归一化                                         | true（推荐）      |
+---------------------------------------+----------------------------------------------------+-------------------+
| ``actor.optim.lr``                    | 学习率                                             | 1e-7 到 5e-6      |
+---------------------------------------+----------------------------------------------------+-------------------+
| ``data.train_batch_size``             | 每步 prompt 数量                                   | 128 到 1024       |
+---------------------------------------+----------------------------------------------------+-------------------+

GRPO 在 Agentic 任务中的优势
-----------------------------

GRPO 特别适合 agentic 训练：

1. **无 Critic 模型** — 节省 GPU 内存，在工具环境占用资源时尤为重要
2. **组内多样性** — 同一 prompt 的多个响应探索不同工具使用策略
3. **简单奖励信号** — 适用于 agentic 任务中常见的二元/稀疏奖励（通过/失败）
4. **可扩展** — 总 rollout 样本 = ``batch_size × n``，天然并行

Agentic GRPO 示例
-----------------

.. code:: yaml

   # GRPO + 多轮工具交互
   actor_ref:
     algorithm:
       adv_estimator: grpo
     actor:
       ppo_mini_batch_size: 128

   rollout:
     n: 8
     flow_function: naive
     multiturn:
       env_type: tool_env
       max_env_turns: 5
       max_assistant_turns: 10

   data:
     train_batch_size: 256
     max_response_length: 8192   # 多轮轨迹需要更长的长度

常见问题
--------

.. list-table::
   :header-rows: 1

   * - 现象
     - 原因
     - 解决方案
   * - 组内所有奖励相同
     - 采样温度过低
     - 提高 ``rollout.temperature``
   * - 优势方差过大
     - 组大小太小
     - 增大 ``rollout.n``
   * - Rollout 速度慢
     - ``n × batch_size`` 过大
     - 减小其中一个，或增加 rollout GPU 数
   * - 训练不稳定
     - ``norm_adv_by_std_in_grpo=false``
     - 设置为 ``true``
   * - Rollout OOM
     - 并发样本过多
     - 减小 ``train_server_concurrency``
