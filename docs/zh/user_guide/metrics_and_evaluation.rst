指标与评估
==========

   **适合谁：** 需要监控训练进度和评估模型性能的用户。

   **你将获得：** 可用指标列表、日志后端配置与评估设置说明。

日志后端
--------

.. code:: yaml

   trainer:
     logger: ["console", "wandb"]
     project_name: siirl_examples
     experiment_name: my_experiment

支持的后端：``console``、``wandb``

核心训练指标
------------

==================== ============================
指标                 说明
==================== ============================
``reward/mean``      每步平均奖励
``reward/std``       奖励标准差
``actor/loss``       Actor 策略损失
``actor/clip_ratio`` 被裁剪更新的比例
``critic/loss``      Critic 价值损失（仅 PPO）
``kl/mean``          与参考模型的 KL 散度
``lr``               当前学习率
==================== ============================

Rollout 指标
------------

================================ =======================
指标                              说明
================================ =======================
``rollout/generation_duration``  LLM 生成耗时
``rollout/reward_duration``      奖励计算耗时
``rollout/total_tokens``         生成的总 token 数
``rollout/response_length_mean`` 平均响应长度
================================ =======================

Agentic 专属指标
----------------

当启用多轮交互时，额外记录以下指标：

================================ =======================
指标                              说明
================================ =======================
``rollout/env_turns_mean``       每个样本平均工具交互轮数
``rollout/tool_call_count``      工具调用总次数
================================ =======================

验证配置
--------

.. code:: yaml

   trainer:
     test_freq: 10          # 每 N 步验证一次
     val_before_train: true # 首步训练前先验证
     log_val_generations: 5 # 记录 N 个验证样本

   data:
     val_batch_size: null   # null = 使用完整验证集

MetricWorker
------------

``MetricWorker`` Ray Actor 聚合来自所有训练 rank 的指标：

- ``siirl/utils/metrics/`` — 指标收集与聚合
- 只有 TrainerGroup 的 rank 0 向日志后端上报
