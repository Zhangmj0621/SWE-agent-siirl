原生 Agentic 轨迹训练
=====================

   **适合谁：** 需要在多轮工具交互轨迹上训练 LLM 策略的用户。

   **你将获得：** 理解 siirl-agentic 如何原生处理多轮轨迹的 token 级 loss mask 和 log-probability 追踪。

核心设计
--------

siirl-agentic 的 ``NaiveFlow``\ （\ ``siirl/execution/rollout/agent_flow/naive_flow.py``\ ）是一个多轮 rollout 状态机：

::

   PENDING → GENERATING → PROCESSING_ENV → GENERATING → ... → TERMINATED

关键特性
~~~~~~~~

1. **按轮 Loss Mask** — 每一轮的模型 token（需要训练）和环境 token（不需要训练）被精确区分。Loss mask 在整个轨迹上是 token 级别的。

2. **完整 Log-Probability 追踪** — 跨越整个多轮交互的 ``rollout_log_probs`` 被保留，用于 PPO/GRPO 的 importance sampling ratio 计算。

3. **并行工具执行** — ``max_parallel_calls`` 支持同一样本内多个工具调用并行执行，减少单轮等待时间。

4. **灵活终止条件** — 支持 ``max_env_turns``\ 、\ ``max_assistant_turns``\ 、环境完成信号等多种终止条件。

关键配置
~~~~~~~~

.. code:: yaml

   rollout:
     multiturn:
       env_type: tool_env
       max_env_turns: 5
       max_assistant_turns: 10
       max_parallel_calls: 4

与通用框架对比
~~~~~~~~~~~~~~

========= ============ =============
特性      通用 RL 框架 siirl-agentic
========= ============ =============
多轮支持  外部 wrapper 原生状态机
Loss mask 整序列       Token 级按轮
工具调用  不支持       一等操作
========= ============ =============

..

   详细英文版请参阅 `Native Agentic Trajectory Training (English) <../../en/highlights/native_agentic_trajectory_training.html>`__
