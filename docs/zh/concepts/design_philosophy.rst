设计哲学
========

   **适合谁：** 希望理解 siirl-agentic 架构设计原则的框架用户和贡献者。

   **你将获得：** 理解系统为何这样设计，从而做出更好的配置和扩展决策。


为什么需要专用 Agentic RL 框架？
----------------------------------

标准 RL 训练框架（verl、OpenRLHF、TRL）为 **单轮文本生成** 设计：一个 prompt 进去，一个 response 出来，算 reward，更新策略。

**Agentic 任务** 根本不同：

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - 维度
     - 单轮 RL
     - Agentic RL
   * - 交互模式
     - prompt → response
     - 多轮：生成 → 工具调用 → 环境反馈 → 生成 → …
   * - 轨迹长度
     - 固定、可预测
     - 可变，10–50+ 轮
   * - 延迟特征
     - 均匀（GPU-bound）
     - 高度可变（工具调用：100ms–30s）
   * - 奖励信号
     - 序列末尾
     - 每轮环境反馈 + 最终奖励
   * - Loss 计算
     - 所有 response tokens
     - 仅模型生成的 tokens（环境 tokens 被 mask）


三大架构原则
--------------

1. 异步 MPMD
~~~~~~~~~~~~~~

每个核心组件作为独立 Ray Actor 运行，互不阻塞。在 Agentic 场景中，工具调用延迟高度可变（100ms–30s），MPMD 架构使训练和 rollout 完全解耦：Trainer 持续更新策略，RolloutManager 持续收集新轨迹。

2. Agentic 优先的数据路径
~~~~~~~~~~~~~~~~~~~~~~~~~~

样本生命周期原生支持多轮轨迹、按轮 loss mask 和可变长序列。``Sample`` dataclass、``NaiveFlow`` 状态机和 loss 计算从设计之初就假设数据是多轮、可变长的。

3. 配置优先于代码
~~~~~~~~~~~~~~~~~~~

任务特定逻辑通过 YAML 配置注入，而非硬编码在框架内部。``AgentFlow`` 协议的三阶段 ``preprocess → generate → reward`` 和动态函数注入（``MethodType``）使得切换任务场景无需修改框架代码。


延伸阅读
----------

- :doc:`architecture_overview` — 完整组件图和数据流
- :doc:`../highlights/why_agentic_rl` — 与标准 RL 框架的详细对比
- :doc:`../user_guide/configuration_system` — 如何配置框架
