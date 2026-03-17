异步训练生命周期
==================

   **适合谁：** 需要理解端到端训练执行流程的框架用户，以及调试生命周期问题的贡献者。

   **你将获得：** 从配置解析到关停的每个阶段的详细走查，包含时序图和代码锚点。


概览
-----

siirl-agentic 的训练生命周期包含三大阶段：

1. **初始化** — 配置解析、资源分配、组件启动
2. **异步训练循环** — 持续的 rollout → 缓冲 → 训练 循环
3. **关停** — 优雅终止、检查点保存、资源清理


阶段 1：初始化
---------------

``siirl/async_train.py`` 中的 ``main()`` 函数驱动初始化：

.. mermaid::

   graph LR
       A[main] --> B[ray.init]
       B --> C[parse_config]
       C --> D[MainRunner.remote]
       D --> E[create_coordinator]
       E --> F[allocate_resources]
       F --> G[init_data_coordinator]
       F --> H[RolloutManager.remote]
       F --> I[TrainerGroup]

步骤：

1. **Ray 初始化** — 启动或连接 Ray 集群
2. **配置解析** — ``parse_config()`` 加载 YAML 配置，创建 ``SiiRLArguments``
3. **MainRunner 启动** — 专用 Ray Actor 编排整个工作流
4. **TaskCoordinator** — 集中式生命周期管理
5. **资源分配** — 根据 ``actor_gpus`` 和 ``rollout_gpus`` 分配 GPU
6. **DataCoordinator** — 初始化分布式 DataBuffer 和 DataLoader
7. **RolloutManager** — 启动 SGLang 推理引擎
8. **TrainerGroup** — 初始化 Actor、Ref、Critic 模型


阶段 2：异步训练循环
----------------------

初始化完成后，异步训练循环开始：

- **RolloutManager** 持续获取 prompt 并分发 rollout，每个 RolloutWorker 运行 NaiveFlow 与工具交互
- **DataBuffer** 收集完成的样本，积累足够样本后提供训练 batch
- **TrainerGroup** 拉取训练 batch 执行 PPO/GRPO 更新，更新后将权重同步到 SGLang 引擎

NaiveFlow 状态机：

::

   PENDING → GENERATING → PROCESSING_ENV → GENERATING → ... → TERMINATED


阶段 3：关停
--------------

通过 ``TaskCoordinator`` 协调关停：

- **正常关停** — 训练 epoch 耗尽 → ``report_completed()`` → 所有组件检测到 ``should_stop()``
- **故障关停** — 任意组件异常 → ``report_failure()`` → 所有组件停止


代码锚点
----------

- ``siirl/async_train.py`` — ``MainRunner.run()`` 编排所有阶段
- ``siirl/utils/task_coordinator.py`` — 生命周期管理
- ``siirl/worker/rollout/rollout_manager.py`` — Rollout 分发
- ``siirl/worker/actor/trainer_group.py`` — 训练协调
- ``siirl/data_coordinator/data_buffer.py`` — 样本缓冲


延伸阅读
----------

- :doc:`architecture_overview` — 组件图和职责说明
- :doc:`design_philosophy` — 系统为何这样设计
- :doc:`../advanced/failure_propagation` — 详细的故障处理行为
