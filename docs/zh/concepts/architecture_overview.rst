架构概览
========

   **适合谁：** 需要理解系统设计的框架用户，以及需要定位代码入口的贡献者。

   **你将获得：** siirl-agentic 的组件架构、数据流和训练生命周期的完整图景。

设计哲学
--------

siirl-agentic 基于三个架构原则构建：

1. **异步 MPMD** — 每个主要组件（Trainer、Rollout、DataCoordinator）作为独立的 Ray actor 运行，互不阻塞。
2. **Agentic 优先的数据路径** — 样本生命周期原生支持多轮轨迹、工具交互、逐 turn loss masking 和变长序列。
3. **配置优于代码** — 任务特定逻辑（AgentFlow、奖励函数、工具环境）通过配置注入，不硬编码在框架内部。

系统架构
--------

.. mermaid::

   graph TB
       subgraph Driver Process
           A[main] --> B[MainRunner Ray Actor]
       end

       B --> C[TaskCoordinator]
       B --> D[allocate_resources]

       D --> E[TrainerGroup]
       D --> F[RolloutManager]
       D --> G[DataCoordinator]

       G -->|DataBuffer| H[init_dataloader]

       F --> I[RolloutWorker 1..N]
       I --> J[SGLang Engine]
       I --> K[NaiveFlow / AgentFlow]
       K --> L[ToolEnv / AIO Proxy]

       E --> M[Trainer 1..N]
       M --> N[Actor Model - Megatron]
       M --> O[Ref Model - Megatron]
       M --> P[Critic Model - PPO only]

       E -->|param sync| F

组件职责
--------

MainRunner (``siirl/async_train.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``MainRunner`` 是一个 Ray actor，编排整个训练生命周期：

1. 解析 ``SiiRLArguments`` 配置
2. 分配 GPU 资源（训练 vs 推理）
3. 初始化 ``DataCoordinator``、``RolloutManager``、``TrainerGroup``
4. 启动异步训练循环
5. 通过 ``TaskCoordinator`` 监控状态
6. 处理清理和失败上报

TrainerGroup (``siirl/worker/actor/trainer_group.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

管理分布式训练 worker：

- **Actor 模型** — 正在训练的策略模型（Megatron 后端）
- **Reference 模型** — 用于 KL 散度计算的冻结副本
- **Critic 模型** — 价值函数估计器（仅 PPO 使用，GRPO 不需要）
- **参数同步** — 每个训练步后将更新的权重推送到 RolloutManager 的 SGLang 引擎

关键文件：

- ``siirl/worker/actor/trainer.py`` — 每个 rank 的训练逻辑（前向、loss、反向、优化器步）
- ``siirl/worker/actor/trainer_group.py`` — 多 rank 协调
- ``siirl/worker/actor/checkpoint_manager.py`` — 保存/加载检查点
- ``siirl/engine/param_sync/`` — 权重同步到推理引擎

RolloutManager (``siirl/worker/rollout/rollout_manager.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

管理推理引擎并调度 rollout：

- **SGLang 引擎** — 一个或多个 SGLang 实例用于快速 LLM 推理
- **Router** — 在多引擎间负载均衡请求
- **NaiveFlow** — 带工具交互的多轮 rollout 执行
- **验证** — 使用不同采样参数的独立验证 rollout

关键文件：

- ``siirl/worker/rollout/rollout_manager.py`` — 引擎生命周期、请求调度
- ``siirl/worker/rollout/rollout_worker.py`` — 每个引擎的 rollout 执行
- ``siirl/execution/rollout/agent_flow/naive_flow.py`` — 多轮状态机
- ``siirl/execution/rollout/concurrency.py`` — 并发参数解析

DataCoordinator (``siirl/data_coordinator/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

管理从 prompt 到训练的样本生命周期：

- **Dataloader** — 加载和分发训练/验证数据集
- **DataBuffer** — 缓冲已完成的 rollout 样本供训练消费
- **Off-policy 支持** — 接受可配置版本窗口内的数据

关键文件：

- ``siirl/data_coordinator/data_buffer.py`` — 分布式缓冲逻辑
- ``siirl/data_coordinator/dataloader/`` — 数据集加载和分发
- ``siirl/data_coordinator/sample.py`` — ``Sample`` 数据类
- ``siirl/data_coordinator/protocol.py`` — 数据交换协议

TaskCoordinator (``siirl/utils/task_coordinator.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

分布式训练的集中式生命周期管理：

- ``should_stop()`` — 所有组件轮询以检查训练是否应结束
- ``report_failure(source, reason)`` — 任何组件上报失败以便传播
- ``report_completed(source)`` — 成功完成信号
- ``request_shutdown(reason, source)`` — 优雅关闭请求
- 事件日志用于事后调试

训练生命周期
------------

初始化阶段
~~~~~~~~~~

.. mermaid::

   graph LR
       A[解析配置] --> B[初始化 Ray]
       B --> C[分配 GPU]
       C --> D[初始化 DataCoordinator]
       C --> E[初始化 RolloutManager]
       D --> F[初始化 Dataloader]
       E --> G[启动 SGLang 引擎]
       C --> H[初始化 TrainerGroup]
       H --> I[加载检查点 if resume]

训练循环（异步）
~~~~~~~~~~~~~~~~

.. mermaid::

   graph TB
       A[DataCoordinator] -->|prompt batch| B[RolloutManager]
       B -->|async rollout| C[NaiveFlow per sample]
       C -->|tool calls| D[ToolEnv / AIO]
       D -->|responses| C
       C -->|completed samples| E[DataBuffer]
       E -->|training batch| F[TrainerGroup]
       F -->|forward + backward| G[Actor Update]
       F -->|compute ref logprob| H[Ref Forward]
       F -->|value estimate| I[Critic Update PPO only]
       G -->|param sync| B

关闭阶段
~~~~~~~~

1. 训练 epoch 用尽 → ``coordinator.report_completed()``
2. 所有组件检测到 ``should_stop() == True``
3. ``MainRunner._cleanup_and_report()`` 记录摘要
4. Ray 关闭

部署模式
--------

Separated 模式（默认）
~~~~~~~~~~~~~~~~~~~~~~

GPU 在训练和推理之间分割：

::

   节点 (8 GPUs):
     GPU 0-3: Actor/Ref/Critic (TrainerGroup)
     GPU 4-7: SGLang Engines (RolloutManager)

配置: ``trainer.actor_gpus=4, trainer.rollout_gpus=4, trainer.colocate=false``

Colocated 模式
~~~~~~~~~~~~~~

训练和推理共享相同的 GPU，通过权重 offload 实现：

::

   节点 (8 GPUs):
     GPU 0-7: 共享（训练时 offload 推理权重，反之亦然）

配置: ``trainer.colocate=true``

Colocated 模式自动：

- 启用参数 offload (``megatron.param_offload=true``)
- 将 ``rollout.gpu_memory_utilization`` 限制为 0.45
- 禁用 ``validate_reuse_train_gpus``

关键数据结构
------------

SiiRLArguments (``siirl/params/training_args.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

顶层配置数据类：

.. code:: python

   @dataclass
   class SiiRLArguments:
       data: DataArguments              # 数据集路径、batch 大小、tokenization
       actor_ref: ActorRefArguments     # Actor、Ref、Algorithm、Checkpoint 配置
       rollout: RolloutArguments        # SGLang 引擎、采样、多轮配置
       critic: CriticArguments          # Critic 模型（仅 PPO）
       trainer: TrainingArguments       # Epoch、GPU 分配、检查点
       custom_reward_function: CustomRewardArguments  # 自定义奖励配置

Sample (``siirl/data_coordinator/sample.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

贯穿整个流水线的数据载体：

::

   Sample:
     prompts: list[int]          # Prompt token IDs
     responses: list[int]        # Response token IDs（模型 + 环境交替）
     response_mask: list[int]    # 1 = 模型 token, 0 = 环境 token
     rollout_log_prob: ndarray   # Token 级 log 概率
     rewards: float              # 标量奖励
     data_source: str            # 数据集来源标识
     reward_model: dict          # 用于奖励计算的 ground truth
