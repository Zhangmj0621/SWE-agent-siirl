Architecture Overview
=====================

   **Who this is for:** Framework Users wanting to understand the system design, and Contributors needing to locate code entry points.

   **What you will get:** A complete picture of siirl-agentic’s component architecture, data flow, and training lifecycle.

Design Philosophy
-----------------

siirl-agentic is built on three architectural principles:

1. **Asynchronous MPMD** — Each major component (Trainer, Rollout, DataCoordinator) runs as an independent Ray actor. No component blocks another.
2. **Agentic-first data path** — The sample lifecycle natively supports multi-turn trajectories with tool interaction, per-turn loss masking, and variable-length sequences.
3. **Configuration over code** — Task-specific logic (AgentFlow, reward functions, tool environments) is injected via config, not hardcoded in framework internals.

System Architecture
-------------------

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


Component Responsibilities
--------------------------

MainRunner (``siirl/async_train.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``MainRunner`` is a Ray actor that orchestrates the entire training lifecycle:

1. Parse ``SiiRLArguments`` configuration
2. Allocate GPU resources (actor vs rollout split)
3. Initialize ``DataCoordinator``, ``RolloutManager``, ``TrainerGroup``
4. Start the async training loop
5. Monitor status via ``TaskCoordinator``
6. Handle cleanup and failure reporting

TrainerGroup (``siirl/worker/actor/trainer_group.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Manages the distributed training workers:

- **Actor model** — Policy model being trained (Megatron backend)
- **Reference model** — Frozen copy for KL divergence computation
- **Critic model** — Value function estimator (PPO only, not used in GRPO)
- **Parameter sync** — Pushes updated weights to RolloutManager's SGLang engines after each training step

Key files:

- ``siirl/worker/actor/trainer.py`` — Per-rank training logic (forward, loss, backward, optimizer step)
- ``siirl/worker/actor/trainer_group.py`` — Multi-rank coordination
- ``siirl/worker/actor/checkpoint_manager.py`` — Save/load checkpoints
- ``siirl/engine/param_sync/`` — Weight synchronization to rollout engines

RolloutManager (``siirl/worker/rollout/rollout_manager.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Manages inference engines and dispatches rollouts:

- **SGLang engines** — One or more SGLang instances for fast LLM inference
- **Router** — Load balances requests across engines (when multiple engines)
- **NaiveFlow** — Multi-turn rollout execution with tool interaction
- **Validation** — Separate validation rollouts with different sampling params

Key files:

- ``siirl/worker/rollout/rollout_manager.py`` — Engine lifecycle, request dispatch
- ``siirl/worker/rollout/rollout_worker.py`` — Per-engine rollout execution
- ``siirl/execution/rollout/agent_flow/naive_flow.py`` — Multi-turn state machine
- ``siirl/execution/rollout/concurrency.py`` — Concurrency parameter resolution

DataCoordinator (``siirl/data_coordinator/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Manages the sample lifecycle from prompt to training:

- **Dataloader** — Loads and distributes training/validation datasets
- **DataBuffer** — Buffers completed rollout samples for training consumption
- **Off-policy support** — Accepts data from configurable version window

Key files:

- ``siirl/data_coordinator/data_buffer.py`` — ``init_data_coordinator()``, distributed buffer logic
- ``siirl/data_coordinator/dataloader/`` — Dataset loading and distribution
- ``siirl/data_coordinator/sample.py`` — ``Sample`` dataclass with prompt, response, mask, reward
- ``siirl/data_coordinator/protocol.py`` — Data exchange protocol

TaskCoordinator (``siirl/utils/task_coordinator.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Centralized lifecycle management for distributed training:

- ``should_stop()`` — Polled by all components to check if training should end
- ``report_failure(source, reason)`` — Any component reports failures for propagation
- ``report_completed(source)`` — Signal successful completion
- ``request_shutdown(reason, source)`` — Graceful shutdown request
- Event logging for post-mortem debugging

Training Lifecycle
------------------

Initialization Phase
~~~~~~~~~~~~~~~~~~~~

.. mermaid::

   graph LR
       A[Parse Config] --> B[Init Ray]
       B --> C[Allocate GPUs]
       C --> D[Init DataCoordinator]
       C --> E[Init RolloutManager]
       D --> F[Init Dataloader]
       E --> G[Start SGLang Engines]
       C --> H[Init TrainerGroup]
       H --> I[Load Checkpoint if resume]

Training Loop (Async)
~~~~~~~~~~~~~~~~~~~~~

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

Shutdown Phase
~~~~~~~~~~~~~~

1. Training epochs exhausted → ``coordinator.report_completed()``
2. All components detect ``should_stop() == True``
3. ``MainRunner._cleanup_and_report()`` logs summary
4. Ray shutdown

Deployment Modes
----------------

Separated Mode (Default)
~~~~~~~~~~~~~~~~~~~~~~~~

GPUs are split between training and rollout:

::

   Node (8 GPUs):
     GPU 0-3: Actor/Ref/Critic (TrainerGroup)
     GPU 4-7: SGLang Engines (RolloutManager)

Config: ``trainer.actor_gpus=4, trainer.rollout_gpus=4, trainer.colocate=false``

Colocated Mode
~~~~~~~~~~~~~~

Training and rollout share the same GPUs via weight offloading:

::

   Node (8 GPUs):
     GPU 0-7: Shared (offload rollout weights during training, vice versa)

Config: ``trainer.colocate=true``

Colocated mode automatically:

- Enables parameter offloading (``megatron.param_offload=true``)
- Clamps ``rollout.gpu_memory_utilization`` to 0.45
- Disables ``validate_reuse_train_gpus``

Key Data Structures
-------------------

SiiRLArguments (``siirl/params/training_args.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Top-level configuration dataclass:

.. code:: python

   @dataclass
   class SiiRLArguments:
       data: DataArguments              # Dataset paths, batch sizes, tokenization
       actor_ref: ActorRefArguments     # Actor, Ref, Algorithm, Checkpoint config
       rollout: RolloutArguments        # SGLang engine, sampling, multi-turn
       critic: CriticArguments          # Critic model (PPO only)
       trainer: TrainingArguments       # Epochs, GPU allocation, checkpointing
       custom_reward_function: CustomRewardArguments  # Custom reward config

Sample (``siirl/data_coordinator/sample.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Carries data through the entire pipeline:

::

   Sample:
     prompts: list[int]          # Prompt token IDs
     responses: list[int]        # Response token IDs (model + env)
     response_mask: list[int]    # 1 = model token, 0 = env token
     rollout_log_prob: ndarray   # Token-level log probabilities
     rewards: float              # Scalar reward
     data_source: str            # Dataset source identifier
     reward_model: dict          # Ground truth for reward computation
