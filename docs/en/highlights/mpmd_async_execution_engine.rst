MPMD Asynchronous Execution Engine
==================================

   **Who this is for:** Users who need high GPU utilization during agentic RL training where tool calls introduce variable latency.

   **What you will get:** Understanding of the Multi-Program Multi-Data architecture and how it keeps GPUs busy during long-horizon agentic rollouts.

Capability
----------

siirl-agentic uses a **Multi-Program Multi-Data (MPMD)** architecture where Trainer, RolloutManager, and DataCoordinator run as independent Ray actors. Training and rollout proceed asynchronously — while one set of samples is being trained on, the next set is already rolling out with tool interactions. This eliminates the synchronous train-then-rollout bottleneck that cripples GPU utilization in agentic settings.

User Value
----------

- **High GPU utilization.** GPUs never idle waiting for tool responses; training proceeds on buffered samples while new rollouts interact with tools.
- **Natural latency absorption.** Variable tool call times (100ms for search, 30s for code sandbox) don’t stall the training pipeline.
- **Scalable concurrency.** Hundreds of concurrent rollouts can be in-flight, each at different stages of their multi-turn trajectory.
- **Off-policy support.** The ``DataBuffer`` supports configurable off-policy steps, allowing training on slightly stale data to further decouple rollout and training.

Code Entry Points
-----------------

+-----------------------+---------------------------------------------+------------------------------------------------------------+
| Component             | File                                        | Responsibility                                             |
+=======================+=============================================+============================================================+
| Main orchestrator     | ``siirl/async_train.py``                    | ``MainRunner`` initializes and coordinates all actors      |
+-----------------------+---------------------------------------------+------------------------------------------------------------+
| Task coordinator      | ``siirl/utils/task_coordinator.py``         | ``TaskCoordinator`` manages lifecycle, failure propagation |
+-----------------------+---------------------------------------------+------------------------------------------------------------+
| Rollout manager       | ``siirl/worker/rollout/rollout_manager.py`` | Manages SGLang engines, dispatches rollouts                |
+-----------------------+---------------------------------------------+------------------------------------------------------------+
| Rollout worker        | ``siirl/worker/rollout/rollout_worker.py``  | Per-engine rollout execution                               |
+-----------------------+---------------------------------------------+------------------------------------------------------------+
| Trainer group         | ``siirl/worker/actor/trainer_group.py``     | Manages distributed Actor/Ref/Critic training              |
+-----------------------+---------------------------------------------+------------------------------------------------------------+
| Data buffer           | ``siirl/data_coordinator/data_buffer.py``   | Async sample buffering between rollout and training        |
+-----------------------+---------------------------------------------+------------------------------------------------------------+
| Concurrency config    | ``siirl/execution/rollout/concurrency.py``  | Resolves concurrency knobs per phase                       |
+-----------------------+---------------------------------------------+------------------------------------------------------------+

Key Configuration
-----------------

.. code:: yaml

   trainer:
     # --- Async pipeline control ---
     async_factor: 1                 # Number of rollout batches to buffer ahead
     off_policy_step: 0              # 0 = on-policy only; N = accept data from [current-N, current]
     off_policy_strategy: fifo       # "fifo" or "oldest_first"

     # --- GPU resource allocation ---
     actor_gpus: 4                   # GPUs for training (Actor/Ref/Critic)
     rollout_gpus: 4                 # GPUs for rollout (SGLang inference)
     colocate: false                 # true = share GPUs between training and rollout

   rollout:
     train_server_concurrency: 256   # Max concurrent client requests per rollout worker
     max_num_seqs: 0                 # Max running requests in SGLang (0 = auto)

How It Works
------------

Async Training Pipeline
~~~~~~~~~~~~~~~~~~~~~~~

.. mermaid::

   graph TB
       subgraph MainRunner
           A[Parse Config] --> B[Allocate Resources]
           B --> C[Init DataCoordinator]
           B --> D[Init RolloutManager]
           B --> E[Init TrainerGroup]
       end

       subgraph Async Loop
           F[DataCoordinator] -->|next batch| G[RolloutManager]
           G -->|rollout with tools| H[NaiveFlow + ToolEnv]
           H -->|completed samples| I[DataBuffer]
           I -->|training batch| J[TrainerGroup]
           J -->|param sync| G
       end

The key insight is that **rollout and training overlap**:

1. **RolloutManager** receives prompts from DataCoordinator and starts agentic rollouts via ``NaiveFlow``. Each rollout may take 5-60 seconds (with tool calls).
2. **DataBuffer** collects completed rollout samples. Once a full training batch is buffered, it’s dispatched to TrainerGroup.
3. **TrainerGroup** performs PPO/GRPO updates on the buffered batch. Meanwhile, RolloutManager continues producing the next batch.
4. **Param sync** pushes updated model weights from Trainer to RolloutManager’s SGLang engines after each training step.

Lifecycle Management
~~~~~~~~~~~~~~~~~~~~

The ``TaskCoordinator`` provides unified lifecycle management:

- All components poll ``should_stop()`` in their loops
- Any component can ``report_failure()`` to propagate errors
- Graceful shutdown via ``request_shutdown()``
- Event logging for post-mortem debugging

Concurrency Resolution
~~~~~~~~~~~~~~~~~~~~~~

The concurrency module (``concurrency.py``) auto-tunes request concurrency:

+------------------------------+-------------------+----------------------------------------------------+
| Parameter                    | Default           | Auto Logic                                         |
+==============================+===================+====================================================+
| ``train_server_concurrency`` | 256               | Per-worker concurrency cap                         |
+------------------------------+-------------------+----------------------------------------------------+
| ``max_num_seqs``             | auto              | ``4 × train_server_concurrency``                   |
+------------------------------+-------------------+----------------------------------------------------+
| Router concurrency           | auto              | ``ceil(max_num_seqs / num_engines) × num_engines`` |
+------------------------------+-------------------+----------------------------------------------------+

vs. General RL Frameworks
-------------------------

   Synchronous RL frameworks execute ``rollout → train → rollout → train`` in lockstep. A single slow tool call blocks the entire pipeline. siirl-agentic’s MPMD architecture keeps **rollout and training running simultaneously**, with the DataBuffer absorbing timing differences. This is critical for agentic tasks where tool latency varies by 100x.
