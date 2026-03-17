Async Training Lifecycle
========================

   **Who this is for:** Framework Users who need to understand the end-to-end training
   execution flow, and Contributors debugging lifecycle issues.

   **What you will get:** A detailed walkthrough of every phase in the training lifecycle,
   from config parsing to shutdown, with sequence diagrams and code anchors.


Overview
--------

siirl-agentic's training lifecycle consists of three major phases:

1. **Initialization** — Config parsing, resource allocation, component startup
2. **Async Training Loop** — Continuous rollout → buffer → train cycle
3. **Shutdown** — Graceful termination, checkpoint saving, resource cleanup


Phase 1: Initialization
-------------------------

The ``main()`` function in ``siirl/async_train.py`` drives initialization:

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
       G --> J[init_dataloader]
       H --> K[rollout_manager.init]
       I --> L[trainer_group.init_actors]

Step-by-step:

1. **Ray initialization** — Start or connect to a Ray cluster with runtime env vars
   (``TOKENIZERS_PARALLELISM``, ``NCCL_DEBUG``, etc.).

2. **Config parsing** — ``parse_config()`` loads YAML config and creates a
   ``SiiRLArguments`` dataclass hierarchy.

3. **MainRunner launch** — A dedicated Ray actor (with ``num_cpus=5`` reservation) is
   created to orchestrate the workflow. This isolates the main process.

4. **TaskCoordinator** — ``create_coordinator()`` creates a named Ray actor for
   centralized lifecycle management. All components reference this actor.

5. **Resource allocation** — ``allocate_resources(config)`` splits available GPUs between
   training and rollout based on ``trainer.actor_gpus`` and ``trainer.rollout_gpus``.

6. **DataCoordinator init** — Creates distributed ``DataBuffer`` instances and starts
   the dataloader for prompt distribution.

7. **RolloutManager init** — Starts SGLang inference engines, initializes the router,
   and prepares AgentFlow instances. In colocated mode, this must complete before
   trainer init.

8. **TrainerGroup init** — Initializes Actor, Reference, and (for PPO) Critic models
   with Megatron backend. Loads checkpoint if ``resume_mode != "disable"``.

Colocated Mode Special Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``trainer.colocate=true``, the initialization order changes:

1. RolloutManager must fully initialize first
2. ``offload_for_train()`` is called to free GPU memory for trainer init
3. Only then does ``trainer_group.init_actors()`` proceed
4. ``gpu_memory_utilization`` is clamped to 0.45
5. ``param_offload`` is forced on for all Megatron configs


Phase 2: Async Training Loop
------------------------------

Once initialization completes, the async training loop begins:

.. mermaid::

   graph TB
       subgraph RolloutManager
           A[run_dataloader] --> B[next_rollout]
           B --> C[RolloutWorker.rollout]
           C --> D[NaiveFlow state machine]
           D --> E[ToolEnv / AIO Proxy]
           E --> D
           D --> F[completed Sample]
       end

       subgraph DataCoordinator
           F --> G[DataBuffer.add]
           G --> H[DataBuffer.get_batch]
       end

       subgraph TrainerGroup
           H --> I[Trainer.train_step]
           I --> J[Actor forward + backward]
           I --> K[Ref forward logprob]
           I --> L[Critic forward + backward PPO]
           J --> M[optimizer.step]
           M --> N[param_sync to SGLang]
       end

       N -.->|updated weights| B

The loop operates asynchronously:

- **RolloutManager** continuously fetches prompts via ``run_dataloader()`` and dispatches
  rollouts. Each ``RolloutWorker`` runs ``NaiveFlow`` which interacts with tools.
- **DataBuffer** collects completed samples. When enough samples accumulate for a training
  batch, they become available for the Trainer.
- **TrainerGroup** pulls training batches and executes PPO or GRPO updates. After each
  update, parameter sync pushes new weights to the SGLang engines.

Key asynchronous properties:

- Rollout and training run **concurrently** — the Trainer does not wait for all rollouts
  to finish before starting a training step.
- **Off-policy tolerance** — ``off_policy_step`` controls how many training steps old a
  sample can be before it is discarded (default: 1).
- **Continuous GPU utilization** — While the Trainer updates weights, the RolloutManager
  is already collecting new trajectories.

NaiveFlow State Machine
~~~~~~~~~~~~~~~~~~~~~~~~~

Each sample goes through a state machine within ``NaiveFlow``:

::

   PENDING → GENERATING → PROCESSING_ENV → GENERATING → ... → TERMINATED

- **PENDING** — Sample created, awaiting first generation
- **GENERATING** — LLM is generating tokens via SGLang
- **PROCESSING_ENV** — Tool calls are being executed (potentially in parallel)
- **TERMINATED** — Max turns reached, or environment signaled completion

Per-turn, the flow:

1. Calls ``model.generate()`` for the next assistant response
2. Parses tool calls from the response
3. Executes tool calls via ``ToolEnv`` or ``AIO Proxy``
4. Appends environment responses to the conversation
5. Updates ``loss_mask`` (1 for model tokens, 0 for environment tokens)
6. Checks termination conditions


Phase 3: Shutdown
------------------

Shutdown is coordinated through the ``TaskCoordinator``:

.. mermaid::

   graph LR
       A[Training epochs exhausted] --> B[coordinator.report_completed]
       B --> C[should_stop returns True]
       C --> D[RolloutManager stops]
       C --> E[TrainerGroup stops]
       D --> F[cleanup_and_report]
       E --> F
       F --> G[Log summary]
       G --> H[ray.shutdown]

**Normal shutdown:**

1. All training epochs are exhausted
2. ``TrainerGroup`` calls ``coordinator.report_completed()``
3. All components polling ``coordinator.should_stop()`` receive ``True``
4. ``MainRunner._cleanup_and_report()`` logs the final summary
5. ``ray.shutdown()`` cleans up the cluster

**Failure shutdown:**

1. Any component catches an exception
2. It calls ``coordinator.report_failure(source, reason)``
3. ``should_stop()`` returns ``True`` for all components
4. ``MainRunner`` detects the failure, logs diagnostics, and re-raises

**Key config parameters:**

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``trainer.total_epochs``
     - 1
     - Number of training epochs
   * - ``trainer.off_policy_step``
     - 1
     - Max staleness of training samples
   * - ``trainer.colocate``
     - false
     - Whether to share GPUs between training and rollout
   * - ``trainer.resume_mode``
     - "disable"
     - Checkpoint resume strategy


Code Anchors
-------------

- ``siirl/async_train.py`` — ``MainRunner.run()`` orchestrates all three phases
- ``siirl/utils/task_coordinator.py`` — ``TaskCoordinator`` lifecycle management
- ``siirl/worker/rollout/rollout_manager.py`` — ``RolloutManager`` rollout dispatch
- ``siirl/worker/actor/trainer_group.py`` — ``TrainerGroup`` training coordination
- ``siirl/data_coordinator/data_buffer.py`` — ``DataBuffer`` sample buffering
- ``siirl/execution/rollout/agent_flow/naive_flow.py`` — ``NaiveFlow`` state machine


Further Reading
----------------

- :doc:`architecture_overview` — Component diagram and responsibilities
- :doc:`design_philosophy` — Why the system is designed this way
- :doc:`../user_guide/configuration_system` — How to configure the training job
- :doc:`../advanced/failure_propagation` — Detailed failure handling behavior
