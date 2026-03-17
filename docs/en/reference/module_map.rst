Module Map
==========

   **Who this is for:** Developers navigating the siirl-agentic source code.

   **What you will get:** A quick reference map from top-level directories to their responsibilities and key files.

Top-Level Structure
-------------------

::

   siirl-agentic/
   ├── siirl/                    # Main package
   │   ├── async_train.py        # Entry point: MainRunner Ray actor
   │   ├── algorithm/            # RL algorithm implementations
   │   ├── data_coordinator/     # Async data buffering & loading
   │   ├── engine/               # GPU compute engines
   │   ├── environment/          # Tool environments & base classes
   │   ├── execution/            # Rollout orchestration
   │   ├── models/               # Model loading & weight management
   │   ├── params/               # Configuration dataclasses
   │   ├── utils/                # Shared utilities
   │   └── worker/               # Ray actor workers
   ├── examples/                 # Training scripts & configs
   ├── scripts/                  # Utility scripts
   ├── tests/                    # Test suite
   └── docs/                     # Documentation (this site)

Module Details
--------------

``siirl/async_train.py``
~~~~~~~~~~~~~~~~~~~~~~~~

The main entry point. Implements ``MainRunner`` as a Ray actor with an 8-phase lifecycle:

1. Parse config → 2. Allocate resources → 3. Init DataCoordinator → 4. Init components → 5. Async training loop → 6. Wait → 7. Check status → 8. Cleanup

``siirl/algorithm/``
~~~~~~~~~~~~~~~~~~~~

+---------------------+-------------------------------------------------------+
| File                | Description                                           |
+=====================+=======================================================+
| ``advantage.py``    | GAE and group-relative advantage estimation (GRPO)    |
+---------------------+-------------------------------------------------------+
| ``loss.py``         | PPO clipped loss, GRPO loss, entropy bonus            |
+---------------------+-------------------------------------------------------+
| ``kl_penalty.py``   | KL divergence computation between actor and reference |
+---------------------+-------------------------------------------------------+

``siirl/data_coordinator/``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

+---------------------+--------------------------------------------------------------------------------+
| File                | Description                                                                    |
+=====================+================================================================================+
| ``data_buffer.py``  | ``DataBuffer`` Ray actor — async sample buffering between rollout and training |
+---------------------+--------------------------------------------------------------------------------+
| ``protocol.py``     | Data exchange protocols between components                                     |
+---------------------+--------------------------------------------------------------------------------+
| ``sample.py``       | Sample packing, padding, and tensor construction                               |
+---------------------+--------------------------------------------------------------------------------+
| ``dataloader/``     | Dataset loading and batching logic                                             |
+---------------------+--------------------------------------------------------------------------------+

``siirl/engine/``
~~~~~~~~~~~~~~~~~

+--------------------------------+-------------------------------------------------------------+
| Directory                      | Description                                                 |
+================================+=============================================================+
| ``actor/``                     | Actor model engine (forward/backward, Megatron integration) |
+--------------------------------+-------------------------------------------------------------+
| ``param_sync/``                | Parameter synchronization between actor and rollout engines |
+--------------------------------+-------------------------------------------------------------+
| ``rollout/``                   | SGLang-based rollout inference engine                       |
+--------------------------------+-------------------------------------------------------------+

``siirl/environment/``
~~~~~~~~~~~~~~~~~~~~~~

+--------------------------------+---------------------------------------------------------+
| File                           | Description                                             |
+================================+=========================================================+
| ``base.py``                    | ``BasEnvironment`` ABC, ``EnvResponse`` Pydantic model  |
+--------------------------------+---------------------------------------------------------+
| ``tool_env/base_tool_env.py``  | ``ToolEnv`` — OpenAI function-calling tool schema       |
+--------------------------------+---------------------------------------------------------+
| ``tool_env/aio_search_env.py`` | ``AIOSearchEnv`` — concrete tool env using AIO HTTP API |
+--------------------------------+---------------------------------------------------------+

``siirl/execution/rollout/``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+--------------------------------------+------------------------------------------------------------------------------------+
| File/Directory                       | Description                                                                        |
+======================================+====================================================================================+
| ``agentflow/base.py``                | ``AgentFlow`` Protocol (``preprocess → generate → reward``), ``Sample``, ``Model`` |
+--------------------------------------+------------------------------------------------------------------------------------+
| ``agentflow/__init__.py``            | ``load_agentflow()`` — dynamic function injection via YAML config                  |
+--------------------------------------+------------------------------------------------------------------------------------+
| ``agent_flow/naive_flow.py``         | ``NaiveFlow`` — multi-turn rollout state machine                                   |
+--------------------------------------+------------------------------------------------------------------------------------+
| ``agent_executor/``                  | Agent execution orchestrator                                                       |
+--------------------------------------+------------------------------------------------------------------------------------+
| ``concurrency.py``                   | Concurrency resolution utilities                                                   |
+--------------------------------------+------------------------------------------------------------------------------------+

``siirl/models/``
~~~~~~~~~~~~~~~~~

============================= =====================================
File/Directory                Description
============================= =====================================
``loader.py``                 Model weight loading utilities
``weight_loader_registry.py`` Registry for weight format converters
``mcore/``                    Megatron-Core model implementations
``llama/``                    LLaMA-specific model utilities
``patcher.py``                Model architecture patching utilities
============================= =====================================

``siirl/params/``
~~~~~~~~~~~~~~~~~

+----------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| File                 | Description                                                                                                                       |
+======================+===================================================================================================================================+
| ``training_args.py`` | ``SiiRLArguments`` (top-level), ``TrainingArguments``                                                                             |
+----------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| ``model_args.py``    | ``ModelArguments``, ``ActorArguments``, ``RolloutArguments``, ``AlgorithmArguments``, ``CriticArguments``, ``MultiturnArguments`` |
+----------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| ``data_args.py``     | ``DataArguments`` — dataset paths, lengths, batch sizes                                                                           |
+----------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| ``parser.py``        | YAML config parser                                                                                                                |
+----------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| ``display_dict.py``  | Config display and serialization                                                                                                  |
+----------------------+-----------------------------------------------------------------------------------------------------------------------------------+

``siirl/utils/``
~~~~~~~~~~~~~~~~

+--------------------------------------+------------------------------------------------------------------+
| File/Directory                       | Description                                                      |
+======================================+==================================================================+
| ``task_coordinator.py``              | ``TaskCoordinator`` Ray actor — distributed lifecycle management |
+--------------------------------------+------------------------------------------------------------------+
| ``timer.py``                         | Training timer and profiling                                     |
+--------------------------------------+------------------------------------------------------------------+
| ``distributed_utils.py``             | Distributed communication helpers                                |
+--------------------------------------+------------------------------------------------------------------+
| ``checkpoint/``                      | Checkpoint save/load logic                                       |
+--------------------------------------+------------------------------------------------------------------+
| ``logger/``                          | Logging configuration (loguru-based)                             |
+--------------------------------------+------------------------------------------------------------------+
| ``megatron/``                        | Megatron-Core integration utilities                              |
+--------------------------------------+------------------------------------------------------------------+
| ``metrics/``                         | Training metrics collection and reporting                        |
+--------------------------------------+------------------------------------------------------------------+
| ``model_utils/``                     | Model utility functions                                          |
+--------------------------------------+------------------------------------------------------------------+
| ``net_utils/``                       | Network and port utilities                                       |
+--------------------------------------+------------------------------------------------------------------+
| ``reward_score/``                    | Reward scoring functions (custom reward implementations)         |
+--------------------------------------+------------------------------------------------------------------+

``siirl/worker/``
~~~~~~~~~~~~~~~~~

+--------------------------------+---------------------------------------------------------------+
| Directory                      | Description                                                   |
+================================+===============================================================+
| ``actor/``                     | Actor worker — wraps actor engine as Ray actor                |
+--------------------------------+---------------------------------------------------------------+
| ``rollout/``                   | RolloutManager — orchestrates rollout across multiple engines |
+--------------------------------+---------------------------------------------------------------+
| ``validate/``                  | Validation worker — runs evaluation during training           |
+--------------------------------+---------------------------------------------------------------+

External Dependencies
---------------------

+-----------------------------+---------------------------+-------------------------------------------------------+
| Component                   | Package                   | Role                                                  |
+=============================+===========================+=======================================================+
| Distributed orchestration   | ``ray``                   | Actor model, resource management                      |
+-----------------------------+---------------------------+-------------------------------------------------------+
| Inference engine            | ``sglang``                | High-throughput LLM inference for rollout             |
+-----------------------------+---------------------------+-------------------------------------------------------+
| Model framework             | ``transformers``          | Model loading, tokenization                           |
+-----------------------------+---------------------------+-------------------------------------------------------+
| GPU training                | ``torch`` + Megatron-Core | Distributed training with tensor/pipeline parallelism |
+-----------------------------+---------------------------+-------------------------------------------------------+
| Tool scheduling             | ``AIO`` (sibling project) | Elastic tool instance management                      |
+-----------------------------+---------------------------+-------------------------------------------------------+

Related
-------

- :doc:`Architecture Overview <../concepts/architecture_overview>` — High-level system design
- :doc:`Code Structure <../developer_guide/code_structure>` — Developer-oriented walkthrough
