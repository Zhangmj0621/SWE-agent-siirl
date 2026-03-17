AIO: Elastic Agentic Tool Infrastructure
========================================

   **Who this is for:** Users running agentic RL training with external tools (code sandboxes, search APIs, MCP servers) that need scalable tool management.

   **What you will get:** Understanding of the AIO three-layer scheduling architecture, auto-scaling, and async request submission.

.. note::

   AIO is a **separate repository** (``AIO/``) in the monorepo, not part of the ``siirl-agentic/`` package. All file paths below are relative to the monorepo root. To install: ``cd AIO && pip install -e .``

Capability
----------

AIO (Agentic I/O) is a **distributed tool scheduling infrastructure** that manages the lifecycle of external tool instances (code sandboxes, retrieval services, MCP tool servers) across heterogeneous compute nodes. It provides three key capabilities:

1. **Three-layer distributed scheduling** — Proxy → ResourcePool → WorkerManager for fine-grained load balancing
2. **Predictive auto-scaling** — Holt-Winters time-series predictor adjusts tool environment count per training step
3. **Async batch request submission** — Absorbs latency spikes from variable tool response times

User Value
----------

- **Eliminate tool-side bottlenecks.** When 1000 rollouts need sandbox access simultaneously, AIO distributes load across available instances with capacity-aware routing.
- **Auto-scale tool environments.** No manual capacity planning — AIO predicts demand from per-step concurrency patterns and scales tool instances accordingly.
- **Cross-node tool management.** Tool instances can run on CPU nodes, GPU nodes, or dedicated tool servers. AIO abstracts the physical topology.
- **Latency spike absorption.** The async batch request submitter queues requests and dispatches them efficiently, preventing cascading timeouts during concurrency bursts.

Code Entry Points
-----------------

+-----------------------+-----------------------------------------------------------------------------------+------------------------------------------------------------+
| Component             | File                                                                              | Responsibility                                             |
+=======================+===================================================================================+============================================================+
| Proxy (entry point)   | ``AIO/aio/Scheduler/proxy.py``                                                    | FastAPI service: request routing, scheduling policies      |
+-----------------------+-----------------------------------------------------------------------------------+------------------------------------------------------------+
| Resource pool         | ``AIO/aio/Scheduler/resource_pool.py``                                            | Capacity tracking, round-robin/least-load server selection |
+-----------------------+-----------------------------------------------------------------------------------+------------------------------------------------------------+
| Worker manager        | ``AIO/aio/Scheduler/Resources/worker_manager.py``                                 | Per-node tool instance lifecycle management                |
+-----------------------+-----------------------------------------------------------------------------------+------------------------------------------------------------+
| Concurrency monitor   | ``AIO/aio/Scheduler/concurrency_monitor.py``                                      | Real-time concurrency tracking per tool type               |
+-----------------------+-----------------------------------------------------------------------------------+------------------------------------------------------------+
| Auto-scaling          | ``AIO/aio/batch_conductor/auto_scaling_algorithm/holt_winters_rolling_punish.py`` | Holt-Winters predictor for demand forecasting              |
+-----------------------+-----------------------------------------------------------------------------------+------------------------------------------------------------+
| Batch conductor       | ``AIO/aio/batch_conductor/batch_conductor.py``                                    | Batch request orchestration                                |
+-----------------------+-----------------------------------------------------------------------------------+------------------------------------------------------------+
| Request submitter     | ``AIO/aio/tools/request_submitter.py``                                            | Async batch request submission with backpressure           |
+-----------------------+-----------------------------------------------------------------------------------+------------------------------------------------------------+
| Base tool interface   | ``AIO/aio/tools/aio_base_tool.py``                                                | Abstract tool interface for AIO-managed tools              |
+-----------------------+-----------------------------------------------------------------------------------+------------------------------------------------------------+
| Search tool           | ``AIO/aio/tools/aio_search_tool.py``                                              | Concrete search tool implementation                        |
+-----------------------+-----------------------------------------------------------------------------------+------------------------------------------------------------+
| Sandbox tool          | ``AIO/aio/tools/aio_sandbox_fusion_tool.py``                                      | Code sandbox tool implementation                           |
+-----------------------+-----------------------------------------------------------------------------------+------------------------------------------------------------+

Key Configuration
-----------------

.. code:: yaml

   # AIO Scheduler configuration
   aio:
     proxy:
       host: 0.0.0.0
       port: 8080
       scheduling_policy: round_robin  # round_robin, least_load

     tools:
       - name: search
         capacity_per_worker: 10      # Max concurrent requests per tool instance
         min_instances: 2             # Minimum tool instances
         max_instances: 20            # Maximum tool instances

       - name: sandbox_fusion
         capacity_per_worker: 5
         min_instances: 4
         max_instances: 50

     auto_scaling:
       enabled: true
       algorithm: holt_winters
       prediction_horizon: 3          # Steps ahead to predict
       scale_up_threshold: 0.8        # Utilization trigger for scale-up
       scale_down_threshold: 0.3      # Utilization trigger for scale-down
       cooldown_steps: 5              # Steps between scaling decisions

How It Works
------------

Three-Layer Architecture
~~~~~~~~~~~~~~~~~~~~~~~~

.. mermaid::

   graph TB
   A[Rollout Workers] -->|tool requests| B[AIO Proxy]
   B -->|route by tool type| C[ResourcePool]
   C -->|select server| D[WorkerManager 1]
   C -->|select server| E[WorkerManager 2]
   C -->|select server| F[WorkerManager N]
   D --> G[Tool Instance 1..K]
   E --> H[Tool Instance 1..K]
   F --> I[Tool Instance 1..K]


**Layer 1: Proxy** — FastAPI service that receives tool call requests from rollout workers. Routes requests based on tool type and scheduling policy (round-robin or least-load).

**Layer 2: ResourcePool** — Maintains a real-time view of all registered tool servers, their capacities, and current load. Uses two strategies: - Round-robin across servers with available capacity - Least-loaded server fallback when all servers are near capacity

**Layer 3: WorkerManager** — Runs on each compute node. Manages the lifecycle (create/destroy) of tool instances and reports available capacity back to the ResourcePool.

Auto-Scaling Flow
~~~~~~~~~~~~~~~~~

.. mermaid::

   graph LR
   A[ConcurrencyMonitor] -->|per-step metrics| B[Holt-Winters Predictor]
   B -->|predicted demand| C[Scaling Decision]
   C -->|scale up| D[WorkerManager.create_instances]
   C -->|scale down| E[WorkerManager.destroy_instances]
   C -->|no change| F[Continue]


The Holt-Winters rolling predictor: 1. Collects per-step concurrency measurements from the ``ConcurrencyMonitor`` 2. Predicts demand N steps ahead using exponential smoothing with trend and seasonality 3. Applies a punish factor for under-prediction (scaling up is cheaper than cascading timeouts) 4. Issues scale-up/down commands to WorkerManagers with cooldown enforcement

Request Flow
~~~~~~~~~~~~

1. Rollout worker encounters a tool call in ``NaiveFlow._handle_processing_envs_state()``
2. Tool environment (e.g., ``AIOSearchEnv``) sends HTTP request to AIO Proxy (``/get_server``)
3. Proxy queries ResourcePool for an available server with capacity
4. Request is forwarded to the selected tool instance
5. On completion, the tool env notifies Proxy (``/complete_task``) to release capacity
6. Response flows back to the rollout worker for prompt continuation

vs. General RL Frameworks
-------------------------

   General RL frameworks provide no tool management infrastructure. Users must manually deploy, scale, and load-balance tool servers. AIO provides a **purpose-built distributed scheduler** with predictive auto-scaling, ensuring tool availability keeps pace with training demand — a critical requirement for agentic RL at scale.
