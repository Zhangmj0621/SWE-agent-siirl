AIO Tool Infrastructure
=======================

   **Who this is for:** Users deploying and configuring AIO for agentic tool management.

   **What you will get:** A complete guide to AIO's three-tier architecture, deployment steps,
   auto-scaling configuration, and integration with the rollout pipeline.

Overview
--------

AIO (Agentic I/O) is a distributed tool scheduling infrastructure that manages external tool
instances across heterogeneous nodes. It solves the **tool bottleneck** that caps throughput
in long-horizon agentic rollout: when 1000 concurrent rollouts all need sandbox executions,
naive single-node tool servers cascade into timeouts.

See the :doc:`AIO Highlights page <../highlights/aio_elastic_agentic_tool_infrastructure>` for
architecture rationale and design decisions.

Architecture
------------

AIO uses a three-tier scheduler:

.. code:: text

   siirl-agentic Rollout (client)
     │  async batched HTTP requests
     ▼
   Proxy                     ← Unified entry point, routes requests, tracks capacity
     │  distributes work
     ▼
   ResourcePool              ← Manages tool registration, round-robin load balancing
     │  dispatches calls
     ▼
   WorkerManager (×N)        ← Runs on each tool node, manages tool instances

.. mermaid::

   graph LR
   A[RolloutManager] -->|batch tool calls| B[AIO Proxy]
   B -->|route| C[ResourcePool]
   C -->|dispatch| D[WorkerManager 1]
   C -->|dispatch| E[WorkerManager 2]
   C -->|dispatch| F[WorkerManager N]
   D -->|result| C
   E -->|result| C
   F -->|result| C
   C -->|response| B
   B -->|response| A

Deployment
----------

Prerequisites
~~~~~~~~~~~~~

Clone and install AIO from the monorepo:

.. code:: bash

   # From the monorepo root (parent of siirl-agentic/)
   cd AIO
   pip install -e .

Step 1: Create AIO Config
~~~~~~~~~~~~~~~~~~~~~~~~~

Create ``aio_config.yaml``:

.. code:: yaml

   proxy:
     host: 0.0.0.0
     port: 8080
   tools:
     - name: search
       capacity_per_worker: 10     # Max concurrent calls per worker
     - name: sandbox_fusion
       capacity_per_worker: 5
   auto_scaling:
     enabled: true
     algorithm: holt_winters        # Time-series demand prediction

Step 2: Start AIO Proxy
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   python -m aio.Scheduler.proxy --config aio_config.yaml

The Proxy exposes:

- ``POST /call`` — Submit tool call requests
- ``GET /status`` — Current resource pool state
- ``GET /health`` — Health check endpoint
- ``GET /metrics`` — Concurrency and latency metrics

Step 3: Start WorkerManagers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On each tool node:

.. code:: bash

   python -m aio.Scheduler.Resources.worker_manager \
       --proxy-url http://proxy-host:8080 \
       --config worker_config.yaml

WorkerManagers auto-register with the Proxy on startup. Verify registration:

.. code:: bash

   curl http://proxy-host:8080/status

Expected output:

.. code:: json

   {
     "tools": {
       "search": {"workers": 3, "total_capacity": 30, "active_calls": 7},
       "sandbox_fusion": {"workers": 2, "total_capacity": 10, "active_calls": 2}
     }
   }

Integration with siirl-agentic
-------------------------------

Configure the rollout to use AIO:

.. code:: yaml

   rollout:
     multiturn:
       env_type: tool_env
       env_kwargs:
         tool_format: hermes
         aio_proxy_url: http://proxy-host:8080

   # Or set via environment variable
   # export AIO_PROXY_URL=http://proxy-host:8080

The ``AIOSearchEnv`` class (``siirl/environment/tool_env/aio_search_env.py``) connects to the
Proxy URL and dispatches tool calls asynchronously.

Auto-Scaling
------------

AIO uses a **Holt-Winters time-series predictor** for demand-based auto-scaling:

How It Works
~~~~~~~~~~~~

1. ``ConcurrencyMonitor`` samples the active concurrent tool call count every second
2. Holt-Winters exponential smoothing forecasts demand for the next time window
3. When predicted demand exceeds current capacity × threshold, additional tool instances activate
4. When load drops, idle instances are reclaimed to free resources

Configuration
~~~~~~~~~~~~~

.. code:: yaml

   auto_scaling:
     enabled: true
     algorithm: holt_winters
     scale_up_threshold: 0.8     # Scale up when utilization > 80%
     scale_down_threshold: 0.3   # Scale down when utilization < 30%
     min_instances: 1            # Minimum tool instances per worker
     max_instances: 20           # Maximum tool instances per worker
     alpha: 0.3                  # Holt-Winters level smoothing factor
     beta: 0.1                   # Holt-Winters trend smoothing factor

Async Batched Request Submitter
--------------------------------

The client side uses a batched async submitter to absorb latency spikes:

.. code:: yaml

   aio:
     client:
       batch_size: 32             # Batch tool calls before submitting
       batch_timeout_ms: 10       # Max wait time to fill a batch
       max_retries: 3             # Retry failed calls

This prevents the rollout from blocking on individual slow tool calls by buffering and
submitting requests in batches, while keeping per-sample latency bounded.

Monitoring
----------

Real-time Status
~~~~~~~~~~~~~~~~

.. code:: bash

   # Current pool state
   curl http://proxy-host:8080/status

   # Detailed metrics
   curl http://proxy-host:8080/metrics

Key metrics to watch:

=============================== ========================================================
Metric                          Description
=============================== ========================================================
``active_calls``                Current concurrent tool calls
``queue_depth``                 Pending calls waiting for capacity
``p50_latency_ms``              Median tool call latency
``p99_latency_ms``              Tail latency (watch for AIO-scale-up trigger)
``scale_events``                Number of auto-scaling events
=============================== ========================================================

Logs
~~~~

Enable verbose AIO logging:

.. code:: bash

   LOGURU_LEVEL=DEBUG python -m aio.Scheduler.proxy --config aio_config.yaml

Look for ``ConcurrencyMonitor`` log lines showing demand forecasts and scale decisions.

Multi-Node Deployment
---------------------

For large-scale training with many tool workers:

.. code:: bash

   # Node 1: AIO Proxy (dedicated coordinator)
   python -m aio.Scheduler.proxy --config aio_config.yaml

   # Node 2–N: WorkerManagers (tool execution nodes)
   # Run on each node:
   python -m aio.Scheduler.Resources.worker_manager \
       --proxy-url http://node1:8080

Capacity planning:

- **Search tools:** 10–20 concurrent calls per worker, 1 CPU + low memory
- **Code sandbox:** 3–5 concurrent calls per worker, 2+ CPUs + 4GB RAM each
- **Scaling rule:** Target 70% average utilization; auto-scaling handles peaks

Common Issues
-------------

+-------------------------------------------+------------------------------+-------------------------------------------+
| Symptom                                   | Cause                        | Fix                                       |
+===========================================+==============================+===========================================+
| ``Error getting server from master node`` | Proxy not running            | Start ``python -m aio.Scheduler.proxy``   |
+-------------------------------------------+------------------------------+-------------------------------------------+
| Tool call timeouts                        | Capacity exhausted           | Add WorkerManagers or increase capacity   |
+-------------------------------------------+------------------------------+-------------------------------------------+
| Tools not registered                      | WorkerManager not started    | Start WorkerManager on each tool node     |
+-------------------------------------------+------------------------------+-------------------------------------------+
| Scaling not triggered                     | Threshold too high           | Lower ``scale_up_threshold``              |
+-------------------------------------------+------------------------------+-------------------------------------------+
| OOM on tool nodes                         | Too many concurrent sandbox  | Reduce ``capacity_per_worker``            |
+-------------------------------------------+------------------------------+-------------------------------------------+

Related
-------

- :doc:`AIO Highlights <../highlights/aio_elastic_agentic_tool_infrastructure>` — Architecture design and motivation
- :doc:`Agentic Multi-Turn <agentic_multiturn>` — How AIO integrates with rollout
- :doc:`First Agentic Training Job <../get_started/first_agentic_training_job>` — End-to-end setup walkthrough
