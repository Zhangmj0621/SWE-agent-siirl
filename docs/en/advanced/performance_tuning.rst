Performance Tuning
==================

   **Who this is for:** Users optimizing training throughput and GPU utilization.

   **What you will get:** Tuning strategies for the async pipeline, rollout concurrency, and memory management.

Pipeline Throughput
-------------------

Async Factor
~~~~~~~~~~~~

.. code:: yaml

   trainer:
     async_factor: 2   # Buffer 2 rollout batches ahead

Higher ``async_factor`` means more decoupling between rollout and training, at the cost of higher off-policy staleness.

Off-Policy Training
~~~~~~~~~~~~~~~~~~~

.. code:: yaml

   trainer:
     off_policy_step: 2       # Accept data from [current-2, current] versions
     off_policy_strategy: fifo

Allows training on slightly stale data, maximizing GPU utilization during long agentic rollouts.

Rollout Concurrency
-------------------

.. code:: yaml

   rollout:
     train_server_concurrency: 256    # Concurrent requests per engine
     max_num_seqs: 0                  # 0 = auto (4 × concurrency)

For agentic workloads with tool calls, increase concurrency to keep engines saturated while individual requests wait for tool responses.

Memory Optimization
-------------------

Parameter Offloading
~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

   actor_ref:
     actor:
       megatron:
         param_offload: true       # Offload params to CPU
         grad_offload: true        # Offload gradients to CPU
         optimizer_offload: true   # Offload optimizer to CPU

Dynamic Batching
~~~~~~~~~~~~~~~~

.. code:: yaml

   actor_ref:
     actor:
       use_dynamic_batch: true       # Token-based batching
       max_tokens_per_gpu: 4096      # Max tokens per GPU
       use_workload_balance: true    # FLOPs-based load balancing

Multi-Node Scaling
------------------

.. code:: yaml

   trainer:
     nnodes: 4
     n_gpus_per_node: 8
     actor_gpus: 16    # 2 nodes for training
     rollout_gpus: 16  # 2 nodes for rollout
