Validate Reuse & Evaluation Scaling
===================================

   **Who this is for:** Users wanting to scale validation throughput by reusing training GPUs.

   **What you will get:** Configuration for validate-reuse mode and evaluation scaling.

Overview
--------

In separated mode, siirl-agentic can **reuse training GPUs** during validation to increase evaluation throughput. When training is idle (waiting for rollout), trainer GPUs temporarily serve as additional rollout engines for validation.

Configuration
-------------

.. code:: yaml

   trainer:
     validate_reuse_train_gpus: true        # Enable training GPU reuse for validation
     validate_reuse_begin_timeout_s: 30     # Timeout for rendezvous
     test_freq: 10                          # Validate every N steps
     val_before_train: true                 # Run validation before first training step

Validation Batch Size
---------------------

.. code:: yaml

   data:
     val_batch_size: null    # null = use entire validation set as one batch
   rollout:
     validate_chunk_size: 0  # 0 = auto (train_server_concurrency × validate_workers)

Notes
-----

- ``validate_reuse_train_gpus`` is automatically disabled in colocated mode
- Validation uses ``rollout.val_kwargs`` for sampling parameters (greedy by default)
