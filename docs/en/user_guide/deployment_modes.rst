Deployment Modes
================

   **Who this is for:** Users choosing between separated and colocated GPU deployment.

   **What you will get:** Understanding of each deployment topology, when to use each, and configuration.

Separated Mode (Default)
------------------------

GPUs are split between training and rollout. Each group has dedicated resources.

.. code:: yaml

   trainer:
     colocate: false
     actor_gpus: 4
     rollout_gpus: 4

**When to use:** Most production training. Clear resource boundaries, predictable performance.

Colocated Mode
--------------

Training and rollout share all GPUs via weight offloading.

.. code:: yaml

   trainer:
     colocate: true

**When to use:** Limited GPU budget, smaller models. Framework automatically manages: - ``megatron.param_offload = true`` - ``rollout.gpu_memory_utilization`` clamped to 0.45 - ``validate_reuse_train_gpus`` disabled

Comparison
----------

=============== ======================== =========================
Aspect          Separated                Colocated
=============== ======================== =========================
GPU efficiency  Dedicated per role       Time-shared
Memory pressure Lower                    Higher
Complexity      Simple                   Offload management
Best for        Production, large models Prototyping, small models
=============== ======================== =========================
