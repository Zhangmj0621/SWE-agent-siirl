Quickstart
==========

   **Who this is for:** Users who want to run their first RL training job with siirl-agentic.

   **What you will get:** A running GRPO training job on a single node with 8 GPUs.

Prerequisites
-------------

- siirl-agentic installed (see :doc:`Installation <installation>`)
- 8 GPUs available (4 for training, 4 for rollout)
- A model (e.g., Qwen3-8B) downloaded locally
- Training data in Parquet format

Step 1: Prepare Your Environment
--------------------------------

.. code:: bash

   # Set paths
   export HOME_DIR=/path/to/your/home
   export MODEL_PATH=$HOME_DIR/data/models/Qwen3-8B
   export TRAIN_DATA_PATH=$HOME_DIR/data/datasets/deepscaler/train.parquet
   export TEST_DATA_PATH=$HOME_DIR/data/datasets/deepscaler/test.parquet

Step 2: Run GRPO Training (Separated Mode)
------------------------------------------

.. code:: bash

   cd siirl-agentic
   bash examples/grpo_train/run_qwen3_8b_separated.sh

This script configures: - **4 GPUs for Actor** (training with TP=4) - **4 GPUs for Rollout** (SGLang inference with TP=2) - **GRPO algorithm** with batch_size=512, n=8 samples per prompt - **Qwen3-8B model** with max_prompt=2048, max_response=4096

**Expected output:**

::

   INFO  | Ray is initialized. Time cost: 150.23 ms
   INFO  | MainRunner started. Beginning workflow setup...
   INFO  | Allocating GPU resources...
   INFO  | Initializing DataCoordinator with 1 distributed DataBuffers...
   SUCCESS | DataCoordinator initialized: 2 batches/epoch, 60 total steps
   SUCCESS | RolloutManager initialized. Router at: http://10.0.0.1:30000
   SUCCESS | TrainerGroup initialized with 4 trainers
   INFO  | Starting async training loop...

Step 3: Run PPO Training
------------------------

.. code:: bash

   bash examples/ppo_train/run_qwen3_8b_separated.sh

PPO adds a Critic model on top of GRPO. The script adjusts resource allocation accordingly.

Step 4: Colocated Mode (Share GPUs)
-----------------------------------

For smaller setups or maximum GPU utilization:

.. code:: bash

   bash examples/grpo_train/run_qwen3_8b_colocate.sh

In colocated mode, training and rollout share all 8 GPUs. The framework automatically manages weight offloading.

Key Parameters to Tune
----------------------

+-----------------------------------+------------------------------+------------------------------+
| Parameter                         | What it controls             | Recommendation               |
+===================================+==============================+==============================+
| ``TRAIN_BATCH_SIZE``              | Samples per training step    | Start with 512               |
+-----------------------------------+------------------------------+------------------------------+
| ``ROLLOUT_N``                     | Samples per prompt           | 8 for GRPO, 1 for PPO        |
+-----------------------------------+------------------------------+------------------------------+
| ``MAX_RESPONSE_LENGTH``           | Max generation length        | Match your task needs        |
+-----------------------------------+------------------------------+------------------------------+
| ``ACTOR_GPUS`` / ``ROLLOUT_GPUS`` | GPU split                    | Even split for most cases    |
+-----------------------------------+------------------------------+------------------------------+
| ``ROLLOUT_TP``                    | Inference tensor parallelism | 2 for 8B models, 4+ for 70B+ |
+-----------------------------------+------------------------------+------------------------------+

Monitoring
----------

Training logs are written to stdout and optionally to WandB:

.. code:: yaml

   trainer:
     logger: ["console", "wandb"]
     project_name: siirl_examples
     experiment_name: my_first_run

Next Steps
----------

- :doc:`First Agentic Training Job <first_agentic_training_job>` — Add tool interaction
- :doc:`GRPO Training Guide <../user_guide/grpo_training>` — Deep dive into GRPO configuration
- :doc:`PPO Training Guide <../user_guide/ppo_training>` — Deep dive into PPO configuration
- :doc:`Configuration System <../user_guide/configuration_system>` — Understand all config options
