PPO Training
============

   **Who this is for:** Users running Proximal Policy Optimization training with siirl-agentic.

   **What you will get:** PPO-specific configuration, architecture details, and tuning guidance.

Overview
--------

PPO (Proximal Policy Optimization) in siirl-agentic uses three models: - **Actor** — Policy model being optimized - **Reference** — Frozen copy for KL divergence computation - **Critic** — Value function estimator for advantage computation

PPO-Specific Configuration
--------------------------

.. code:: yaml

   actor_ref:
     algorithm:
       adv_estimator: ppo           # Use PPO advantage estimator
       gamma: 1.0                   # Discount factor
       lam: 1.0                     # GAE lambda
       kl_penalty: kl               # KL penalty type

     actor:
       ppo_mini_batch_size: 256     # Mini-batch size
       ppo_micro_batch_size_per_gpu: 8  # Per-GPU micro-batch
       clip_ratio: 0.2              # PPO clipping ratio
       ppo_epochs: 1                # PPO update epochs per batch
       entropy_coeff: 0.0           # Entropy regularization
       use_kl_loss: false           # Additional KL loss
       loss_agg_mode: token-mean    # Loss aggregation

     ref:
       log_prob_micro_batch_size_per_gpu: 8  # Ref forward batch size

   critic:
     ppo_mini_batch_size: 256
     ppo_micro_batch_size_per_gpu: 8
     ppo_epochs: 1
     cliprange_value: 0.5          # Value function clipping
     optim:
       lr: 1e-5                    # Critic learning rate (typically higher than actor)

   rollout:
     n: 1                          # PPO uses 1 sample per prompt

Minimal Runnable Example
------------------------

.. code:: bash

   export MODEL_PATH=/path/to/Qwen3-8B
   export TRAIN_DATA_PATH=/path/to/train.parquet
   export TEST_DATA_PATH=/path/to/test.parquet

   bash examples/ppo_train/run_qwen3_8b_separated.sh

PPO Training Flow
-----------------

.. mermaid::

   graph TB
   A[DataCoordinator] -->|prompt batch| B[RolloutManager]
   B -->|generate responses| C[Completed Samples]
   C -->|samples| D[DataBuffer]
   D -->|training batch| E[Actor Forward]
   E --> F[Ref Forward - compute ref logprobs]
   F --> G[Critic Forward - compute values]
   G --> H[Compute GAE Advantages]
   H --> I[Actor Loss - PPO clipped]
   H --> J[Critic Loss - value clipped]
   I --> K[Actor Backward + Update]
   J --> L[Critic Backward + Update]
   K --> M[Param Sync to Rollout]


Key Parameters
--------------

+----------------------------+------------------------------+-------------------+
| Parameter                  | Impact                       | Recommended Range |
+============================+==============================+===================+
| ``actor.optim.lr``         | Policy learning rate         | 1e-7 to 1e-5      |
+----------------------------+------------------------------+-------------------+
| ``critic.optim.lr``        | Value function learning rate | 1e-6 to 1e-4      |
+----------------------------+------------------------------+-------------------+
| ``actor.clip_ratio``       | PPO clipping range           | 0.1 to 0.3        |
+----------------------------+------------------------------+-------------------+
| ``critic.cliprange_value`` | Value clipping range         | 0.2 to 0.5        |
+----------------------------+------------------------------+-------------------+
| ``algorithm.gamma``        | Discount factor              | 0.99 to 1.0       |
+----------------------------+------------------------------+-------------------+
| ``algorithm.lam``          | GAE lambda                   | 0.95 to 1.0       |
+----------------------------+------------------------------+-------------------+
| ``trainer.critic_warmup``  | Critic warmup steps          | 0 to 10           |
+----------------------------+------------------------------+-------------------+

PPO vs GRPO
-----------

============= ====================== =====================
Aspect        PPO                    GRPO
============= ====================== =====================
Models        Actor + Ref + Critic   Actor + Ref
GPU usage     Higher (3 models)      Lower (2 models)
``rollout.n`` 1 (single sample)      8+ (group sampling)
Advantage     GAE from Critic values Group-relative reward
Stability     More stable            Simpler, but noisier
============= ====================== =====================

Common Issues
-------------

+-----------------------------+------------------------+---------------------------------------------------+
| Symptom                     | Cause                  | Fix                                               |
+=============================+========================+===================================================+
| Value loss diverges         | Critic lr too high     | Reduce ``critic.optim.lr``                        |
+-----------------------------+------------------------+---------------------------------------------------+
| No reward improvement       | Clip ratio too tight   | Increase ``actor.clip_ratio``                     |
+-----------------------------+------------------------+---------------------------------------------------+
| OOM on critic               | Micro-batch too large  | Reduce ``critic.ppo_micro_batch_size_per_gpu``    |
+-----------------------------+------------------------+---------------------------------------------------+
| KL divergence spikes        | Learning rate too high | Reduce ``actor.optim.lr``, enable ``use_kl_loss`` |
+-----------------------------+------------------------+---------------------------------------------------+
