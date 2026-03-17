GRPO Training
=============

   **Who this is for:** Users running Group Relative Policy Optimization with siirl-agentic.

   **What you will get:** GRPO-specific configuration, group sampling mechanics, and tuning guidance.

Overview
--------

GRPO (Group Relative Policy Optimization) is a critic-free RL algorithm that estimates advantages by comparing rewards within a **group of responses** generated from the same prompt. This eliminates the need for a separate Critic model, reducing GPU memory and simplifying the training pipeline.

GRPO-Specific Configuration
---------------------------

.. code:: yaml

   actor_ref:
     algorithm:
       adv_estimator: grpo               # Use GRPO advantage estimator
       norm_adv_by_std_in_grpo: true     # Normalize advantages by group std

     actor:
       ppo_mini_batch_size: 256
       ppo_micro_batch_size_per_gpu: 8
       clip_ratio: 0.2
       ppo_epochs: 1
       optim:
         lr: 1e-6

   rollout:
     n: 8                                # Generate 8 responses per prompt
     temperature: 1.0                    # Sampling temperature
     do_sample: true

   data:
     train_batch_size: 512               # Prompts per step (total samples = 512 × 8 = 4096)

Minimal Runnable Example
------------------------

.. code:: bash

   export MODEL_PATH=/path/to/Qwen3-8B
   export TRAIN_DATA_PATH=/path/to/train.parquet
   export TEST_DATA_PATH=/path/to/test.parquet

   bash examples/grpo_train/run_qwen3_8b_separated.sh

How GRPO Works
--------------

.. mermaid::

   graph TB
   A[Prompt] --> B[Generate N=8 responses]
   B --> C[Response 1: reward=0.8]
   B --> D[Response 2: reward=0.2]
   B --> E[Response 3: reward=1.0]
   B --> F[...]
   B --> G[Response 8: reward=0.5]
   C --> H[Group mean=0.6, std=0.3]
   D --> H
   E --> H
   F --> H
   G --> H
   H --> I[Advantage = reward - mean / std]
   I --> J[PPO-style clipped loss]


For each prompt, GRPO: 1. Generates ``n`` responses (default 8) 2. Computes reward for each response 3. Calculates group-relative advantage: ``adv_i = (reward_i - mean) / std`` 4. Updates the policy using PPO-style clipped objective

Key Parameters
--------------

+---------------------------------------+----------------------------------------------------+-----------------------------------+
| Parameter                             | Impact                                             | Recommended Range                 |
+=======================================+====================================================+===================================+
| ``rollout.n``                         | Group size (more = better estimates, more compute) | 4 to 16                           |
+---------------------------------------+----------------------------------------------------+-----------------------------------+
| ``rollout.temperature``               | Response diversity                                 | 0.8 to 1.2                        |
+---------------------------------------+----------------------------------------------------+-----------------------------------+
| ``algorithm.norm_adv_by_std_in_grpo`` | Normalize advantages                               | true (recommended)                |
+---------------------------------------+----------------------------------------------------+-----------------------------------+
| ``actor.optim.lr``                    | Learning rate                                      | 1e-7 to 5e-6                      |
+---------------------------------------+----------------------------------------------------+-----------------------------------+
| ``data.train_batch_size``             | Prompts per step                                   | 128 to 1024                       |
+---------------------------------------+----------------------------------------------------+-----------------------------------+

GRPO Advantages for Agentic Tasks
---------------------------------

GRPO is particularly well-suited for agentic training:

1. **No Critic model** — Saves GPU memory, critical when tool environments consume resources
2. **Group diversity** — Multiple responses from the same prompt explore different tool-use strategies
3. **Simple reward signal** — Works with binary/sparse rewards common in agentic tasks (pass/fail)
4. **Scalable** — Total rollout samples = ``batch_size × n``, naturally parallelized

Agentic GRPO Example
--------------------

.. code:: yaml

   # GRPO + multi-turn tool interaction
   actor_ref:
     algorithm:
       adv_estimator: grpo
     actor:
       ppo_mini_batch_size: 128

   rollout:
     n: 8
     flow_function: naive
     multiturn:
       env_type: tool_env
       max_env_turns: 5
       max_assistant_turns: 10

   data:
     train_batch_size: 256
     max_response_length: 8192   # Longer for multi-turn trajectories

Common Issues
-------------

+--------------------------------+-----------------------------------+--------------------------------------+
| Symptom                        | Cause                             | Fix                                  |
+================================+===================================+======================================+
| All rewards identical in group | Temperature too low               | Increase ``rollout.temperature``     |
+--------------------------------+-----------------------------------+--------------------------------------+
| Advantage variance too high    | Small group size                  | Increase ``rollout.n``               |
+--------------------------------+-----------------------------------+--------------------------------------+
| Slow rollout                   | Large ``n`` × ``batch_size``      | Reduce one or both, add rollout GPUs |
+--------------------------------+-----------------------------------+--------------------------------------+
| Training instability           | ``norm_adv_by_std_in_grpo=false`` | Set to ``true``                      |
+--------------------------------+-----------------------------------+--------------------------------------+
| OOM during rollout             | Too many concurrent samples       | Reduce ``train_server_concurrency``  |
+--------------------------------+-----------------------------------+--------------------------------------+
