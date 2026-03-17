Configuration System
====================

   **Who this is for:** Users who need to understand and customize training configuration.

   **What you will get:** Complete understanding of siirl-agentic’s configuration hierarchy, file format, and override mechanism.

Overview
--------

siirl-agentic uses a **hierarchical dataclass-based configuration system** parsed from YAML files with CLI overrides. All configuration is represented by the ``SiiRLArguments`` dataclass, which contains nested dataclasses for each subsystem.

Configuration Hierarchy
-----------------------

.. code:: text

   SiiRLArguments
   ├── data: DataArguments                    # Dataset and tokenization
   ├── actor_ref: ActorRefArguments           # Actor, Ref, Algorithm, Checkpoint
   │   ├── model: ModelArguments              #   Base model path and settings
   │   ├── actor: ActorArguments              #   Actor training hyperparameters
   │   │   └── optim: OptimizerArguments      #     Optimizer (lr, scheduler, ...)
   │   │   └── megatron: MegatronArguments    #     Megatron backend config
   │   ├── ref: RefArguments                  #   Reference model settings
   │   ├── algorithm: AlgorithmArguments      #   PPO/GRPO algorithm params
   │   └── checkpoint: CheckpointArguments    #   Save/load contents
   ├── rollout: RolloutArguments              # Inference engine and sampling
   │   ├── val_kwargs: EvalSamplingArguments  #   Validation sampling params
   │   └── multiturn: MultiturnArguments      #   Multi-turn agentic config
   ├── critic: CriticArguments                # Critic model (PPO only)
   │   └── optim: OptimizerArguments          #   Critic optimizer
   │   └── megatron: MegatronArguments        #   Critic Megatron config
   │   └── checkpoint: CheckpointArguments    #   Critic checkpoint config
   ├── trainer: TrainingArguments             # Training loop and resources
   └── custom_reward_function: CustomRewardArguments  # Custom reward config

Configuration Files
-------------------

Source Files
~~~~~~~~~~~~

+-----------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| File                              | Contains                                                                                                                                                                     |
+===================================+==============================================================================================================================================================================+
| ``siirl/params/training_args.py`` | ``TrainingArguments``, ``SiiRLArguments``, ``CustomRewardArguments``                                                                                                         |
+-----------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``siirl/params/model_args.py``    | ``ModelArguments``, ``ActorArguments``, ``RefArguments``, ``RolloutArguments``, ``MultiturnArguments``, ``AlgorithmArguments``, ``CriticArguments``, ``CheckpointArguments`` |
+-----------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``siirl/params/data_args.py``     | ``DataArguments``                                                                                                                                                            |
+-----------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``siirl/params/parser.py``        | ``parse_config()`` function                                                                                                                                                  |
+-----------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Example YAML Config
~~~~~~~~~~~~~~~~~~~

.. code:: yaml

   data:
     train_files: ["/data/train.parquet"]
     val_files: ["/data/test.parquet"]
     prompt_key: prompt
     max_prompt_length: 2048
     max_response_length: 4096
     train_batch_size: 512

   actor_ref:
     model:
       path: /models/Qwen3-8B
     actor:
       train_backend: megatron
       ppo_mini_batch_size: 256
       clip_ratio: 0.2
       ppo_epochs: 1
       optim:
         lr: 1e-6
         lr_decay_style: linear
     algorithm:
       adv_estimator: grpo
       norm_adv_by_std_in_grpo: true

   rollout:
     name: sglang
     temperature: 1.0
     top_p: 1.0
     n: 8
     gpu_memory_utilization: 0.7
     tensor_model_parallel_size: 2
     max_model_len: 8192

   trainer:
     total_epochs: 30
     actor_gpus: 4
     rollout_gpus: 4
     save_freq: 10
     test_freq: 5

Key Configuration Groups
------------------------

Data Configuration (``data:``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-------------------------+-----------+----------------------------------+---------------------------------------------+
| Parameter               | Type      | Default                          | Description                                 |
+=========================+===========+==================================+=============================================+
| ``train_files``         | list[str] | ``["~/data/.../train.parquet"]`` | Training dataset paths                      |
+-------------------------+-----------+----------------------------------+---------------------------------------------+
| ``val_files``           | list[str] | ``["~/data/.../test.parquet"]``  | Validation dataset paths                    |
+-------------------------+-----------+----------------------------------+---------------------------------------------+
| ``train_batch_size``    | int       | 1024                             | Samples per training step                   |
+-------------------------+-----------+----------------------------------+---------------------------------------------+
| ``max_prompt_length``   | int       | 512                              | Max prompt token length                     |
+-------------------------+-----------+----------------------------------+---------------------------------------------+
| ``max_response_length`` | int       | 512                              | Max response token length                   |
+-------------------------+-----------+----------------------------------+---------------------------------------------+
| ``mask_history``        | bool      | false                            | Mask earlier turns, train on last turn only |
+-------------------------+-----------+----------------------------------+---------------------------------------------+
| ``train_on_prompt``     | bool      | false                            | Include prompt tokens in loss               |
+-------------------------+-----------+----------------------------------+---------------------------------------------+

Rollout Configuration (``rollout:``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+--------------------------------+----------+---------------+----------------------------------------------+
| Parameter                      | Type     | Default       | Description                                  |
+================================+==========+===============+==============================================+
| ``name``                       | str      | “sglang”      | Inference engine                             |
+--------------------------------+----------+---------------+----------------------------------------------+
| ``temperature``                | float    | 1.0           | Sampling temperature                         |
+--------------------------------+----------+---------------+----------------------------------------------+
| ``n``                          | int      | 1             | Responses per prompt (GRPO typically uses 8) |
+--------------------------------+----------+---------------+----------------------------------------------+
| ``gpu_memory_utilization``     | float    | 0.5           | SGLang GPU memory fraction                   |
+--------------------------------+----------+---------------+----------------------------------------------+
| ``tensor_model_parallel_size`` | int      | 1             | Inference TP size                            |
+--------------------------------+----------+---------------+----------------------------------------------+
| ``train_server_concurrency``   | int      | 256           | Max concurrent requests per engine           |
+--------------------------------+----------+---------------+----------------------------------------------+
| ``flow_function``              | str      | “naive”       | Rollout flow implementation                  |
+--------------------------------+----------+---------------+----------------------------------------------+
| ``executor_module``            | str      | “naive”       | Batch executor module                        |
+--------------------------------+----------+---------------+----------------------------------------------+

Multi-turn Configuration (``rollout.multiturn:``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+--------------------------------+----------+---------------+---------------------------------------------+
| Parameter                      | Type     | Default       | Description                                 |
+================================+==========+===============+=============================================+
| ``env_type``                   | str      | null          | Environment type: ``tool_env``, ``vla_env`` |
+--------------------------------+----------+---------------+---------------------------------------------+
| ``max_env_turns``              | int      | 1             | Max environment interaction rounds          |
+--------------------------------+----------+---------------+---------------------------------------------+
| ``max_assistant_turns``        | int      | 1             | Max model generation turns                  |
+--------------------------------+----------+---------------+---------------------------------------------+
| ``max_parallel_calls``         | int      | 1             | Concurrent tool calls per sample            |
+--------------------------------+----------+---------------+---------------------------------------------+
| ``max_env_response_length``    | int      | 256           | Max env response tokens                     |
+--------------------------------+----------+---------------+---------------------------------------------+
| ``env_response_truncate_side`` | str      | “middle”      | Truncation: left/middle/right               |
+--------------------------------+----------+---------------+---------------------------------------------+

Training Configuration (``trainer:``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

=================== ==== ======= =======================================
Parameter           Type Default Description
=================== ==== ======= =======================================
``total_epochs``    int  30      Training epochs
``actor_gpus``      int  2       GPUs for training
``rollout_gpus``    int  6       GPUs for rollout
``colocate``        bool false   Share GPUs between training and rollout
``async_factor``    int  1       Rollout batch buffer ahead count
``off_policy_step`` int  0       Off-policy version window
``save_freq``       int  -1      Checkpoint save frequency
``resume_mode``     str  “auto”  Resume: auto/disable/resume_path
=================== ==== ======= =======================================

Common Mistakes
---------------

+--------------------------------------------------------+------------------------------+---------------------------------------+
| Mistake                                                | Symptom                      | Fix                                   |
+========================================================+==============================+=======================================+
| ``max_response_length`` too small for multi-turn       | Trajectories truncated early | Increase to 4096+ for agentic tasks   |
+--------------------------------------------------------+------------------------------+---------------------------------------+
| ``n > 1`` with PPO                                     | Unexpected behavior          | Use ``n=1`` for PPO, ``n=8`` for GRPO |
+--------------------------------------------------------+------------------------------+---------------------------------------+
| ``actor_gpus + rollout_gpus > total GPUs``             | Resource allocation failure  | Ensure sum matches available GPUs     |
+--------------------------------------------------------+------------------------------+---------------------------------------+
| Missing ``multiturn.env_type``                         | No tool interaction          | Set ``env_type: tool_env``            |
+--------------------------------------------------------+------------------------------+---------------------------------------+
| ``colocate=true`` with high ``gpu_memory_utilization`` | OOM errors                   | Framework auto-clamps to 0.45         |
+--------------------------------------------------------+------------------------------+---------------------------------------+

CLI Overrides
-------------

Override any config parameter from the command line:

.. code:: bash

   python -m siirl.async_train \
       --config config.yaml \
       trainer.total_epochs=50 \
       rollout.temperature=0.8 \
       data.train_batch_size=256

Next Steps
----------

- :doc:`Config Reference <../reference/config_reference>` — Complete parameter listing with defaults
- :doc:`PPO Training <ppo_training>` — PPO-specific configuration
- :doc:`GRPO Training <grpo_training>` — GRPO-specific configuration
