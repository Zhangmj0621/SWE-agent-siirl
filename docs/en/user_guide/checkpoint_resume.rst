Checkpoint & Resume
===================

   **Who this is for:** Users managing checkpoints and resuming training.

   **What you will get:** Checkpoint configuration, resume modes, and HuggingFace model export.

Save Configuration
------------------

.. code:: yaml

   trainer:
     save_freq: 10                    # Save every N steps (-1 = disable)
     max_actor_ckpt_to_keep: 5       # Keep last N actor checkpoints
     max_critic_ckpt_to_keep: 5      # Keep last N critic checkpoints
     default_local_dir: checkpoints/  # Checkpoint directory

   actor_ref:
     checkpoint:
       save_contents: ["model", "optimizer", "extra"]  # What to save
       # Add "hf_model" to also export HuggingFace format

Resume Training
---------------

.. code:: yaml

   trainer:
     resume_mode: auto           # auto-detect latest checkpoint
     # Or:
     resume_mode: resume_path
     resume_from_path: /path/to/checkpoint/step_100

Resume Modes
~~~~~~~~~~~~

=============== ===============================================
Mode            Behavior
=============== ===============================================
``auto``        Find latest checkpoint in ``default_local_dir``
``disable``     Start fresh, ignore existing checkpoints
``resume_path`` Resume from specific ``resume_from_path``
=============== ===============================================

Checkpoint Contents
-------------------

============= ==================================== =========
Content       Description                          Default
============= ==================================== =========
``model``     Model weights (Megatron format)      Saved
``optimizer`` Optimizer states                     Saved
``extra``     RNG states, lr_scheduler, step count Saved
``hf_model``  HuggingFace format (converted)       Not saved
============= ==================================== =========

Async Checkpoint Save
---------------------

.. code:: yaml

   actor_ref:
     checkpoint:
       async_save: true    # Experimental: non-blocking checkpoint save
