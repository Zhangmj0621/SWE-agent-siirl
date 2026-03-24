Metrics & Evaluation
====================

   **Who this is for:** Users monitoring training progress and evaluating model performance.

   **What you will get:** Available metrics, logging backends, and evaluation configuration.

Logging Backends
----------------

.. code:: yaml

   trainer:
     logger: ["console", "wandb"]
     project_name: siirl_examples
     experiment_name: my_experiment

Supported backends: ``console``, ``wandb``

Key Training Metrics
--------------------

==================== ============================
Metric               Description
==================== ============================
``reward/mean``      Average reward per step
``reward/std``       Reward standard deviation
``actor/loss``       Actor policy loss
``actor/clip_ratio`` Fraction of clipped updates
``critic/loss``      Critic value loss (PPO only)
``kl/mean``          KL divergence from reference
``lr``               Current learning rate
==================== ============================

Rollout Metrics
---------------

================================ =======================
Metric                           Description
================================ =======================
``rollout/generation_duration``  LLM generation time
``rollout/reward_duration``      Reward computation time
``rollout/total_tokens``         Total tokens generated
``rollout/response_length_mean`` Average response length
================================ =======================

Validation
----------

.. code:: yaml

   trainer:
     test_freq: 10          # Validate every N steps
     val_before_train: true # Validate before first step
     log_val_generations: 5 # Log N validation samples

   data:
     val_batch_size: null   # null = entire validation set

MetricWorker
------------

The ``MetricWorker`` Ray actor aggregates metrics from all training ranks: - ``siirl/utils/metrics/`` — Metric collection and aggregation - Only rank 0 of the TrainerGroup reports to logging backends
