siirl-agentic Documentation
============================

**siirl-agentic** is an asynchronous, agentic reinforcement learning training framework purpose-built
for multi-turn agent trajectories with tool interaction. It natively supports PPO/GRPO training on
SWE-style tasks where agents call tools, receive environment feedback, and accumulate rewards across
long interaction horizons.

.. toctree::
   :maxdepth: 1
   :caption: Highlights

   highlights/why_agentic_rl
   highlights/native_agentic_trajectory_training
   highlights/pluggable_agentflow_protocol
   highlights/mpmd_async_execution_engine
   highlights/aio_elastic_agentic_tool_infrastructure

.. toctree::
   :maxdepth: 1
   :caption: Concepts

   concepts/design_philosophy
   concepts/architecture_overview
   concepts/async_training_lifecycle

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/installation
   get_started/quickstart
   get_started/first_agentic_training_job

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/configuration_system
   user_guide/ppo_training
   user_guide/grpo_training
   user_guide/agentic_multiturn
   user_guide/tool_env_and_swe
   user_guide/aio_tool_infrastructure
   user_guide/deployment_modes
   user_guide/validate_reuse_and_eval_scaling
   user_guide/checkpoint_resume
   user_guide/metrics_and_evaluation

.. toctree::
   :maxdepth: 1
   :caption: Advanced

   advanced/performance_tuning
   advanced/failure_propagation

.. toctree::
   :maxdepth: 1
   :caption: Reference

   reference/config_reference
   reference/module_map
   reference/compatibility_matrix

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   developer_guide/code_structure
   developer_guide/adding_new_executor_or_flow
   developer_guide/contributing

.. toctree::
   :maxdepth: 1
   :caption: FAQ

   faq/troubleshooting
