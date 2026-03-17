siirl-agentic 文档
===================

**siirl-agentic** 是一个异步 agentic 强化学习训练框架，专为多轮 agent 轨迹与工具交互而设计。
原生支持在 SWE-style 任务上进行 PPO/GRPO 训练——agent 调用工具、接收环境反馈，并在长交互 horizon 上累积奖励。

.. toctree::
   :maxdepth: 1
   :caption: 亮点

   highlights/why_agentic_rl
   highlights/native_agentic_trajectory_training
   highlights/pluggable_agentflow_protocol
   highlights/mpmd_async_execution_engine
   highlights/aio_elastic_agentic_tool_infrastructure

.. toctree::
   :maxdepth: 1
   :caption: 核心概念

   concepts/design_philosophy
   concepts/architecture_overview
   concepts/async_training_lifecycle

.. toctree::
   :maxdepth: 1
   :caption: 快速开始

   get_started/installation
   get_started/quickstart
   get_started/first_agentic_training_job

.. toctree::
   :maxdepth: 2
   :caption: 用户指南

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
   :caption: 进阶

   advanced/performance_tuning
   advanced/failure_propagation

.. toctree::
   :maxdepth: 1
   :caption: 参考

   reference/config_reference
   reference/module_map
   reference/compatibility_matrix

.. toctree::
   :maxdepth: 1
   :caption: 开发者指南

   developer_guide/code_structure
   developer_guide/adding_new_executor_or_flow
   developer_guide/contributing

.. toctree::
   :maxdepth: 1
   :caption: 常见问题

   faq/troubleshooting
