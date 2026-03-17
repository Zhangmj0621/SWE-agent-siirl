Native Agentic Trajectory Training
==================================

   **Who this is for:** Users training LLM agents on tasks involving tool calls, environment interaction, and multi-turn decision making.

   **What you will get:** Understanding of how siirl-agentic natively trains on agentic trajectories without offline data collection or custom wrappers.

Capability
----------

siirl-agentic trains PPO and GRPO **directly on multi-turn agentic trajectories** — sequences where the model generates text, invokes tools, receives environment feedback, and continues generating. The training loop handles variable-length rollouts with per-turn loss masking and token-level log-probability tracking as built-in primitives.

User Value
----------

- **No offline trajectory collection.** Rollouts happen online during training; the model generates, calls tools, and receives rewards in a single forward pass.
- **Per-turn loss masking.** Environment tokens (tool responses, observations) are automatically masked from the loss computation. Only model-generated tokens contribute to policy gradients.
- **Token-level log-probability tracking.** Log-probs are tracked across the entire multi-turn trajectory, enabling correct importance sampling for PPO/GRPO.
- **Environment reward integration.** Tool environments can return per-step rewards (e.g., code execution success) that are accumulated alongside final outcome rewards.

Code Entry Points
-----------------

+--------------------------+------------------------------------------------------+---------------------------------------------------------------------------------------+
| Component                | File                                                 | Responsibility                                                                        |
+==========================+======================================================+=======================================================================================+
| Multi-turn state machine | ``siirl/execution/rollout/agent_flow/naive_flow.py`` | ``NaiveFlow`` implements ``PENDING → GENERATING → PROCESSING_ENV → TERMINATED`` cycle |
+--------------------------+------------------------------------------------------+---------------------------------------------------------------------------------------+
| AgentFlow protocol       | ``siirl/execution/rollout/agentflow/base.py``        | ``AgentFlow`` Protocol with ``preprocess``, ``generate``, ``reward``                  |
+--------------------------+------------------------------------------------------+---------------------------------------------------------------------------------------+
| Sample data structure    | ``siirl/execution/rollout/agentflow/base.py``        | ``Sample`` tracks ``tokens``, ``loss_mask``, ``rollout_log_probs``, ``conversations`` |
+--------------------------+------------------------------------------------------+---------------------------------------------------------------------------------------+
| Tool environment         | ``siirl/environment/tool_env/base_tool_env.py``      | ``ToolEnv`` base class for tool interaction                                           |
+--------------------------+------------------------------------------------------+---------------------------------------------------------------------------------------+
| AIO search env           | ``siirl/environment/tool_env/aio_search_env.py``     | Concrete tool env using AIO scheduler                                                 |
+--------------------------+------------------------------------------------------+---------------------------------------------------------------------------------------+

Key Configuration
-----------------

.. code:: yaml

   rollout:
     flow_function: naive           # Use NaiveFlow for multi-turn rollout
     multiturn:
       env_type: tool_env           # Enable tool environment
       max_env_turns: 5             # Max environment interaction rounds
       max_assistant_turns: 10      # Max model generation turns
       max_parallel_calls: 4        # Concurrent tool calls per sample
       max_env_response_length: 256 # Truncate long tool responses
       env_response_truncate_side: middle  # left/middle/right truncation

   data:
     max_response_length: 4096      # Max total response length (model + env tokens)
     mask_history: false             # Whether to mask earlier turns

How It Works
------------

The ``NaiveFlow`` rollout operates as an async state machine for each sample:

.. mermaid::

   graph TB
   A[PENDING] --> B[Apply chat template + tool schemas]
   B --> C[GENERATING]
   C --> D{Tool calls detected?}
   D -->|Yes| E[PROCESSING_ENV]
   D -->|No| F[TERMINATED]
   E --> G[Execute tool calls in parallel]
   G --> H{Max turns reached?}
   H -->|No| C
   H -->|Yes| F
   F --> I[Compute reward]


At each state transition:

1. **PENDING → GENERATING**: The prompt is tokenized with ``apply_chat_template``, including tool schemas. The tokenized prompt becomes the initial ``prompts_ids``.
2. **GENERATING**: The SGLang engine generates a response. Output tokens get ``response_mask = 1`` and log-probs are recorded. The tool parser checks for function calls.
3. **PROCESSING_ENV**: Tool calls are executed in parallel (up to ``max_parallel_calls``). Tool response tokens get ``response_mask = 0`` (masked from loss). Environment rewards are accumulated.
4. **TERMINATED**: Final reward is computed using the custom reward function or ``default_compute_score``.

The resulting ``Sample`` contains:

- ``prompts``: Original prompt token IDs
- ``responses``: All response tokens (model + env interleaved)
- ``response_mask``: ``1`` for model tokens, ``0`` for env tokens
- ``rollout_log_prob``: Log-probs for all tokens
- ``rewards``: Scalar reward (env rewards or computed reward)

vs. General RL Frameworks
-------------------------

   General RL frameworks treat rollout as a single ``generate(prompt) → response`` call. siirl-agentic’s ``NaiveFlow`` implements a **full multi-turn state machine** with async tool execution, per-turn loss masking, and environment reward accumulation — all within the standard PPO/GRPO training loop.
