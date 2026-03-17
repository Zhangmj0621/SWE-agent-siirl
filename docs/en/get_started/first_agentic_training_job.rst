First Agentic Training Job
==========================

   **Who this is for:** Users who want to train an LLM agent with multi-turn tool interaction.

   **What you will get:** A running GRPO training job where the agent uses search tools during rollout.

Prerequisites
-------------

- siirl-agentic installed and verified (see :doc:`Installation <installation>`)
- Completed :doc:`Quickstart <quickstart>` (basic GRPO training works)
- **AIO tool infrastructure** cloned and installed. AIO is a separate repository in the monorepo:

  .. code:: bash

     # From the monorepo root (parent of siirl-agentic/)
     cd AIO
     pip install -e .

  If you do not have the AIO repository, contact your team for access or see :doc:`AIO Tool Infrastructure <../user_guide/aio_tool_infrastructure>`.

What Makes This “Agentic”
-------------------------

In a standard RL training job, the model generates a single response and receives a reward. In an **agentic** training job:

1. The model generates text that may include **tool calls** (e.g., search queries, code execution)
2. Tool calls are **executed in real environments** via the AIO infrastructure
3. Tool responses are **appended to the conversation** as environment observations
4. The model continues generating based on tool responses
5. This multi-turn loop repeats until termination (max turns or model decision)
6. **Only model-generated tokens** contribute to the policy gradient (environment tokens are masked)

Step 1: Configure Multi-Turn Rollout
------------------------------------

Add multi-turn configuration to your training config:

.. code:: yaml

   rollout:
     flow_function: naive
     multiturn:
       env_type: tool_env
       max_env_turns: 5
       max_assistant_turns: 10
       max_parallel_calls: 4
       max_env_response_length: 256
       env_response_truncate_side: middle
       env_path: /path/to/tool_env_config.yaml
       env_kwargs:
         tool_format: hermes

Step 2: Configure Tool Environment
----------------------------------

Create a tool environment config (``tool_env_config.yaml``):

.. code:: yaml

   tools:
     - name: search
       type: aio_search
       config:
         topk: 3

Step 3: Start AIO Infrastructure
--------------------------------

Before training, launch the AIO Proxy and WorkerManagers:

.. code:: bash

   # Start AIO Proxy
   python -m aio.Scheduler.proxy --config aio_config.yaml

   # Start WorkerManager on each tool node
   python -m aio.Scheduler.Resources.worker_manager --proxy-url http://proxy-host:8080

Step 4: Launch Agentic Training
-------------------------------

.. code:: bash

   # Using the AIO example script
   cd siirl-agentic
   bash examples/AIO/run_qwen3_8b.sh

**Expected output:**

::

   INFO  | Ray is initialized. Time cost: 150.23 ms
   INFO  | MainRunner started. Beginning workflow setup...
   INFO  | Initializing DataCoordinator with 1 distributed DataBuffers...
   SUCCESS | RolloutManager initialized. Router at: http://10.0.0.1:30000
   INFO  | Starting async training loop...
   INFO  | [NaiveFlow] PENDING -> GENERATING (sample 0)
   INFO  | [NaiveFlow] GENERATING -> PROCESSING_ENV (sample 0, 2 tool calls)
   INFO  | [AIOSearchTool] search query dispatched to worker
   INFO  | [NaiveFlow] PROCESSING_ENV -> GENERATING (sample 0, env_turn 1)
   INFO  | [NaiveFlow] GENERATING -> TERMINATED (sample 0, reward=1.0)

Step 5: Monitor the Training
----------------------------

Key metrics to watch:

- **reward/mean** — Average reward per step (should increase)
- **rollout/generation_duration** — Time spent in LLM generation
- **rollout/reward_duration** — Time spent computing rewards
- **rollout/env_turns_mean** — Average tool interaction rounds per sample

Understanding the Trajectory
----------------------------

A typical agentic trajectory looks like:

::

   [User] Solve: what is the population of Tokyo?
   [Assistant] I'll search for this information.
               <tool_call>search(query_list=["Tokyo population"])</tool_call>
   [Tool]      Tokyo has a population of approximately 13.96 million...
   [Assistant] Based on the search results, the population of Tokyo is
               approximately 13.96 million people.

In token space: - User prompt → ``response_mask = 0`` (not trained on) - Assistant turn 1 → ``response_mask = 1`` (trained on) - Tool response → ``response_mask = 0`` (masked from loss) - Assistant turn 2 → ``response_mask = 1`` (trained on)

Minimal Runnable Example
------------------------

.. code:: python

   # Verify multi-turn config is parsed correctly
   from siirl.params import parse_config, SiiRLArguments

   config = parse_config()  # Reads from YAML/CLI
   print(f"env_type: {config.rollout.multiturn.env_type}")
   print(f"max_env_turns: {config.rollout.multiturn.max_env_turns}")
   print(f"max_assistant_turns: {config.rollout.multiturn.max_assistant_turns}")

Common Issues
-------------

+-----------------------------------------+-------------------------------------------+-----------------------------------------+
| Symptom                                 | Cause                                     | Fix                                     |
+=========================================+===========================================+=========================================+
| ``AIOSearchTool: Error getting server`` | AIO Proxy not running                     | Start ``python -m aio.Scheduler.proxy`` |
+-----------------------------------------+-------------------------------------------+-----------------------------------------+
| ``TERMINATED after 1 turn``             | ``max_assistant_turns=1``                 | Increase ``max_assistant_turns``        |
+-----------------------------------------+-------------------------------------------+-----------------------------------------+
| Tool response truncated                 | ``max_env_response_length`` too small     | Increase the value                      |
+-----------------------------------------+-------------------------------------------+-----------------------------------------+
| All rewards = 0                         | Reward function doesn’t handle multi-turn | Check custom reward logic               |
+-----------------------------------------+-------------------------------------------+-----------------------------------------+

Next Steps
----------

- :doc:`Agentic Multi-turn Guide <../user_guide/agentic_multiturn>` — Deep dive into multi-turn configuration
- :doc:`AIO Tool Infrastructure <../user_guide/aio_tool_infrastructure>` — Configure and scale tool environments
- :doc:`Pluggable AgentFlow <../highlights/pluggable_agentflow_protocol>` — Customize the rollout flow
