Agentic Multi-Turn Training
===========================

   **Who this is for:** Users configuring multi-turn agentic rollouts with tool interaction.

   **What you will get:** Deep understanding of multi-turn configuration, tool environment setup, state machine behavior, and loss masking.

Overview
--------

Multi-turn agentic training is the core differentiator of siirl-agentic. The rollout engine executes a state machine where the model generates text, calls tools, receives environment feedback, and continues generating — all within a single training step.

Core Components
---------------

NaiveFlow State Machine
~~~~~~~~~~~~~~~~~~~~~~~

The ``NaiveFlow`` class (``siirl/execution/rollout/agent_flow/naive_flow.py``) implements the multi-turn rollout:

.. mermaid::

   graph LR
   A[PENDING] --> B[GENERATING]
   B --> C{Tool calls?}
   C -->|Yes| D[PROCESSING_ENV]
   C -->|No| E[TERMINATED]
   D --> F{Max turns?}
   F -->|No| B
   F -->|Yes| E


AgentData (``siirl/execution/rollout/utils.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Internal state tracking for each rollout sample:

.. code:: python

   class AgentData:
       def __init__(self, raw_prompt: list[dict[str, Any]]):
           self.messages = raw_prompt           # OpenAI-format conversation history
           self.prompts_ids = []                # Current full sequence (prompt + all turns)
           self.response_ids = []               # Current turn response tokens
           self.response_mask = []              # 1=model token, 0=env token
           self.rollout_log_prob = []           # Log-probs for all response tokens
           self.env_calls: list[FunctionCall] = []  # Pending tool calls
           self.env_rewards = []                # Per-turn environment rewards
           self.env_turns = 0                   # Current environment turn count
           self.assistant_turns = 0             # Current assistant turn count
           self.state = AgentState.PENDING      # Current state machine state
           self.env_kwargs = {}                 # Extra kwargs passed to tool env

   class AgentState(Enum):
       PENDING = "pending"
       GENERATING = "generating"
       BEFORE_PROCESSING_ENV = "before_processing_envs"
       PROCESSING_ENV = "processing_envs"
       TERMINATED = "terminated"
       ABORTED = "aborted"

Configuration
-------------

Full Multi-Turn Config
~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

   rollout:
     flow_function: naive               # Use NaiveFlow
     max_model_len: 8192                # Max total context length

     multiturn:
       env_type: tool_env               # Environment type
       max_env_turns: 5                 # Max tool interaction rounds
       max_assistant_turns: 10          # Max model generation turns
       max_parallel_calls: 4            # Concurrent tool calls
       max_env_response_length: 256     # Max env response tokens
       env_response_truncate_side: middle  # left/middle/right
       env_path: /path/to/env_config.yaml
       env_kwargs:
         tool_format: hermes            # Tool call format: hermes, gpt-oss

   data:
     max_response_length: 4096          # Max total response tokens
     mask_history: false                # Train on all turns vs last only

Parameter Guide
~~~~~~~~~~~~~~~

+--------------------------------+-------------------------------+-------------------------------------+
| Parameter                      | Impact                        | Recommendation                      |
+================================+===============================+=====================================+
| ``max_env_turns``              | Tool interaction depth        | 3-10 for SWE, 1-3 for search        |
+--------------------------------+-------------------------------+-------------------------------------+
| ``max_assistant_turns``        | Total model generation rounds | 2× max_env_turns                    |
+--------------------------------+-------------------------------+-------------------------------------+
| ``max_parallel_calls``         | Parallel tool execution       | 1-4 (higher = more tool throughput) |
+--------------------------------+-------------------------------+-------------------------------------+
| ``max_env_response_length``    | Tool response truncation      | 256-1024 depending on tool          |
+--------------------------------+-------------------------------+-------------------------------------+
| ``env_response_truncate_side`` | Where to truncate             | “middle” preserves start+end        |
+--------------------------------+-------------------------------+-------------------------------------+
| ``max_response_length``        | Total token budget            | Sum of all turns’ tokens            |
+--------------------------------+-------------------------------+-------------------------------------+

Tool Call Formats
-----------------

siirl-agentic supports multiple tool call formats via ``ToolParser``:

Hermes Format (Default)
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: xml

   <tool_call>
   {"name": "search", "arguments": {"query_list": ["Tokyo population"]}}
   </tool_call>

GPT-OSS Format
~~~~~~~~~~~~~~

.. code:: json

   {"name": "search", "arguments": "{\"query_list\": [\"Tokyo population\"]}"}

Configure via ``env_kwargs.tool_format``:

.. code:: yaml

   rollout:
     multiturn:
       env_kwargs:
         tool_format: hermes    # or "gpt-oss"

Loss Masking
------------

The key to multi-turn training is **correct loss masking**:

::

   Tokens:       [prompt] [assistant-1] [tool-response] [assistant-2] [tool-response] [assistant-3]
   response_mask: 0 0 0    1 1 1 1 1     0 0 0 0 0       1 1 1 1       0 0 0 0 0       1 1 1 1

- **Model tokens** (assistant turns): ``response_mask = 1`` → included in policy gradient
- **Environment tokens** (tool responses): ``response_mask = 0`` → masked from loss
- **Prompt tokens**: Not in response → not in loss

This ensures the model learns from its own decisions, not from environment outputs.

Termination Conditions
----------------------

A rollout terminates when **any** of these conditions is met:

1. ``assistant_turns >= max_assistant_turns``
2. ``env_turns >= max_env_turns``
3. ``len(response_mask) >= max_response_length``
4. No tool calls detected in model output (single-turn completion)
5. Environment returns ``complete=True``

Custom Tool Environments
------------------------

Implement the ``ToolEnv`` interface:

.. code:: python

   from siirl.environment.tool_env.base_tool_env import ToolEnv
   from siirl.environment.base import EnvResponse

   class MyCustomTool(ToolEnv):
       async def create(self, instance_id=None, **kwargs):
           # Initialize tool instance
           return instance_id, EnvResponse()

       async def step(self, action: dict) -> EnvResponse:
           # Execute tool action
           result = await my_tool_logic(action)
           return EnvResponse(
               text=result,
               rewards=0.5,       # Optional per-step reward
               complete=False,    # Set True to end trajectory
           )

       async def release(self, instance_id: str):
           # Cleanup tool instance
           pass

Common Mistakes
---------------

+---------------------------------------------------+---------------------------+---------------------------------------+
| Mistake                                           | Symptom                   | Fix                                   |
+===================================================+===========================+=======================================+
| ``max_response_length`` < total multi-turn tokens | Premature truncation      | Increase to 4096-8192                 |
+---------------------------------------------------+---------------------------+---------------------------------------+
| ``max_env_turns=1`` for SWE tasks                 | Only one tool call        | Increase to 5+                        |
+---------------------------------------------------+---------------------------+---------------------------------------+
| Missing ``env_path``                              | Tool env not loaded       | Provide path to tool config           |
+---------------------------------------------------+---------------------------+---------------------------------------+
| ``tool_format`` mismatch                          | Tool calls not parsed     | Match format to model’s training data |
+---------------------------------------------------+---------------------------+---------------------------------------+
| ``max_parallel_calls`` too high                   | Tool server overload      | Start with 1, increase gradually      |
+---------------------------------------------------+---------------------------+---------------------------------------+
