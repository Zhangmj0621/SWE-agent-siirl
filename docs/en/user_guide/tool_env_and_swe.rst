Tool Environment & SWE Tasks
============================

   **Who this is for:** Users building or configuring tool environments for SWE-bench and similar coding tasks.

   **What you will get:** Guide to configuring tool environments, implementing custom tools, and running SWE-style training.

Overview
--------

siirl-agentic supports agentic training on SWE-style tasks where the model interacts with code sandboxes, file systems, and test runners. The tool environment system provides a unified interface for these interactions.

Tool Environment Architecture
-----------------------------

.. mermaid::

   graph TB
   A[NaiveFlow] -->|tool_call| B[ToolEnv Manager]
   B --> C[ToolParser - extract calls]
   C --> D{Tool Type}
   D -->|search| E[AIOSearchEnv]
   D -->|sandbox| F[SandboxEnv]
   D -->|custom| G[CustomToolEnv]


Built-in Tool Environments
--------------------------

AIO Search Environment
~~~~~~~~~~~~~~~~~~~~~~

For retrieval-augmented tasks:

.. code:: yaml

   rollout:
     multiturn:
       env_type: tool_env
       env_kwargs:
         tool_format: hermes

Key file: ``siirl/environment/tool_env/aio_search_env.py``

Custom Tool Environment
~~~~~~~~~~~~~~~~~~~~~~~

Implement ``ToolEnv`` for your own tools:

.. code:: python

   from siirl.environment.tool_env.base_tool_env import ToolEnv
   from siirl.environment.base import EnvResponse

   class MySWETool(ToolEnv):
       async def step(self, action: dict) -> EnvResponse:
           # Run code in sandbox, execute tests, etc.
           result = await self.run_in_sandbox(action["code"])
           return EnvResponse(
               text=result.output,
               rewards=1.0 if result.tests_passed else 0.0,
               complete=result.all_tests_passed,
           )

SWE-Bench Configuration
-----------------------

.. code:: yaml

   rollout:
     flow_function: naive
     multiturn:
       env_type: tool_env
       max_env_turns: 10
       max_assistant_turns: 20
       max_env_response_length: 1024
       env_kwargs:
         tool_format: hermes

   data:
     max_response_length: 16384   # SWE tasks need long context

Common Mistakes
---------------

+-------------------------------+---------------------------+-------------------------------+
| Mistake                       | Symptom                   | Fix                           |
+===============================+===========================+===============================+
| Sandbox not accessible        | Tool execution timeout    | Check AIO Proxy connectivity  |
+-------------------------------+---------------------------+-------------------------------+
| Short ``max_response_length`` | Incomplete edits          | Increase to 16384+            |
+-------------------------------+---------------------------+-------------------------------+
| Wrong ``tool_format``         | Tool calls not parsed     | Match model’s expected format |
+-------------------------------+---------------------------+-------------------------------+
