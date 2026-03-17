Adding a New Executor or AgentFlow
==================================

   **Who this is for:** Framework contributors extending siirl-agentic with new task types or rollout strategies.

   **What you will get:** A step-by-step tutorial for implementing and registering a custom AgentFlow.

Prerequisites
-------------

- Familiarity with the :doc:`AgentFlow Protocol <../highlights/pluggable_agentflow_protocol>`
- Understanding of the :doc:`Architecture Overview <../concepts/architecture_overview>`
- Local development environment (:doc:`Installation <../get_started/installation>`)

The AgentFlow Protocol
----------------------

Every AgentFlow must implement three stages:

.. code:: python

   class AgentFlow(Protocol):
       def preprocess(self, data: dict) -> list[Sample]:
           """Convert raw data into Sample objects."""
           ...

       def generate(self, model: Model, samples: list[Sample]) -> list[Sample]:
           """Run model inference and environment interaction."""
           ...

       def reward(self, samples: list[Sample]) -> list[Sample]:
           """Compute rewards for completed trajectories."""
           ...

Where ``Sample`` is a dataclass with fields:

+-----------------------+----------------+---------------------------------------------------------+
| Field                 | Type           | Description                                             |
+=======================+================+=========================================================+
| ``tokens``            | ``Tensor``     | Full token sequence (prompt + response)                 |
+-----------------------+----------------+---------------------------------------------------------+
| ``loss_mask``         | ``Tensor``     | 1 for model tokens to train on, 0 for env/prompt tokens |
+-----------------------+----------------+---------------------------------------------------------+
| ``rollout_log_probs`` | ``Tensor``     | Log-probabilities from the rollout policy               |
+-----------------------+----------------+---------------------------------------------------------+
| ``conversations``     | ``list[dict]`` | Full conversation history                               |
+-----------------------+----------------+---------------------------------------------------------+
| ``reward``            | ``float``      | Scalar reward for the trajectory                        |
+-----------------------+----------------+---------------------------------------------------------+

Step 1: Create the Flow Module
------------------------------

Create a new file in ``siirl/execution/rollout/agentflow/``:

.. code:: python

   # siirl/execution/rollout/agentflow/my_task_flow.py

   from dataclasses import dataclass
   from siirl.execution.rollout.agentflow.base import AgentFlow, Sample, Model, ModelResponse


   class MyTaskFlow:
       """
       Custom AgentFlow for [describe your task type].

       Implements the three-stage AgentFlow protocol:
       preprocess → generate → reward
       """

       def __init__(self, config=None):
           self.config = config or {}

       def preprocess(self, data: dict) -> list[Sample]:
           """
           Convert raw dataset items into Sample objects.

           Args:
               data: Dictionary from the dataset loader.
                     Expected keys: 'prompt', 'reference', etc.

           Returns:
               List of Sample objects ready for generation.
           """
           samples = []
           for item in data['items']:
               sample = Sample(
                   conversations=[{"role": "user", "content": item['prompt']}],
                   # Initialize other fields as needed
               )
               samples.append(sample)
           return samples

       def generate(self, model: Model, samples: list[Sample]) -> list[Sample]:
           """
           Run inference loop.

           For single-turn: one model.generate() call.
           For multi-turn: loop with environment interaction.
           """
           for sample in samples:
               # Single-turn example:
               response: ModelResponse = model.generate(sample.conversations)
               sample.tokens = response.tokens
               sample.rollout_log_probs = response.log_probs
               sample.loss_mask = response.loss_mask
               sample.conversations.append({
                   "role": "assistant",
                   "content": response.text
               })
           return samples

       def reward(self, samples: list[Sample]) -> list[Sample]:
           """
           Compute reward for each trajectory.
           """
           for sample in samples:
               # Example: binary correctness reward
               sample.reward = self._evaluate(sample)
           return samples

       def _evaluate(self, sample: Sample) -> float:
           """Custom evaluation logic."""
           # Implement your reward logic here
           return 1.0  # placeholder

Step 2: Register the Flow
-------------------------

Option A: YAML Config Reference (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reference the flow directly in your training config:

.. code:: yaml

   rollout:
     agentflow: "siirl.execution.rollout.agentflow.my_task_flow:MyTaskFlow"

The ``load_agentflow()`` function will dynamically import and instantiate the class.

Option B: Add to Built-in Registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add an entry in ``siirl/execution/rollout/agentflow/__init__.py``:

.. code:: python

   BUILTIN_FLOW = {
       "swe": ".swe:agentflow",
       "my_task": ".my_task_flow:MyTaskFlow",  # Add this line
   }

Then reference by alias:

.. code:: yaml

   rollout:
     agentflow: "my_task"

Step 3: Custom Function Injection
---------------------------------

The AgentFlow system supports dynamic function injection. Instead of subclassing, you can override individual stages via config:

.. code:: yaml

   rollout:
     agentflow: "swe"  # Use built-in SWE flow
     agentflow_preprocess_fn: "my_project.preprocess:custom_preprocess"
     agentflow_reward_fn: "my_project.rewards:custom_reward"

The injected functions are bound to the flow instance via ``MethodType``:

.. code:: python

   # my_project/rewards.py
   def custom_reward(self, samples):
       """
       'self' is the AgentFlow instance.
       Access self.config, self.tokenizer, etc.
       """
       for sample in samples:
           sample.reward = my_scoring_function(sample.conversations)
       return samples

Step 4: Add a Multi-Turn Flow
-----------------------------

For tasks with environment interaction, implement the generate loop:

.. code:: python

   def generate(self, model: Model, samples: list[Sample]) -> list[Sample]:
       for sample in samples:
           for turn in range(self.config.get('max_turns', 5)):
               # 1. Generate model response
               response = model.generate(sample.conversations)

               # 2. Check for tool calls
               if not response.has_tool_call:
                   break

               # 3. Execute tool call via environment
               env_response = self.environment.step(response.tool_call)

               # 4. Append to conversation
               sample.conversations.extend([
                   {"role": "assistant", "content": response.text},
                   {"role": "tool", "content": env_response.text},
               ])

               # 5. Update loss mask (mask environment tokens)
               sample.loss_mask = self._update_loss_mask(
                   sample.loss_mask, response, env_response
               )

       return samples

Step 5: Test Your Flow
----------------------

Create a test file:

.. code:: python

   # tests/unit/test_my_task_flow.py
   import pytest
   from siirl.execution.rollout.agentflow.my_task_flow import MyTaskFlow

   class TestMyTaskFlow:
       def test_preprocess(self):
           flow = MyTaskFlow()
           data = {"items": [{"prompt": "Hello"}]}
           samples = flow.preprocess(data)
           assert len(samples) == 1
           assert samples[0].conversations[0]["content"] == "Hello"

       def test_reward(self):
           flow = MyTaskFlow()
           samples = [Sample(conversations=[...])]
           results = flow.reward(samples)
           assert all(s.reward is not None for s in results)

Run:

.. code:: bash

   pytest tests/unit/test_my_task_flow.py -v

Checklist
---------

- ☐ Flow implements all three protocol methods (``preprocess``, ``generate``, ``reward``)
- ☐ ``preprocess`` returns properly initialized ``Sample`` objects
- ☐ ``generate`` populates ``tokens``, ``rollout_log_probs``, and ``loss_mask``
- ☐ ``reward`` assigns a scalar ``reward`` to each sample
- ☐ Flow is registered (YAML path or built-in registry)
- ☐ Unit tests cover all three stages
- ☐ Documentation updated if adding a built-in flow

Related
-------

- :doc:`Pluggable AgentFlow Protocol <../highlights/pluggable_agentflow_protocol>` — Design rationale
- :doc:`Code Structure <code_structure>` — Full codebase walkthrough
- :doc:`Agentic Multi-Turn <../user_guide/agentic_multiturn>` — Multi-turn configuration
