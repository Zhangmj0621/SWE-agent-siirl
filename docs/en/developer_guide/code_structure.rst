Code Structure
==============

   **Who this is for:** Framework contributors who need to navigate and modify the siirl-agentic codebase.

   **What you will get:** A developer-oriented walkthrough of the code organization, call chains, and extension points.

Entry Point
-----------

Everything starts from ``siirl/async_train.py``:

.. code:: python

   # MainRunner is a Ray actor that orchestrates the entire training lifecycle
   @ray.remote
   class MainRunner:
       def run(self):
           # Phase 1: Parse YAML config → SiiRLArguments
           # Phase 2: Allocate GPU resources (actor_gpus, rollout_gpus, critic_gpus)
           # Phase 3: Initialize DataCoordinator (DataBuffer + DataLoader)
           # Phase 4: Initialize components (Trainer, RolloutManager, Critic)
           # Phase 5: Launch async training loop
           # Phase 6: Wait for completion
           # Phase 7: Check status via TaskCoordinator
           # Phase 8: Cleanup

Call Chain: Training Step
-------------------------

::

   MainRunner.run()
     └─► Trainer.train()           # siirl/worker/actor/
           ├─► DataBuffer.get()    # siirl/data_coordinator/data_buffer.py
           ├─► Actor.forward()     # siirl/engine/actor/
           ├─► Critic.forward()    # (PPO only)
           ├─► compute_advantage() # siirl/algorithm/advantage.py
           ├─► compute_loss()      # siirl/algorithm/loss.py
           └─► Actor.backward()    # gradient update

Call Chain: Rollout Step
------------------------

::

   RolloutManager.rollout()          # siirl/worker/rollout/
     └─► AgentExecutor.execute()     # siirl/execution/rollout/agent_executor/
           └─► NaiveFlow.run()       # siirl/execution/rollout/agent_flow/naive_flow.py
                 ├─► preprocess()    # AgentFlow protocol
                 ├─► generate()      # SGLang inference
                 ├─► Environment.step()  # Tool execution
                 ├─► (repeat until terminated)
                 └─► reward()        # Compute trajectory reward

Call Chain: AgentFlow Loading
-----------------------------

::

   load_agentflow(config)            # siirl/execution/rollout/agentflow/__init__.py
     ├─► Load AgentFlow class (e.g., NaiveFlow)
     ├─► Import preprocess_fn from config
     ├─► Import generate_fn from config
     ├─► Import reward_fn from config
     └─► Inject as MethodType on flow instance

Configuration Flow
------------------

::

   YAML file
     └─► parser.py → SiiRLArguments     # siirl/params/parser.py
           ├─► .data: DataArguments       # siirl/params/data_args.py
           ├─► .actor_ref: ActorArguments # siirl/params/model_args.py
           ├─► .rollout: RolloutArguments # siirl/params/model_args.py
           ├─► .critic: CriticArguments   # siirl/params/model_args.py
           └─► .trainer: TrainingArguments # siirl/params/training_args.py

Key Extension Points
--------------------

1. Adding a New AgentFlow
~~~~~~~~~~~~~~~~~~~~~~~~~

Create a new flow module and register it:

.. code:: python

   # siirl/execution/rollout/agentflow/my_flow.py
   from .base import AgentFlow, Sample, Model

   class MyFlow:
       """Implements the AgentFlow protocol."""

       def preprocess(self, data: dict) -> list[Sample]:
           ...

       def generate(self, model: Model, samples: list[Sample]) -> list[Sample]:
           ...

       def reward(self, samples: list[Sample]) -> list[Sample]:
           ...

Reference it in YAML config:

.. code:: yaml

   rollout:
     agentflow: "siirl.execution.rollout.agentflow.my_flow:MyFlow"

2. Adding a New Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subclass ``BasEnvironment``:

.. code:: python

   # siirl/environment/my_env.py
   from siirl.environment.base import BasEnvironment, EnvResponse

   class MyEnvironment(BasEnvironment):
       async def step(self, action: str) -> EnvResponse:
           ...

       async def reset(self) -> EnvResponse:
           ...

3. Adding a New Reward Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a reward function and inject via config:

.. code:: python

   # my_rewards.py
   def custom_reward(samples):
       for sample in samples:
           sample.reward = compute_reward(sample.conversations)
       return samples

.. code:: yaml

   custom_reward_function: "my_rewards:custom_reward"

4. Adding a New Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~

Extend the algorithm module:

- Add advantage estimation in ``siirl/algorithm/advantage.py``
- Add loss function in ``siirl/algorithm/loss.py``
- Wire into the training loop in the worker

Testing Structure
-----------------

::

   tests/
   ├── unit/            # Fast, isolated unit tests
   ├── integration/     # Multi-component tests
   ├── e2e/             # End-to-end training tests (GPU required)
   └── conftest.py      # Shared fixtures

Run tests:

.. code:: bash

   pytest tests/unit/              # Fast unit tests
   pytest tests/integration/       # Integration tests
   pytest -m gpu                   # GPU-dependent tests

Related
-------

- :doc:`Module Map <../reference/module_map>` — Quick-reference directory guide
- :doc:`Adding New Executor or Flow <adding_new_executor_or_flow>` — Step-by-step tutorial
- :doc:`Architecture Overview <../concepts/architecture_overview>` — High-level design
