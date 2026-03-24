Pluggable AgentFlow Protocol
============================

   **Who this is for:** Users who need to define custom agentic task logic (e.g., SWE coding, search-augmented QA, math verification) without modifying framework internals.

   **What you will get:** Understanding of the three-stage AgentFlow protocol and how to inject custom logic via configuration.

Capability
----------

siirl-agentic defines a **three-stage protocol** — ``preprocess → generate → reward`` — that encapsulates all task-specific logic for agentic RL training. Each stage can be independently overridden via YAML configuration, allowing users to switch between entirely different agentic scenarios (coding, search, math) by changing a config file.

User Value
----------

- **Zero framework code changes.** Switch from SWE-agent to search-agent by changing ``flow_config`` in YAML.
- **Independent stage overrides.** Override just the reward function while keeping the default generate logic, or vice versa.
- **Dynamic function injection.** Custom ``preprocess_fn``, ``generate_fn``, and ``reward_fn`` are loaded at runtime via Python import paths.
- **Built-in + custom flows.** Use built-in flows (e.g., ``swe``) or point to your own Python module.

Code Entry Points
-----------------

+-----------------------+------------------------------------------------------+-----------------------------------------------------------------+
| Component             | File                                                 | Responsibility                                                  |
+=======================+======================================================+=================================================================+
| AgentFlow Protocol    | ``siirl/execution/rollout/agentflow/base.py``        | Defines ``AgentFlow``, ``Sample``, ``Model``, ``ModelResponse`` |
+-----------------------+------------------------------------------------------+-----------------------------------------------------------------+
| Flow loader           | ``siirl/execution/rollout/agentflow/__init__.py``    | ``load_agentflow()`` resolves config to flow instance           |
+-----------------------+------------------------------------------------------+-----------------------------------------------------------------+
| Built-in SWE flow     | ``siirl/execution/rollout/agentflow/swe/``           | SWE-bench agentic flow implementation                           |
+-----------------------+------------------------------------------------------+-----------------------------------------------------------------+
| Custom reward         | ``siirl/utils/reward_score/custom_reward.py``        | Custom reward function loading                                  |
+-----------------------+------------------------------------------------------+-----------------------------------------------------------------+
| NaiveFlow (runtime)   | ``siirl/execution/rollout/agent_flow/naive_flow.py`` | Production rollout flow with tool interaction                   |
+-----------------------+------------------------------------------------------+-----------------------------------------------------------------+
| Config params         | ``siirl/params/model_args.py``                       | ``RolloutArguments.flow_function``, ``flow_config``             |
+-----------------------+------------------------------------------------------+-----------------------------------------------------------------+

Key Configuration
-----------------

.. code:: yaml

   rollout:
     # Select flow implementation
     flow_function: naive        # "naive" for NaiveFlow, or custom module path
     flow_config: config.yaml    # Path to flow-specific config

     # AgentFlow-level config (loaded by load_agentflow)
     agentflow:
       name: swe                         # Built-in alias or "module.path:ClassName"
       preprocess_fn: null               # Optional: "my_module:my_preprocess"
       generate_fn: null                 # Optional: "my_module:my_generate"
       reward_fn: my_evaluator:parse_reward  # Override reward computation
       python_path:                      # Additional import paths
         - /path/to/custom/agents

   # Custom reward function (alternative to agentflow.reward_fn)
   custom_reward_function:
     path: /path/to/reward.py
     name: reward_function
     reward_kwargs:
       timeout: 30

How It Works
------------

The AgentFlow Protocol
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   class AgentFlow(Protocol):
       def preprocess(self, sample: dict) -> Sample:
           """Convert raw dataset record to Sample object"""

       async def generate(self, sample: Sample):
           """Run multi-turn rollout with tool interaction"""

       async def reward(self, sample: Sample):
           """Evaluate trajectory and compute reward"""

Dynamic Injection
~~~~~~~~~~~~~~~~~

The ``load_agentflow()`` function composes an AgentFlow instance from config:

.. mermaid::

   graph LR
   A[YAML Config] --> B[load_agentflow]
   B --> C[Import agent class]
   B --> D[Import preprocess_fn]
   B --> E[Import generate_fn]
   B --> F[Import reward_fn]
   C --> G[Instantiate AgentFlow]
   D --> G
   E --> G
   F --> G
   G --> H[Ready for rollout]


Each function override uses ``MethodType`` binding:

.. code:: python

   agent = agent_class(config, model)

   if preprocess_fn is not None:
       agent.preprocess = MethodType(preprocess_fn, agent)
   if generate_fn is not None:
       agent.generate = MethodType(generate_fn, agent)
   if reward_fn is not None:
       agent.reward = MethodType(reward_fn, agent)

Example: Custom Reward Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # my_evaluator.py
   async def parse_reward(self, sample):
       """Custom reward: check if code passes test suite."""
       code_output = sample.conversations[-1]["content"]
       sample.reward = 1.0 if "PASSED" in code_output else 0.0

Config:

.. code:: yaml

   rollout:
     agentflow:
       name: swe
       reward_fn: my_evaluator:parse_reward
       python_path: [/path/to/my_evaluator]

vs. General RL Frameworks
-------------------------

   General RL frameworks require subclassing internal classes or modifying the training loop to support new task types. siirl-agentic’s AgentFlow protocol provides a **clean injection boundary** — all task-specific logic lives outside the framework, loaded via configuration, and hot-swappable between experiments.
