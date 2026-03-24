Why Agentic RL?
===============

   **Who this is for:** Anyone evaluating RL training frameworks for LLM agent tasks.

   **What you will get:** A clear understanding of why agentic RL requires purpose-built training infrastructure, and how siirl-agentic addresses gaps that general-purpose RL frameworks leave open.

The Problem: General RL Frameworks Hit a Wall on Agentic Tasks
--------------------------------------------------------------

Standard RL training frameworks (verl, OpenRLHF, TRL) are designed for **single-turn** text generation: prompt in, response out, reward computed, policy updated. This works well for math reasoning, summarization, and instruction following.

But **agentic tasks** are fundamentally different:

+-------------------+---------------------------+-----------------------------------------------------------------------+
| Dimension         | Single-Turn RL            | Agentic RL                                                            |
+===================+===========================+=======================================================================+
| Interaction       | One prompt → one response | Multi-turn: generate → tool call → env feedback → generate → …        |
+-------------------+---------------------------+-----------------------------------------------------------------------+
| Trajectory length | Fixed, predictable        | Variable, potentially 10-50+ turns                                    |
+-------------------+---------------------------+-----------------------------------------------------------------------+
| Latency profile   | Uniform (GPU-bound)       | Highly variable (tool calls: 100ms–30s)                               |
+-------------------+---------------------------+-----------------------------------------------------------------------+
| Reward signal     | End-of-sequence           | Per-turn environment rewards + final outcome                          |
+-------------------+---------------------------+-----------------------------------------------------------------------+
| Loss masking      | Simple response mask      | Per-turn loss mask distinguishing model tokens vs. environment tokens |
+-------------------+---------------------------+-----------------------------------------------------------------------+
| Tool management   | N/A                       | Must provision, scale, and load-balance tool instances                |
+-------------------+---------------------------+-----------------------------------------------------------------------+

When you try to force-fit agentic workloads into single-turn frameworks, you encounter:

1. **Trajectory mismatch** — No native support for multi-turn rollouts with tool interaction. Users must build custom wrappers that break framework assumptions.
2. **GPU starvation** — Synchronous rollout blocks on tool calls. A 5-second sandbox execution stalls the entire GPU batch.
3. **Tool bottleneck** — No built-in tool scheduling. 1000 concurrent rollouts competing for 50 sandbox instances creates cascading timeouts.
4. **Reward complexity** — Per-turn rewards, environment-provided signals, and custom scoring functions don’t fit the ``(prompt, response) → float`` reward API.

How siirl-agentic Solves These Problems
---------------------------------------

siirl-agentic is architecturally designed for agentic workloads from the ground up:

1. Native Multi-Turn Trajectory Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``NaiveFlow`` rollout engine implements a state machine (``PENDING → GENERATING → PROCESSING_ENV → TERMINATED``) that handles tool calls as first-class operations. Token-level log-probabilities and per-turn loss masks are tracked across the entire trajectory.

**Contrast with general RL:** No wrapper hacks, no offline trajectory collection, no external reward server.

2. Asynchronous MPMD Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Trainer and Rollout run as independent Ray actors. While one rollout waits for a sandbox response, other rollouts continue generating. The ``DataCoordinator`` buffers completed samples for training consumption.

**Contrast with general RL:** Synchronous rollout-then-train pipelines waste GPU cycles on tool wait times.

3. Pluggable Task Definition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``AgentFlow`` protocol (``preprocess → generate → reward``) lets users define new agentic tasks via YAML config injection. Switching from SWE tasks to search-augmented QA requires changing a config file, not framework code.

**Contrast with general RL:** New task types require subclassing framework internals and modifying the training loop.

4. Elastic Tool Infrastructure (AIO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AIO’s three-layer scheduler (Proxy → ResourcePool → WorkerManager) auto-scales tool instances based on Holt-Winters demand prediction. The async batch request submitter absorbs latency spikes.

**Contrast with general RL:** Users must manually provision and manage tool servers, with no load balancing or auto-scaling.

Code Evidence
-------------

+-----------------------------------+---------------------------------------------------------------------------------------------------+
| Capability                        | Key Files                                                                                         |
+===================================+===================================================================================================+
| Multi-turn state machine          | ``siirl/execution/rollout/agent_flow/naive_flow.py``                                              |
+-----------------------------------+---------------------------------------------------------------------------------------------------+
| AgentFlow protocol                | ``siirl/execution/rollout/agentflow/base.py``, ``siirl/execution/rollout/agentflow/__init__.py``  |
+-----------------------------------+---------------------------------------------------------------------------------------------------+
| Async MPMD execution              | ``siirl/async_train.py``, ``siirl/worker/rollout/rollout_manager.py``                             |
+-----------------------------------+---------------------------------------------------------------------------------------------------+
| Tool environment                  | ``siirl/environment/tool_env/base_tool_env.py``, ``siirl/environment/tool_env/aio_search_env.py`` |
+-----------------------------------+---------------------------------------------------------------------------------------------------+
| AIO scheduling                    | ``AIO/aio/Scheduler/proxy.py``, ``AIO/aio/Scheduler/resource_pool.py``                            |
+-----------------------------------+---------------------------------------------------------------------------------------------------+

Key Configuration
-----------------

.. code:: yaml

   rollout:
     multiturn:
       env_type: tool_env          # Enable multi-turn tool environment
       max_env_turns: 5            # Max tool interaction rounds
       max_assistant_turns: 10     # Max model generation turns
       max_parallel_calls: 4       # Concurrent tool calls per sample

vs. General RL Frameworks
-------------------------

   siirl-agentic treats multi-turn tool interaction as a **first-class training primitive**, not an afterthought. The entire stack — from rollout engine to data coordinator to tool scheduler — is designed around the assumption that trajectories are variable-length, tool-dependent, and latency-heterogeneous.
