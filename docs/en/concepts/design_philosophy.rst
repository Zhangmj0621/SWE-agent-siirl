Design Philosophy
=================

   **Who this is for:** Framework Users and Contributors who want to understand the guiding
   principles behind siirl-agentic's architecture.

   **What you will get:** Clarity on *why* the system is designed the way it is, enabling
   better configuration decisions and extension choices.


Why a Dedicated Agentic RL Framework?
--------------------------------------

Standard RL post-training frameworks (verl, OpenRLHF, TRL) are designed for **single-turn
text generation**: one prompt in, one response out, compute reward, update policy. This works
well for tasks like math reasoning, summarization, and instruction following.

**Agentic tasks** are fundamentally different:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Dimension
     - Single-Turn RL
     - Agentic RL
   * - Interaction pattern
     - prompt → response
     - Multi-turn: generate → tool call → env feedback → generate → …
   * - Trajectory length
     - Fixed, predictable
     - Variable, 10–50+ turns
   * - Latency profile
     - Uniform (GPU-bound)
     - Highly variable (tool calls: 100ms–30s)
   * - Reward signal
     - End-of-sequence
     - Per-turn environment feedback + final reward
   * - Loss computation
     - All response tokens
     - Only model-generated tokens (environment tokens masked)

These differences require purpose-built infrastructure. siirl-agentic was designed from
scratch to handle them natively.


Three Architectural Principles
-------------------------------

1. Asynchronous MPMD
~~~~~~~~~~~~~~~~~~~~~

**Principle:** Each major component runs as an independent Ray actor. No component blocks
another.

**Why it matters:** In agentic workloads, rollout latency is dominated by tool calls with
wildly varying response times (100ms for local tools, 30s for web APIs). A synchronous
pipeline would force GPUs to idle while waiting for the slowest tool call. The MPMD
architecture decouples training from rollout: the Trainer keeps updating the policy while
the RolloutManager collects new trajectories.

**Implementation:**

- ``MainRunner`` — Orchestrator (resource allocation, lifecycle management)
- ``TrainerGroup`` — Distributed training workers (Actor, Ref, Critic models)
- ``RolloutManager`` — SGLang inference + AgentFlow execution
- ``DataCoordinator`` — Async sample buffering between rollout and training

See :doc:`architecture_overview` for the full component diagram.

2. Agentic-First Data Path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Principle:** The sample lifecycle natively supports multi-turn trajectories with tool
interaction, per-turn loss masking, and variable-length sequences.

**Why it matters:** Retrofitting multi-turn support onto a single-turn framework leads to
fragile workarounds. siirl-agentic's ``Sample`` dataclass, ``NaiveFlow`` state machine,
and loss computation all assume variable-length, multi-turn data from the start.

**Key design decisions:**

- **Per-turn loss mask** — Each turn's model tokens and environment tokens are precisely
  distinguished. Only model-generated tokens contribute to the policy gradient.
- **Full log-probability tracking** — ``rollout_log_probs`` are maintained across the
  entire multi-turn trajectory for PPO/GRPO importance sampling ratio computation.
- **Parallel tool execution** — Multiple tool calls within a single turn can execute
  concurrently, reducing wall-clock time.

See :doc:`../highlights/native_agentic_trajectory_training` for the deep dive.

3. Configuration Over Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Principle:** Task-specific logic is injected via configuration, not hardcoded in
framework internals.

**Why it matters:** Agentic RL research iterates rapidly across diverse task types
(coding agents, search agents, math verification, browser interaction). Each task type
needs different preprocessing, generation strategies, reward functions, and tool
environments. Requiring framework code changes for each new task would be prohibitively
slow.

**Implementation:**

- **AgentFlow Protocol** — Three-stage ``preprocess → generate → reward`` protocol.
  Users specify a flow class and optional custom functions in YAML.
- **Dynamic function injection** — ``load_agentflow()`` uses ``MethodType`` to bind
  custom Python functions onto flow instances at runtime.
- **Built-in flow registry** — ``BUILTIN_FLOW = {"swe": ".swe:agentflow"}`` provides
  ready-made flows for common task types.

See :doc:`../highlights/pluggable_agentflow_protocol` for the full protocol specification.


Design Trade-offs
------------------

Separation vs Simplicity
~~~~~~~~~~~~~~~~~~~~~~~~~~

The MPMD architecture introduces complexity in debugging and deployment compared to
monolithic designs. We accept this trade-off because:

- GPU utilization gains from async overlap far outweigh operational complexity
- Ray provides robust actor-level error isolation and retry
- The ``TaskCoordinator`` centralizes lifecycle management for observability

Configuration Flexibility vs Safety
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dynamic function injection system (``MethodType`` binding) is powerful but requires
users to write correct Python functions. We mitigate risks via:

- Type-annotated protocol interfaces (``AgentFlow``, ``Model``, ``Sample``)
- Validation at load time with clear error messages
- Built-in flows as reference implementations

Async Data Freshness vs Throughput
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Off-policy data from older model versions may reduce training signal quality. We handle
this via:

- Configurable ``off_policy_step`` window (default: 1) to bound data staleness
- Version tagging on all samples for auditability
- The throughput gain from continuous GPU utilization outweighs mild off-policy degradation
  in practice


Further Reading
----------------

- :doc:`architecture_overview` — Full component diagram and data flow
- :doc:`../highlights/why_agentic_rl` — Detailed comparison with standard RL frameworks
- :doc:`../user_guide/configuration_system` — How to configure the framework
