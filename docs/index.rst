siirl-agentic
=============

**siirl-agentic** is an asynchronous, agentic reinforcement learning training framework purpose-built
for multi-turn agent trajectories with tool interaction. It natively supports PPO/GRPO training on
SWE-style tasks where agents call tools, receive environment feedback, and accumulate rewards across
long interaction horizons.

.. raw:: html

   <div style="display:flex;gap:12px;margin:1.5rem 0">
     <a href="en/index.html" style="flex:1;padding:1rem;background:#0969da;color:#fff;border-radius:8px;text-align:center;text-decoration:none;font-weight:600">
       English Docs →
     </a>
     <a href="zh/index.html" style="flex:1;padding:1rem;background:#0969da;color:#fff;border-radius:8px;text-align:center;text-decoration:none;font-weight:600">
       中文文档 →
     </a>
   </div>

----

Why siirl-agentic?
------------------

Standard RL frameworks assume single-turn generation. Agentic tasks are fundamentally different:
variable-length multi-turn loops, highly variable latency from tool calls, and the need for
per-turn loss masking. siirl-agentic is built for this from day one.

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Dimension
     - Single-Turn RL
     - Agentic RL (siirl-agentic)
   * - Interaction
     - One prompt → one response
     - Multi-turn: generate → tool → env → generate …
   * - Trajectory length
     - Fixed, predictable
     - Variable, 10–50+ turns
   * - Latency profile
     - Uniform (GPU-bound)
     - Heterogeneous (tool calls: 100 ms – 30 s)
   * - Reward signal
     - End-of-sequence scalar
     - Per-turn environment + final outcome
   * - Tool management
     - N/A
     - Distributed AIO scheduler, auto-scaling

----

Key Features
------------

**Native Agentic Trajectory Training**
  Token-level log-probabilities and per-turn loss masks tracked across the entire multi-turn
  trajectory. No wrapper hacks, no offline collection. The NaiveFlow state machine
  (``PENDING → GENERATING → PROCESSING_ENV → TERMINATED``) handles tool calls as first-class
  training primitives.

**Pluggable AgentFlow Protocol**
  Three-stage contract — ``preprocess → generate → reward`` — loaded at runtime from YAML.
  Switch from SWE-bench to search-augmented QA by editing one config file.

**MPMD Asynchronous Execution Engine**
  Trainer and RolloutManager run as independent Ray actor loops. Generation and optimization
  proceed at independent clock rates, eliminating pipeline stalls under variable tool-call
  latency.

**AIO: Elastic Agentic Tool Infrastructure**
  Three-tier distributed scheduler (Proxy → ResourcePool → WorkerManager) with Holt-Winters
  auto-scaling and async batched request submission. Handles 1000+ concurrent tool calls
  without cascading timeouts.

----

Architecture
------------

::

    ┌──────────────────────────────────────────────────────────────────┐
    │  MainRunner (Ray Actor)                                          │
    │                                                                  │
    │  ┌────────────────┐   ┌─────────────────┐   ┌────────────────┐  │
    │  │ DataCoordinator│   │ RolloutManager  │   │ TrainerGroup   │  │
    │  │                │──>│                 │   │                │  │
    │  │  Dataloader    │   │  SGLang Engine  │   │  Actor Model   │  │
    │  │  DataBuffer    │   │  AgentFlow      │   │  Ref Model     │  │
    │  │                │<──│  ToolEnv / AIO  │   │  Critic (PPO)  │  │
    │  └────────────────┘   └─────────────────┘   └────────────────┘  │
    │                              │  param sync  ▲                    │
    │                              └──────────────┘                    │
    └──────────────────────────────────────────────────────────────────┘

----

Quickstart
----------

**Prerequisites:** 8 GPUs, a model (e.g., Qwen3-8B), training data in Parquet format.

.. code:: bash

   git clone https://github.com/sii-research/siirl-agentic.git
   cd siirl-agentic && pip install -e ".[gpu]"

   export MODEL_PATH=/path/to/Qwen3-8B
   export TRAIN_DATA_PATH=/path/to/train.parquet
   export TEST_DATA_PATH=/path/to/test.parquet

   # GRPO training: 4 GPUs training + 4 GPUs rollout
   bash examples/grpo_train/run_qwen3_8b_separated.sh

   # Agentic training with tool interaction
   bash examples/AIO/run_qwen3_8b.sh

----

.. toctree::
   :maxdepth: 1
   :caption: Language / 语言
   :hidden:

   en/index
   zh/index
