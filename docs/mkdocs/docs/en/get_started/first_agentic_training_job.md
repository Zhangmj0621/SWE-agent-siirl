# First Agentic Training Job

*Launch a GRPO training job where the agent uses search tools during multi-turn rollout.*

## Prerequisites

-   siirl-agentic installed and verified (see [Installation](installation.md))
-   Completed [Quickstart](quickstart.md) (basic GRPO training works)
-   **AIO tool infrastructure** cloned and installed. AIO is a separate repository in the monorepo:

    ``` bash
    # From the monorepo root (parent of siirl-agentic/)
    cd AIO
    pip install -e .
    ```

    If you do not have the AIO repository, contact your team for access or see [AIO Tool Infrastructure](../guides/aio_tool_infrastructure.md).

## What Makes This "Agentic"

In a standard RL training job, the model generates a single response and receives a reward. In an **agentic** training job:

1.  The model generates text that may include **tool calls** (e.g., search queries, code execution)
2.  Tool calls are **executed in real environments** via the AIO infrastructure
3.  Tool responses are **appended to the conversation** as environment observations
4.  The model continues generating based on tool responses
5.  This multi-turn loop repeats until termination (max turns or model decision)
6.  **Only model-generated tokens** contribute to the policy gradient (environment tokens are masked)

!!! tip "Tool Environment Setup Order"
    Always start AIO infrastructure **before** launching the training script. The rollout engine attempts to connect to the AIO Proxy at startup — if the Proxy isn't running, rollout initialization will fail with `AIOSearchTool: Error getting server`.

## Step 1: Configure Multi-Turn Rollout

Add multi-turn configuration to your training config:

``` yaml
rollout:
  flow_function: naive
  multiturn:
    env_type: tool_env
    max_env_turns: 5            # Max tool interaction rounds (default: 1)
    max_assistant_turns: 10     # Max model generation turns (default: 1)
    max_parallel_calls: 4       # Concurrent tool calls per sample (default: 1)
    max_env_response_length: 256  # Max env response characters for truncation
    env_response_truncate_side: middle
    env_path: /path/to/tool_env_config.yaml
    env_kwargs:
      tool_format: hermes
```

!!! warning
    `max_env_response_length` truncates based on **character count**, not token count. A value of 256 means 256 characters, which may be fewer or more than 256 tokens depending on the tokenizer.

## Step 2: Configure Tool Environment

Create a tool environment config (`tool_env_config.yaml`):

``` yaml
tools:
  - name: search
    type: aio_search
    config:
      topk: 3
```

## Step 3: Start AIO Infrastructure

Before training, launch the AIO Proxy and WorkerManagers:

``` bash
# Start AIO Proxy
python -m aio.Scheduler.proxy --config aio_config.yaml

# Start WorkerManager on each tool node
python -m aio.Scheduler.Resources.worker_manager --proxy-url http://proxy-host:8080
```

## Step 4: Launch Agentic Training

``` bash
# Using the AIO example script
cd siirl-agentic
bash examples/AIO/run_qwen3_8b.sh
```

**Expected output:**

```
INFO  | Ray is initialized. Time cost: 150.23 ms
INFO  | MainRunner started. Beginning workflow setup...
INFO  | Initializing DataCoordinator...
SUCCESS | RolloutManager initialized. Router at: http://10.0.0.1:30000
INFO  | Starting async training loop...
INFO  | [NaiveFlow] PENDING -> GENERATING (sample 0)
INFO  | [NaiveFlow] GENERATING -> PROCESSING_ENV (sample 0, 2 tool calls)
INFO  | [AIOSearchTool] search query dispatched to worker
INFO  | [NaiveFlow] PROCESSING_ENV -> GENERATING (sample 0, env_turn 1)
INFO  | [NaiveFlow] GENERATING -> TERMINATED (sample 0, reward=1.0)
```

## Step 5: Monitor the Training

Key metrics to watch:

-   **reward/mean** — Average reward per step (should increase)
-   **rollout/generation_duration** — Time spent in LLM generation
-   **rollout/reward_duration** — Time spent computing rewards
-   **rollout/env_turns_mean** — Average tool interaction rounds per sample

## Understanding the Trajectory

A typical agentic trajectory looks like:

```
[User]      Solve: what is the population of Tokyo?
[Assistant] I'll search for this information.
            <tool_call>search(query_list=["Tokyo population"])</tool_call>
[Tool]      Tokyo has a population of approximately 13.96 million...
[Assistant] Based on the search results, the population of Tokyo is
            approximately 13.96 million people.
```

In the `response_mask` (which applies to the **response portion only**, not the prompt):

- Assistant turn 1 tokens → `response_mask = 1` (trained on, included in policy gradient)
- Tool response tokens → `response_mask = 0` (masked from loss)
- Assistant turn 2 tokens → `response_mask = 1` (trained on, included in policy gradient)

The prompt tokens are tracked separately in the `prompts` field and are never included in the loss computation.

## Minimal Runnable Example

``` python
# Verify multi-turn config is parsed correctly
from siirl.params import parse_config, SiiRLArguments

# parse_config() uses argparse + OmegaConf.from_cli() to parse CLI arguments
config = parse_config()
print(f"env_type: {config.rollout.multiturn.env_type}")
print(f"max_env_turns: {config.rollout.multiturn.max_env_turns}")
print(f"max_assistant_turns: {config.rollout.multiturn.max_assistant_turns}")
```

## Common setup mistakes

| Mistake                             | What happens                                    | Correct config                                     |
| ----------------------------------- | ----------------------------------------------- | -------------------------------------------------- |
| `max_env_turns=1`                   | Agent can only call one tool before terminating | Use `max_env_turns=5` for multi-step tasks         |
| Forgetting `env_type=tool_env`      | Single-turn rollout, tools are never called     | Set `rollout.multiturn.env_type=tool_env`          |
| `rollout.n=32` on first run         | OOM or very slow rollout                        | Start with `rollout.n=4`, scale up after verifying |
| `max_response_length=512` (default) | Agentic trajectories truncated after one turn   | Set `data.max_response_length=4096` or higher      |
| Starting training before AIO Proxy  | Rollout init fails immediately                  | Start AIO Proxy first, then launch training        |

## Common Issues

| Symptom                               | Cause                                     | Fix                                   |
| ------------------------------------- | ----------------------------------------- | ------------------------------------- |
| `AIOSearchTool: Error getting server` | AIO Proxy not running                     | Start `python -m aio.Scheduler.proxy` |
| `TERMINATED after 1 turn`             | `max_assistant_turns=1`                   | Increase `max_assistant_turns`        |
| Tool response truncated               | `max_env_response_length` too small       | Increase the value                    |
| All rewards = 0                       | Reward function doesn't handle multi-turn | Check custom reward logic             |

## What success looks like for multi-turn agentic training

After completing this guide, you should see:

- Log lines showing state machine transitions: `PENDING -> GENERATING -> PROCESSING_ENV -> GENERATING -> TERMINATED`
- `rollout/env_turns_mean` metric is **greater than 1** — this confirms the agent is actually using tools and generating multiple turns, not just terminating after one pass
- `reward/mean` starts non-zero and trends upward over training steps
- Tool call success rate visible in logs via AIO Proxy: `[AIOSearchTool] search query dispatched to worker`
- AIO Proxy logs show incoming `tool_call` requests being dispatched to workers
- No `TERMINATED after 1 turn` warnings in steady state

**Sample log output from a healthy agentic run:**

```
INFO  | [Step 1] rollout done. reward/mean=0.08, env_turns/mean=2.3, tool_calls=184
INFO  | [Step 1] training done. loss=1.51
INFO  | [Step 10] rollout done. reward/mean=0.24, env_turns/mean=2.7, tool_calls=216
INFO  | [Step 10] training done. loss=1.27
```

`env_turns/mean=2.3` means the agent is averaging 2.3 tool interactions per sample. If this value is stuck at `1.0`, the agent terminates after one turn — check `max_env_turns` and `max_assistant_turns` in your config.

If something went wrong, see [Troubleshooting](../reference/troubleshooting.md).

## Next steps

- [Agentic Multi-Turn](../guides/agentic_multiturn.md) — Deep dive into multi-turn configuration, state machine behavior, and loss masking
- [AIO Tool Infrastructure](../guides/aio_tool_infrastructure.md) — Configure and scale the distributed tool scheduling system for production workloads
- [How It Works](../concepts/how_it_works.md) — Build a mental model of the async training loop you just ran
