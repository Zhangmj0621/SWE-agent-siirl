# Tutorial: Training a SWE Agent

> Train an agent to solve software engineering tasks with multi-turn tool interaction.

This tutorial covers the end-to-end process of training a software engineering (SWE) agent using siirl-agentic. The agent learns to fix real-world GitHub issues by interacting with a sandboxed code environment through bash commands, with RL rewards derived from automated test evaluation.


## Prerequisites

!!! tip "Key Takeaway"
    SWE agent training combines multi-turn rollout (agent issues commands, observes output, iterates) with containerized sandboxes (Docker or Kubernetes) for safe code execution. This is fundamentally different from single-turn math/code RL.

| Requirement       | Minimum                            | Recommended            |
| ----------------- | ---------------------------------- | ---------------------- |
| Python            | 3.10+                              | 3.12                   |
| GPUs              | 8x A100 80GB                       | 8x A100 80GB           |
| Container Runtime | Docker 24+ **or** Kubernetes 1.28+ | Kubernetes (for scale) |
| RAM               | 256 GB                             | 512 GB                 |
| Disk              | 200 GB free (for container images) | 500 GB free            |

Install siirl-agentic with all dependencies:

```bash
pip install -e "siirl-agentic/[all]"
pip install swebench datasets pyarrow ray[default] sglang jinja2 pydantic
```


## Step 1: Prepare the Environment

!!! tip "Key Takeaway"
    The SWE module decouples three concerns: **environment** (container lifecycle), **agent** (rollout logic), and **runtime** (dataset-specific setup/eval). This separation is defined in the README at `swe/README.md`.

### Architecture Overview

The SWE agent flow consists of three pluggable components configured via `config.yaml`:

| Component       | Responsibility                                                | Implementations                                                                         |
| --------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **Environment** | Container lifecycle (start, execute, cleanup)                 | `DockerEnvBuilder` (`environment/docker.py`), `K8sEnvBuilder` (`environment/k8s.py`)    |
| **Agent**       | Multi-turn rollout: query model, parse action, observe result | `MiniSWEAgentBuilder` (`agent/minisweagent.py`)                                         |
| **Runtime**     | Dataset-specific: bootstrap repo, generate diff, run eval     | `SWEBenchBuilder` (`runtime/swebench.py`), `SWEFactoryBuiler` (`runtime/swefactory.py`) |

### Docker Setup

For single-node development, Docker is the simplest option:

```bash
# Verify Docker is available
docker info

# Pre-pull SWE-bench environment images (these are large)
# Images are named by the SWE-bench test spec, e.g.:
docker pull swebench/sweb.env.x86_64:latest
```

The `DockerEnvBuilder` (`docker.py:226`) creates containers with `docker run -d`, executes commands via `docker exec`, and cleans up with `docker rm -f`. Default command timeout is 180 seconds per `ContainerEnv.execute` (`environment/base.py:155`).

### Kubernetes Setup

For production-scale training with many parallel sandbox instances:

```bash
# Ensure kubectl is configured
kubectl cluster-info

# Create a dedicated namespace
kubectl create namespace swebench
```

The `K8sEnvBuilder` (`k8s.py:319`) creates pods via `kubectl apply`, executes commands via `kubectl exec`, and deletes pods on cleanup. Configuration options from `K8sEnvConfig` (`k8s.py:310`):

| Field             | Default       | Description                           |
| ----------------- | ------------- | ------------------------------------- |
| `namespace`       | `"default"`   | Kubernetes namespace for sandbox pods |
| `kubeconfig`      | `None`        | Path to kubeconfig file               |
| `context`         | `None`        | Kubernetes context to use             |
| `pod_name_prefix` | `"agentflow"` | Prefix for generated pod names        |


## Step 2: Prepare the Dataset

!!! tip "Key Takeaway"
    SWE-bench instances must be converted to Parquet format. Each row must contain the full instance metadata needed by the runtime to bootstrap the container and run evaluation.

### SWE-bench Dataset Format

Download SWE-bench and convert to Parquet:

```python title="prepare_swebench.py"
import json
import pandas as pd
from datasets import load_dataset

# Load SWE-bench verified split
ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

records = []
for row in ds:
    records.append({
        "prompt": [
            {"role": "user", "content": row["problem_statement"]}
        ],
        "data_source": "swebench",
        "reward_model": {"ground_truth": ""},  # reward from eval, not string match
        "metadata": {
            "instance_id": row["instance_id"],
            "repo": row["repo"],
            "base_commit": row["base_commit"],
            "patch": row["patch"],
            "test_patch": row["test_patch"],
            "problem_statement": row["problem_statement"],
            "hints_text": row.get("hints_text", ""),
            "version": row.get("version", ""),
        },
    })

df = pd.DataFrame(records)
df.to_parquet("swebench_verified.parquet")
print(f"Prepared {len(df)} SWE-bench instances")
```

The runtime's `parse_sampledata` method (e.g., `SWEBenchBuiler.parse_sampledata` at `swebench.py:120`) extracts container arguments and the problem statement from each row's metadata. The `SWESampleData` dataclass (`runtime/base.py:9`) holds:

- `container_args`: A `ContainerStartArgs` specifying the Docker image and working directory (`/testbed`)
- `problem_statement`: The GitHub issue text used as the initial prompt
- `runtime_meta`: Dataset-specific metadata (SWE-bench test spec)


## Step 3: Configure AgentFlow

!!! tip "Key Takeaway"
    The SWE flow config is a YAML file with three sections: `environment`, `runtime`, and `agent`. The agent section controls step limits, prompt templates, and output parsing.

### Flow Configuration

Create a `swe_flow_config.yaml` based on the reference config at `swe/config.yaml`:

```yaml title="swe_flow_config.yaml"
environment:
  # Choose one:
  # name: "swe.environment.docker:DockerEnvBuilder"   # for Docker
  name: "swe.environment.k8s:K8sEnvBuilder"           # for Kubernetes
  namespace: "swebench"

runtime:
  name: "swe.runtime.swebench.SWEBenchBuilder"

agent:
  name: "swe.agent.minisweagent:MiniSWEAgentBuilder"
  step_limit: 30        # Max agent steps before forced termination
  cost_limit: 3.0       # Max cost in dollars (for API-based models)
  system_template: |
    You are a helpful assistant that can interact with a computer.

    Your response must contain exactly ONE bash code block with ONE command
    (or commands connected with && or ||).
    Include a THOUGHT section before your command where you explain your reasoning.

    To finish, issue: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
  instance_template: |
    Please solve this issue: {{task}}

    ## Recommended Workflow
    1. Analyze the codebase by finding and reading relevant files
    2. Create a script to reproduce the issue
    3. Edit the source code to resolve the issue
    4. Verify your fix works by running your script again
    5. Test edge cases to ensure your fix is robust

    ## Important Rules
    1. Every response must contain exactly one action
    2. The action must be enclosed in triple backticks
    3. Directory or environment variable changes are not persistent
    4. To finish: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
  action_observation_template: |
    <returncode>{{output.returncode}}</returncode>
    {% if output.output | length < 10000 -%}
    <output>
    {{ output.output -}}
    </output>
    {%- else -%}
    <warning>
    The output of your last command was too long.
    </warning>
    <output_head>
    {{ output.output[:5000] }}
    </output_head>
    <elided_chars>
    {{ output.output | length - 10000 }} characters elided
    </elided_chars>
    <output_tail>
    {{ output.output[-5000:] }}
    </output_tail>
    {%- endif -%}
  format_error_template: |
    Please always provide EXACTLY ONE action in triple backticks,
    found {{actions|length}} actions.
```

### Key Agent Parameters

The `MiniSWEAgent` (`minisweagent.py:43`) implements the agent loop:

| Parameter                     | Default                         | Description                                                                                     |
| ----------------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------- |
| `step_limit`                  | `0` (unlimited)                 | Max interaction steps; `LimitsExceeded` raised when hit                                         |
| `cost_limit`                  | `3.0`                           | Dollar cost limit (relevant for API models)                                                     |
| `action_regex`                | `` r"```bash\s*\n(.*?)\n```" `` | Regex to extract bash commands from model output                                                |
| `system_template`             | (see config)                    | Jinja2 template for the system prompt                                                           |
| `instance_template`           | (see config)                    | Jinja2 template for the initial user message; `{{task}}` is replaced with the problem statement |
| `action_observation_template` | (see config)                    | Jinja2 template to format command output; receives `output.returncode` and `output.output`      |
| `format_error_template`       | (see config)                    | Shown when model output does not contain exactly one code block                                 |

### Agent Loop Flow

The `MiniSWEAgent.run` method (`minisweagent.py:56`) implements this loop:

1. Add system message and instance message to conversation
2. **Query** the model (`query`, line 79): tokenize conversation, call inference, append assistant response
3. **Parse action** (`parse_action`, line 106): extract bash command via regex
4. **Execute** in container (`execute_action`, line 113): run via `env.execute`, check for submission
5. **Observe** (`get_observation`, line 91): format output via template, append as user message
6. Repeat until `Submitted` (agent outputs `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`) or `LimitsExceeded`

The agent sets the rollout result to `SWERolloutResult.SUBMIT`, `EXCEED`, or `FAILURE` depending on how the loop terminates (`base.py:12`).


## Step 4: Configure Training

!!! tip "Key Takeaway"
    SWE agent training uses the same `siirl.async_train` entry point as single-turn GRPO, but with multiturn parameters in `rollout.multiturn.*` and a flow function pointing to the SWE config.

### Multi-turn Rollout Parameters

These are defined in `MultiturnArguments` (`model_args.py:152`):

| Parameter Path                                 | Default         | SWE Value                      | Description                                 |
| ---------------------------------------------- | --------------- | ------------------------------ | ------------------------------------------- |
| `rollout.multiturn.env_type`                   | `None`          | `"tool_env"`                   | Environment type for multi-turn interaction |
| `rollout.multiturn.max_env_turns`              | `1`             | `30`                           | Max environment interaction turns           |
| `rollout.multiturn.max_assistant_turns`        | `1`             | `30`                           | Max model generation turns                  |
| `rollout.multiturn.max_parallel_calls`         | `1`             | `32`                           | Concurrent sandbox instances                |
| `rollout.multiturn.max_env_response_length`    | `256`           | `2048`                         | Max tokens for environment observation      |
| `rollout.multiturn.env_response_truncate_side` | `"middle"`      | `"middle"`                     | Truncation strategy for long outputs        |
| `rollout.flow_function`                        | `"naive"`       | `"swe"`                        | Flow function to use                        |
| `rollout.flow_config`                          | `"config.yaml"` | Path to `swe_flow_config.yaml` | Flow configuration file                     |

### Full Launch Command

```bash title="run_swe_agent.sh"
#!/usr/bin/env bash
set -euo pipefail

# --- Paths ---
export MODEL_PATH=/path/to/Qwen3-8B
export TRAIN_DATA_PATH=./swebench_verified.parquet

# --- Start Ray ---
ray stop --force 2>/dev/null || true
ray start --head --num-gpus 8 \
    --object-store-memory 100000000000 \
    --memory 100000000000

# --- Launch Training ---
python3 -m siirl.async_train \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    data.train_files=$TRAIN_DATA_PATH \
    data.train_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    actor_ref.model.path=$MODEL_PATH \
    actor_ref.model.trust_remote_code=True \
    actor_ref.actor.optim.lr=5e-7 \
    actor_ref.actor.ppo_mini_batch_size=32 \
    actor_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_ref.actor.use_dynamic_batch=True \
    actor_ref.actor.max_tokens_per_gpu=16384 \
    actor_ref.actor.use_kl_loss=True \
    actor_ref.actor.kl_loss_coef=0.01 \
    actor_ref.actor.kl_loss_type=low_var_kl \
    actor_ref.actor.megatron.param_offload=True \
    actor_ref.actor.megatron.use_mbridge=True \
    actor_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_ref.ref.megatron.param_offload=True \
    rollout.name=sglang \
    rollout.tensor_model_parallel_size=2 \
    rollout.gpu_memory_utilization=0.6 \
    rollout.n=4 \
    rollout.trust_remote_code=True \
    rollout.max_model_len=16384 \
    rollout.flow_function=swe \
    rollout.flow_config=./swe_flow_config.yaml \
    rollout.multiturn.env_type='tool_env' \
    rollout.multiturn.max_env_turns=30 \
    rollout.multiturn.max_assistant_turns=30 \
    rollout.multiturn.max_parallel_calls=32 \
    rollout.multiturn.max_env_response_length=2048 \
    rollout.multiturn.env_response_truncate_side=middle \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.colocate=True \
    trainer.total_epochs=10 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.default_local_dir=ckpts/swe_agent \
    trainer.project_name=siirl_swe_agent \
    trainer.experiment_name=swe_grpo_qwen3_8b \
    trainer.logger="['console','tensorboard','wandb']" \
    trainer.resume_mode=auto \
    trainer.tensor_model_parallel_size=2

ray stop --force
```

### Key Differences from Single-Turn Training

| Aspect                | Single-Turn (DeepScaleR) | Multi-Turn (SWE)               |
| --------------------- | ------------------- | ------------------------------ |
| Batch size            | 512                 | 64 (longer sequences)          |
| `rollout.n`           | 8                   | 4 (each rollout is expensive)  |
| `max_response_length` | 2048                | 8192+ (multi-turn accumulates) |
| Learning rate         | 1e-6                | 5e-7 (more conservative)       |
| Reward source         | String match        | Test suite pass/fail           |
| Rollout time          | Seconds             | Minutes (per instance)         |


## Step 5: Launch and Monitor

!!! tip "Key Takeaway"
    SWE agent rollouts are much slower than single-turn generation. Monitor sandbox health alongside training metrics. Each rollout instance runs a full agent loop (up to `step_limit` turns) inside a container.

### SWE-Specific Metrics

| Metric                 | Description                              | Healthy Signal                                  |
| ---------------------- | ---------------------------------------- | ----------------------------------------------- |
| `reward/mean`          | Fraction of resolved instances           | Any upward trend is good; SWE-bench is hard     |
| `rollout/avg_steps`    | Average agent steps per instance         | Should decrease as agent becomes more efficient |
| `rollout/submit_rate`  | Fraction of rollouts reaching submission | Should increase (vs. timeout)                   |
| `rollout/timeout_rate` | Fraction hitting step_limit              | Should decrease over training                   |
| `kl_divergence`        | KL from reference policy                 | Keep below 10                                   |

### Inspecting Generated Patches

After the runtime's `diff` phase (`swebench.py:43`), each sample's patch is stored in `SWERolloutMeta.patch`. The evaluation phase (`swebench.py:47`) applies the patch to a clean checkout and runs the test suite via `get_eval_report`. A fully resolved instance receives reward `1.0`; otherwise `0.0` (`swebench.py:88-91`).

To inspect patches from a completed rollout, check the logged outputs or the checkpoint data. The `SWERolloutResult` enum (`base.py:12`) records whether each instance was `SUBMIT`, `EXCEED` (step limit), or `FAILURE`.


## Expected Results

!!! warning
    SWE-bench results are highly sensitive to agent design, prompt engineering, step limits, and base model capability. The ranges below are approximate and based on published literature.

| Model              | GPUs     | Step Limit | Approx. SWE-bench Verified Resolve Rate |
| ------------------ | -------- | ---------- | --------------------------------------- |
| Qwen3-8B (GRPO)    | 8x A100  | 30         | 5--15%                                  |
| LLaMA-3-70B (GRPO) | 32x A100 | 30         | 15--25%                                 |

These are rough ranges. State-of-the-art systems with specialized scaffolding and larger models achieve 30-50%+ on SWE-bench Verified.


## Common Mistakes

| Symptom                                               | Cause                                                          | Fix                                                                                                                    |
| ----------------------------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `RuntimeError: Container environment is closed`       | Container cleaned up before agent finished                     | Increase `container_timeout` in `ContainerStartArgs` (default `"2h"`)                                                  |
| All rollouts timeout at step_limit                    | `step_limit` too low, or model not learning submission command | Increase `agent.step_limit` in config; verify `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` is in system prompt              |
| `docker exec failed (exit code 127)`                  | Command not found inside container                             | Ensure SWE-bench images have required tools installed; use `swebench.harness` to build images                          |
| `TimeoutError: Command timed out after 180s`          | Single command running too long (e.g., full test suite)        | The default timeout per `ContainerEnv.execute` is 180s; for heavy test suites, customize in runtime                    |
| `FormatError: EXACTLY ONE action in triple backticks` | Model output multiple or zero code blocks                      | Agent auto-retries via `NonTerminatingException` (`minisweagent.py:140`); if persistent, improve system prompt         |
| `kubectl cp failed` / pod creation errors             | K8s namespace missing or RBAC issues                           | Verify `kubectl create namespace swebench`; check pod creation permissions                                             |
| OOM during multi-turn rollout                         | Conversation too long accumulating over turns                  | Reduce `rollout.multiturn.max_env_response_length`, set `env_response_truncate_side=middle`                            |
| Reward is always 0.0                                  | Patches not applying or tests failing                          | Check `SWERolloutResult` -- if most are `EXCEED`, increase step limit; if `SUBMIT` but reward=0, inspect patch quality |
