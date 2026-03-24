# Building a Custom Agent

*Implement your own agentic task without modifying framework code.*

## AgentFlow Three-Method Contract

!!! tip "Key Insight"
    Every AgentFlow implements exactly three methods: `preprocess()`, `async generate()`, and `async reward()`. The generate and reward methods **must** be `async def`. This contract is the only interface between your agent logic and the RL training loop.

The `AgentFlow` protocol (`base.py:12`) defines the contract:

```python
class AgentFlow(Protocol):
    def preprocess(self, sample: dict) -> Sample:
        """Convert a raw dataset row (dict) into a Sample object."""
        ...

    async def generate(self, sample: Sample):
        """Run scaffold rollout / solution generation."""
        ...

    async def reward(self, sample: Sample):
        """Run evaluation / verification and set sample.reward."""
        ...
```

Each method has a distinct responsibility:

| Method       | Sync/Async | Input                | Output                    | Responsibility                                             |
| ------------ | ---------- | -------------------- | ------------------------- | ---------------------------------------------------------- |
| `preprocess` | Sync       | `dict` (dataset row) | `Sample`                  | Parse raw data, build prompt, initialize metadata          |
| `generate`   | **Async**  | `Sample`             | Mutates `Sample` in-place | Query model, record tokens/log_probs/loss_mask, set status |
| `reward`     | **Async**  | `Sample`             | Mutates `Sample` in-place | Evaluate correctness, set `sample.reward` float            |

The `Sample` dataclass (`base.py:118`) carries all state through the pipeline:

| Field               | Type                  | Set By                | Purpose                                                |
| ------------------- | --------------------- | --------------------- | ------------------------------------------------------ |
| `m`                 | `AgentMeta` (generic) | `preprocess`          | Your custom metadata (task info, parsed answers, etc.) |
| `model`             | `Model`               | `preprocess`          | LLM handle for querying during generate                |
| `status`            | `Status` enum         | `generate` / `reward` | Lifecycle: `PENDING` -> `ROLLEDOUT` -> `COMPLETED`     |
| `tokens`            | `list[int]`           | `generate`            | Full token sequence (prompt + response + observations) |
| `loss_mask`         | `list[int]`           | `generate`            | 1 for model-generated tokens, 0 for input/observation  |
| `rollout_log_probs` | `list[float]`         | `generate`            | Per-token log probabilities from rollout               |
| `conversations`     | `list[dict]`          | `generate`            | OpenAI-format message history                          |
| `reward`            | `float`               | `reward`              | Final scalar reward for RL training                    |

## Step-by-Step: Math Reasoning Agent

This walkthrough builds a complete agent based on the reference implementation in `agentflow/swe/example/__init__.py`.

### 1. Define Your Metadata

```python
from dataclasses import dataclass, field
from siirl.execution.rollout.agentflow.base import (
    AgentFlow, Model, ModelResponse, Sample
)

@dataclass
class MathMeta:
    """Custom metadata for our math agent."""
    question: str = ""
    ground_truth: float = 0.0
    parsed_answer: str = ""

MathSample = Sample[MathMeta]
```

### 2. Implement the AgentFlow Class

```python
import re

class MathReasoningFlow:
    """AgentFlow for mathematical reasoning tasks."""

    def __init__(self, config: dict, model: Model):
        self.model = model
        self.max_tokens = config.get("max_tokens", 1024)

    def preprocess(self, sample: dict) -> MathSample:
        meta = MathMeta(
            question=sample["question"],
            ground_truth=float(sample["answer"]),
        )
        return MathSample(meta, self.model)

    async def generate(self, sample: MathSample):
        # Build the user message
        message = {
            "role": "user",
            "content": (
                f"Solve this math problem step by step.\n\n"
                f"{sample.m.question}\n\n"
                f"Put your final answer in <answer></answer> tags."
            ),
        }
        input_tokens = self.model.tokenizer.apply_chat_template(
            [message], add_generation_prompt=True
        )
        sample.conversations.append(message)
        sample.append_input_tokens(input_tokens)

        try:
            response: ModelResponse = await self.model.query(
                input_tokens=sample.tokens,
                messages=sample.conversations,
                max_tokens=self.max_tokens,
                timeout=60,
            )
            sample.conversations.append(
                {"role": "assistant", "content": response.output}
            )
            sample.append_output(response)

            # Parse the answer
            match = re.search(
                r"<answer>\s*(.*?)\s*</answer>", response.output, re.DOTALL
            )
            if match:
                sample.m.parsed_answer = match.group(1).strip()
                sample.status = MathSample.Status.ROLLEDOUT
            else:
                sample.status = MathSample.Status.ABORTED
                sample.errors.append("No <answer> tags found")

        except Exception as e:
            sample.status = MathSample.Status.ABORTED
            sample.errors.append(f"Generate failed: {e}")

    async def reward(self, sample: MathSample):
        try:
            parsed = float(sample.m.parsed_answer)
            correct = abs(parsed - sample.m.ground_truth) < 1e-6
            sample.reward = 1.0 if correct else 0.0
            sample.status = MathSample.Status.COMPLETED
        except (ValueError, TypeError):
            sample.reward = 0.0
            sample.status = MathSample.Status.FAILED
            sample.errors.append("Cannot parse answer as number")
```

### 3. Sample Lifecycle

The `Sample.Status` enum tracks lifecycle progression:

```
PENDING ──preprocess()──> (sample created)
         ──generate()───> ROLLEDOUT  (success)
                        > TRUNCATED  (context limit hit)
                        > ABORTED    (generation failure)
         ──reward()────> COMPLETED   (success)
                        > FAILED     (reward computation failure)
```

The framework filters out non-`COMPLETED` samples before sending data to the training loop. Only `COMPLETED` samples with valid `reward`, `tokens`, and `loss_mask` contribute to gradient updates.

## Step-by-Step: Tool-Calling Agent (Pedagogical)

!!! note
    This is a pedagogical example. Only the math (example.py) and SWE agents exist in the codebase. This shows how you would build a custom tool-calling agent.

A tool-calling agent interacts with external tools across multiple turns. The key pattern: query the model, parse tool calls from the response, execute tools, feed results back as observations, and loop.

```python
import json
import re

@dataclass
class ToolMeta:
    task: str = ""
    tool_results: list[dict] = field(default_factory=list)
    max_turns: int = 5

ToolSample = Sample[ToolMeta]


class SearchAgentFlow:
    """Multi-turn agent that searches for information and answers questions."""

    def __init__(self, config: dict, model: Model):
        self.model = model
        self.max_turns = config.get("max_turns", 5)

    def preprocess(self, sample: dict) -> ToolSample:
        meta = ToolMeta(
            task=sample["question"],
            max_turns=self.max_turns,
        )
        return ToolSample(meta, self.model)

    async def generate(self, sample: ToolSample):
        # System message with tool description
        system_msg = {
            "role": "system",
            "content": (
                "You can use the search tool by writing: "
                '<tool_call>{"name": "search", "args": {"query": "..."}}</tool_call>\n'
                "Put your final answer in <answer></answer> tags."
            ),
        }
        user_msg = {"role": "user", "content": sample.m.task}

        messages = [system_msg, user_msg]
        for msg in messages:
            tokens = self.model.tokenizer.apply_chat_template(
                [msg], add_generation_prompt=False
            )
            sample.conversations.append(msg)
            sample.append_input_tokens(tokens)

        for turn in range(sample.m.max_turns):
            # Query the model
            input_tokens = self.model.tokenizer.apply_chat_template(
                sample.conversations, add_generation_prompt=True
            )
            response = await self.model.query(
                input_tokens=input_tokens,
                messages=sample.conversations,
                max_tokens=512,
            )
            sample.add_message("assistant", response)

            # Check for tool calls
            tool_match = re.search(
                r"<tool_call>(.*?)</tool_call>", response.output, re.DOTALL
            )
            if tool_match:
                # Execute tool and add observation
                tool_call = json.loads(tool_match.group(1))
                result = await self._execute_tool(tool_call)
                sample.m.tool_results.append(result)
                observation = f"Search result: {result['output']}"
                sample.add_message("tool", observation)
                continue

            # Check for final answer
            if "<answer>" in response.output:
                sample.status = ToolSample.Status.ROLLEDOUT
                return

        # Exhausted turns
        sample.status = ToolSample.Status.TRUNCATED

    async def _execute_tool(self, tool_call: dict) -> dict:
        """Execute a tool call. Replace with your actual tool implementation."""
        # In production, this would call an actual search API
        return {"output": f"Mock result for: {tool_call['args']['query']}"}

    async def reward(self, sample: ToolSample):
        # Extract and evaluate the answer
        last_assistant = [
            m for m in sample.conversations if m["role"] == "assistant"
        ][-1]["content"]
        match = re.search(r"<answer>(.*?)</answer>", last_assistant, re.DOTALL)
        if match:
            sample.reward = 1.0  # Replace with actual evaluation
            sample.status = ToolSample.Status.COMPLETED
        else:
            sample.reward = 0.0
            sample.status = ToolSample.Status.FAILED
```

Key patterns in the tool-calling agent:

- **`sample.add_message("assistant", response)`** automatically calls `append_output()` for `ModelResponse` objects, setting `loss_mask=1` for generated tokens.
- **`sample.add_message("tool", observation)`** calls `append_input_tokens()` internally, setting `loss_mask=0` for observation tokens. Only model-generated tokens contribute to the RL loss.
- The loop respects a maximum turn count and sets `TRUNCATED` status if exceeded.

## Registering Your Agent

!!! tip "Key Insight"
    Use `load_agentflow()` for the AgentFlow protocol (token-level control). Use `flow_function` and `flow_config` in `RolloutArguments` for NaiveFlow-based agents. Both support dynamic method injection without modifying framework code.

### AgentFlow Registration (via `load_agentflow`)

The `load_agentflow()` function (`__init__.py:11`) loads your agent from a config dict:

```python
config = {
    "name": "my_agents.math:MathReasoningFlow",  # module:class format
    "reward_fn": "my_agents.math:custom_eval_reward",  # optional override
    "python_path": ["/path/to/your/agents"],  # optional import paths
}
agent = load_agentflow(config, model)
```

Configuration keys:

| Key             | Required | Format                    | Purpose                      |
| --------------- | -------- | ------------------------- | ---------------------------- |
| `name`          | Yes      | `"module.path:ClassName"` | Agent class to instantiate   |
| `preprocess_fn` | No       | `"module.path:function"`  | Override `preprocess` method |
| `generate_fn`   | No       | `"module.path:function"`  | Override `generate` method   |
| `reward_fn`     | No       | `"module.path:function"`  | Override `reward` method     |
| `python_path`   | No       | `["/path/to/dir", ...]`   | Additional import paths      |

### Dynamic Method Injection

The `load_agentflow()` function supports injecting individual methods at load time. This is useful for mixing and matching preprocessing, generation, and reward logic:

```python
# The framework calls:
agent = agent_class(config, model)  # Your __init__

# Then overrides methods if specified in config:
if reward_fn is not None:
    agent.reward = MethodType(reward_fn, agent)
```

The injected function receives `self` (the agent instance) as the first argument:

```python
# my_rewards.py
async def custom_eval_reward(self, sample: MathSample):
    """Injected reward function. `self` is the AgentFlow instance."""
    # You can access self.model, self.config, etc.
    sample.reward = 1.0 if sample.m.parsed_answer == "42" else 0.0
    sample.status = MathSample.Status.COMPLETED
```

### NaiveFlow Registration (via Hydra config)

For simpler agents that use the built-in `NaiveFlow` pipeline, configure via Hydra YAML:

```yaml
rollout:
  flow_function: naive          # Uses NaiveFlow
  flow_config: config.yaml      # YAML config path for the flow
  multiturn:
    env_type: tool_env           # Enable tool environment
    max_env_turns: 5             # Max tool interaction rounds
    max_assistant_turns: 10      # Max model generation rounds
    max_parallel_calls: 1        # Parallel tool executions per sample
```

### SWE Agent Registration

The SWE agent uses a builder pattern with three pluggable components:

```yaml
name: swe
agent:
  name: minisweagent            # Built-in SWE agent
environment:
  name: k8s                     # k8s | docker | kr8s
  namespace: swe
runtime:
  name: swebench_sii            # swebench | swebench_sii | swefactory
```

## Testing Your Agent Locally

!!! tip "Key Insight"
    Test each method independently before running a full training loop. Use `DummyModel` from `base.py` for unit tests that don't require a real LLM.

### Unit Testing with DummyModel

```python
import asyncio
from siirl.execution.rollout.agentflow.base import DummyModel
from siirl.execution.rollout.agentflow import load_agentflow

# DummyModel returns pre-configured responses
model = DummyModel(["<answer> 42 </answer>"])

agent = load_agentflow(
    {
        "name": "my_agents.math:MathReasoningFlow",
        "python_path": ["/path/to/your/agents"],
    },
    model,
)

# Test preprocess
sample = agent.preprocess({"question": "What is 6*7?", "answer": "42"})
assert sample.status == sample.Status.PENDING

# Test generate
asyncio.run(agent.generate(sample))
assert sample.status == sample.Status.ROLLEDOUT
assert len(sample.tokens) > 0
assert sum(sample.loss_mask) > 0  # Some tokens are model-generated

# Test reward
asyncio.run(agent.reward(sample))
assert sample.status == sample.Status.COMPLETED
assert sample.reward == 1.0
```

### Testing with a Real Model

For integration tests with SGLang or another inference engine, run the agent script directly:

```bash
python -m siirl.execution.rollout.agentflow.swe.example
```

This pattern (using `__main__.py`) is the recommended way to create a standalone test harness for your agent.

## Debugging Agent Behavior

### Inspecting Conversations

```python
# After generate(), examine the full conversation
for i, msg in enumerate(sample.conversations):
    print(f"Turn {i} [{msg['role']}]: {msg['content'][:200]}")
```

### Verifying Token Alignment

```python
# Ensure tokens, loss_mask, and rollout_log_probs are aligned
assert len(sample.tokens) == len(sample.loss_mask)
assert len(sample.tokens) == len(sample.rollout_log_probs)

# Count model-generated vs. input tokens
model_tokens = sum(sample.loss_mask)
input_tokens = len(sample.loss_mask) - model_tokens
print(f"Total: {len(sample.tokens)}, Model: {model_tokens}, Input: {input_tokens}")
```

### Checking Status Transitions

```python
# Log status after each phase for debugging
print(f"After preprocess: {sample.status}")  # PENDING
await agent.generate(sample)
print(f"After generate: {sample.status}")    # ROLLEDOUT / ABORTED / TRUNCATED
await agent.reward(sample)
print(f"After reward: {sample.status}")      # COMPLETED / FAILED
print(f"Errors: {sample.errors}")            # Any error messages accumulated
```

### Common Mistakes

| Symptom                                               | Cause                                              | Fix                                                                                 |
| ----------------------------------------------------- | -------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `TypeError: object NoneType can't be used in 'await'` | `generate` or `reward` not declared as `async def` | Add `async` keyword to method definition                                            |
| `loss_mask` all zeros                                 | Forgot to call `sample.append_output(response)`    | Use `sample.add_message("assistant", response)` or `sample.append_output()`         |
| `reward` is `None` after reward phase                 | Reward function didn't set `sample.reward`         | Ensure every code path sets `sample.reward` to a float                              |
| `ImportError` on agent load                           | Wrong module path format                           | Use `"module.path:ClassName"` with colon separator, not dot                         |
| Injected reward function missing `self`               | Function signature lacks first `self` parameter    | Injected functions receive the agent as `self`: `async def my_reward(self, sample)` |
| Training loss is NaN                                  | `tokens` list is empty or `loss_mask` has no 1s    | Verify `generate` populates tokens and calls `append_output`                        |

## Next Steps

- [Custom Rewards](custom_rewards.md) -- Plug in reward functions without implementing a full AgentFlow
- [Agentic Multi-Turn Training](agentic_multiturn.md) -- Configure multi-turn rollout parameters
- [Tool Environment & SWE](tool_env_and_swe.md) -- Use the built-in tool infrastructure and SWE agent
