# 构建自定义 Agent

*实现自定义的 Agent 任务，无需修改框架代码。*

## AgentFlow 三方法契约

!!! tip "核心要点"
    每个 AgentFlow 必须实现三个方法：`preprocess()`、`async generate()` 和 `async reward()`。其中 generate 和 reward **必须**是 `async def`。这三个方法是你的 Agent 逻辑与 RL 训练循环之间的唯一接口。

`AgentFlow` 协议（`base.py:12`）定义了契约：

```python
class AgentFlow(Protocol):
    def preprocess(self, sample: dict) -> Sample:
        """将原始数据集行（dict）转换为 Sample 对象。"""
        ...

    async def generate(self, sample: Sample):
        """运行 scaffold rollout / 解题生成。"""
        ...

    async def reward(self, sample: Sample):
        """运行评估 / 验证，并设置 sample.reward。"""
        ...
```

各方法职责：

| 方法           | 同步/异步  | 输入           | 输出            | 职责                                      |
| ------------ | ------ | ------------ | ------------- | --------------------------------------- |
| `preprocess` | 同步     | `dict`（数据集行） | `Sample`      | 解析原始数据、构建 prompt、初始化元数据                 |
| `generate`   | **异步** | `Sample`     | 原地修改 `Sample` | 查询模型、记录 tokens/log_probs/loss_mask、设置状态 |
| `reward`     | **异步** | `Sample`     | 原地修改 `Sample` | 评估正确性、设置 `sample.reward` 浮点值            |

`Sample` 数据类（`base.py:118`）在整个流程中承载所有状态：

| 字段                  | 类型              | 设置方                   | 用途                                           |
| ------------------- | --------------- | --------------------- | -------------------------------------------- |
| `m`                 | `AgentMeta`（泛型） | `preprocess`          | 自定义元数据（任务信息、解析后的答案等）                         |
| `model`             | `Model`         | `preprocess`          | 用于 generate 阶段查询的 LLM 句柄                     |
| `status`            | `Status` 枚举     | `generate` / `reward` | 生命周期：`PENDING` -> `ROLLEDOUT` -> `COMPLETED` |
| `tokens`            | `list[int]`     | `generate`            | 完整 token 序列（prompt + 回复 + 观测）                |
| `loss_mask`         | `list[int]`     | `generate`            | 模型生成的 token 为 1，输入/观测为 0                     |
| `rollout_log_probs` | `list[float]`   | `generate`            | 来自 rollout 的逐 token 对数概率                     |
| `conversations`     | `list[dict]`    | `generate`            | OpenAI 格式的消息历史                               |
| `reward`            | `float`         | `reward`              | 用于 RL 训练的最终标量奖励                              |

## 分步教程：数学推理 Agent

以下教程基于参考实现 `agentflow/swe/example/__init__.py` 构建完整的 Agent。

### 1. 定义元数据

```python
from dataclasses import dataclass, field
from siirl.execution.rollout.agentflow.base import (
    AgentFlow, Model, ModelResponse, Sample
)

@dataclass
class MathMeta:
    """数学 Agent 的自定义元数据。"""
    question: str = ""
    ground_truth: float = 0.0
    parsed_answer: str = ""

MathSample = Sample[MathMeta]
```

### 2. 实现 AgentFlow 类

```python
import re

class MathReasoningFlow:
    """用于数学推理任务的 AgentFlow。"""

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
        # 构建用户消息
        message = {
            "role": "user",
            "content": (
                f"请逐步解决这个数学问题。\n\n"
                f"{sample.m.question}\n\n"
                f"将最终答案放在 <answer></answer> 标签中。"
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

            # 解析答案
            match = re.search(
                r"<answer>\s*(.*?)\s*</answer>", response.output, re.DOTALL
            )
            if match:
                sample.m.parsed_answer = match.group(1).strip()
                sample.status = MathSample.Status.ROLLEDOUT
            else:
                sample.status = MathSample.Status.ABORTED
                sample.errors.append("未找到 <answer> 标签")

        except Exception as e:
            sample.status = MathSample.Status.ABORTED
            sample.errors.append(f"生成失败: {e}")

    async def reward(self, sample: MathSample):
        try:
            parsed = float(sample.m.parsed_answer)
            correct = abs(parsed - sample.m.ground_truth) < 1e-6
            sample.reward = 1.0 if correct else 0.0
            sample.status = MathSample.Status.COMPLETED
        except (ValueError, TypeError):
            sample.reward = 0.0
            sample.status = MathSample.Status.FAILED
            sample.errors.append("无法将答案解析为数字")
```

### 3. Sample 生命周期

`Sample.Status` 枚举追踪生命周期的推进：

```
PENDING ──preprocess()──> （sample 已创建）
         ──generate()───> ROLLEDOUT  （成功）
                        > TRUNCATED  （上下文长度超限）
                        > ABORTED    （生成失败）
         ──reward()────> COMPLETED   （成功）
                        > FAILED     （奖励计算失败）
```

框架在将数据发送到训练循环之前会过滤掉非 `COMPLETED` 状态的样本。只有状态为 `COMPLETED` 且具有有效 `reward`、`tokens` 和 `loss_mask` 的样本才会参与梯度更新。

## 分步教程：工具调用 Agent（教学示例）

!!! note
    这是一个教学示例。代码库中目前只有数学（example.py）和 SWE Agent 的实际实现。本节展示如何构建自定义的工具调用 Agent。

工具调用 Agent 通过多轮交互与外部工具互动。核心模式：查询模型、从回复中解析工具调用、执行工具、将结果作为观测反馈回模型，然后循环。

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
    """多轮搜索 Agent，搜索信息并回答问题。"""

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
        # 包含工具描述的系统消息
        system_msg = {
            "role": "system",
            "content": (
                "你可以使用搜索工具，格式如下："
                '<tool_call>{"name": "search", "args": {"query": "..."}}</tool_call>\n'
                "将最终答案放在 <answer></answer> 标签中。"
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
            # 查询模型
            input_tokens = self.model.tokenizer.apply_chat_template(
                sample.conversations, add_generation_prompt=True
            )
            response = await self.model.query(
                input_tokens=input_tokens,
                messages=sample.conversations,
                max_tokens=512,
            )
            sample.add_message("assistant", response)

            # 检查工具调用
            tool_match = re.search(
                r"<tool_call>(.*?)</tool_call>", response.output, re.DOTALL
            )
            if tool_match:
                # 执行工具并添加观测
                tool_call = json.loads(tool_match.group(1))
                result = await self._execute_tool(tool_call)
                sample.m.tool_results.append(result)
                observation = f"搜索结果：{result['output']}"
                sample.add_message("tool", observation)
                continue

            # 检查最终答案
            if "<answer>" in response.output:
                sample.status = ToolSample.Status.ROLLEDOUT
                return

        # 轮次用尽
        sample.status = ToolSample.Status.TRUNCATED

    async def _execute_tool(self, tool_call: dict) -> dict:
        """执行工具调用。请替换为实际的工具实现。"""
        return {"output": f"模拟结果：{tool_call['args']['query']}"}

    async def reward(self, sample: ToolSample):
        # 提取并评估答案
        last_assistant = [
            m for m in sample.conversations if m["role"] == "assistant"
        ][-1]["content"]
        match = re.search(r"<answer>(.*?)</answer>", last_assistant, re.DOTALL)
        if match:
            sample.reward = 1.0  # 替换为实际评估逻辑
            sample.status = ToolSample.Status.COMPLETED
        else:
            sample.reward = 0.0
            sample.status = ToolSample.Status.FAILED
```

工具调用 Agent 的关键模式：

- **`sample.add_message("assistant", response)`** 自动为 `ModelResponse` 对象调用 `append_output()`，将生成的 token 的 `loss_mask` 设为 1。
- **`sample.add_message("tool", observation)`** 内部调用 `append_input_tokens()`，将观测 token 的 `loss_mask` 设为 0。只有模型生成的 token 参与 RL 损失计算。
- 循环遵守最大轮次限制，超过后设置 `TRUNCATED` 状态。

## 注册你的 Agent

!!! tip "核心要点"
    对于 AgentFlow 协议（token 级控制），使用 `load_agentflow()`。对于基于 NaiveFlow 的 Agent，使用 `RolloutArguments` 中的 `flow_function` 和 `flow_config`。两者都支持动态方法注入，无需修改框架代码。

### AgentFlow 注册（通过 `load_agentflow`）

`load_agentflow()` 函数（`__init__.py:11`）从配置字典加载你的 Agent：

```python
config = {
    "name": "my_agents.math:MathReasoningFlow",  # module:class 格式
    "reward_fn": "my_agents.math:custom_eval_reward",  # 可选覆盖
    "python_path": ["/path/to/your/agents"],  # 可选的导入路径
}
agent = load_agentflow(config, model)
```

配置键说明：

| 键               | 必需  | 格式                        | 用途                 |
| --------------- | --- | ------------------------- | ------------------ |
| `name`          | 是   | `"module.path:ClassName"` | 要实例化的 Agent 类      |
| `preprocess_fn` | 否   | `"module.path:function"`  | 覆盖 `preprocess` 方法 |
| `generate_fn`   | 否   | `"module.path:function"`  | 覆盖 `generate` 方法   |
| `reward_fn`     | 否   | `"module.path:function"`  | 覆盖 `reward` 方法     |
| `python_path`   | 否   | `["/path/to/dir", ...]`   | 额外的导入路径            |

### 动态方法注入

`load_agentflow()` 函数支持在加载时注入单个方法。这对于混合搭配预处理、生成和奖励逻辑非常有用：

```python
# 框架调用：
agent = agent_class(config, model)  # 你的 __init__

# 如果配置中指定了方法，则覆盖：
if reward_fn is not None:
    agent.reward = MethodType(reward_fn, agent)
```

注入的函数接收 `self`（Agent 实例）作为第一个参数：

```python
# my_rewards.py
async def custom_eval_reward(self, sample: MathSample):
    """注入的奖励函数。`self` 是 AgentFlow 实例。"""
    # 可以访问 self.model、self.config 等
    sample.reward = 1.0 if sample.m.parsed_answer == "42" else 0.0
    sample.status = MathSample.Status.COMPLETED
```

### NaiveFlow 注册（通过 Hydra 配置）

对于使用内置 `NaiveFlow` 管线的简单 Agent，通过 Hydra YAML 配置：

```yaml
rollout:
  flow_function: naive          # 使用 NaiveFlow
  flow_config: config.yaml      # flow 的 YAML 配置路径
  multiturn:
    env_type: tool_env           # 启用工具环境
    max_env_turns: 5             # 最大工具交互轮次
    max_assistant_turns: 10      # 最大模型生成轮次
    max_parallel_calls: 1        # 每个样本的并行工具执行数
```

### SWE Agent 注册

SWE Agent 使用构建器模式，包含三个可插拔组件：

```yaml
name: swe
agent:
  name: minisweagent            # 内置 SWE Agent
environment:
  name: k8s                     # k8s | docker | kr8s
  namespace: swe
runtime:
  name: swebench_sii            # swebench | swebench_sii | swefactory
```

## 本地测试你的 Agent

!!! tip "核心要点"
    在运行完整训练循环之前，独立测试每个方法。使用 `base.py` 中的 `DummyModel` 进行不需要真实 LLM 的单元测试。

### 使用 DummyModel 进行单元测试

```python
import asyncio
from siirl.execution.rollout.agentflow.base import DummyModel
from siirl.execution.rollout.agentflow import load_agentflow

# DummyModel 返回预配置的响应
model = DummyModel(["<answer> 42 </answer>"])

agent = load_agentflow(
    {
        "name": "my_agents.math:MathReasoningFlow",
        "python_path": ["/path/to/your/agents"],
    },
    model,
)

# 测试 preprocess
sample = agent.preprocess({"question": "6*7 等于多少？", "answer": "42"})
assert sample.status == sample.Status.PENDING

# 测试 generate
asyncio.run(agent.generate(sample))
assert sample.status == sample.Status.ROLLEDOUT
assert len(sample.tokens) > 0
assert sum(sample.loss_mask) > 0  # 部分 token 是模型生成的

# 测试 reward
asyncio.run(agent.reward(sample))
assert sample.status == sample.Status.COMPLETED
assert sample.reward == 1.0
```

### 使用真实模型测试

对于使用 SGLang 或其他推理引擎的集成测试，直接运行 Agent 脚本：

```bash
python -m siirl.execution.rollout.agentflow.swe.example
```

这种模式（使用 `__main__.py`）是为你的 Agent 创建独立测试工具的推荐方式。

## 调试 Agent 行为

### 检查对话内容

```python
# 在 generate() 之后，检查完整对话
for i, msg in enumerate(sample.conversations):
    print(f"轮次 {i} [{msg['role']}]: {msg['content'][:200]}")
```

### 验证 Token 对齐

```python
# 确保 tokens、loss_mask 和 rollout_log_probs 对齐
assert len(sample.tokens) == len(sample.loss_mask)
assert len(sample.tokens) == len(sample.rollout_log_probs)

# 统计模型生成 vs. 输入 token
model_tokens = sum(sample.loss_mask)
input_tokens = len(sample.loss_mask) - model_tokens
print(f"总计: {len(sample.tokens)}, 模型生成: {model_tokens}, 输入: {input_tokens}")
```

### 检查状态转换

```python
# 在每个阶段后记录状态以便调试
print(f"preprocess 后: {sample.status}")  # PENDING
await agent.generate(sample)
print(f"generate 后: {sample.status}")    # ROLLEDOUT / ABORTED / TRUNCATED
await agent.reward(sample)
print(f"reward 后: {sample.status}")      # COMPLETED / FAILED
print(f"错误信息: {sample.errors}")        # 累积的错误消息
```

### 常见错误

| 症状                                                    | 原因                                     | 修复方法                                                                      |
| ----------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------- |
| `TypeError: object NoneType can't be used in 'await'` | `generate` 或 `reward` 未声明为 `async def` | 在方法定义前添加 `async` 关键字                                                      |
| `loss_mask` 全为零                                       | 忘记调用 `sample.append_output(response)`  | 使用 `sample.add_message("assistant", response)` 或 `sample.append_output()` |
| reward 阶段后 `reward` 为 `None`                          | 奖励函数未设置 `sample.reward`                | 确保每条代码路径都将 `sample.reward` 设为浮点数                                          |
| Agent 加载时 `ImportError`                               | 模块路径格式错误                               | 使用冒号分隔的 `"module.path:ClassName"` 格式                                      |
| 注入的奖励函数缺少 `self`                                      | 函数签名缺少第一个 `self` 参数                    | 注入函数接收 Agent 作为 `self`：`async def my_reward(self, sample)`                |
| 训练损失为 NaN                                             | `tokens` 列表为空或 `loss_mask` 没有值为 1 的元素  | 验证 `generate` 填充了 tokens 并调用了 `append_output`                             |

## 下一步

- [自定义奖励函数](custom_rewards.md) -- 无需实现完整 AgentFlow 即可接入奖励函数
- [多轮 Agent 训练](agentic_multiturn.md) -- 配置多轮 rollout 参数
- [工具环境与 SWE](tool_env_and_swe.md) -- 使用内置工具基础设施和 SWE Agent
