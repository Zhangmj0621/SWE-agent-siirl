from dataclasses import dataclass, field
from enum import Enum
from types import MethodType
from typing import Any, Generic, Protocol, TypeVar

from loguru import logger

# from transformers import PreTrainedTokenizer # This is slow!
PreTrainedTokenizer = Any


class AgentFlow(Protocol):
    """一套 dataset-rollout-evaluate 任务定义"""

    def preprocess(self, sample: dict, model: "Model") -> "Sample":
        """把数据集的一条数据 (dict) 处理成 Sample 对象；可能会 raise exception

        Args:
            sample (dict): 数据集的一条数据
            model (Model): 模型实例（每个样本独立）

        Returns:
            Sample: rollout 并 evaluate 的算例
        """
        raise NotImplementedError

    async def generate(self, sample: "Sample"):
        """运行 scaffold rollout / math solution generation

        Args:
            sample (Sample): 本 TaskFlow 兼容的算例
        """
        raise NotImplementedError

    async def reward(self, sample: "Sample"):
        """运行 evaluate / verification

        Args:
            sample (Sample): 本 TaskFlow 兼容的算例
        """
        raise NotImplementedError


@dataclass
class ModelResponse:
    # 输出文字（raw，不经过 toolcall/thinking 解析）
    output: str
    # token list; 等同于 tokenizer(output)
    output_tokens: list[int]
    # Log probs for each token
    log_probs: list[float]
    # moe specific; experts of each token
    experts: list[list[int]] | None
    # raw http response / other metadata
    raw: dict
    # tool calls
    tool_calls: list[dict] = field(default=None, metadata="extract tool calls in model response")
    reasoning_content: str = field(default=None, metadata="extract reasoning_content in model response")
    thinking_blocks: str = field(default=None, metadata="extract thinking_blocks in model response")


class Model(Protocol):
    """Protocol for language models."""

    async def query(
        self,
        input_tokens: list[int],
        messages: list[dict],
        max_tokens: int | None = None,
        timeout: int | None = None,
    ) -> ModelResponse:
        """使用 language model 生成回复，最好支持 token-in, token-out；异步进行 (aiohttp)

        Args:
            input_token (list[int]): 输入 token；等同于 tokenizer.apply_chat_template(messages)
            messages (list[dict]): 输入消息；fallback 使用
            max_tokens (Optional[int]): 最大输出长度；None 则使用默认值
            timeout (Optional[int]): 调用最大超时时间

        Returns:
            ModelResponse: 模型输出
        """
        raise NotImplementedError

    def query_for_swe(
        self,
        history: "list[int] | list[dict]",
        action_prompt: str = "> ",
    ) -> dict:
        """Query method compatible with SWE-agent AbstractModel.

        This method provides a synchronous interface that matches SWE-agent's
        AbstractModel.query() signature, enabling direct use with SWE-agent's
        RLTokenAgent and other components.

        Args:
            history: Chat history (list of dict) or token IDs (list of int)
            action_prompt: Action prompt string (default: ">")

        Returns:
            dict with keys:
                - message: str - model output text
                - output_tokens: list[int] - generated token IDs
                - log_probs: list[float] - log probabilities
                - tool_calls: list[dict] - optional tool calls
                - reasoning_content: str - optional reasoning
                - thinking_blocks: list[dict] - optional thinking blocks
        """
        import asyncio

        # Handle both token IDs and message history
        if isinstance(history, list) and len(history) > 0 and isinstance(history[0], int):
            # history is token IDs
            input_tokens = history
            messages = []  # Token IDs mode - messages not used
        else:
            # history is message dict format
            messages = history if isinstance(history, list) else []
            input_tokens = []  # Will be computed by model

        # Run async query in event loop
        loop = asyncio.get_event_loop()
        response: ModelResponse = loop.run_until_complete(
            self.query(
                input_tokens=input_tokens,
                messages=messages,
                max_tokens=None,
                timeout=None,
            )
        )

        # Return format expected by SWE-agent
        result = {
            "message": response.output,
            "output_tokens": response.output_tokens,
            "log_probs": response.log_probs,
        }

        # Add optional fields if present
        if response.tool_calls:
            result["tool_calls"] = response.tool_calls
        if response.reasoning_content:
            result["reasoning_content"] = response.reasoning_content
        if response.thinking_blocks:
            result["thinking_blocks"] = response.thinking_blocks

        return result

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """language model 使用的 tokenizer；inference train 应使用相同的"""
        raise NotImplementedError


class DummyTokenizer:
    def __getattribute__(self, name: str) -> Any:
        def call(self, *args, **kargs):
            return [1]

        return MethodType(call, self)

    def __call__(self, *args, **kargs):
        return [1]


class DummyModel:
    """dumb model used for unit test; its tokenizer always returns [1]"""

    def __init__(self, responses: list[str] | None = None):
        self.tokenizer = DummyTokenizer()
        self.responses = responses if responses is not None else []
        self.cursor = 0

    async def query(self, *args, **kwargs) -> ModelResponse:
        if self.cursor >= len(self.responses):
            msg = ""
        else:
            msg = self.responses[self.cursor]
            self.cursor += 1
        return ModelResponse(output=msg, output_tokens=[1], log_probs=[0.0], experts=None, raw={})


AgentMeta = TypeVar("AgentMeta")


@dataclass
class Sample(Generic[AgentMeta]):
    """
    算例，包括一条数据和其 generate, eval 等训练相关信息（见 SWESample）
    """

    class Status(Enum):
        PENDING = "pending"  # before preprocess
        ROLLEDOUT = "rolledout"  # complete rollout
        COMPLETED = "completed"  # complete rollout and reward
        TRUNCATED = "truncated"  # rollout cancel for context limit
        ABORTED = "aborted"  # rollout cancel for failure
        FAILED = "failed"  # reward cancel for failure

    # metadata，RL 框架可以忽略
    m: AgentMeta  # 由 agentflow 生成；使用泛型以有更好的编译器支持
    model: Model

    # RL 框架需要处理不同的 status
    status: Status = Status.PENDING
    errors: list[str] = field(default_factory=list)  # 报错信息

    # language model RL 需要用到的信息.
    # input_token, used for postprocess in executor
    prompts: list[int] = field(default_factory=list)
    # 这些由 generate 生成
    # 所有 token，包括  model generated 和 observation
    tokens: list[int] = field(default_factory=list)
    # 只有 model generated 为 1，其余为 0
    loss_mask: list[int] = field(default_factory=list)
    # Log probabilities from rollout engine; may be none so need re-forward
    # on loss calculation
    rollout_log_probs: list[float] = field(default_factory=list)
    # moe experts, len(experts)==len(tokens)
    experts: list[list[int]] = field(default_factory=list)
    # fields: role:str, content:str, ... (openai format)
    conversations: list[dict] = field(default_factory=list)

    # 这些由 reward 生成
    reward: float | None = None

    def append_input_tokens(self, input_tokens: list[int]):
        n_tokens = len(input_tokens)
        self.tokens += input_tokens
        self.loss_mask += [0] * n_tokens
        self.rollout_log_probs += [0.0] * n_tokens
        self.experts += [[0]] * n_tokens

    def append_output(self, o: ModelResponse):
        n_tokens = len(o.output_tokens)
        self.tokens += o.output_tokens
        self.loss_mask += [1] * n_tokens
        self.rollout_log_probs += o.log_probs
        self.experts += [[]] * n_tokens if o.experts is None else o.experts

    def add_message(self, role: str, content: str | ModelResponse, **kwargs):
        # TODO: evaluate tokenin-tokenout
        if isinstance(content, ModelResponse):
            assert role == "assistant"
            self.append_output(content)
            content = content.output
        else:
            assert role != "assistant"
            tokens: list[int] = self.model.tokenizer(content)
            self.append_input_tokens(tokens)
        logger.debug(
            f"[Agentflow Sample] message {len(self.conversations)}",
            "role",
            role,
            "content",
            content,
        )
        self.conversations.append({"role": role, "content": content, **kwargs})
