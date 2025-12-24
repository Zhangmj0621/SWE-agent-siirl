from dataclasses import dataclass, field
from typing import Optional, Any, Protocol, Generic, TypeVar
from types import MethodType
from enum import Enum

# from transformers import PreTrainedTokenizer # This is slow!
type PreTrainedTokenizer = Any


class AgentFlow(Protocol):
    """一套 dataset-rollout-evaluate 任务定义"""

    def preprocess(self, sample: dict) -> "Sample":
        """把数据集的一条数据 (dict) 处理成 Sample 对象；可能会 raise exception

        Args:
            sample (dict): 数据集的一条数据

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
    experts: Optional[list[list[int]]]
    # raw http response / other metadata
    raw: dict


class Model(Protocol):
    """Protocol for language models."""

    async def query(
        self,
        input_tokens: list[int],
        messages: list[dict],
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
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

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """language model 使用的 tokenizer；inference train 应使用相同的"""
        raise NotImplementedError


class DummyModel:
    """dumb model used for unit test; its tokenizer always returns [1]"""

    def __init__(self, responses: list[str] = []):
        class DummyTokenizer:
            def __getattribute__(self, name: str) -> Any:
                def call(self, *args, **kargs):
                    return [1]

                return MethodType(call, self)

        self.tokenizer = DummyTokenizer()
        self.responses = responses
        self.cursor = 0

    async def query(self, *args, **kwargs) -> ModelResponse:
        if self.cursor >= len(self.responses):
            msg = ""
        else:
            msg = self.responses[self.cursor]
            self.cursor += 1
        return ModelResponse(
            output=msg, output_tokens=[1], log_probs=[0.0], experts=None, raw={}
        )


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
    reward: Optional[float] = None

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
        else:
            assert role != "assistant"
            tokens: list[int] = self.model.tokenizer(content)
            self.append_input_tokens(tokens)
        self.conversations.append({"role": role, "content": content, **kwargs})
