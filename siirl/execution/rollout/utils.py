from enum import Enum
from typing import Any

from siirl.environment.tool_env.utils.tool_parser import FunctionCall


class RolloutGenerationAborted(Exception):
    """Raised when rollout generation is aborted and should be resumed later."""

    def __init__(self, sample: Any):
        super().__init__("rollout generation aborted")
        self.sample = sample


class SglangGenerationAborted(Exception):
    """Raised when SGLang returns an aborted generation with partial tokens."""

    def __init__(
        self,
        responses: list[int],
        rollout_log_prob: list[float],
        routed_experts: Any = None,
        rid: str | None = None,
    ):
        super().__init__("sglang generation aborted")
        self.responses = responses
        self.rollout_log_prob = rollout_log_prob
        self.routed_experts = routed_experts
        self.rid = rid


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_ENV = "processing_envs"
    TERMINATED = "terminated"
    BEFORE_PROCESSING_ENV = "before_processing_envs"
    ABORTED = "aborted"


class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(self, raw_prompt: list[dict[str, Any]], ground_truth: Any = None):
        self.messages: list[dict[str, Any]] = raw_prompt
        self.prompts_ids = []
        self.rollout_log_prob = []
        self.response_ids = []
        self.response_mask = []
        self.env_calls: list[FunctionCall] = []
        self.env_turns = 0
        self.assistant_turns = 0
        self.state = AgentState.PENDING
        self.env_kwargs = {}
        self.env_rewards = []
        self.routed_experts = None  # Raw flat np.int32 array from SGLang MoE routing
        # SGLang request id. Assigned once by naive_flow._load_agent_data and shared
        # across every generate() call of this sample's multi-turn lifetime (so the
        # inference engine can correlate turns as one logical request), and
        # preserved across abort/resume via partial_agent_data.
        self.rid: str | None = None
        if ground_truth:
            self.env_kwargs["ground_truth"] = ground_truth


def format_gpt_oss_tool_response_manually(tool_response: str, tool_call_name: str) -> str:
    """Format tool response for gpt-oss model.
    Args:
        tool_response: Tool response string
        tool_call_name: Name of the tool that was called

    Returns:
        Formatted tool response string
    """
    return f"<|start|>functions.{tool_call_name} to=assistant<|channel|>commentary<|message|>{tool_response}<|end|>"


def add_generation_prompt_for_gpt_oss(message_content: str) -> str:
    """Add generation prompt for gpt-oss model.
    Args:
        message_content: Message content string

    Returns:
        Message content string with generation prompt
    """
    return message_content + "<|start|>assistant"
