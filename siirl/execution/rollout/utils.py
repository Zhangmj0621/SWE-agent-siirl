from enum import Enum
from typing import Any

from siirl.environment.tool_env.utils.tool_parser import FunctionCall


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
