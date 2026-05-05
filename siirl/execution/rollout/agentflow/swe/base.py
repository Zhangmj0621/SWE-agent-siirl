from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, TypeVar

from ..base import Sample
from ..utils import null_abc
from .agent.base import Agent
from .runtime.base import Runtime, SWESampleData


class SWERolloutResult(Enum):
    PENDING = "pending"
    SUBMIT = "submit"
    EXCEED = "limit exceed"
    FAILURE = "unknown failure"


Patch = TypeVar("Patch")
T = TypeVar("T", bound=ABC)


@dataclass
class SWERolloutMeta(Generic[Patch]):
    # 状态记录，可用于 reward
    result: SWERolloutResult = SWERolloutResult.PENDING
    # 由 runtime 自己决定
    patch: Patch | None = None
    # SWE agent 退出状态（来自 result.info["exit_status"]），用于决定是否进入 eval
    exit_status: str | None = None
    # 当前样本是否为 validate 阶段，用于在 reward 阶段决定过滤规则
    is_validate: bool = False
    # Partial rollout (aligned with naive_flow semantics):
    #   partial_state —  inbound, preprocess 写入；_generate_async 读后走 resume 路径
    #   partial_agent_data — outbound, _generate_async abort 分支写入；
    #                        AgentFlowCallable.__call__ 搬到 siirl Sample 上
    partial_state: dict | None = None
    partial_agent_data: dict | None = None


@dataclass
class SWERewardMeta:
    succeed: bool = False


def null_field(base_cls: type[T]) -> T:
    return field(default_factory=null_abc(base_cls))


@dataclass
class SWEAgentMeta:
    data: SWESampleData
    rollout: SWERolloutMeta = field(default_factory=SWERolloutMeta)
    agent: Agent = null_field(Agent)
    runtime: Runtime = null_field(Runtime)
    # Weight version (training step) when this sample was generated
    # Used for organizing eval logs by training step
    weight_version: int | None = None


SWESample = Sample[SWEAgentMeta]
