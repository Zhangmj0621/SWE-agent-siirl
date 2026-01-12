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


SWESample = Sample[SWEAgentMeta]
