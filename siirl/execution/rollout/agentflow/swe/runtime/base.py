from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from ...base import Sample
from ..environment import ContainerStartArgs, ContainerEnv


@dataclass
class SWESampleData:
    """SWE 单任务的数据集内容；
    应当包含 repo, base_commit, patch, test_patch, eval 等信息
    """

    container_args: ContainerStartArgs
    problem_statement: str  # 问题描述，作为最初 prompt
    runtime_meta: Any  # original dict


class RuntimeBuilder(ABC):
    """A RuntimeBuilder parses config and starts runtime"""

    @abstractmethod
    def __init__(self, config: dict):
        raise NotImplementedError

    @abstractmethod
    def parse_sampledata(self, sample: dict) -> SWESampleData:
        """Parse raw sample into SampleData, including container args and parse sample."""
        raise NotImplementedError

    @abstractmethod
    def build(self, sample: Sample) -> "Runtime":
        """Build a runtime on sample"""
        raise NotImplementedError


class Runtime(ABC):
    """A Runtime enables dataset-related operation by one sample. It specifies
    container args, parse sample, generate and apply patch, run eval"""

    @abstractmethod
    async def bootstrap(self, env: ContainerEnv):
        """Bootstrap environment so agent can work on"""
        raise NotImplementedError

    @abstractmethod
    async def diff(self, env: ContainerEnv):
        """Build patch and update sample"""
        raise NotImplementedError

    @abstractmethod
    async def eval(self, env: ContainerEnv):
        """Apply patch in sample to sample env, run evaluation, and set reward"""
        raise NotImplementedError
