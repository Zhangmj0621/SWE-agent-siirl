from abc import ABC, abstractmethod

from ...base import Sample
from ..environment import ContainerEnv


class AgentBuilder(ABC):
    """Abstract base class for agent builder

    Args:
        config (dict): Static configuration for the agent (immutable).
    """

    @abstractmethod
    def __init__(self, config: dict):
        raise NotImplementedError

    @abstractmethod
    def build(self, sample: Sample) -> "Agent":
        """Build an agent.

        Args:
            env (ContainerEnv): The container environment in which the agent operates.
            sample (Sample): The sample data associated with this agent instance.
        """
        raise NotImplementedError


class Agent(ABC):
    """Abstract base class for agents that interact with a containerized environment.

    Each agent corresponds to a single sample and is responsible for executing
    tasks within a container environment using a specified model.
    """

    @abstractmethod
    async def run(self, env: ContainerEnv):
        """Executes the agent's main logic asynchronously.

        This method should contain the core logic for the agent's operation,
        including interactions with the model and environment. The implementation
        should be asynchronous to support concurrent execution of multiple agents.

        Returns:
            The return type depends on the specific implementation in subclasses.

        Raises:
            Implementation-specific exceptions may be raised by subclasses.
        """
        raise NotImplementedError
