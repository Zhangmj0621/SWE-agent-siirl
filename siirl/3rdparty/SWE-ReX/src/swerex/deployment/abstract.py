import asyncio
import logging
from abc import ABC, abstractmethod

from swerex.deployment.hooks.abstract import DeploymentHook
from swerex.runtime.abstract import AbstractRuntime, IsAliveResponse

__all__ = ["AbstractDeployment"]


class AbstractDeployment(ABC):
    def __init__(self, *args, **kwargs):
        self.logger: logging.Logger

    @abstractmethod
    def add_hook(self, hook: DeploymentHook): ...

    @abstractmethod
    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        """Checks if the runtime is alive. The return value can be
        tested with bool().

        Raises:
            DeploymentNotStartedError: If the deployment was not started.
        """

    @abstractmethod
    async def start(self, *args, **kwargs):
        """Starts the runtime."""

    @abstractmethod
    async def stop(self, *args, **kwargs):
        """Stops the runtime."""

    @property
    @abstractmethod
    def runtime(self) -> AbstractRuntime:
        """Returns the runtime if running.

        Raises:
            DeploymentNotStartedError: If the deployment was not started.
        """

    def __del__(self):
        # Intentional no-op. GC-triggered pod deletion races with in-flight
        # requests still held by env/agent, producing spurious
        # ConnectionRefusedError on a pod we just force-killed. Cleanup is
        # the caller's responsibility via explicit stop(); the K8s adapter
        # performs a detached kubectl delete as a last-resort fallback.
        return

