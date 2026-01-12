from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import BinaryIO


@dataclass
class ContainerStartArgs:
    """Container Start Arguments

    Args:
        image: Container image to use.
        cmd: Override image start command. Defaults to None. May do some preparation
            and sleep.
        cwd: Working directory for command execution. Defaults to None.
        env: Environment variables to set in the container. Defaults to None.
        forward_env: Environment variables to forward from the host. Variables
            are only forwarded if set in the host environment. In case of conflict
            with `env`, the `env` variables take precedence. Defaults to None.
        container_timeout: Maximum duration to keep container running. Uses the
            same format as the sleep command. Defaults to "2h".
        startup_timeout: Maximum time in seconds to wait for the runtime to start.
            Defaults to 180.0.
        resource_requests: Resource requests for the container (e.g., memory, cpu).
            Defaults to None (typically `{"memory": "4Gi", "cpu": "2"}`).
        resource_limits: Resource limits for the container (e.g., memory, cpu).
            Defaults to None (typically `{"memory": "4Gi", "cpu": "2"}`).
    """

    image: str
    cmd: str | None = None
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    forward_env: list[str] = field(default_factory=list)
    container_timeout: str = "2h"
    startup_timeout: float = 180.0
    resource_requests: dict[str, int | str] | None = None
    resource_limits: dict[str, int | str] | None = None


@dataclass
class ContainerBuildArgs:
    """Arguments for building a container image.

    Attributes:
        tag: The tag to apply to the built container image (e.g., 'myapp:latest').
        build_dir: Path to the directory containing the build context.
        dockerfile_path: Optional path to the Dockerfile. If None, defaults to
            'Dockerfile' in the build_dir.
        nocache: If True, do not use cache when building the image. Defaults to False.
        rm: If True, remove intermediate containers after a successful build.
            Defaults to True.
        push: If True, push the built image to a registry after building.
            Defaults to False.
        timeout: Build timeout in seconds. A value of 0.0 means no timeout.
            Defaults to 0.0.
        resource_limits: Optional dictionary of resource limits for the build process.
            Can include keys like 'memory' (int or str) and 'cpus' (int or str).
            Defaults to None.
    """

    tag: str
    build_dir: str
    dockerfile_path: str | None = None
    nocache: bool = False
    rm: bool = True
    push: bool = False
    timeout: float = 0.0
    resource_limits: dict[str, int | str] | None = None


class ContainerEnvBuilder(ABC):
    """Protocol for building container execution environments."""

    @abstractmethod
    def __init__(self, conf: dict):
        raise NotImplementedError

    @abstractmethod
    async def build(self, args: ContainerBuildArgs):
        """Start a container from the given configuration.

        TODO: design signature

        Args:
            args (ContainerBuildArgs): container build arguments

        Raises:
            TimeoutError: Startup timeout.
            Exception: Unexpected failure
        """
        raise NotImplementedError

    @abstractmethod
    async def start(self, args: ContainerStartArgs) -> "ContainerEnv":
        """Start a container from the given configuration.

        Args:
            args (ContainerStartArgs): container start up arguments

        Returns:
            ContainerEnv instance for executing commands.

        Raises:
            TimeoutError: Startup timeout.
            Exception: Unexpected failure

        Notes:
            The builder itself may have its own configuration that affects behavior.

            All envs (cwd, env, forward_env) should act as default, and be merged when running cmd.
        """
        raise NotImplementedError


@dataclass
class ContainerOutput:
    """Output from a container command execution.

    Attributes:
        output: Combined stdout and stderr from the command.
        returncode: Exit code returned by the command.
    """

    output: bytes
    returncode: int


class ContainerEnv(ABC):
    """Abstract base class for container execution environments."""

    @abstractmethod
    async def popen(
        self,
        cmd: str,
        cwd: str | None = None,
        env: dict[str, str] = {},
        forward_env: list[str] = [],
        timeout: float = 180.0,
    ) -> ContainerOutput:
        """TODO: design a Popen class like interface, that supports async interaction with
        process state, stdin, stdout, stderr, returncode...
        """
        raise NotImplementedError

    @abstractmethod
    async def execute(
        self,
        cmd: str,
        stdin: BinaryIO | None = None,
        cwd: str | None = None,
        env: dict[str, str] = {},
        forward_env: list[str] = [],
        timeout: float = 180.0,
        check: bool = True,
    ) -> ContainerOutput:
        """Execute a command in the container environment.

        Args:
            cmd: The command to execute.
            stdin: The stdin put to command. Defaults to None.
            cwd: Working directory in which to execute the command. Defaults to None.
            env: Environment variables to set for the command. Defaults to {}.
            forward_env: Environment variables to forward from the host. Defaults to [].
            timeout: Maximum time in seconds to wait for command completion. Defaults to 180.0.
            check: Raise on non-0 return code if set to True. Defaults to True.

        Returns:
            ContainerOutput containing the command's output text and return code.

        Raises:
            Exception: If unexpected errors occur (network failure, OOM, etc.).

        Notes:
            cwd, env, forward_env will merge with those on startup; so no need to repass it.
        """
        raise NotImplementedError

    async def copy(
        self,
        src: str,
        dst: str,
        upload: bool = True,
        cwd: str | None = None,
        timeout: float = 180.0,
    ):
        """Copies files between the host and container.
        Args:
            src (str): Source path. If upload is True, this is a host path; otherwise,
                it's a container path.
            dst (str): Destination path. If upload is True, this is a container path;
                otherwise, it's a host path.
            upload (bool): If True, copies from host to container. If False, copies from
                container to host. Defaults to True.
            cwd (Optional[str]): Current working directory for resolving relative paths.
                If None, uses the container's default working directory. Defaults to None.
            timeout: Maximum time in seconds to wait for command completion. Defaults to 180.0.
        """
        # TODO: implement as https://github.com/kubernetes/kubectl/blob/master/pkg/cmd/cp/cp.go
        raise NotImplementedError

    @abstractmethod
    async def cleanup(self):
        """Cleanup the container environment and release resources."""
        raise NotImplementedError

    @property
    @abstractmethod
    def alive(self) -> bool:
        """Check whether the environment is alive.

        Returns:
            True if the environment is running, False otherwise.
        """
        raise NotImplementedError

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.alive:
            await self.cleanup()

    def __del__(self):
        """Destructor that warns if cleanup wasn't called properly."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, skip async cleanup
            return

        if loop.is_running():

            async def _check_and_cleanup():
                if self.alive:
                    import logging

                    logging.warning("Container should cleanup manually or use `async with as`; run in background")
                    await self.cleanup()

            asyncio.create_task(_check_and_cleanup())
