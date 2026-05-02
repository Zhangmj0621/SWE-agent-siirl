import asyncio
import contextlib
import json
import logging
import os
import re
import shlex
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import PurePath
from subprocess import CalledProcessError
from typing import TYPE_CHECKING, Any, BinaryIO, Literal, Union

if TYPE_CHECKING:
    from sweagent.environment.repo import GithubRepoConfig, LocalRepoConfig, PreExistingRepoConfig

import pexpect
from pydantic import BaseModel, Field

from siirl.execution.rollout.agentflow.swe.environment.base import ContainerBuildArgs

from .base import ContainerEnv, ContainerEnvBuilder, ContainerOutput, ContainerStartArgs

# Type for repo configuration (compatible with SWE-agent's repo types)
RepoConfig = Union[Any, "LocalRepoConfig", "GithubRepoConfig", "PreExistingRepoConfig", None]


async def _retry_async(
    logger: logging.Logger,
    func: callable,
    *args,
    max_retries: int = 3,
    backoff_base: float = 2.0,
    max_backoff: float = 30.0,
    retryable_exceptions: tuple = (Exception,),
    is_retryable_fn: callable = None,
    operation_name: str = "operation",
    **kwargs,
):
    """Generic async retry wrapper with exponential backoff.

    Args:
        logger: Logger instance for logging
        func: Async function to retry
        *args: Positional arguments for func
        max_retries: Maximum number of retry attempts (default: 3)
        backoff_base: Base for exponential backoff (default: 2.0)
        max_backoff: Maximum backoff time in seconds (default: 30.0)
        retryable_exceptions: Tuple of exception types to retry (default: all Exceptions)
        is_retryable_fn: Optional function to check if error is retryable (receives exception)
        operation_name: Name of the operation for logging (default: "operation")
        **kwargs: Keyword arguments for func

    Returns:
        Result of func on first successful call

    Raises:
        The last exception if all retries fail
    """
    for attempt in range(max_retries + 1):
        try:
            result = await func(*args, **kwargs)
            if attempt > 0:
                logger.info(f"{operation_name} succeeded on attempt {attempt + 1}")
            return result

        except retryable_exceptions as e:
            # Check if this error is retryable
            if is_retryable_fn and not is_retryable_fn(e):
                logger.error(f"{operation_name} failed with non-retryable error: {e}")
                raise

            if attempt < max_retries:
                # Calculate backoff time with exponential increase, capped at max_backoff
                backoff_time = min(backoff_base**attempt, max_backoff)
                logger.warning(
                    f"{operation_name} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. " f"Retrying in {backoff_time:.1f}s..."
                )
                await asyncio.sleep(backoff_time)
            else:
                # All retries exhausted
                error_msg = f"{operation_name} failed after {max_retries + 1} attempts. Last error: {e}"
                logger.error(error_msg)
                raise Exception(error_msg) from e


# Response data classes matching remote.py interface
@dataclass
class IsAliveResponse:
    """Response from is_alive check."""

    is_alive: bool
    message: str = ""


@dataclass
class CreateSessionResponse:
    """Response from create_session operation."""

    session_id: str
    message: str = ""


@dataclass
class Observation:
    """Response from run_in_session operation."""

    output: str
    return_code: int
    error_output: str = ""


@dataclass
class CloseSessionResponse:
    """Response from close_session operation."""

    success: bool
    message: str = ""


@dataclass
class ReadFileResponse:
    """Response from read_file operation."""

    content: str


@dataclass
class WriteFileResponse:
    """Response from write_file operation."""

    success: bool


@dataclass
class CloseResponse:
    """Response from close operation."""

    success: bool


@dataclass
class BashObservation:
    """Observation from running a bash command."""

    output: str
    exit_code: int | None = None
    expect_string: str | None = None


def _strip_control_chars(s: str) -> str:
    """Strip ANSI control characters from string."""
    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", s).replace("\r\n", "\n")


class K8sBashSession:
    """Manages a persistent bash session within a Kubernetes pod using pexpect."""

    _PS1 = "SHELLPS1PREFIX"
    _EXIT_CODE_PREFIX = "EXITCODESTART"
    _EXIT_CODE_SUFFIX = "EXITCODEEND"

    def __init__(
        self,
        k8s_env: "K8sEnv",
        session_id: str,
        startup_source: list[str] | None = None,
        startup_timeout: float = 30.0,
        logger: logging.Logger | None = None,
    ):
        """Initialize the bash session.

        Args:
            k8s_env: The K8sEnv instance
            session_id: Unique identifier for this session
            startup_source: List of files to source on startup (e.g., ["/root/.bashrc"])
            startup_timeout: Timeout for startup operations (default: 30s, increased from 10s)
            logger: Logger instance
        """
        self.k8s_env = k8s_env
        self.session_id = session_id
        self.startup_source = startup_source or []
        self.startup_timeout = startup_timeout
        self.logger = logger or logging.root
        self.process: pexpect.spawn | None = None
        self._closed = False

    def _build_kubectl_command(self) -> str:
        """Build the kubectl exec command to start a bash session."""
        parts = ["kubectl"]
        if self.k8s_env.kubeconfig:
            parts.extend(["--kubeconfig", self.k8s_env.kubeconfig])
        if self.k8s_env.context:
            parts.extend(["--context", self.k8s_env.context])
        parts.extend(
            [
                "exec",
                "-i",  # Keep -i for TTY (required for persistent shell)
                "-n",
                self.k8s_env.namespace,
                self.k8s_env.pod_name,
                "-c",
                self.k8s_env.container_name,
                "--",
                "/bin/bash",
                "-i",
            ]
        )
        return " ".join(parts)

    async def start(self) -> str:
        """Start the bash session.

        Returns:
            Output from session initialization
        """
        if self.process is not None:
            raise RuntimeError(f"Session {self.session_id} already started")

        self.logger.info(f"Starting bash session {self.session_id}")

        # Build the kubectl command
        cmd = self._build_kubectl_command()
        self.logger.info(f"Kubectl command: {cmd}")

        # Spawn the process
        try:
            self.process = pexpect.spawn(
                cmd,
                encoding="utf-8",
                codec_errors="backslashreplace",
                echo=False,
                timeout=self.startup_timeout,
                maxread=100000,  # Increase maxread to handle long commands/outputs
            )
            self.logger.info(f"pexpect.spawn() succeeded for session {self.session_id}")
        except Exception as e:
            self.logger.error(f"pexpect.spawn() failed: {e}")
            raise RuntimeError(f"Failed to spawn kubectl process: {e}") from e

        # Wait for bash to start
        time.sleep(0.5)

        # Check if process is still alive
        if not self.process.isalive():
            output = self.process.before
            self.logger.error(f"Process died immediately. Output: {output}")
            raise RuntimeError(f"kubectl exec process died immediately: {output}")

        # Initialize the session - skip bashrc if it doesn't exist
        init_commands = []
        if self.startup_source:
            for path in self.startup_source:
                # Check if file exists before sourcing
                init_commands.append(f"[ -f {path} ] && source {path} || true")
            init_commands.append("sleep 0.2")
        init_commands.extend(
            [
                f"export PS1='{self._PS1}'",
                "export PS2=''",
                "export PS0=''",
                # Disable pagers to prevent blocking
                # "export PAGER=cat",
                # "export MANPAGER=cat",
                # "export GIT_PAGER=cat",
                # "export LESS='-R -F'",
                # "export LV='-c'",
            ]
        )

        init_cmd = " && ".join(init_commands)
        self.logger.debug(f"Sending init command: {init_cmd[:200]}...")
        self.process.sendline(init_cmd)

        try:
            self.process.expect(self._PS1, timeout=self.startup_timeout)
        except pexpect.TIMEOUT:
            # Get debug output
            before = self.process.before or ""
            after = self.process.after or ""
            self.logger.error(f"Timeout waiting for PS1 prompt '{self._PS1}' in session {self.session_id}")
            self.logger.error(f"Process.before: {before[:500]}")
            self.logger.error(f"Process.after: {after}")
            self.logger.error(f"Process.isalive(): {self.process.isalive()}")
            self.logger.error(f"Process.exitstatus: {self.process.exitstatus}")

            # Try to read any remaining output
            with contextlib.suppress(Exception):
                if self.process.isalive():
                    remaining = self.process.read_nonblocking(size=2000, timeout=0.5)
                    self.logger.error(f"Remaining output: {remaining}")

            self.close()
            raise TimeoutError(f"Failed to initialize bash session {self.session_id}. " f"Process output: {before[:200]}") from None

        # Clear the buffer after initialization (important!)
        _ = self.process.before
        if self.process.buffer:
            with contextlib.suppress(Exception):
                self.process.read_nonblocking(timeout=0.1)

        output = _strip_control_chars(self.process.before or "")
        self.logger.info(f"Bash session {self.session_id} started successfully")
        return output

    async def run(
        self,
        command: str,
        timeout: float = 25.0,
        check: Literal["silent", "ignore", "raise"] = "ignore",
    ) -> BashObservation:
        """Run a command in the bash session.

        Args:
            command: The command to execute
            timeout: Timeout in seconds
            check: How to handle exit codes ("silent", "ignore", or "raise")

        Returns:
            BashObservation with output and exit code

        Raises:
            RuntimeError: If session is not initialized
            TimeoutError: If command times out
            RuntimeError: If check="raise" and exit code is non-zero
        """
        if self.process is None:
            raise RuntimeError(f"Session {self.session_id} not initialized")

        if self._closed:
            raise RuntimeError(f"Session {self.session_id} is closed")

        self.logger.debug(f"Running command in session {self.session_id}: {command[:100]}...")

        # Critical: Ensure we're at a clean PS1 prompt before sending the command
        # Use a unique marker to synchronize and flush all pending output
        if self.process.isalive():
            try:
                # Send a unique marker and wait for it to ensure all pending output is flushed
                sync_marker = f"_SYNC_MARKER_{id(command)}_"
                self.process.sendline(f"echo '{sync_marker}'")
                # Wait for the marker - this ensures all previous output is consumed
                self.process.expect(sync_marker, timeout=2.0)
                # Now wait for PS1 to get back to prompt
                self.process.expect(self._PS1, timeout=1.0)
                # Read and discard any remaining output
                _ = self.process.before
            except Exception:
                # If sync fails, try a simpler approach
                try:
                    self.process.sendline("")
                    self.process.expect(self._PS1, timeout=1.0)
                    _ = self.process.before
                except Exception:
                    pass

        # Send the command
        self.process.sendline(command)

        try:
            self.process.expect(self._PS1, timeout=timeout)
        except pexpect.TIMEOUT:
            # Try to interrupt the command
            self.process.sendintr()
            with contextlib.suppress(pexpect.TIMEOUT):
                self.process.expect(self._PS1, timeout=5)
            raise TimeoutError(f"Command timed out after {timeout}s: {command[:100]}") from None

        raw_output = self.process.before or ""
        output = _strip_control_chars(raw_output)

        # Critical: Clear the buffer immediately after reading to prevent pollution
        # This is essential for kubectl exec due to TTY buffering
        if self.process.isalive():
            try:
                # Read and discard all buffered data
                while True:
                    try:
                        chunk = self.process.read_nonblocking(timeout=0.05)
                        if not chunk:
                            break
                    except Exception:
                        break
            except Exception:
                pass

        # Debug: log raw output to see if commands are being truncated
        if len(output) > 0 and "printf" in output and "$?" not in output:
            self.logger.warning(f"Possible command truncation detected. Output preview: {output[:200]}")

        # For kubectl exec, both exit code extraction and output-based error detection
        # are unreliable due to TTY buffering. Output from previous commands can mix
        # with current commands, causing false positives. Therefore, we always return
        # exit_code=None and don't raise errors based on output patterns.
        # If command validation is needed, it should be done at the application level
        # by checking the return value/output explicitly.
        exit_code = None

        # Note: We don't use exit codes or output pattern matching in kubectl exec because:
        # 1. Exit code extraction is unreliable
        # 2. Output from previous commands mixes with current commands
        # 3. Pattern matching causes false positives from buffered output
        # If you need to verify a command succeeded, check the output content explicitly

        return BashObservation(output=output, exit_code=exit_code)

    async def interrupt(self, timeout: float = 5.0) -> BashObservation:
        """Interrupt the currently running command.

        Args:
            timeout: Timeout for interrupt operation

        Returns:
            BashObservation with output from interrupt
        """
        if self.process is None:
            raise RuntimeError(f"Session {self.session_id} not initialized")

        self.logger.info(f"Interrupting session {self.session_id}")

        output = ""
        # Send Ctrl+C
        self.process.sendintr()

        try:
            expect_index = self.process.expect([self._PS1, pexpect.TIMEOUT], timeout=timeout)
            if expect_index == 0:
                output = _strip_control_chars(self.process.before or "")
        except pexpect.TIMEOUT:
            # Try to send Ctrl+Z and kill the job
            self.process.sendcontrol("z")
            try:
                self.process.expect(self._PS1, timeout=timeout)
                output = _strip_control_chars(self.process.before or "")
                self.process.sendline("kill -9 %1")
                self.process.expect(self._PS1, timeout=timeout)
                output += _strip_control_chars(self.process.before or "")
            except pexpect.TIMEOUT:
                pass

        return BashObservation(output=output, exit_code=0)

    def close(self) -> None:
        """Close the bash session."""
        if self._closed:
            return

        self.logger.info(f"Closing bash session {self.session_id}")
        self._closed = True

        if self.process is not None:
            with contextlib.suppress(Exception):
                self.process.close()
            self.process = None


class K8sEnv(ContainerEnv):
    """Kubernetes Pod-based container environment using kubectl.

    Supports SWEEnv-compatible interface for running bash sessions in pods.
    """

    def __init__(
        self,
        pod_name: str,
        container_name: str,
        namespace: str,
        kubeconfig: str | None = None,
        context: str | None = None,
        default_cwd: str | None = None,
        default_env: dict[str, str] | None = None,
        default_forward_env: list[str] | None = None,
        repo_name: str | None = None,
        instance: dict | None = None,
        logger: logging.Logger = logging.root,
    ):
        self.pod_name = pod_name
        self.container_name = container_name
        self.namespace = namespace
        self.kubeconfig = kubeconfig
        self.context = context
        self.default_cwd = default_cwd
        self.default_env = default_env or {}
        self.default_forward_env = default_forward_env or []
        self.repo_name = repo_name
        self.logger = logger
        self._closed = False
        self.name = "main"  # todo: support param
        # FIFO-based persistent bash session
        self._fifo_initialized = False

        # Legacy session management (for execute, kept for compatibility)
        self._sessions: dict[str, K8sBashSession] = {}
        self._default_session: K8sBashSession | None = None
        self._session_counter = 0
        try:
            from pathlib import Path

            from sweagent.environment.repo import GithubRepoConfig, LocalRepoConfig, PreExistingRepoConfig

            if not self.repo_name:
                repo = None
            elif "github" in self.repo_name:
                repo = GithubRepoConfig(github_url=self.repo_name, base_commit=instance.get("base_commit", "HEAD"))
            elif "/" not in self.repo_name:
                repo = PreExistingRepoConfig(repo_name=self.repo_name, base_commit=instance.get("base_commit", "HEAD"))
            else:
                repo = LocalRepoConfig(path=Path(self.repo_name), base_commit=instance.get("base_commit", "HEAD"))
            self.repo = repo
        except ModuleNotFoundError:
            logger.info("Can't import SweAgent, Not Use repo in env")
            self.repo = None

    async def _init_fifo_session(self) -> None:
        """Verify HTTP server is ready in the pod."""
        if self._fifo_initialized:
            return

        self.logger.info("Verifying HTTP server for persistent bash session")

        # Verify that the HTTP server is running
        # The HTTP server is started by the pod's main command, we just need to verify it's ready
        verify_cmd = """
# Wait for HTTP server to be ready (with timeout)
for i in $(seq 1 60); do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo "HTTP server is ready"
        exit 0
    fi
    echo "Waiting for HTTP server to start..."
    sleep 0.5
done
echo "HTTP server failed to start"
exit 1
"""

        result = await self.execute(verify_cmd, timeout=60.0, check=False)

        if result.returncode != 0:
            error_output = result.output.decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP server not ready: {error_output}")

        self._fifo_initialized = True
        self.logger.debug("HTTP server verified and ready")

    def _get_kubectl_base_args(self) -> list[str]:
        """Get base kubectl arguments with kubeconfig and context."""
        args = []
        if self.kubeconfig:
            args.extend(["--kubeconfig", self.kubeconfig])
        if self.context:
            args.extend(["--context", self.context])
        return args

    def _merge_env(self, env: dict[str, str], forward_env: list[str]) -> dict[str, str]:
        """Merge environment variables with defaults."""
        merged_env = {}
        for key in self.default_forward_env:
            if key in os.environ:
                merged_env[key] = os.environ[key]
        merged_env.update(self.default_env)
        for key in forward_env:
            if key in os.environ:
                merged_env[key] = os.environ[key]
        # Filter out empty strings from env to avoid overriding container defaults
        for key, value in env.items():
            if value != "" and value is not None:
                merged_env[key] = value

        return merged_env

    def _build_shell_command(self, cmd: str, cwd: str | None, env: dict[str, str]) -> str:
        """Build a shell command with working directory and environment variables."""
        parts = []

        if effective_cwd := cwd or self.default_cwd:
            parts.append(f"cd {effective_cwd}")

        if env:
            for key, value in env.items():
                parts.append(f"export {key}={shlex.quote(str(value))}")

        parts.append(cmd)

        return "; ".join(parts)

    async def popen(
        self,
        cmd: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        forward_env: list[str] | None = None,
        timeout: float = 180.0,
    ) -> ContainerOutput:
        """Execute command and return combined output."""
        raise NotImplementedError

    async def execute(
        self,
        cmd: str,
        stdin: BinaryIO | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        forward_env: list[str] | None = None,
        timeout: float = 180.0,
        check=True,
    ) -> ContainerOutput:
        """Execute a command in the pod container. (noexcept)

        Args:
            cmd: Command to execute
            stdin: Input stream for the command
            cwd: Working directory (merged with default)
            env: Environment variables (merged with defaults)
            forward_env: Environment variables to forward (merged with defaults)
            timeout: Execution timeout in seconds

        Returns:
            ContainerOutput with combined output and return code

        Raises:
            RuntimeError: If container is closed
            TimeoutError: If execution exceeds timeout
            Exception: On execution failure
        """
        if self._closed:
            raise RuntimeError("Container environment is closed")

        if env is None:
            env = {}
        if forward_env is None:
            forward_env = []
        merged_env = self._merge_env(env, forward_env)
        shell_cmd = self._build_shell_command(cmd, cwd, merged_env)
        kubectl_args = self._get_kubectl_base_args()
        # Build base exec arguments
        exec_args = [
            "exec",
            "-n",
            self.namespace,
            self.pod_name,
            "-c",
            self.container_name,
        ]
        # Only add -i if stdin data is provided
        if stdin is not None:
            exec_args.append("-i")
        exec_args.extend(
            [
                "--",
                "/bin/sh",
                "-c",
                shell_cmd,
            ]
        )
        kubectl_args.extend(exec_args)

        try:
            self.logger.debug(f"Executing in pod {self.pod_name}: {shell_cmd[:100]}...")

            # Create subprocess with parent environment to ensure kubectl can find kubeconfig
            process = await asyncio.create_subprocess_exec(
                "kubectl",
                *kubectl_args,
                stdin=asyncio.subprocess.PIPE if stdin is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout
                # env=dict(os.environ),  # Not needed - inherits parent env by default
            )
            stdin_data = stdin and stdin.read()
            try:
                stdout, _ = await asyncio.wait_for(
                    process.communicate(input=stdin_data),
                    timeout=timeout,
                )
            except asyncio.TimeoutError as e:
                process.kill()
                await process.wait()
                raise TimeoutError(f"Command timed out after {timeout}s") from e

            returncode = process.returncode if process.returncode is not None else 127

            self.logger.debug(f"Command completed with exit code {returncode}, " f"output length: {len(stdout)}")

            if check and returncode != 0:
                raise CalledProcessError(returncode, shell_cmd, output=stdout)

            return ContainerOutput(output=stdout, returncode=returncode)

        except asyncio.TimeoutError as e:
            self.logger.error(f"Command execution timed out after {timeout}s: {cmd[:100]}")
            raise e
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            raise e

    async def copy(
        self,
        src: str,
        dst: str,
        upload: bool = True,
        cwd: str | None = None,
        timeout=180.0,
    ):
        """Copy files between host and container using kubectl cp.

        Args:
            src: Source path (host path if upload=True, container path otherwise)
            dst: Destination path (container path if upload=True, host path otherwise)
            upload: True to copy to container, False to copy from container
            cwd: Working directory for resolving relative paths

        Note:
            This implementation uses kubectl cp command.
        """
        if self._closed:
            raise RuntimeError("Container environment is closed")

        effective_cwd = cwd or self.default_cwd or "/"

        try:
            if upload:
                # Check if source file exists on host
                if not os.path.exists(src):
                    raise FileNotFoundError(f"Source file does not exist: {src}")

                # Resolve container destination path
                if not dst.startswith("/"):
                    dst = f"{effective_cwd}/{dst}"

                # Build kubectl cp command
                pod_path = f"{self.namespace}/{self.pod_name}:{dst}"
                kubectl_args = self._get_kubectl_base_args()
                # Add --request-timeout to prevent kubectl's internal context from timing out
                # IMPORTANT: This is a global flag and must come BEFORE the "cp" subcommand
                # Use 0 (infinite) to let Python's timeout handle it
                kubectl_args.extend(["--request-timeout=0"])
                kubectl_args.extend(["cp", src, pod_path, "-c", self.container_name])

                self.logger.debug(f"Uploading {src} to {pod_path}")

            else:
                # Resolve container source path
                if not src.startswith("/"):
                    src = f"{effective_cwd}/{src}"

                # Check if source file exists in container
                check_cmd = f"test -e {src}"
                result = await self.execute(check_cmd, timeout=10.0)
                if result.returncode != 0:
                    raise FileNotFoundError(f"Source file does not exist in container: {src}")

                # Build kubectl cp command
                pod_path = f"{self.namespace}/{self.pod_name}:{src}"
                kubectl_args = self._get_kubectl_base_args()
                # Add --request-timeout to prevent kubectl's internal context from timing out
                # IMPORTANT: This is a global flag and must come BEFORE the "cp" subcommand
                kubectl_args.extend(["--request-timeout=0"])
                kubectl_args.extend(["cp", pod_path, dst, "-c", self.container_name])

                self.logger.info(f"Downloading {pod_path} to {dst}")

            # Execute kubectl cp
            # Note: We're using --request-timeout=0 flag (already added to kubectl_args)
            # The env parameter is omitted to automatically inherit parent process environment
            # This ensures PATH, KUBECONFIG, and other necessary variables are preserved

            # Log the command for debugging
            cmd_str = "kubectl " + " ".join(kubectl_args)
            self.logger.debug(f"Executing kubectl cp command: {cmd_str}")

            process = await asyncio.create_subprocess_exec(
                "kubectl",
                *kubectl_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # env=dict(os.environ),  # Not needed - inherits parent env by default
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)

            returncode = process.returncode if process.returncode is not None else 0

            if returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace").strip()
                raise Exception(f"kubectl cp failed (exit code {returncode}): {error_msg}")

            self.logger.debug("Copy operation completed successfully")

        except asyncio.TimeoutError as e:
            self.logger.error(f"Copy operation timed out after {timeout}s")
            raise TimeoutError(f"Copy operation timed out after {timeout}s") from e
        except Exception as e:
            self.logger.error(f"Copy operation failed: {e}")
            raise

    async def _copy_with_retry(
        self,
        src: str,
        dst: str,
        upload: bool = True,
        cwd: str | None = None,
        timeout: float = 180.0,
        max_retries: int = 2,
    ):
        """Copy files with retry logic for handling transient K8s failures.

        Args:
            src: Source path
            dst: Destination path
            upload: True to copy to container
            cwd: Working directory
            timeout: Timeout for each attempt
            max_retries: Maximum number of retry attempts

        Raises:
            Exception: If all retry attempts fail
        """
        await _retry_async(
            self.logger,
            self.copy,
            src,
            dst,
            upload=upload,
            cwd=cwd,
            timeout=timeout,
            max_retries=max_retries,
            retryable_exceptions=(TimeoutError, Exception),
            operation_name=f"Copy ({src} -> {dst})",
        )

    async def _upload_via_exec(
        self,
        local_path: str,
        remote_path: str,
        timeout: float = 60.0,  # Reduced from 1800s to 60s
        max_retries: int = 3,  # Added retry support
        chunk_size: int = 1024 * 1024,  # 1MB chunks
    ) -> None:
        """Upload a file to the container using kubectl exec + base64 encoding.

        This method is a fallback when kubectl cp fails due to timeout issues.
        It reads the file in chunks, encodes each chunk as base64, and sends
        it through kubectl exec to be decoded and written on the remote side.

        Args:
            local_path: Path to the local file
            remote_path: Destination path in the container
            timeout: Timeout for each upload attempt (default: 60s)
            max_retries: Maximum number of retry attempts (default: 3)
            chunk_size: Size of chunks to read and transmit (default: 1MB)

        Raises:
            Exception: If upload fails after all retries
        """
        import base64

        async def _do_upload():
            """Internal function that performs the actual upload."""
            self.logger.debug(f"Uploading {local_path} to {remote_path} via exec+base64")

            file_size = os.path.getsize(local_path)
            self.logger.debug(f"File size: {file_size / (1024*1024):.1f}MB")

            # Read file and encode as base64
            with open(local_path, "rb") as f:
                file_data = f.read()

            # Encode entire file as base64 (for efficiency)
            encoded_data = base64.b64encode(file_data).decode("utf-8")
            encoded_size = len(encoded_data)
            self.logger.debug(f"Encoded size: {encoded_size / (1024*1024):.1f}MB")

            # Build kubectl exec command
            kubectl_args = self._get_kubectl_base_args()

            # Create the remote write command
            # Use base64 -d to decode and write directly
            remote_cmd = f"base64 -d > {shlex.quote(remote_path)}"

            kubectl_args.extend(
                ["exec", "-n", self.namespace, self.pod_name, "-c", self.container_name, "--", "/bin/bash", "-c", remote_cmd]
            )

            self.logger.debug(f"Exec command: kubectl {' '.join(kubectl_args)}")

            # Start kubectl exec process
            process = await asyncio.create_subprocess_exec(
                "kubectl",
                *kubectl_args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Write the base64 data to stdin
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(input=encoded_data.encode("utf-8")), timeout=timeout)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError(f"Upload via exec timed out after {timeout}s") from None

            returncode = process.returncode if process.returncode is not None else -1

            if returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace") if stderr else "Unknown error"
                raise Exception(f"Upload via exec failed (exit code {returncode}): {error_msg}")

            self.logger.info(f"Successfully uploaded {local_path} to {remote_path}")

        # Use retry wrapper for the upload operation
        await _retry_async(
            self.logger,
            _do_upload,
            max_retries=max_retries,
            retryable_exceptions=(TimeoutError, Exception),
            operation_name=f"Upload via exec ({local_path} -> {remote_path})",
        )

    # ======================================================================
    # Session Management (SWEEnv compatibility)
    # ======================================================================

    async def create_session(
        self,
        session_id: str = "default",
        startup_source: list[str] | None = None,
        startup_timeout: float = 30.0,
    ) -> K8sBashSession:
        """Create a new bash session or return an existing one.

        Args:
            session_id: Unique identifier for the session
            startup_source: List of files to source on startup (e.g., ["/root/.bashrc"])
            startup_timeout: Timeout for startup operations (default: 30s, increased for slow clusters)

        Returns:
            The created or existing K8sBashSession
        """
        if session_id in self._sessions:
            return self._sessions[session_id]

        if self._closed:
            raise RuntimeError("Container environment is closed")

        session = K8sBashSession(
            k8s_env=self,
            session_id=session_id,
            startup_source=startup_source or [],
            startup_timeout=startup_timeout,
            logger=self.logger,
        )
        await session.start()
        self._sessions[session_id] = session

        if session_id == "default":
            self._default_session = session

        return session

    async def get_default_session(self) -> K8sBashSession:
        """Get or create the default session.

        Returns:
            The default K8sBashSession
        """
        if self._default_session is None or self._default_session._closed:
            await self.create_session("default", startup_source=["/root/.bashrc"])
        return self._default_session

    # ======================================================================
    # SWEEnv Compatibility - Pure async interface
    # ======================================================================

    async def communicate(
        self,
        input: str,
        timeout: float = 25.0,
        *,
        check: Literal["warn", "ignore", "raise"] = "ignore",
        error_msg: str = "Command failed",
    ) -> str:
        """Execute a command using HTTP server for persistent bash session.

        Uses HTTP server running in the pod to execute commands in a persistent bash session.

        Args:
            input: Command to execute
            timeout: Timeout in seconds
            check: How to handle exit codes ("warn", "ignore", or "raise")
            error_msg: Error message to use if command fails

        Returns:
            Command output as a string
        """
        check = "warn"
        self.logger.debug(f"Communicate (HTTP): {input[:100]}...")

        # Ensure HTTP server is initialized
        if not self._fifo_initialized:
            await self._init_fifo_session()

        try:
            # Prepare request data and use base64 to avoid shell escaping issues
            import base64
            import json

            request_data = json.dumps({"cmd": input, "timeout": timeout})
            # Base64 encode the JSON to avoid any shell escaping issues with quotes
            encoded_data = base64.b64encode(request_data.encode()).decode()

            # Execute curl to send HTTP request to the server in the pod
            # Use base64 decoding in the shell to avoid quoting issues
            curl_cmd = (
                f"echo '{encoded_data}' | base64 -d | "
                f"curl -s -X POST http://localhost:8080/execute "
                f"-H 'Content-Type: application/json' "
                f"--data-binary @-"
            )

            result = await self.execute(curl_cmd, timeout=timeout + 5, check=False)

            if result.returncode != 0:
                raise RuntimeError(f"HTTP request failed: {result.output.decode()}")

            # Parse JSON response
            try:
                response = json.loads(result.output.decode())
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse server response: {result.output.decode()}") from e

            output = response.get("output", "")
            error = response.get("error", "")
            returncode = response.get("returncode", 0)

            # Combine stdout and stderr
            full_output = output
            if error:
                full_output += "\n" + error if output else error

            # Handle exit code
            if check != "ignore" and returncode != 0:
                self.logger.error(f"{error_msg}:\n{full_output}")
                if check == "raise" and not self._closed:
                    raise RuntimeError(error_msg)

            return full_output

        except Exception as e:
            if check == "raise" and not self._closed:
                raise RuntimeError(f"{error_msg}: {e}") from e
            if check == "warn":
                self.logger.error(f"{error_msg}: {e}")
            return str(e)

    async def interrupt_session(self) -> None:
        """Interrupt the currently running command.

        Note: HTTP server doesn't support interrupting running commands,
        so this is a no-op for now.
        """
        self.logger.debug("Interrupt requested (not supported with HTTP server)")
        # TODO: Implement interrupt via HTTP server if needed
        return

    async def read_file(
        self,
        path: str | PurePath,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        """Read file contents from container (SWEEnv-compatible).

        Args:
            path: Path to the file in the container
            encoding: Encoding to use (default: utf-8)
            errors: Error handling strategy (default: strict)

        Returns:
            File contents as a string
        """
        self.logger.info(f"[read_file] Starting to read file: {path}")
        if self._closed:
            raise RuntimeError("Container environment is closed")

        # Use cat command to read the file
        cmd = f"cat {shlex.quote(str(path))}"
        self.logger.info(f"[read_file] Executing command: {cmd}")
        result = await self.execute(cmd, timeout=30.0, check=False)
        self.logger.info(f"[read_file] Command completed, returncode={result.returncode}, output_length={len(result.output)}")

        if result.returncode != 0:
            error_output = result.output.decode(errors="replace")
            self.logger.error(f"[read_file] File read failed: {error_output}")
            raise FileNotFoundError(f"Failed to read file {path}: {error_output}")

        decoded = result.output.decode(encoding or "utf-8", errors=errors or "strict")
        self.logger.info(f"[read_file] Successfully decoded {len(decoded)} characters")
        return decoded

    async def write_file(self, path: str | PurePath, content: str) -> None:
        """Write content to a file in the container (SWEEnv-compatible).

        Args:
            path: Path to the file in the container
            content: Content to write
        """
        if self._closed:
            raise RuntimeError("Container environment is closed")

        # Write to a temporary file first, then use kubectl cp
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix="_k8s_write") as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()
            local_path = tmp_file.name

        try:
            await self.copy(local_path, str(path), upload=True)
        finally:
            os.unlink(local_path)

    async def execute_command(
        self,
        command: str,
        shell: bool = True,
        check: bool = False,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        """Execute a command independent of the session (SWEEnv-compatible).

        This runs the command as a one-off subprocess, not in the persistent bash session.

        Args:
            command: Command to execute
            shell: Whether to use shell
            check: Whether to raise on non-zero exit code
            env: Environment variables
            cwd: Working directory
        """
        if self._closed:
            raise RuntimeError("Container environment is closed")

        await self.execute(command, cwd=cwd, env=env, check=check)

    async def set_env_variables(self, env_variables: dict[str, str], timeout: float = 90.0) -> None:
        """Set environment variables in the current session (SWEEnv-compatible).

        Args:
            env_variables: Dictionary of environment variables to set
            timeout: Timeout in seconds for the command execution (default: 90.0)
        """
        if not env_variables:
            self.logger.debug("No environment variables to set")
            return

        # Filter out locale variables since they're already configured in the HTTP server
        # This prevents corruption through the environment persistence mechanism
        filtered_vars = {k: v for k, v in env_variables.items() if k not in ("LANG", "LC_ALL")}

        if not filtered_vars:
            self.logger.debug("No non-locale environment variables to set")
            return

        _env_setters = [f"export {k}={shlex.quote(str(v))}" for k, v in filtered_vars.items()]
        command = " && ".join(_env_setters)

        await self.communicate(command, timeout=timeout, check="raise")

    async def upload(
        self,
        source_path: str,
        target_path: str,
    ) -> None:
        """Upload a file or directory from local machine to container (SWE-ReX compatible).

        Args:
            source_path: Path to the local file or directory to upload
            target_path: Target path in the container
        """
        if self._closed:
            raise RuntimeError("Container environment is closed")

        import shutil
        from pathlib import Path

        source = Path(source_path).resolve()
        self.logger.debug(f"Uploading {source_path} to {target_path}")

        if source.is_dir():
            # For directories, create a zip file, upload, then extract (SWE-ReX compatible)
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_name = source.name
                zip_path = os.path.join(temp_dir, f"{zip_name}.zip")
                shutil.make_archive(zip_path.replace(".zip", ""), "zip", root_dir=str(source.parent), base_dir=source.name)

                # Get zip size for timeout calculation
                zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
                # Use 30 minutes base timeout, plus 1 minute per 10MB
                # This is much more conservative for slow K8s clusters
                copy_timeout = max(1800.0, zip_size_mb * 60)
                self.logger.debug(
                    f"Uploading zip ({zip_size_mb:.1f}MB) to {target_path} with timeout {copy_timeout:.0f}s ({copy_timeout/60:.1f}min)"
                )

                # Upload zip to /tmp with increased timeout and retry
                try:
                    await self._copy_with_retry(zip_path, "/tmp/", upload=True, timeout=copy_timeout, max_retries=2)
                except Exception as e:
                    self.logger.warning(f"kubectl cp failed: {e}, falling back to exec+base64 method")
                    await self._upload_via_exec(zip_path, f"/tmp/{zip_name}.zip", timeout=copy_timeout)

                # Extract in container to target_path using Python zipfile
                # Create parent directory and extract to target_path (like SWE-ReX server.py does)
                target_parent = str(Path(target_path).parent)
                extract_cmd = (
                    f"mkdir -p {target_parent} && "
                    f"python3 -c \"import zipfile; zipfile.ZipFile('/tmp/{zip_name}.zip').extractall('{target_parent}')\" && "
                    f"rm /tmp/{zip_name}.zip"
                )
                self.logger.debug(f"Extract command: {extract_cmd[:200]}...")
                result = await self.execute(extract_cmd, timeout=600.0, check=False)
                if result.returncode != 0:
                    raise RuntimeError(f"Failed to extract uploaded directory: {result.output.decode()}")

                # Verify extraction succeeded
                # verify_cmd = f"ls -la {target_path} 2>&1 || echo 'EXTRACT_VERIFY_FAILED'"
                # verify_result = await self.execute(verify_cmd, timeout=10.0, check=False)
                # self.logger.info(f"[Upload verification] After extract for {target_path}: {verify_result.output.decode()[:500]}")
                # if verify_result.returncode != 0 or b'EXTRACT_VERIFY_FAILED' in verify_result.output:
                #     raise RuntimeError(f"Extraction verification failed for {target_path}: {verify_result.output.decode()}")
        else:
            # For single files, use kubectl cp with exec+base64 fallback
            try:
                await self.copy(str(source), target_path, upload=True, timeout=300.0)
            except Exception as e:
                self.logger.warning(f"kubectl cp for single file failed: {e}, falling back to exec+base64")
                await self._upload_via_exec(str(source), target_path, timeout=300.0)

        # Change ownership to root if uploading to /
        if target_path.startswith("/") and target_path[1:-1].strip("/") == "/":
            # Check if we need to change ownership
            top_dir = target_path.strip("/").split("/")[0] if "/" in target_path.strip("/") else "."
            if top_dir and top_dir != ".":
                chown_cmd = f"chown -R root:root {target_path.split('/')[0] if '/' in target_path else '.'}"
                await self.execute(chown_cmd, timeout=30.0, check=False)

    # ======================================================================
    # Repo Management (SWEEnv compatibility)
    # ======================================================================

    async def _copy_repo(self) -> None:
        """Clone/copy repository/codebase in container (SWEEnv-compatible)."""
        if self.repo is None:
            return

        # Check if repo already exists
        result = await self.execute("ls", timeout=10.0, check=False)
        existing_folders = result.output.decode().split()
        if hasattr(self.repo, "repo_name") and self.repo.repo_name in existing_folders:
            self.logger.info(f"Repository {self.repo.repo_name} already exists in container")
            return

        self.logger.info("Copying repository to container")

        # Handle different repo types
        if hasattr(self.repo, "copy"):
            # For SWE-agent repo configs, use their copy method
            # We need to adapt this to work with kubectl
            if hasattr(self.repo, "type") and self.repo.type == "github":
                # Clone GitHub repo
                await self._clone_github_repo(self.repo)
            elif hasattr(self.repo, "type") and self.repo.type == "local":
                # Upload local repo
                await self._upload_local_repo(self.repo)
            elif hasattr(self.repo, "type") and self.repo.type == "preexisting":
                # Pre-existing repo, skip copying
                pass
        else:
            self.logger.warning(f"Unknown repo type: {type(self.repo)}")

    async def _clone_github_repo(self, repo) -> None:
        """Clone a GitHub repository into the container."""
        github_url = repo.github_url
        base_commit = repo.base_commit if hasattr(repo, "base_commit") else "HEAD"
        repo_name = repo.repo_name

        # Get GitHub token if available
        github_token = os.getenv("GITHUB_TOKEN", "")
        url = github_url
        if github_token and "@" not in url:
            _, _, url_no_protocol = url.partition("://")
            url = f"https://{github_token}@{url_no_protocol}"

        clone_timeout = getattr(repo, "clone_timeout", 500.0)

        clone_cmd = " && ".join(
            [
                f"mkdir /{repo_name}",
                f"cd /{repo_name}",
                "git init",
                f"git remote add origin {shlex.quote(url)}",
                f"git fetch --depth 1 origin {shlex.quote(base_commit)}",
                "git checkout FETCH_HEAD",
                "cd ..",
            ]
        )

        result = await self.execute(clone_cmd, timeout=clone_timeout, check=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to clone GitHub repo: {result.output.decode()}")

    async def _upload_local_repo(self, repo) -> None:
        """Upload a local repository to the container using kubectl cp."""
        import shutil

        local_path = str(repo.path)
        repo_name = repo.repo_name

        # Create a temporary tarball
        with tempfile.TemporaryDirectory() as tmp_dir:
            tar_path = os.path.join(tmp_dir, f"{repo_name}.tar.gz")
            shutil.make_archive(
                os.path.join(tmp_dir, repo_name),
                "gztar",
                root_dir=os.path.dirname(local_path),
                base_dir=os.path.basename(local_path),
            )

            # Copy tarball to container
            await self.copy(tar_path, f"/tmp/{repo_name}.tar.gz", upload=True)

            # Extract in container
            extract_cmd = f"cd / && tar xzf /tmp/{repo_name}.tar.gz && rm /tmp/{repo_name}.tar.gz"
            result = await self.execute(extract_cmd, timeout=180.0, check=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to extract local repo: {result.output.decode()}")

        # Change ownership to root
        chown_cmd = f"chown -R root:root /{repo_name}"
        result = await self.execute(chown_cmd, timeout=30.0, check=False)
        if result.returncode != 0:
            self.logger.warning(f"Failed to change ownership: {result.output.decode()}")

    async def _reset_repository(self) -> None:
        """Reset repository to base commit (SWEEnv-compatible)."""
        if self.repo is None:
            return

        if not hasattr(self.repo, "repo_name"):
            return

        base_commit = self.repo.base_commit if hasattr(self.repo, "base_commit") else "HEAD"
        repo_name = self.repo.repo_name

        self.logger.info(f"Resetting repository {repo_name} to commit {base_commit}")

        reset_commands = [
            f"cd /{repo_name}",
            "export ROOT=$(pwd -P)",
            "git fetch",
            "git status",
            "git restore .",
            "git reset --hard",
            f"git checkout {shlex.quote(base_commit)}",
            "git clean -fdq",
        ]

        # Skip reset if repo is preexisting and reset is False
        if hasattr(self.repo, "type") and self.repo.type == "preexisting" and hasattr(self.repo, "reset") and not self.repo.reset:
            return

        await self.communicate(" && ".join(reset_commands), check="raise")

    async def reset(self) -> None:
        """Reset the environment to a clean state (SWEEnv-compatible)."""
        self.logger.info("Resetting environment")

        # Navigate to root
        await self.communicate("cd /", check="raise")

        # Copy/re-setup repository
        await self._copy_repo()

        # Reset repository state
        await self._reset_repository()

    # ======================================================================
    # Cleanup
    # ======================================================================

    async def cleanup(self):
        """Delete the pod and clean up resources."""
        if self._closed:
            return

        # Kill HTTP server process in pod
        if self._fifo_initialized:
            try:
                self.logger.info("Stopping HTTP server")
                await self.execute("pkill -f 'http.server' || true", timeout=10.0, check=False)
                self._fifo_initialized = False
            except Exception as e:
                self.logger.warning(f"Error stopping HTTP server: {e}")

        # Close all sessions first
        for session in list(self._sessions.values()):
            try:
                session.close()
            except Exception as e:
                self.logger.warning(f"Error closing session: {e}")
        self._sessions.clear()
        self._default_session = None

        try:
            self.logger.info(f"Cleaning up pod {self.pod_name}")

            kubectl_args = self._get_kubectl_base_args()
            kubectl_args.extend(
                [
                    "delete",
                    "pod",
                    "-n",
                    self.namespace,
                    self.pod_name,
                    "--wait=false",
                ]
            )

            process = await asyncio.create_subprocess_exec(
                "kubectl",
                *kubectl_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # env=dict(os.environ),  # Not needed - inherits parent env by default
            )

            await asyncio.wait_for(process.communicate(), timeout=30.0)
            self.logger.info(f"Pod {self.pod_name} deleted")

        except Exception as e:
            self.logger.error(f"Failed to delete pod {self.pod_name}: {e}")
        finally:
            self._closed = True

    @property
    def alive(self) -> bool:
        """Check whether the pod is still running."""
        return not self._closed


class K8sEnvConfig(BaseModel):
    """Configuration for Kubernetes environment."""

    namespace: str = Field(default="default", description="Kubernetes namespace")
    kubeconfig: str | None = Field(default=None, description="Path to kubeconfig file")
    context: str | None = Field(default=None, description="Kubernetes context to use")
    pod_name_prefix: str = Field(default="agentflow", description="Prefix for generated pod names")
    repo_name: str | None = Field(default=None, description="Repo name in Env")


class K8sEnvBuilder(ContainerEnvBuilder):
    """Builder for creating Kubernetes-based container environments."""

    def __init__(self, conf: dict):
        """Initialize the K8s environment builder.

        Args:
            conf: Configuration dictionary with optional fields:
                - namespace: Kubernetes namespace (default: "default")
                - kubeconfig: Path to kubeconfig file (optional)
                - context: Kubernetes context to use (optional)
                - pod_name_prefix: Prefix for generated pod names (default: "agentflow")
        """
        self.config = K8sEnvConfig(**conf)

        self.namespace = self.config.namespace
        self.kubeconfig = self.config.kubeconfig
        self.context = self.config.context
        self.pod_name_prefix = self.config.pod_name_prefix
        self.logger = logging.Logger("SWE K8s EnvBuilder")

    def _get_kubectl_base_args(self) -> list[str]:
        """Get base kubectl arguments with kubeconfig and context."""
        args = []
        if self.kubeconfig:
            args.extend(["--kubeconfig", self.kubeconfig])
        if self.context:
            args.extend(["--context", self.context])
        return args

    def _is_retryable_error(self, stderr: bytes, returncode: int) -> bool:
        """Check if the error is retryable (e.g., network errors)."""
        if returncode == 0:
            return False

        stderr_str = stderr.decode("utf-8", errors="replace").lower()
        retryable_patterns = [
            "client connection lost",
            "connection reset",
            "connection refused",
            "timeout",
            "network",
            "temporary failure",
        ]
        return any(pattern in stderr_str for pattern in retryable_patterns)

    async def _run_with_retry(
        self,
        kubectl_args: list[str],
        stdin_data: bytes | None = None,
        timeout: float = 10.0,
        max_retries: int = 3,
    ) -> tuple[bytes, bytes, int]:
        """Run kubectl command with automatic retry on timeout and network errors.

        Args:
            kubectl_args: Arguments to pass to kubectl
            stdin_data: Optional data to pass to stdin
            timeout: Timeout for each attempt in seconds
            max_retries: Maximum number of retry attempts

        Returns:
            Tuple of (stdout, stderr, return_code)

        Raises:
            Exception: If all retries fail
        """

        async def _do_run():
            """Internal function that executes kubectl command once."""
            process = await asyncio.create_subprocess_exec(
                "kubectl",
                *kubectl_args,
                stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=stdin_data),
                timeout=timeout,
            )

            returncode = process.returncode if process.returncode is not None else -1

            # Check if this is a retryable error
            if self._is_retryable_error(stderr, returncode):
                # Raise an exception to trigger retry
                error_msg = stderr.decode("utf-8", errors="replace")
                raise Exception(f"kubectl failed (retryable): {error_msg}")

            return stdout, stderr, returncode

        async def _is_retryable_kubectl_error(e: Exception) -> bool:
            """Check if a kubectl error is retryable based on stderr content."""
            # For asyncio.TimeoutError, always retry
            if isinstance(e, asyncio.TimeoutError):
                return True
            # For other exceptions, check if error message contains retryable patterns
            error_str = str(e).lower()
            retryable_patterns = [
                "timeout",
                "connection",
                "network",
                "temporary failure",
            ]
            return any(pattern in error_str for pattern in retryable_patterns)

        # Use retry wrapper for kubectl execution
        return await _retry_async(
            self.logger,
            _do_run,
            max_retries=max_retries,
            retryable_exceptions=(asyncio.TimeoutError, Exception),
            is_retryable_fn=_is_retryable_kubectl_error,
            operation_name=f"kubectl {' '.join(kubectl_args[:3])}",
        )

    async def build(self, args: ContainerBuildArgs):
        raise NotImplementedError("K8s does not support container building")

    async def start(self, args: ContainerStartArgs, instance: dict = None) -> K8sEnv:
        """Start a Kubernetes pod from the given configuration.

        Args:
            args: Container start arguments
            instance: instance info, like base-commit
        Returns:
            K8sEnv instance for executing commands

        Raises:
            TimeoutError: Pod failed to become ready within startup_timeout
            Exception: Pod creation or startup failed
        """
        # Generate unique pod name
        pod_name = f"{self.pod_name_prefix}-{uuid.uuid4().hex[:8]}"

        # Merge environment variables
        env_vars = []

        # Add forwarded environment variables
        for env_key in args.forward_env:
            if env_key in os.environ:
                env_vars.append({"name": env_key, "value": os.environ[env_key]})

        # Add explicit environment variables (these override forwarded ones)
        for key, value in args.env.items():
            # Remove any existing entry with same name
            env_vars = [e for e in env_vars if e["name"] != key]
            env_vars.append({"name": key, "value": value})

        # Add locale variables to avoid setlocale warnings (set at container level to avoid escaping issues)
        locale_vars = {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
        for key, value in locale_vars.items():
            # Only add if not already specified in args.env
            if not any(e["name"] == key for e in env_vars):
                env_vars.append({"name": key, "value": value})

        # Build container spec with HTTP server for persistent bash session
        # Create the Python script file inside the container at startup
        http_server_python = """import http.server
import socketserver
import subprocess
import json
import os
import sys

class CommandHandler(http.server.BaseHTTPRequestHandler):
    # Initialize environment with locale settings to prevent setlocale warnings
    env = os.environ.copy()
    env['LANG'] = 'C.UTF-8'
    env['LC_ALL'] = 'C.UTF-8'
    cwd = os.getcwd()

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
        elif self.path == '/ping':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'ok', 'cwd': self.cwd}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/execute':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data)
                cmd = data.get('cmd', '')
                timeout = data.get('timeout', 300)
                env_file = f'/tmp/env_update_{os.getpid()}'
                # Use Python to serialize environment to JSON instead of parsing export -p
                shell_script = '''#!/bin/bash
''' + cmd + '''
exit_code=$?
''' + f'''python3 -c "import json, os; json.dump(dict(os.environ), open({repr(env_file)}, 'w'))"
exit $exit_code
'''
                result = subprocess.run(
                    ['/bin/bash'],
                    input=shell_script,
                    cwd=self.cwd,
                    env=self.env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                try:
                    with open(env_file, 'r') as ef:
                        new_env = json.load(ef)
                        # Update environment, but preserve locale variables
                        for key, value in new_env.items():
                            if key not in ('LANG', 'LC_ALL'):
                                self.env[key] = value
                    os.remove(env_file)
                except:
                    pass
                try:
                    pwd_result = subprocess.run('pwd', shell=True, cwd=self.cwd, env=self.env, capture_output=True, text=True, timeout=1)
                    if pwd_result.returncode == 0:
                        self.cwd = pwd_result.stdout.strip()
                except:
                    pass
                response = {'output': result.stdout, 'error': result.stderr, 'returncode': result.returncode}
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                import traceback
                error_msg = str(e)
                if 'timeout' in str(type(e).__name__).lower():
                    response = {'output': '', 'error': f'Command timed out', 'returncode': -1}
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode())
                else:
                    response = {'output': '', 'error': error_msg, 'returncode': -1}
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

try:
    # Create a custom TCPServer with allow_reuse_address to avoid "port already in use" errors
    class ReuseAddressTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReuseAddressTCPServer(("", 8080), CommandHandler) as httpd:
        print(f"HTTP server started on port 8080", file=sys.stderr)
        httpd.serve_forever()
except Exception as e:
    print(f"HTTP server error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""

        # Escape the Python code for shell heredoc
        # Use base64 encoding to safely pass the Python code
        import base64

        python_code_b64 = base64.b64encode(http_server_python.encode()).decode()

        # Startup script: decode base64 and run Python
        startup_script = f"echo {python_code_b64} | base64 -d | python3 & wait"

        container_spec = {
            "name": "main",
            "image": args.image,
            "env": env_vars,
            "command": ["/bin/sh", "-c", startup_script],
        }

        if args.cwd:
            container_spec["workingDir"] = args.cwd

        # Add resource requests and limits
        resources = {}
        if args.resource_requests:
            resources["requests"] = args.resource_requests
        if args.resource_limits:
            resources["limits"] = args.resource_limits
        if resources:
            container_spec["resources"] = resources

        # Create pod specification
        pod_spec = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": pod_name,
                "namespace": self.namespace,
                "labels": {
                    "app": "agentflow",
                    "component": "swe-environment",
                },
            },
            "spec": {
                "containers": [container_spec],
                "restartPolicy": "Never",
            },
        }

        try:
            # Create the pod using kubectl apply
            self.logger.info(f"Creating pod {pod_name} in namespace {self.namespace}")

            kubectl_args = self._get_kubectl_base_args()
            kubectl_args.extend(["apply", "--validate=false", "-f", "-"])

            pod_json = json.dumps(pod_spec).encode()
            stdout, stderr, returncode = await self._run_with_retry(
                kubectl_args,
                stdin_data=pod_json,
                timeout=60.0,
                max_retries=3,
            )

            if returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace")
                raise Exception(f"Failed to create pod: {error_msg}")

            # Wait for pod to be ready
            self.logger.info(f"Waiting for pod {pod_name} to be ready (timeout: {args.startup_timeout}s)")
            start_time = time.monotonic()

            while True:
                # Check timeout
                elapsed = time.monotonic() - start_time
                if elapsed > args.startup_timeout:
                    await self._delete_pod(pod_name)
                    raise TimeoutError(f"Pod {pod_name} failed to become ready within {args.startup_timeout}s")

                # Get pod status
                kubectl_args = self._get_kubectl_base_args()
                kubectl_args.extend(
                    [
                        "get",
                        "pod",
                        "-n",
                        self.namespace,
                        pod_name,
                        "-o",
                        "json",
                    ]
                )

                stdout, stderr, returncode = await self._run_with_retry(
                    kubectl_args,
                    timeout=60,
                    max_retries=3,
                )

                if returncode != 0:
                    await self._delete_pod(pod_name)
                    error_msg = stderr.decode("utf-8", errors="replace")
                    raise Exception(f"Failed to get pod status: {error_msg}")

                pod_status = json.loads(stdout.decode())
                phase = pod_status.get("status", {}).get("phase", "")

                if phase == "Running":
                    # Check if container is ready
                    container_statuses = pod_status.get("status", {}).get("containerStatuses", [])
                    if container_statuses and container_statuses[0].get("ready", False):
                        self.logger.info(f"Pod {pod_name} is ready")
                        break
                elif phase in ["Failed", "Succeeded"]:
                    # Get pod logs before deleting to see why it failed
                    self.logger.error(f"Pod {pod_name} entered phase {phase} before becoming ready")
                    self.logger.error(f"Fetching container logs for pod {pod_name}...")
                    try:
                        kubectl_args = self._get_kubectl_base_args()
                        kubectl_args.extend(["logs", "-n", self.namespace, pod_name, "--tail", "100"])
                        stdout, stderr, returncode = await self._run_with_retry(kubectl_args, timeout=10.0, max_retries=1)
                        if returncode == 0:
                            self.logger.error(f"Container logs:\n{stdout.decode('utf-8', errors='replace')}")
                        else:
                            self.logger.error(f"Failed to get logs: {stderr.decode('utf-8', errors='replace')}")
                    except Exception as log_err:
                        self.logger.error(f"Exception while fetching logs: {log_err}")
                    await self._delete_pod(pod_name)
                    raise Exception(f"Pod {pod_name} entered phase {phase} before becoming ready")

                # Wait before checking again
                await asyncio.sleep(1)

            # Create K8sEnv instance
            k8s_env = K8sEnv(
                pod_name=pod_name,
                container_name="main",
                namespace=self.namespace,
                kubeconfig=self.kubeconfig,
                context=self.context,
                default_cwd=args.cwd,
                default_env=args.env,
                default_forward_env=args.forward_env,
                repo_name=self.config.repo_name,
                instance=instance,
            )

            # Initialize FIFO-based persistent bash session
            # This will be done lazily on first communicate() call

            # Note: Locale variables (LANG, LC_ALL) are now set directly in container spec
            # to avoid shell escaping issues through the HTTP server

            return k8s_env

        except Exception as e:
            self.logger.error(f"Failed to start pod {pod_name}: {e}")
            # Try to clean up if pod was created
            with contextlib.suppress(Exception):
                await self._delete_pod(pod_name)
            raise

    async def _delete_pod(self, pod_name: str):
        """Delete a pod by name."""
        kubectl_args = self._get_kubectl_base_args()
        kubectl_args.extend(
            [
                "delete",
                "pod",
                "-n",
                self.namespace,
                pod_name,
                "--wait=false",
            ]
        )

        process = await asyncio.create_subprocess_exec(
            "kubectl",
            *kubectl_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=dict(os.environ),  # Inherit parent environment
        )

        await asyncio.wait_for(process.communicate(), timeout=30.0)
