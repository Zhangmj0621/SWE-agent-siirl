import asyncio
import logging
import os
import time
from typing import Optional, BinaryIO

import kr8s
from kr8s.asyncio.objects import Pod
from kr8s._exec import Exec

from .base import ContainerEnv, ContainerEnvBuilder, ContainerStartArgs, ContainerOutput

logger = logging.getLogger(__name__)


class K8rsEnv(ContainerEnv):
    """Kubernetes Pod-based container environment."""

    def __init__(
        self,
        pod: Pod,
        container_name: str,
        namespace: str,
        kubeconfig: Optional[str] = None,
        context: Optional[str] = None,
        default_cwd: Optional[str] = None,
        default_env: Optional[dict[str, str]] = None,
        default_forward_env: Optional[list[str]] = None,
    ):
        self.pod = pod
        self.container_name = container_name
        self.namespace = namespace
        self.kubeconfig = kubeconfig
        self.context = context
        self.default_cwd = default_cwd
        self.default_env = default_env or {}
        self.default_forward_env = default_forward_env or []
        self._closed = False

    def _merge_env(self, env: dict[str, str], forward_env: list[str]) -> dict[str, str]:
        """Merge environment variables with defaults."""
        merged_env = {}

        # Start with default forwarded env
        for key in self.default_forward_env:
            if key in os.environ:
                merged_env[key] = os.environ[key]

        # Add default env (overrides forwarded)
        merged_env.update(self.default_env)

        # Add runtime forwarded env (overrides defaults)
        for key in forward_env:
            if key in os.environ:
                merged_env[key] = os.environ[key]

        # Add runtime env (highest priority)
        merged_env.update(env)

        return merged_env

    def _build_shell_command(
        self, cmd: str, cwd: Optional[str], env: dict[str, str]
    ) -> str:
        """Build a shell command with working directory and environment variables."""
        parts = []

        # Change directory if needed
        effective_cwd = cwd or self.default_cwd
        if effective_cwd:
            parts.append(f"cd {effective_cwd}")

        # Export environment variables
        if env:
            for key, value in env.items():
                # Escape single quotes in value
                escaped_value = value.replace("'", "'\"'\"'")
                parts.append(f"export {key}='{escaped_value}'")

        # Add the actual command
        parts.append(cmd)

        return " && ".join(parts)

    async def popen(
        self,
        cmd: str,
        cwd: Optional[str] = None,
        env: dict[str, str] = {},
        forward_env: list[str] = [],
        timeout: float = 180.0,
    ) -> ContainerOutput:
        """Execute command and return combined output.

        Note: This is a simplified version. For true popen-like behavior with
        streaming stdin/stdout/stderr, we would need a more complex implementation.
        """
        return await self.execute(
            cmd=cmd,
            stdin=None,
            cwd=cwd,
            env=env,
            forward_env=forward_env,
            timeout=timeout,
        )

    async def _exec_with_stdin_support(
        self,
        command: list[str],
        stdin: Optional[BinaryIO] = None,
        check: bool = False,
    ):
        """Execute command with v5 protocol support for stdin.

        This is a custom implementation that forces v5 protocol when stdin is provided.
        """
        # Create custom Exec instance
        exec_obj = Exec(
            resource=self.pod,
            command=command,
            container=self.container_name,
            stdin=stdin,
            check=check,
            capture_output=True,
        )

        # Override the protocol if stdin is provided
        if stdin is not None:
            # Force v5 protocol for stdin support
            async with self.pod.api.open_websocket(  # type: ignore
                version=self.pod.version,
                url=f"{self.pod.endpoint}/{self.pod.name}/exec",
                subprotocols=("v5.channel.k8s.io",),  # Use v5 instead of v4
                namespace=self.pod.namespace,
                params={
                    "command": command,
                    "container": self.container_name,
                    "stdout": "true",
                    "stderr": "true",
                    "stdin": "true",
                },
            ) as ws:
                # Send stdin data
                if isinstance(stdin, str):
                    stdin_data = stdin.encode()
                else:
                    stdin_data = stdin.read()

                # Send stdin with channel prefix
                await ws.send_bytes(b"\x00" + stdin_data)
                # Close stdin channel
                await ws.send_bytes(b"\xff\x00")

                stdout = b""
                stderr = b""
                returncode = 0

                # Read output
                while True:
                    message = await ws.receive_bytes()
                    channel = message[0]
                    data = message[1:]

                    if channel == 1:  # STDOUT
                        stdout += data
                    elif channel == 2:  # STDERR
                        stderr += data
                    elif channel == 3:  # ERROR
                        import json

                        error = json.loads(data.decode())
                        if error.get("status") == "Success":
                            returncode = 0
                            break
                        # Extract return code
                        if "details" in error and "causes" in error["details"]:
                            for cause in error["details"]["causes"]:
                                if cause.get("reason") == "ExitCode":
                                    returncode = int(cause["message"])
                                    break
                        else:
                            returncode = 1
                        break

                from kr8s._exec import CompletedExec

                return CompletedExec(
                    args=command,
                    stdout=stdout,
                    stderr=stderr,
                    returncode=returncode,
                )
        else:
            # Use default exec for commands without stdin
            async with exec_obj.run():
                await exec_obj.wait()
            return exec_obj.as_completed()

    async def execute(
        self,
        cmd: str,
        stdin: Optional[BinaryIO] = None,
        cwd: Optional[str] = None,
        env: dict[str, str] = {},
        forward_env: list[str] = [],
        timeout: float = 180.0,
        check: bool = True,
    ) -> ContainerOutput:
        """Execute a command in the pod container.

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

        # Merge environment variables
        merged_env = self._merge_env(env, forward_env)

        # Build the shell command
        shell_cmd = self._build_shell_command(cmd, cwd, merged_env)

        try:
            # Execute command with timeout
            logger.debug(f"Executing in pod {self.pod.name}: {shell_cmd[:100]}...")

            if stdin is not None:
                exec_result = await asyncio.wait_for(
                    self._exec_with_stdin_support(
                        ["/bin/sh", "-c", shell_cmd],
                        stdin=stdin,
                        check=check,
                    ),
                    timeout=timeout,
                )
            else:
                exec_result = await asyncio.wait_for(
                    self.pod.exec(
                        ["/bin/sh", "-c", shell_cmd],
                        container=self.container_name,
                        check=check,
                    ),
                    timeout=timeout,
                )

            # kr8s exec returns a CompletedExec object with stdout, stderr, and returncode
            stdout = exec_result.stdout if hasattr(exec_result, "stdout") else b""
            stderr = exec_result.stderr if hasattr(exec_result, "stderr") else b""
            returncode = (
                exec_result.returncode if hasattr(exec_result, "returncode") else 0
            )

            # Ensure bytes
            # if isinstance(stdout, str):
            #     stdout = stdout.encode()
            # if isinstance(stderr, str):
            #     stderr = stderr.encode()

            # Combine stdout and stderr
            combined_output = stdout + stderr

            logger.debug(
                f"Command completed with exit code {returncode}, "
                f"output length: {len(combined_output)}"
            )

            return ContainerOutput(output=combined_output, returncode=returncode)

        except asyncio.TimeoutError:
            logger.error(f"Command execution timed out after {timeout}s: {cmd[:100]}")
            raise TimeoutError(f"Command timed out after {timeout}s")
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise

    async def _run_kubectl(
        self, args: list[str], timeout: float = 60.0
    ) -> tuple[bytes, bytes, int]:
        """Run kubectl command and return stdout, stderr, returncode.

        Args:
            args: kubectl command arguments (without 'kubectl' itself)
            timeout: Command timeout in seconds (default: 60.0)

        Returns:
            Tuple of (stdout, stderr, returncode)
        """
        cmd = ["kubectl"] + args

        # Add kubeconfig if specified
        if self.kubeconfig:
            cmd.extend(["--kubeconfig", self.kubeconfig])

        # Add context if specified
        if self.context:
            cmd.extend(["--context", self.context])

        logger.debug(f"Running kubectl command: {' '.join(cmd)}")

        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )

            returncode = process.returncode if process.returncode is not None else 0

            return stdout, stderr, returncode

        except asyncio.TimeoutError:
            if process is not None:
                process.kill()
                await process.wait()
            raise TimeoutError(f"kubectl command timed out after {timeout}s")

    async def copy(
        self,
        src: str,
        dst: str,
        upload: bool = True,
        cwd: Optional[str] = None,
        timeout: float = 180.0,
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

                # Build kubectl cp command: kubectl cp <src> <namespace>/<pod>:<dst> -c <container>
                pod_path = f"{self.namespace}/{self.pod.name}:{dst}"
                kubectl_args = ["cp", src, pod_path, "-c", self.container_name]

                logger.info(f"Uploading {src} to {pod_path}")

            else:
                # Resolve container source path
                if not src.startswith("/"):
                    src = f"{effective_cwd}/{src}"

                # Check if source file exists in container
                check_cmd = f"test -e {src}"
                result = await self.execute(check_cmd, timeout=10.0)
                if result.returncode != 0:
                    raise FileNotFoundError(
                        f"Source file does not exist in container: {src}"
                    )

                # Build kubectl cp command: kubectl cp <namespace>/<pod>:<src> <dst> -c <container>
                pod_path = f"{self.namespace}/{self.pod.name}:{src}"
                kubectl_args = ["cp", pod_path, dst, "-c", self.container_name]

                logger.info(f"Downloading {pod_path} to {dst}")

            # Execute kubectl cp
            stdout, stderr, returncode = await self._run_kubectl(
                kubectl_args, timeout=300.0
            )

            if returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace").strip()
                raise Exception(
                    f"kubectl cp failed (exit code {returncode}): {error_msg}"
                )

            logger.info("Copy operation completed successfully")

        except Exception as e:
            logger.error(f"Copy operation failed: {e}")
            raise

    async def cleanup(self):
        """Delete the pod and clean up resources."""
        if self._closed:
            return

        try:
            logger.info(f"Cleaning up pod {self.pod.name}")
            await self.pod.delete()
            logger.info(f"Pod {self.pod.name} deleted")
        except Exception as e:
            logger.error(f"Failed to delete pod {self.pod.name}: {e}")
        finally:
            self._closed = True

    @property
    def alive(self) -> bool:
        """Check whether the pod is still running."""
        return not self._closed


class K8rsEnvBuilder(ContainerEnvBuilder):
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
        self.namespace = conf.get("namespace", "default")
        self.kubeconfig = conf.get("kubeconfig")
        self.context = conf.get("context")
        self.pod_name_prefix = conf.get("pod_name_prefix", "agentflow")
        self._api = None

    async def _get_api(self):
        """Get or create kr8s API client."""
        if self._api is None:
            self._api = await kr8s.asyncio.api(
                kubeconfig=self.kubeconfig,
                context=self.context,
            )
        return self._api

    async def start(self, args: ContainerStartArgs) -> K8rsEnv:
        """Start a Kubernetes pod from the given configuration.

        Args:
            args: Container start arguments

        Returns:
            K8sEnv instance for executing commands

        Raises:
            TimeoutError: Pod failed to become ready within startup_timeout
            Exception: Pod creation or startup failed
        """
        api = await self._get_api()

        # Generate unique pod name
        import uuid

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

        # Build container spec
        container_spec = {
            "name": "main",
            "image": args.image,
            "env": env_vars,
        }

        if args.cmd:
            container_spec["command"] = ["/bin/sh", "-c"]
            container_spec["args"] = [args.cmd]
        else:
            # Default command: sleep for the specified timeout
            container_spec["command"] = ["/bin/sh", "-c"]
            container_spec["args"] = [f"sleep {args.container_timeout}"]

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

        pod = None
        try:
            # Create the pod
            logger.info(f"Creating pod {pod_name} in namespace {self.namespace}")
            pod = await Pod(pod_spec, api=api)
            await pod.create()

            # Wait for pod to be ready
            logger.info(
                f"Waiting for pod {pod_name} to be ready (timeout: {args.startup_timeout}s)"
            )
            start_time = time.monotonic()

            while True:
                await pod.refresh()

                # Check timeout
                elapsed = time.monotonic() - start_time
                if elapsed > args.startup_timeout:
                    await pod.delete()
                    raise TimeoutError(
                        f"Pod {pod_name} failed to become ready within {args.startup_timeout}s"
                    )

                # Check pod phase
                phase = pod.status.phase
                if phase == "Running":
                    # Check if container is ready
                    container_statuses = pod.status.containerStatuses or []
                    if container_statuses and container_statuses[0].ready:
                        logger.info(f"Pod {pod_name} is ready")
                        break
                elif phase in ["Failed", "Succeeded"]:
                    await pod.delete()
                    raise Exception(
                        f"Pod {pod_name} entered phase {phase} before becoming ready"
                    )

                # Wait before checking again
                await asyncio.sleep(1)

            return K8rsEnv(
                pod=pod,
                container_name="main",
                namespace=self.namespace,
                kubeconfig=self.kubeconfig,
                context=self.context,
                default_cwd=args.cwd,
                default_env=args.env,
                default_forward_env=args.forward_env,
            )

        except Exception as e:
            logger.error(f"Failed to start pod {pod_name}: {e}")
            # Try to clean up if pod was created
            raise
