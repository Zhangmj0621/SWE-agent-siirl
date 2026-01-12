import asyncio
import contextlib
import json
import logging
import os
import time
import uuid
from subprocess import CalledProcessError
from typing import BinaryIO

from pydantic import BaseModel, Field

from siirl.execution.rollout.agentflow.swe.environment.base import ContainerBuildArgs

from .base import ContainerEnv, ContainerEnvBuilder, ContainerOutput, ContainerStartArgs


class K8sEnv(ContainerEnv):
    """Kubernetes Pod-based container environment using kubectl."""

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
        self.logger = logger
        self._closed = False

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
        merged_env.update(env)

        return merged_env

    def _build_shell_command(self, cmd: str, cwd: str | None, env: dict[str, str]) -> str:
        """Build a shell command with working directory and environment variables."""
        parts = []

        if effective_cwd := cwd or self.default_cwd:
            parts.append(f"cd {effective_cwd}")

        if env:
            for key, value in env.items():
                escaped_value = value.replace("'", "'\"'\"'")
                parts.append(f"export {key}='{escaped_value}'")

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
        kubectl_args.extend(
            [
                "exec",
                "-n",
                self.namespace,
                self.pod_name,
                "-c",
                self.container_name,
                "-i",
                "--",
                "/bin/sh",
                "-c",
                shell_cmd,
            ]
        )

        try:
            self.logger.debug(f"Executing in pod {self.pod_name}: {shell_cmd[:100]}...")

            # Create subprocess with merged stdout/stderr
            process = await asyncio.create_subprocess_exec(
                "kubectl",
                *kubectl_args,
                stdin=asyncio.subprocess.PIPE if stdin is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout
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
                kubectl_args.extend(["cp", src, pod_path, "-c", self.container_name])

                self.logger.info(f"Uploading {src} to {pod_path}")

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
                kubectl_args.extend(["cp", pod_path, dst, "-c", self.container_name])

                self.logger.info(f"Downloading {pod_path} to {dst}")

            # Execute kubectl cp
            process = await asyncio.create_subprocess_exec(
                "kubectl",
                *kubectl_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)

            returncode = process.returncode if process.returncode is not None else 0

            if returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace").strip()
                raise Exception(f"kubectl cp failed (exit code {returncode}): {error_msg}")

            self.logger.info("Copy operation completed successfully")

        except asyncio.TimeoutError as e:
            self.logger.error("Copy operation timed out")
            raise TimeoutError("Copy operation timed out after 300s") from e
        except Exception as e:
            self.logger.error(f"Copy operation failed: {e}")
            raise

    async def cleanup(self):
        """Delete the pod and clean up resources."""
        if self._closed:
            return

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

    async def build(self, args: ContainerBuildArgs):
        raise NotImplementedError("K8s does not support container building")

    async def start(self, args: ContainerStartArgs) -> K8sEnv:
        """Start a Kubernetes pod from the given configuration.

        Args:
            args: Container start arguments

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

        try:
            # Create the pod using kubectl apply
            self.logger.info(f"Creating pod {pod_name} in namespace {self.namespace}")

            kubectl_args = self._get_kubectl_base_args()
            kubectl_args.extend(["apply", "-f", "-"])

            process = await asyncio.create_subprocess_exec(
                "kubectl",
                *kubectl_args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            pod_json = json.dumps(pod_spec).encode()
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=pod_json),
                timeout=30.0,
            )

            if process.returncode != 0:
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

                process = await asyncio.create_subprocess_exec(
                    "kubectl",
                    *kubectl_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=10.0,
                )

                if process.returncode != 0:
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
                    await self._delete_pod(pod_name)
                    raise Exception(f"Pod {pod_name} entered phase {phase} before becoming ready")

                # Wait before checking again
                await asyncio.sleep(1)

            return K8sEnv(
                pod_name=pod_name,
                container_name="main",
                namespace=self.namespace,
                kubeconfig=self.kubeconfig,
                context=self.context,
                default_cwd=args.cwd,
                default_env=args.env,
                default_forward_env=args.forward_env,
            )

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
        )

        await asyncio.wait_for(process.communicate(), timeout=30.0)
