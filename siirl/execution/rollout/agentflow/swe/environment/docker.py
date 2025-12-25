import asyncio
import os
import logging
import uuid
from typing import Optional, BinaryIO
from loguru import logger
from siirl.execution.rollout.agentflow.swe.environment.base import ContainerBuildArgs

from .base import ContainerEnv, ContainerEnvBuilder, ContainerStartArgs, ContainerOutput


class DockerEnv(ContainerEnv):
    """Docker container environment using docker CLI."""

    def __init__(
        self,
        container_id: str,
        image: str,
        default_cwd: Optional[str] = None,
        default_env: Optional[dict[str, str]] = None,
        default_forward_env: Optional[list[str]] = None,
    ):
        self.container_id = container_id
        self.image = image
        self.default_cwd = default_cwd
        self.default_env = default_env or {}
        self.default_forward_env = default_forward_env or []
        self._closed = False

    def _merge_env(self, env: dict[str, str], forward_env: list[str]) -> dict[str, str]:
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

    def _build_docker_exec_args(
        self,
        container_id: str,
        cwd: Optional[str],
        env: dict[str, str],
    ) -> list[str]:
        """Build docker exec arguments for working directory and environment variables."""
        args = ["exec", "-i"]
        if cwd:
            args += ["-w", cwd]
        for key, value in env.items():
            args += ["-e", f"{key}={value}"]
        args.append(container_id)
        return args

    async def popen(
        self,
        cmd: str,
        cwd: Optional[str] = None,
        env: dict[str, str] = {},
        forward_env: list[str] = [],
        timeout: float = 180.0,
    ) -> ContainerOutput:
        """Execute command and return combined output (not implemented for DockerEnv)."""
        raise NotImplementedError

    async def execute(
        self,
        cmd: str,
        stdin: Optional[BinaryIO] = None,
        cwd: Optional[str] = None,
        env: dict[str, str] = {},
        forward_env: list[str] = [],
        timeout: float = 180.0,
        check=True,
    ) -> ContainerOutput:
        if self._closed:
            raise RuntimeError("Container environment is closed")

        merged_env = self._merge_env(env, forward_env)
        effective_cwd = cwd or self.default_cwd
        docker_args = self._build_docker_exec_args(
            self.container_id, effective_cwd, merged_env
        )
        docker_args += ["/bin/sh", "-c", cmd]
        logger.debug("[DockerEnv] running", docker_args)

        try:
            process = await asyncio.create_subprocess_exec(
                "docker",
                *docker_args,
                stdin=asyncio.subprocess.PIPE if stdin is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdin_data = stdin.read() if stdin is not None else None
            try:
                stdout, _ = await asyncio.wait_for(
                    process.communicate(input=stdin_data),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError(f"Command timed out after {timeout}s")

            returncode = process.returncode if process.returncode is not None else 127

            logger.debug(
                f"[Dockerenv] Command completed with exit code {returncode}, output length: {len(stdout)}"
            )

            if check and returncode != 0:
                raise Exception(
                    f"docker exec failed (exit code {returncode}): {stdout.decode(errors='replace')[:200]}"
                )

            return ContainerOutput(output=stdout, returncode=returncode)

        except asyncio.TimeoutError as e:
            logger.debug(
                f"[Dockerenv] Command execution timed out after {timeout}s",
                "cmd",
                cmd,
                "container",
                self.container_id,
            )
            raise e
        except Exception as e:
            logger.debug(
                f"[Dockerenv] Command `docker {docker_args}` execution failed: {e}",
                "container",
                self.container_id,
            )
            raise e

    async def copy(
        self,
        src: str,
        dst: str,
        upload: bool = True,
        cwd: Optional[str] = None,
        timeout=180.0,
    ):
        if self._closed:
            raise RuntimeError("Container environment is closed")
        logger.debug(
            "[Dockerenv] Copying",
            "src",
            src,
            "dst",
            dst,
            ", upload",
            upload,
            "container",
            self.container_id,
        )

        effective_cwd = cwd or self.default_cwd or "/"

        try:
            if upload:
                # Host to container
                if not os.path.exists(src):
                    raise FileNotFoundError(f"Source file does not exist: {src}")
                container_dst = dst
                if not container_dst.startswith("/"):
                    container_dst = f"{effective_cwd}/{container_dst}"
                docker_args = [
                    "cp",
                    src,
                    f"{self.container_id}:{container_dst}",
                ]
            else:
                # Container to host
                container_src = src
                if not container_src.startswith("/"):
                    container_src = f"{effective_cwd}/{container_src}"
                docker_args = [
                    "cp",
                    f"{self.container_id}:{container_src}",
                    dst,
                ]

            process = await asyncio.create_subprocess_exec(
                "docker",
                *docker_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)

            returncode = process.returncode if process.returncode is not None else 0

            if returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace").strip()
                raise Exception(
                    f"docker cp failed (exit code {returncode}): {error_msg}"
                )

        except asyncio.TimeoutError:
            raise TimeoutError("Copy operation timed out after 180s")

    async def cleanup(self):
        if self._closed:
            return
        try:
            logger.info(f"[DockerEnv] Cleaning up container {self.container_id}")
            process = await asyncio.create_subprocess_exec(
                "docker",
                "rm",
                "-f",
                self.container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(process.communicate(), timeout=30.0)
        except Exception as e:
            logger.debug(
                f"[DockerEnv] Failed to delete container {self.container_id}: {e}"
            )
        finally:
            self._closed = True

    @property
    def alive(self) -> bool:
        return not self._closed


class DockerEnvBuilder(ContainerEnvBuilder):
    """Builder for creating Docker-based container environments."""

    def __init__(self, conf: dict):
        self.config = conf

    async def build(self, args: ContainerBuildArgs):
        """
        Build a Docker image using the provided build arguments.
        """
        dockerfile_path = args.dockerfile_path or os.path.join(
            args.build_dir, "Dockerfile"
        )
        build_args = [
            "build",
            "-t",
            args.tag,
            "-f",
            dockerfile_path,
            args.build_dir,
        ]
        if args.nocache:
            build_args.append("--no-cache")
        if args.rm:
            build_args.append("--rm")
        if args.resource_limits:
            # Docker build supports --memory and --cpus
            if "memory" in args.resource_limits:
                build_args.extend(["--memory", str(args.resource_limits["memory"])])
            if "cpus" in args.resource_limits:
                build_args.extend(["--cpus", str(args.resource_limits["cpus"])])

        logger.debug(
            f"[DockerEnvBuilder] Building Docker image {args.tag} from {args.build_dir} with Dockerfile {dockerfile_path}"
        )

        timeout = args.timeout if args.timeout > 0 else None

        try:
            process = await asyncio.create_subprocess_exec(
                "docker",
                *build_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
            returncode = process.returncode if process.returncode is not None else 127

            if returncode != 0:
                raise Exception(
                    f"Docker build failed (exit code {returncode}): {stdout.decode(errors='replace')[:200]}"
                )

            # Optionally push the image
            if args.push:
                logger.debug(f"[DockerEnvBuilder] Pushing Docker image {args.tag}")
                push_process = await asyncio.create_subprocess_exec(
                    "docker",
                    "push",
                    args.tag,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                push_stdout, _ = await asyncio.wait_for(
                    push_process.communicate(), timeout=timeout
                )
                push_returncode = (
                    push_process.returncode
                    if push_process.returncode is not None
                    else 127
                )
                if push_returncode != 0:
                    raise Exception(
                        f"Docker push failed (exit code {push_returncode}): {push_stdout.decode(errors='replace')[:200]}"
                    )
                return {"output": stdout + push_stdout, "returncode": 0, "pushed": True}
            else:
                return {"output": stdout, "returncode": 0, "pushed": False}

        except asyncio.TimeoutError:
            raise TimeoutError(f"Docker build timed out after {timeout}s")

    async def start(self, args: ContainerStartArgs) -> DockerEnv:
        # Generate unique container name
        container_name = f"swebench-{uuid.uuid4().hex[:8]}"
        logger.info(
            f"[DockerEnvBuilder] Creating container {container_name} with image {args.image}"
        )
        env_args = []
        for env_key in args.forward_env:
            if env_key in os.environ:
                env_args.extend(["-e", f"{env_key}={os.environ[env_key]}"])
        for key, value in args.env.items():
            env_args.extend(["-e", f"{key}={value}"])

        docker_run_args = [
            "run",
            "-d",  # detached
            "--name",
            container_name,
        ]
        if args.cwd:
            docker_run_args.extend(["-w", args.cwd])
        docker_run_args += env_args
        docker_run_args.append(args.image)
        if args.cmd:
            docker_run_args.extend(["/bin/sh", "-c", args.cmd])
        else:
            docker_run_args.extend(["/bin/sh", "-c", f"sleep {args.container_timeout}"])

        try:
            process = await asyncio.create_subprocess_exec(
                "docker",
                *docker_run_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)
            if process.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace")
                raise Exception(f"Failed to create container: {error_msg}")
            container_id = container_name
            return DockerEnv(
                container_id=container_id,
                image=args.image,
                default_cwd=args.cwd,
                default_env=args.env,
                default_forward_env=args.forward_env,
            )
        except Exception as e:
            logger.debug(f"Failed to start container {container_name}: {e}")
            raise
