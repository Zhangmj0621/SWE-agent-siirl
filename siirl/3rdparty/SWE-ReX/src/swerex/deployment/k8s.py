import asyncio
import logging
import subprocess
import time
import socket
import uuid
import re
import os
from typing import Any

from typing_extensions import Self

from swerex import PACKAGE_NAME, REMOTE_EXECUTABLE_NAME
from swerex.deployment.abstract import AbstractDeployment
from swerex.deployment.config import K8sDeploymentConfig
from swerex.deployment.hooks.abstract import CombinedDeploymentHook, DeploymentHook
from swerex.exceptions import DeploymentNotStartedError
from swerex.runtime.abstract import IsAliveResponse
from swerex.runtime.config import RemoteRuntimeConfig
from swerex.runtime.remote import RemoteRuntime
from swerex.utils.log import get_logger
from swerex.utils.temp import get_repo_temp_dir
from swerex.utils.wait import _wait_until_alive

__all__ = ["K8sDeployment", "K8sDeploymentConfig"]

# ACR (Alibaba Container Registry) configuration for SWE-bench images
# Read from environment variables, with defaults for backward compatibility
ACR_REGISTRY = os.environ.get("ACR_REGISTRY", "sii-wulan-registry-vpc.cn-wulanchabu.cr.aliyuncs.com")
ACR_NAMESPACE = os.environ.get("ACR_NAMESPACE", "sii-wulan/dockerhub-mirror")
# Tag mapping follows SWE-scripts/batch_upload.py convert_to_acr_tag


def map_image_to_acr(image: str) -> str:
    """
    Map a Docker image name to ACR format using the same rules as
    [convert_to_acr_tag()](SWE-scripts/batch_upload.py:140).

    Rules:
      - Ensure a tag; default to "latest" if missing
      - Replace "/" with "--" in the image path
      - Construct final ACR reference as: <registry>/<namespace>:<converted_tag>

    Examples:
      - "library/ubuntu:latest" -> ".../dockerhub-mirror:library--ubuntu--latest"
      - "swebench/sweb.eval.x86_64.repo__issue:latest" -> ".../dockerhub-mirror:swebench--sweb.eval.x86_64.repo__issue--latest"
    """
    # Ensure image has a tag
    if ":" in image:
        image_part, tag = image.rsplit(":", 1)
    else:
        image_part, tag = image, "latest"

    # Strip registry prefix if present (e.g., docker.io/, ghcr.io/, quay.io/, localhost:5000/)
    segments = image_part.split("/")
    if len(segments) > 1 and ('.' in segments[0] or ':' in segments[0] or segments[0] == 'localhost'):
        image_part = "/".join(segments[1:])
    if "latest" not in image:
        acr_tag = image_part.replace("/", "--") 
    # Replace "/" with "--" to flatten into a single repo tag
    else:
        acr_tag = image_part.replace("/", "--") + f"--{tag}"
    return f"{ACR_REGISTRY}/{ACR_NAMESPACE}:{acr_tag}"

def map_image_to_acr_swefactory(image: str) -> str:
    """
    Map a swefactory image name to ACR format.
    """
    # Remove namespace prefix if present
    acr_image = f"{ACR_REGISTRY}/sii-wulan/swe-factory:{image.lower()}"

    return acr_image



class K8sDeployment(AbstractDeployment):
    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ):
        """Deployment to Kubernetes pod.

        Args:
            **kwargs: Keyword arguments (see `K8sDeploymentConfig` for details).
        """
        self._config = K8sDeploymentConfig(**kwargs)
        self._runtime: RemoteRuntime | None = None
        self._pod_name: str | None = None
        self._pod_ip: str | None = None
        self.logger = logger or get_logger("rex-deploy-k8s")
        self._runtime_timeout = self._config.runtime_timeout
        self._hooks = CombinedDeploymentHook()

    def add_hook(self, hook: DeploymentHook):
        self._hooks.add_hook(hook)

    @classmethod
    def from_config(cls, config: K8sDeploymentConfig) -> Self:
        return cls(**config.model_dump())

    def _get_pod_name(self) -> str:
        """Returns a concise, DNS-1123 compliant pod name.

        Uses only the most informative piece of the image (last segment, after the last "__" if present),
        plus a short random suffix to avoid collisions.
        """
        raw = self._config.image.lower()

        # Remove registry-like prefix if present (docker.io/, ghcr.io/, quay.io/, localhost:*, etc.)
        parts = [p for p in raw.split("/") if p]
        if len(parts) > 1 and ('.' in parts[0] or ':' in parts[0] or parts[0] == 'localhost'):
            parts = parts[1:]

        # Take only the last path segment (may contain tag)
        last = parts[-1] if parts else raw
        # Strip tag if present
        last_no_tag = last.split(":", 1)[0]

        # If SWE-bench-style naming with "__", keep only the final component (e.g., "matplotlib-24970")
        core = last_no_tag.rsplit("__", 1)[-1] if "__" in last_no_tag else last_no_tag

        # Sanitize to [a-z0-9-]
        core = re.sub(r"[^a-z0-9-]", "-", core)
        core = re.sub(r"-{2,}", "-", core).strip("-") or "image"

        # Compose final pod name with 8-char uuid suffix, respecting 63-char limit
        prefix = "swerex-"
        suffix = uuid.uuid4().hex[:8]
        max_core_len = 63 - len(prefix) - 1 - len(suffix)
        core = core[:max_core_len]
        return f"{prefix}{core}-{suffix}"

    @property
    def pod_name(self) -> str | None:
        return self._pod_name

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        """Checks if the runtime is alive.

        Raises:
            DeploymentNotStartedError: If the deployment was not started.
        """
        if self._runtime is None:
            msg = "Runtime not started"
            raise DeploymentNotStartedError(msg)
        if self._pod_name is None:
            msg = "Pod not started"
            raise DeploymentNotStartedError(msg)
        
        # Check if pod is still running
        try:
            subprocess.check_call(
                ["kubectl", "get", "pod", self._pod_name, f"--namespace={self._config.namespace}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            msg = f"Pod {self._pod_name} is not running"
            raise RuntimeError(msg)

        result = subprocess.run(
            ["kubectl", "get", "pod", self._pod_name, f"--namespace={self._config.namespace}", "-o", "json"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            import json
            pod_json = json.loads(result.stdout)
            status = pod_json.get("status", {})
            conditions = status.get("conditions", [])
            not_ready = any(c.get("reason","") == "PodFailed" and c.get("status") == "False" for c in conditions)
            if not_ready:
                msg = f"Pod {self._pod_name} is not ready since PodFailed"
                raise RuntimeError(msg)
        return await self._runtime.is_alive(timeout=timeout)

    async def _wait_until_alive(self, timeout: float = 10.0):
        try:
            return await _wait_until_alive(self.is_alive, timeout=timeout, function_timeout=self._runtime_timeout)
        except TimeoutError as e:
            self.logger.error("Runtime did not start within timeout. Checking pod logs...")
            try:
                result = subprocess.run(
                    ["kubectl", "logs", self._pod_name, f"--namespace={self._config.namespace}"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.logger.error(f"Pod logs:\n{result.stdout}\n{result.stderr}")
            except Exception as log_error:
                self.logger.error(f"Failed to get pod logs: {log_error}")
            await self.stop()
            raise e

    def _get_token(self) -> str:
        return str(uuid.uuid4())

    def _get_swerex_start_cmd(self, token: str) -> list[str]:
        rex_args = f"--auth-token {token}"

        # Optional: Configure APT source for Ubuntu/Debian systems (e.g., Aliyun mirror for CN regions)
        apt_setup = ""
        if getattr(self._config, "apt_source_url", None):
            apt_url = self._config.apt_source_url
            apt_url_ubuntu = apt_url
            apt_url_debian = apt_url
            apt_url_debian_security = apt_url
            if "ubuntu" in apt_url:
                apt_url_debian = apt_url.replace("ubuntu", "debian")
                apt_url_debian_security = apt_url.replace(
                    "ubuntu", "debian-security"
                )
            elif "debian" in apt_url and "debian-security" not in apt_url:
                apt_url_debian_security = apt_url.replace(
                    "debian", "debian-security"
                )
                apt_url_ubuntu = apt_url.replace("debian", "ubuntu")
            # Update sources.list to use the custom mirror
            apt_setup = (
                f"(sed -i 's|http://archive.ubuntu.com/ubuntu/|{apt_url_ubuntu}|g' /etc/apt/sources.list 2>/dev/null || true) && "
                f"(sed -i 's|http://security.ubuntu.com/ubuntu/|{apt_url_ubuntu}|g' /etc/apt/sources.list 2>/dev/null || true) && "
                f"(sed -i 's|http://deb.debian.org/debian|{apt_url_debian}|g' /etc/apt/sources.list 2>/dev/null || true) && "
                f"(sed -i 's|http://security.debian.org/debian-security|{apt_url_debian_security}|g' /etc/apt/sources.list 2>/dev/null || true) && "
                f"(sed -i 's|https://deb.debian.org/debian|{apt_url_debian}|g' /etc/apt/sources.list 2>/dev/null || true) && "
                f"(sed -i 's|https://security.debian.org/debian-security|{apt_url_debian_security}|g' /etc/apt/sources.list 2>/dev/null || true) && "
                f"(find /etc/apt/sources.list.d -type f -name \"*.sources\" -print0 2>/dev/null | "
                f"xargs -0 -r sed -i "
                f" -e 's|http://deb.debian.org/debian|{apt_url_debian}|g' "
                f" -e 's|https://deb.debian.org/debian|{apt_url_debian}|g' "
                f" -e 's|http://security.debian.org/debian-security|{apt_url_debian_security}|g' "
                f" -e 's|https://security.debian.org/debian-security|{apt_url_debian_security}|g' "
                f" -e 's|http://archive.ubuntu.com/ubuntu/|{apt_url_ubuntu}|g' "
                f" -e 's|http://security.ubuntu.com/ubuntu/|{apt_url_ubuntu}|g' "
                f") && "
            )

        # Optional: Configure PyPI source (e.g., Aliyun mirror) for faster installs in CN regions.
        # We use CLI flags for `pip install pipx` and environment variables for `pipx run` (which uses pip under the hood).
        index_flag = ""
        trusted_part = ""
        if getattr(self._config, "pypi_index_url", None):
            index_flag = f"-i {self._config.pypi_index_url}"
        if getattr(self._config, "pypi_trusted_hosts", None):
            trusted_hosts: list[str] = self._config.pypi_trusted_hosts or []
            if trusted_hosts:
                trusted_part = " ".join(f"--trusted-host {h}" for h in trusted_hosts)

        pip_flags = " ".join(p for p in [index_flag, trusted_part] if p)
        pip_install_flags = f" {pip_flags}" if pip_flags else ""

        # Environment for pipx-run installation path (honored by pip)
        pipx_env_parts: list[str] = []
        if getattr(self._config, "pypi_index_url", None):
            pipx_env_parts.append(f"PIP_INDEX_URL='{self._config.pypi_index_url}'")
        if getattr(self._config, "pypi_trusted_hosts", None):
            hosts = " ".join(self._config.pypi_trusted_hosts or [])
            if hosts:
                pipx_env_parts.append(f"PIP_TRUSTED_HOST='{hosts}'")
        run_prefix = ((" ".join(pipx_env_parts)) + " ") if pipx_env_parts else ""

        cmd = (
            f"{REMOTE_EXECUTABLE_NAME} {rex_args} "
            "|| ("
            "PYTHON_BIN=python3; "
            "if ! command -v \"$PYTHON_BIN\" >/dev/null 2>&1; then "
            "  PYTHON_BIN=python; "
            "fi; "
            "RUN_OK=0; "
            "if command -v \"$PYTHON_BIN\" >/dev/null 2>&1; then "
            "  PIPX_OK=0; "
            "  if command -v pipx >/dev/null 2>&1; then "
            "    PIPX_OK=1; "
            "  fi; "
            "  if [ \"$PIPX_OK\" -ne 1 ] && command -v apt-get >/dev/null 2>&1; then "
            f"    {apt_setup}apt-get update && apt-get install -y pipx; "
            "  fi; "
            "  if command -v pipx >/dev/null 2>&1; then "
            "    PIPX_OK=1; "
            "  fi; "
            "  if [ \"$PIPX_OK\" -eq 1 ]; then "
            f"    {run_prefix}PIPX_DEFAULT_PYTHON=\"$PYTHON_BIN\" pipx run --spec {PACKAGE_NAME} {REMOTE_EXECUTABLE_NAME} {rex_args} && RUN_OK=1; "
            "  else "
            "    VENV_DIR=/tmp/swerex-venv; "
            "    \"$PYTHON_BIN\" -m venv \"$VENV_DIR\" && "
            f"    {run_prefix}\"$VENV_DIR/bin/python\" -m pip install --upgrade pip && "
            f"    {run_prefix}\"$VENV_DIR/bin/python\" -m pip install{pip_install_flags} {PACKAGE_NAME} && "
            f"    \"$VENV_DIR/bin/{REMOTE_EXECUTABLE_NAME}\" {rex_args} && RUN_OK=1; "
            "  fi; "
            "fi; "
            "if [ \"$RUN_OK\" -eq 1 ]; then "
            "  exit 0; "
            "fi; "
            "PYTHON_OK=0; "
            "if command -v \"$PYTHON_BIN\" >/dev/null 2>&1; then "
            "  if \"$PYTHON_BIN\" -c \"import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)\" >/dev/null 2>&1; then "
            "    PYTHON_OK=1; "
            "  fi; "
            "fi; "
            "if [ \"$PYTHON_OK\" -ne 1 ]; then "
            "  if command -v python3.11 >/dev/null 2>&1; then "
            "    PYTHON_BIN=python3.11; "
            "  else "
            "    if command -v apt-get >/dev/null 2>&1; then "
            f"      {apt_setup}apt-get update && apt-get install -y python3.11 python3.11-venv; "
            "    else "
            "      echo 'python3.11 not available and apt-get is missing' >&2; "
            "      exit 1; "
            "    fi; "
            "    if command -v python3.11 >/dev/null 2>&1; then "
            "      PYTHON_BIN=python3.11; "
            "    else "
            "      echo 'python3.11 install failed' >&2; "
            "      exit 1; "
            "    fi; "
            "  fi; "
            "fi; "
            "PIPX_OK=0; "
            "if command -v pipx >/dev/null 2>&1; then "
            "  PIPX_OK=1; "
            "fi; "
            "if [ \"$PIPX_OK\" -ne 1 ] && command -v apt-get >/dev/null 2>&1; then "
            f"  {apt_setup}apt-get update && apt-get install -y pipx; "
            "fi; "
            "if command -v pipx >/dev/null 2>&1; then "
            "  PIPX_OK=1; "
            "fi; "
            "if [ \"$PIPX_OK\" -eq 1 ]; then "
            f"  {run_prefix}PIPX_DEFAULT_PYTHON=\"$PYTHON_BIN\" pipx run --spec {PACKAGE_NAME} {REMOTE_EXECUTABLE_NAME} {rex_args}; "
            "else "
            "  VENV_DIR=/tmp/swerex-venv; "
            "  \"$PYTHON_BIN\" -m venv \"$VENV_DIR\"; "
            f"  {run_prefix}\"$VENV_DIR/bin/python\" -m pip install --upgrade pip; "
            f"  {run_prefix}\"$VENV_DIR/bin/python\" -m pip install{pip_install_flags} {PACKAGE_NAME}; "
            f"  \"$VENV_DIR/bin/{REMOTE_EXECUTABLE_NAME}\" {rex_args}; "
            "fi"
            ")"
        )
        return [
            *self._config.exec_shell,
            cmd,
        ]

    def _create_pod(self, image: str) -> bool:
        """Create the Kubernetes pod."""
        import json
        import tempfile
        from pathlib import Path

        # Create pod manifest
        container_env = [
            {
                "name": "HTTPBIN_URL",
                "value": "http://httpbin-new.opensii.ai/",
            },
            {
                "name": "SWEREX_LAST_CMD_PID_PATH",
                "value": "/tmp/swerex_last_cmd.pid",
            },
            {
                "name": "SWEREX_LAST_CMD_LOG_PATH",
                "value": "/tmp/swerex_last_cmd.log",
            },
        ]
        manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": self._pod_name,
                "namespace": self._config.namespace,
                "labels": {
                    "app": "swerex",
                    "role": "deployment",
                },
            },
            "spec": {
                "containers": [
                    {
                        "name": "swerex",
                        "image": image,
                        "command": self._get_swerex_start_cmd(self._token),
                        "env": container_env,
                        "resources": {
                            "requests": self._config.resource_requests,
                            "limits": self._config.resource_limits,
                        },
                        "ports": [
                            {
                                "containerPort": 8000,
                                "name": "http",
                            }
                        ],
                        "startupProbe": {
                            "tcpSocket": {"port": 8000},
                            "initialDelaySeconds": 10,
                            "periodSeconds": 5,
                            "timeoutSeconds": 5,
                            "failureThreshold": 360,  # 30 minutes: 360 * 5s = 1800s
                        },
                        "readinessProbe": {
                            "tcpSocket": {"port": 8000},
                            "initialDelaySeconds": 5,
                            "periodSeconds": 2,
                            "timeoutSeconds": 2,
                            "failureThreshold": 3,  # Reduced to 3 for quick failure detection after startup
                        },
                    }
                ],
                "nodeSelector": {
                    "kubernetes.io/arch": "amd64",
                },
                "restartPolicy": "Never",
                "hostname": self._pod_name[:63].rstrip("-."),
            },
        }
        if self._config.active_deadline_seconds is not None:
            manifest["spec"]["activeDeadlineSeconds"] = self._config.active_deadline_seconds

        # Write manifest to temp file
        temp_dir_root = get_repo_temp_dir()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir=temp_dir_root) as f:
            yaml_path = f.name
            json.dump(manifest, f)

        try:
            # Create pod
            self.logger.info(f"Creating pod {self._pod_name} with image {self._config.image}")
            result = subprocess.run(
                ["kubectl", "apply", "-f", yaml_path],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                self.logger.error(f"Failed to create pod: {result.stderr}")
                return False

            # Wait for pod to be ready
            self.logger.info(f"Waiting for pod {self._pod_name} to be ready...")
            deadline = time.time() + self._config.startup_timeout
            while time.time() < deadline:
                result = subprocess.run(
                    ["kubectl", "get", "pod", self._pod_name, f"--namespace={self._config.namespace}", "-o", "json"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode == 0:
                    import json
                    pod_json = json.loads(result.stdout)
                    status = pod_json.get("status", {})
                    conditions = status.get("conditions", [])
                    ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in conditions)
                    not_ready = any(c.get("type") == "Ready" and c.get("status") == "True" and c.get("reason","") == "PodFailed" for c in conditions)
                    if not_ready:
                        msg = f"Pod {self._pod_name} is not ready since PodFailed"
                        return False
                    if ready:
                        self._pod_ip = status.get("podIP")
                        self.logger.info(f"Pod {self._pod_name} is ready with IP {self._pod_ip}")
                        return True
                time.sleep(2)

            self.logger.error(f"Pod {self._pod_name} did not become ready within timeout")
            return False
        finally:
            Path(yaml_path).unlink(missing_ok=True)


    async def start(self):
        """Starts the runtime."""
        assert self._pod_name is None
        self._pod_name = self._get_pod_name()
        self._token = self._get_token()

        # Determine image to use
        image = self._config.image
        
        # Map to ACR if it's a SWE-bench image
        if self._config.use_acr and ("swebench" in image.lower() or "sweb.eval" in image):
            image = map_image_to_acr(image)
            self.logger.info(f"Mapped to ACR image: {image}")
        if self._config.use_acr and "swefactory" in image.lower():
            image = map_image_to_acr_swefactory(image)

        # Build with standalone Python if configured
        if self._config.python_standalone_dir:
            # Not implemented
            image = self._build_image_with_standalone_python(image)

        # Create pod
        self._hooks.on_custom_step("Creating Kubernetes pod")
        if not self._create_pod(image):
            msg = f"Failed to create pod {self._pod_name}"
            raise RuntimeError(msg)

        # Resolve pod IP (if not already captured during readiness)
        self._hooks.on_custom_step("Resolving pod IP")
        if not self._pod_ip:
            try:
                result = subprocess.run(
                    ["kubectl", "get", "pod", self._pod_name, f"--namespace={self._config.namespace}", "-o", "json"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode == 0:
                    import json
                    pod_json = json.loads(result.stdout)
                    self._pod_ip = pod_json.get("status", {}).get("podIP")
            except Exception as e:
                self.logger.warning(f"Failed to fetch pod IP: {e}")
        if not self._pod_ip:
            await self.stop()
            msg = f"Failed to determine IP for pod {self._pod_name}"
            raise RuntimeError(msg)
        
        # Connect to runtime directly via Pod IP:8000
        self._hooks.on_custom_step("Starting runtime")
        target_host = f"http://{self._pod_ip}"
        target_port = 8000
        self.logger.info(f"Connecting to runtime at {target_host}:{target_port}")
        self._runtime = RemoteRuntime.from_config(
            RemoteRuntimeConfig(
                host=target_host,
                port=target_port,
                timeout=self._runtime_timeout,
                auth_token=self._token,
            )
        )

        t0 = time.time()
        await self._wait_until_alive(timeout=self._config.startup_timeout)
        self.logger.info(f"Runtime started in {time.time() - t0:.2f}s")

    async def stop(self):
        """Stops the runtime."""
        if self._runtime is not None:
            try:
                if self._config.close_timeout is None:
                    await self._runtime.close()
                else:
                    await asyncio.wait_for(self._runtime.close(), timeout=self._config.close_timeout)
            except Exception as e:
                self.logger.warning("Runtime close failed or timed out: %s", e)
            finally:
                self._runtime = None


        # Delete pod asynchronously to avoid blocking event loop
        if self._pod_name is not None:
            try:
                self.logger.info(f"Deleting pod {self._pod_name}")
                t0 = time.time()
                # Use asyncio.create_subprocess_exec to avoid blocking
                process = await asyncio.create_subprocess_exec(
                    "kubectl",
                    "delete",
                    "pod",
                    self._pod_name,
                    f"--namespace={self._config.namespace}",
                    "--force",
                    "--grace-period=0",
                    "--wait=false",
                    "--request-timeout=10s",  # Reduced from 60s to 10s
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                # Wait for process with timeout
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=15.0  # Total timeout
                    )
                    elapsed = time.time() - t0
                    stdout_str = stdout.decode().strip() if stdout else ""
                    stderr_str = stderr.decode().strip() if stderr else ""
                    self.logger.info(
                        "kubectl delete pod finished in %.2fs (returncode=%s)",
                        elapsed,
                        process.returncode,
                    )
                    if stdout_str:
                        self.logger.info("kubectl delete pod stdout: %s", stdout_str)
                    if stderr_str:
                        self.logger.warning("kubectl delete pod stderr: %s", stderr_str)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    elapsed = time.time() - t0
                    self.logger.warning(f"kubectl delete pod timed out after {elapsed:.2f}s, killing process")
            except Exception as e:
                self.logger.warning(f"Failed to delete pod {self._pod_name}: {e}")
            finally:
                self._pod_name = None

    @property
    def runtime(self) -> RemoteRuntime:
        """Returns the runtime if running.

        Raises:
            DeploymentNotStartedError: If the deployment was not started.
        """
        if self._runtime is None:
            raise DeploymentNotStartedError()
        return self._runtime
