"""
K8s environment adapter — thin wrapper over upstream SWEEnv + SWE-ReX K8sDeployment.

All heavy lifting (pod lifecycle, bash session, file I/O) lives in SWEEnv. This
module only provides:
- the siirl ``ContainerEnv`` interface (execute/popen/copy/cleanup/...),
- a hard-timeout cleanup with detached ``kubectl delete`` fallback, and
- a builder that parses a SWE-bench sample into a ``K8sDeploymentConfig``.

All methods are async; the entire SWE rollout stack runs in a single event loop
so that multiple agents can interleave at ``await`` points (step-level concurrency).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Any

from sweagent.environment.swe_env import SWEEnv
from swerex.deployment.config import K8sDeploymentConfig
from swerex.deployment.k8s import K8sDeployment
from swerex.runtime.abstract import BashAction, CreateBashSessionRequest

from .base import ContainerBuildArgs, ContainerEnv, ContainerEnvBuilder, ContainerOutput, ContainerStartArgs

logger = logging.getLogger(__name__)


# ============================================================================
# K8sEnvAdapter
# ============================================================================


class K8sEnvAdapter(ContainerEnv):
    """Wraps a vanilla ``SWEEnv`` and exposes siirl's ``ContainerEnv`` interface.

    Lazy-initialized: the pod isn't created until the first async method is awaited.
    """

    def __init__(
        self,
        args: ContainerStartArgs,
        instance: dict | None = None,
        repo_name: str | None = None,
        namespace: str | None = None,
        config: dict | None = None,
        deployment_config: K8sDeploymentConfig | None = None,
        skip_reset: bool = False,  # Eval pods skip reset() to keep the image state.
        resume_handle: dict | None = None,  # Partial rollout: attach to existing pod.
    ):
        self.args = args
        self._instance = instance or {}
        self._repo_name = repo_name
        self._namespace = namespace
        self._config = config or {}
        self._deployment_config = deployment_config
        self._skip_reset = skip_reset
        # Partial rollout: when set, ``_start_swe_env`` attaches to the existing
        # pod described by ``resume_handle`` instead of creating a new one.
        # Shape: {"pod_name", "namespace", "auth_token", "pod_ip": optional}.
        self._resume_handle = resume_handle
        self._swe_env: SWEEnv | None = None
        self._closed = False

    # ----- initialization ----------------------------------------------------

    def _build_deployment_config(self) -> K8sDeploymentConfig:
        """Construct a ``K8sDeploymentConfig`` from yaml config + this instance."""
        if self._deployment_config is not None:
            logger.info("[K8sEnvAdapter] Using pre-built deployment_config")
            return self._deployment_config

        cfg = self._config
        kwargs: dict[str, Any] = {
            "image": self.args.image,
            "namespace": self._namespace or "default",
            "startup_timeout": cfg.get("startup_timeout", getattr(self.args, "startup_timeout", 1800.0)),
            "runtime_timeout": cfg.get("runtime_timeout", 1800.0),
            "close_timeout": cfg.get("close_timeout", 30.0),
            "use_acr": True,
            "resource_requests": cfg.get("resource_requests", {"memory": "4Gi", "cpu": "2"}),
            "resource_limits": cfg.get("resource_limits", {"memory": "4Gi", "cpu": "2"}),
            "active_deadline_seconds": cfg.get("active_deadline_seconds", 3600),
        }
        if pypi_index_url := cfg.get("pypi_index_url"):
            kwargs["pypi_index_url"] = pypi_index_url
            logger.info(f"[K8sEnvAdapter] Using PyPI mirror: {pypi_index_url}")
        if pypi_trusted_hosts := cfg.get("pypi_trusted_hosts"):
            kwargs["pypi_trusted_hosts"] = pypi_trusted_hosts
        if apt_source_url := cfg.get("apt_source_url"):
            kwargs["apt_source_url"] = apt_source_url
            logger.info(f"[K8sEnvAdapter] Using APT mirror: {apt_source_url}")

        logger.info(f"[K8sEnvAdapter] Building deployment_config for image: {self.args.image}")
        return K8sDeploymentConfig(**kwargs)

    def _create_repo_config(self):
        """Map ``repo_name`` string to the right swe-agent repo config object."""
        try:
            from sweagent.environment.repo import GithubRepoConfig, LocalRepoConfig, PreExistingRepoConfig

            repo_name = self._repo_name
            base_commit = self._instance.get("base_commit", "HEAD")
            if not repo_name:
                return None
            if "github" in repo_name:
                return GithubRepoConfig(github_url=repo_name, base_commit=base_commit)
            if "/" not in repo_name:
                return PreExistingRepoConfig(repo_name=repo_name, base_commit=base_commit)
            return LocalRepoConfig(path=Path(repo_name), base_commit=base_commit)
        except Exception as e:
            logger.exception("[K8sEnvAdapter] Failed to create repo config: %s", e)
            return None

    async def _start_swe_env(self) -> SWEEnv:
        """Create, initialize, and optionally reset ``SWEEnv``.

        Replicates ``SWEEnv.start()`` but honours ``skip_reset`` by calling
        ``_init_deployment()`` and then optionally ``reset()``.

        If ``self._resume_handle`` is set, attach to the existing pod
        described by the handle instead of creating a new one — used by
        SWEAgentFlow's partial-rollout resume path so that accumulated
        filesystem state (edits, test artefacts) from the previous attempt
        is preserved.
        """
        if self._resume_handle is not None:
            return await self._attach_to_existing_pod()

        deployment = self._build_deployment_config().get_deployment()
        repo = self._create_repo_config()
        logger.info(
            f"[K8sEnvAdapter] Repo config: {repo}, "
            f"base_commit: {getattr(repo, 'base_commit', 'HEAD') if repo else 'N/A'}"
        )

        swe_env = SWEEnv(deployment=deployment, repo=repo, post_startup_commands=[], name="swe_task")
        await swe_env._init_deployment()
        if self._skip_reset:
            logger.info("[K8sEnvAdapter] Skipping reset() for eval pod")
        else:
            await swe_env.reset()
        return swe_env

    async def _attach_to_existing_pod(self) -> SWEEnv:
        """Reconnect to an already-running pod (no pod create, no repo init).

        Replaces ``_init_deployment`` / ``reset`` for the resume path. We:
          1. Build a ``K8sDeployment`` that points at the existing pod via
             ``K8sDeployment.from_existing_pod`` (swerex upstream addition).
          2. Skip all pod-creation / binary-install / repo-clone steps — the
             pod's filesystem already has what the previous attempt left.
          3. Open a fresh bash session (the old Python client's session is
             gone; the swerex server inside the pod is still running and
             accepts a new ``create_session`` call).
          4. Re-apply the env variables ``SWEEnv._init_deployment`` would
             normally set, so subsequent ``run_in_session`` calls behave the
             same as a fresh SWEEnv.
        """
        handle = self._resume_handle
        assert handle is not None  # guarded by caller
        pod_name = handle.get("pod_name")
        namespace = handle.get("namespace") or self._namespace or "default"
        auth_token = handle.get("auth_token")
        pod_ip = handle.get("pod_ip")  # optional — from_existing_pod fetches if None

        if not pod_name or not auth_token:
            raise RuntimeError(
                f"[K8sEnvAdapter] resume_handle missing required fields: "
                f"pod_name={pod_name!r}, auth_token={'<set>' if auth_token else None!r}"
            )

        startup_timeout = float(
            self._config.get("startup_timeout", getattr(self.args, "startup_timeout", 1800.0))
        )
        runtime_timeout = float(self._config.get("runtime_timeout", 1800.0))

        logger.info(f"[K8sEnvAdapter] Attaching to existing pod {pod_name} in namespace {namespace}")
        deployment = await K8sDeployment.from_existing_pod(
            pod_name=pod_name,
            namespace=namespace,
            auth_token=auth_token,
            pod_ip=pod_ip,
            startup_timeout=startup_timeout,
            runtime_timeout=runtime_timeout,
        )

        # Build SWEEnv pointing at this deployment. Repo is intentionally None
        # so SWEEnv won't try to re-clone — filesystem is already warm.
        swe_env = SWEEnv(
            deployment=deployment,
            repo=None,
            post_startup_commands=[],
            name="swe_task",
        )

        # Open a fresh bash session — swe_env._init_deployment would normally
        # do this. Match the params that SWEEnv uses internally.
        await deployment.runtime.create_session(
            CreateBashSessionRequest(
                startup_source=["/root/.bashrc"],
                startup_timeout=10,
            )
        )
        # Re-apply the env vars that SWEEnv._init_deployment sets. Accessed
        # via swe_env (wraps `deployment.runtime.run_in_session`) so we don't
        # have to reach into the runtime API directly.
        try:
            await swe_env.set_env_variables(
                {
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "PIP_PROGRESS_BAR": "off",
                    "PAGER": "cat",
                }
            )
        except Exception as e:
            logger.warning(f"[K8sEnvAdapter] Failed to re-apply env vars on attach: {e}")

        return swe_env

    async def _ensure_initialized(self) -> SWEEnv:
        if self._swe_env is not None:
            return self._swe_env
        startup_timeout = self._config.get("startup_timeout", getattr(self.args, "startup_timeout", 1800.0))
        try:
            self._swe_env = await asyncio.wait_for(self._start_swe_env(), timeout=startup_timeout)
            logger.info("[K8sEnvAdapter] SWEEnv ready")
        except Exception as e:
            logger.exception("[K8sEnvAdapter] Failed to initialize SWEEnv: %s", e)
            if self._swe_env is not None:
                with contextlib.suppress(Exception):
                    await self._swe_env.close()
                self._swe_env = None
            raise
        return self._swe_env

    # ----- ContainerEnv: exec / files ---------------------------------------

    async def _run_in_session(self, cmd: str, cwd: str | None, timeout: float, check: bool) -> ContainerOutput:
        """Execute a bash command in the persistent session.

        Bypasses ``SWEEnv.communicate`` to surface the exit_code directly.
        """
        swe_env = await self._ensure_initialized()
        full_cmd = f"cd {cwd} && {cmd}" if cwd else cmd
        result = await swe_env.deployment.runtime.run_in_session(
            BashAction(command=full_cmd, timeout=int(timeout), check="silent")
        )
        output_bytes = result.output.encode() if isinstance(result.output, str) else result.output
        exit_code = result.exit_code if result.exit_code is not None else 0
        if check and exit_code != 0:
            instance_id = self._instance.get("instance_id", "UNKNOWN")
            preview = output_bytes.decode("utf-8", errors="replace")[:500] if output_bytes else ""
            # Match SWEEnv.communicate(check="raise"): tear down the env on failure.
            with contextlib.suppress(Exception):
                await swe_env.close()
            raise RuntimeError(
                f"[instance_id={instance_id}] Command {cmd!r} failed (exit_code={exit_code}): {preview}"
            )
        return ContainerOutput(output=output_bytes, returncode=exit_code)

    async def execute(
        self,
        cmd: str,
        stdin: BytesIO | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        forward_env: list[str] | None = None,
        timeout: float = 900.0,
        check: bool = True,
    ) -> ContainerOutput:
        if stdin is not None:
            raise NotImplementedError("stdin not supported")
        return await self._run_in_session(cmd, cwd, timeout, check)

    async def popen(
        self,
        cmd: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        forward_env: list[str] | None = None,
        timeout: float = 900.0,
    ) -> ContainerOutput:
        return await self.execute(cmd, cwd=cwd, env=env, timeout=timeout, check=False)

    async def copy(self, *args, **kwargs):
        raise NotImplementedError("Copy not available for K8s pods")

    async def read_file(self, path: str, encoding: str = "utf-8", errors: str = "strict") -> str:
        swe_env = await self._ensure_initialized()
        return await swe_env.read_file(str(path), encoding=encoding, errors=errors)

    async def write_file(self, path: str, content: str) -> None:
        swe_env = await self._ensure_initialized()
        await swe_env.write_file(str(path), content)

    # ----- cleanup -----------------------------------------------------------

    def get_handle(self) -> dict | None:
        """Snapshot pod identity for partial-rollout resume.

        Returned dict feeds ``K8sEnvAdapter(resume_handle=...)`` on the next
        attempt — see ``_attach_to_existing_pod``. Call this BEFORE ``detach``
        while the deployment still holds references to pod_name / auth_token.
        Returns None if the env wasn't initialised (nothing to resume from).
        """
        if self._swe_env is None:
            return None
        pod_name, namespace = self._capture_pod_identity()
        if not pod_name:
            return None
        deployment = getattr(self._swe_env, "deployment", None)
        auth_token = getattr(deployment, "_token", None) if deployment is not None else None
        pod_ip = getattr(deployment, "_pod_ip", None) if deployment is not None else None
        if not auth_token:
            # Without auth_token the attached worker can't talk to the swerex
            # server. Refuse to produce a half-usable handle.
            logger.warning(
                f"[K8sEnvAdapter] get_handle: pod {pod_name} has no captured auth_token; "
                "partial-rollout resume will be impossible"
            )
            return None
        return {
            "pod_name": pod_name,
            "namespace": namespace or "default",
            "auth_token": auth_token,
            "pod_ip": pod_ip,
        }

    async def detach(self) -> None:
        """Close Python-side runtime state but LEAVE THE POD RUNNING.

        Used by SWEAgentFlow on the ``SglangGenerationAborted`` path so the
        next worker that picks up the sample can attach (see
        ``_attach_to_existing_pod``). Pair with ``get_handle`` called just
        before this so the handle is still populated.

        NOT symmetric with ``cleanup`` — ``cleanup`` deletes the pod.
        """
        if self._swe_env is None:
            return
        self._closed = True
        try:
            deployment = getattr(self._swe_env, "deployment", None)
            runtime = getattr(deployment, "runtime", None) if deployment is not None else None
            if runtime is not None and hasattr(runtime, "close"):
                with contextlib.suppress(Exception):
                    await runtime.close()
            # Null out deployment's runtime ref so subsequent ``.runtime``
            # access raises ``DeploymentNotStartedError`` instead of handing
            # back a closed client. Mirrors what K8sDeployment.stop does,
            # minus the kubectl delete.
            if deployment is not None:
                deployment._runtime = None
        finally:
            self._swe_env = None

    async def cleanup(self):
        """Async close with hard timeout + detached ``kubectl delete`` fallback.

        ``SWEEnv.close()`` talks to the K8s API and can hang if the pod stops
        responding; we never want that to block the rollout. If close exceeds
        the timeout, we abandon it and fire-and-forget a ``kubectl delete
        --force`` so the pod still gets reaped.
        """
        if self._swe_env is None:
            return
        self._closed = True

        pod_name, namespace = self._capture_pod_identity()
        close_timeout = 60
        try:
            await asyncio.wait_for(self._swe_env.close(), timeout=close_timeout)
        except asyncio.TimeoutError:
            logger.error(
                f"[K8sEnvAdapter] cleanup() timed out after {close_timeout}s, "
                "abandoning stuck close(). Falling back to detached kubectl delete."
            )
            self._detached_kubectl_delete(pod_name, namespace)
        except Exception as e:
            logger.warning(f"[K8sEnvAdapter] close() raised: {e}; falling back to kubectl delete")
            self._detached_kubectl_delete(pod_name, namespace)
        finally:
            self._swe_env = None

    def _capture_pod_identity(self) -> tuple[str | None, str | None]:
        try:
            deployment = getattr(self._swe_env, "deployment", None)
            if deployment is None:
                return None, None
            pod_name = getattr(deployment, "pod_name", None) or getattr(deployment, "_pod_name", None)
            cfg = getattr(deployment, "_config", None)
            namespace = getattr(cfg, "namespace", None) if cfg is not None else None
            return pod_name, namespace
        except Exception as e:
            logger.warning(f"[K8sEnvAdapter] Failed to capture pod identity: {e}")
            return None, None

    @staticmethod
    def _detached_kubectl_delete(pod_name: str | None, namespace: str | None) -> None:
        if not pod_name:
            logger.error("[K8sEnvAdapter] No pod_name captured; manual cleanup may be needed.")
            return
        ns = namespace or "default"
        try:
            subprocess.Popen(
                [
                    "kubectl",
                    "delete",
                    "pod",
                    pod_name,
                    f"--namespace={ns}",
                    "--force",
                    "--grace-period=0",
                    "--wait=false",
                    "--request-timeout=10s",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            logger.warning(
                f"[K8sEnvAdapter] Dispatched detached kubectl delete for pod {pod_name} in namespace {ns}"
            )
        except Exception as e:
            logger.error(
                f"[K8sEnvAdapter] Failed to dispatch detached kubectl delete for {pod_name}: {e}. "
                "Pod may need manual cleanup."
            )

    # ----- misc --------------------------------------------------------------

    @property
    def alive(self) -> bool:
        if self._closed or self._swe_env is None:
            return False
        deployment = getattr(self._swe_env, "deployment", None)
        if deployment is None:
            return False
        return bool(getattr(deployment, "_pod_name", None))

    @property
    def _env(self):
        """Expose the underlying SWEEnv for consumers that need it directly
        (e.g. ``RLTokenAgentWrapper.setup``)."""
        return self._swe_env

    def __getattr__(self, name: str):
        """Delegate unknown attributes to the underlying SWEEnv."""
        if self._swe_env is None:
            raise AttributeError(
                f"SWEEnv not initialized; cannot delegate {name!r}. Await an env method first."
            )
        return getattr(self._swe_env, name)


# ============================================================================
# K8sEnvAdapterBuilder
# ============================================================================


def _extract_sample_and_eval_flag(runtime_meta: Any) -> tuple[dict, bool]:
    if runtime_meta is None:
        raise ValueError("sample is required in runtime_meta")
    if isinstance(runtime_meta, dict):
        sample = runtime_meta.get("sample") or runtime_meta.get("instance")
        is_eval_pod = runtime_meta.get("_is_eval_pod", False)
    else:
        sample = getattr(runtime_meta, "instance", None)
        is_eval_pod = getattr(runtime_meta, "_is_eval_pod", False)
    if sample is None:
        raise ValueError("sample is required in runtime_meta")
    return sample, is_eval_pod


def _extract_resume_handle(runtime_meta: Any) -> dict | None:
    """Extract ``_resume_handle`` (partial-rollout pod identity) from runtime_meta.

    Set by SWEAgentFlow before calling env.start when ``sample.partial_agent_data``
    carries a ``swe_env_handle``. ``None`` means "no resume, start a fresh pod."
    """
    if runtime_meta is None:
        return None
    if isinstance(runtime_meta, dict):
        return runtime_meta.get("_resume_handle")
    return getattr(runtime_meta, "_resume_handle", None)


def _resolve_image_name(sample: dict) -> str:
    """Prefer the dataset's ``image_name`` (covers swerebench / swefactory prefixes);
    fall back to SWE-bench's ``make_test_spec`` only when the sample lacks one."""
    from swebench.harness.test_spec.test_spec import make_test_spec

    if "extra_info" in sample and isinstance(sample["extra_info"], dict):
        sample = {**sample, **sample["extra_info"]}
    if "image_name" in sample and not sample["image_name"]:
        del sample["image_name"]
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        if key in sample and isinstance(sample[key], str):
            with contextlib.suppress(json.JSONDecodeError):
                sample[key] = json.loads(sample[key])

    if image_name := sample.get("image_name"):
        logger.info(f"[K8sEnvAdapterBuilder] Using dataset image_name: {image_name}")
        return image_name
    spec = make_test_spec(sample, namespace="swebench")
    logger.info(f"[K8sEnvAdapterBuilder] Fallback to TestSpec, image_name: {spec.instance_image_key}")
    return spec.instance_image_key


class K8sEnvAdapterBuilder(ContainerEnvBuilder):
    """Parses a SWE-bench sample into a ``K8sEnvAdapter``."""

    def __init__(self, config: dict):
        self.config = config

    async def build(self, args: ContainerBuildArgs):
        raise NotImplementedError("K8s uses pre-built images. Use 'start' instead.")

    def _make_adapter(self, runtime_meta: Any) -> K8sEnvAdapter:
        sample, is_eval_pod = _extract_sample_and_eval_flag(runtime_meta)
        resume_handle = _extract_resume_handle(runtime_meta)
        image_name = _resolve_image_name(sample)
        if is_eval_pod:
            logger.info("[K8sEnvAdapterBuilder] Starting eval pod (skip_reset=True)")
        if resume_handle is not None:
            logger.info(
                f"[K8sEnvAdapterBuilder] Partial-rollout resume: attaching to "
                f"pod {resume_handle.get('pod_name')} in namespace "
                f"{resume_handle.get('namespace')}"
            )

        container = ContainerStartArgs(
            image=image_name,
            cwd="/testbed",
            startup_timeout=self.config.get("startup_timeout", 1800.0),
        )
        return K8sEnvAdapter(
            args=container,
            instance=sample,
            repo_name=self.config.get("repo_name", "testbed"),
            namespace=self.config.get("namespace", "swe"),
            config=self.config,
            skip_reset=is_eval_pod,
            resume_handle=resume_handle,
        )

    async def start(self, args: ContainerStartArgs, runtime_meta: Any = None) -> K8sEnvAdapter:
        env = self._make_adapter(runtime_meta)
        await env._ensure_initialized()
        return env
