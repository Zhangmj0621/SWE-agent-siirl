"""
SWE-agent / RLTokenAgent expect a SWEEnv-like object (communicate, repo, deployment, …).

E2B sandboxes are driven via ``E2BEnv.execute`` (stateless shells). This module wraps
``E2BEnv`` in a thin shim so ``RLTokenAgentWrapper`` can pass ``env._env`` to upstream
swe-agent while rollout/reward still use ``ContainerEnv`` (execute/read_file/write_file)
on the outer wrapper.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
from pathlib import Path, PurePath
from typing import Any, BinaryIO, Literal

from sweagent.environment.repo import PreExistingRepoConfig, RepoConfig
from swerex.exceptions import NonZeroExitCodeError
from swerex.runtime.abstract import (
    AbstractRuntime,
    Action,
    BashAction,
    BashInterruptAction,
    BashObservation,
    CancelResponse,
    CloseBashSessionResponse,
    CloseResponse,
    CloseSessionRequest,
    CloseSessionResponse,
    Command as RexCommand,
    CommandResponse,
    CreateBashSessionResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    IsAliveResponse,
    Observation,
    ReadFileRequest,
    ReadFileResponse,
    UploadRequest,
    UploadResponse,
    WriteFileRequest,
    WriteFileResponse,
)

from .base import ContainerEnv, ContainerOutput
from .e2b import E2BEnv

_log = logging.getLogger(__name__)


class _E2BRuntimeShim(AbstractRuntime):
    """Maps SWE-ReX ``AbstractRuntime`` calls onto ``E2BEnv`` (stateless shell + file APIs)."""

    def __init__(self, env_shim: "E2BSWEEnvShim"):
        super().__init__()
        self.logger = _log
        self._env = env_shim

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        del timeout
        try:
            alive = bool(self._env._e2b.alive)
            return IsAliveResponse(is_alive=alive, message="" if alive else "sandbox closed")
        except Exception as e:
            return IsAliveResponse(is_alive=False, message=str(e))

    async def create_session(self, request: CreateSessionRequest) -> CreateSessionResponse:
        out_parts: list[str] = []
        tout = float(getattr(request, "startup_timeout", 10.0) or 10.0)
        for src in getattr(request, "startup_source", ()) or ():
            script = f"source {shlex.quote(str(src))} 2>/dev/null || true"
            inner = self._env._bash_one_shot(script)
            r = await self._env._e2b.execute(inner, check=False, timeout=max(5.0, tout))
            out_parts.append(r.output.decode("utf-8", errors="replace"))
            await self._env._sync_cwd()
        return CreateBashSessionResponse(output="".join(out_parts), session_type="bash")

    async def run_in_session(self, action: Action) -> Observation:
        if self._env._step_pause_enabled and not self._env._step_pause_active:
            self._env._step_pause_active = True
            _log.info("[_E2BRuntimeShim] Step-pause now ACTIVE (first run_in_session call)")
        if isinstance(action, BashInterruptAction):
            await self._env.interrupt_session()
            return BashObservation(output="", exit_code=0, session_type="bash")
        if not isinstance(action, BashAction):
            msg = f"Unsupported session action: {type(action).__name__}"
            raise TypeError(msg)
        if action.is_interactive_command or action.is_interactive_quit:
            out = await self._env.communicate(
                action.command,
                timeout=float(action.timeout or 1800.0),
                check="ignore",
            )
            return BashObservation(output=out, exit_code=0, session_type="bash")

        await self._env._ensure_resumed()
        self._env._inflight += 1
        try:
            inner = action.command
            script = self._env._bash_one_shot(inner)
            tout = float(action.timeout) if action.timeout is not None else 1800.0
            r = await self._env._e2b.execute(script, check=False, timeout=tout)
            await self._env._sync_cwd()
            text = r.output.decode("utf-8", errors="replace")
        finally:
            self._env._inflight -= 1
        await self._env._maybe_pause()
        if action.check == "ignore":
            return BashObservation(output=text, exit_code=None, session_type="bash")
        exit_code = int(r.returncode)
        if action.check == "raise" and exit_code != 0:
            msg = f"Command {inner!r} failed with exit code {exit_code}. Output:\n{text!r}"
            if action.error_msg:
                msg = f"{action.error_msg}: {msg}"
            raise NonZeroExitCodeError(msg)
        return BashObservation(output=text, exit_code=exit_code, session_type="bash")

    async def close_session(self, request: CloseSessionRequest) -> CloseSessionResponse:
        del request
        return CloseBashSessionResponse()

    async def execute(self, command: RexCommand) -> CommandResponse:
        await self._env._ensure_resumed()
        self._env._inflight += 1
        try:
            cmd = command.command
            if isinstance(cmd, list):
                inner = " ".join(shlex.quote(str(x)) for x in cmd) if command.shell else shlex.join([str(x) for x in cmd])
            else:
                inner = str(cmd)
            script = inner
            if command.cwd:
                script = f"cd {shlex.quote(command.cwd)} && ({inner})"
            if command.env:
                exports = " && ".join(f"export {k}={shlex.quote(str(v))}" for k, v in command.env.items())
                script = f"{exports} && {script}"
            tout = float(command.timeout) if command.timeout is not None else 1800.0
            r = await self._env._e2b.execute(script, cwd=None, env={}, check=False, timeout=tout)
            text = r.output.decode("utf-8", errors="replace")
        finally:
            self._env._inflight -= 1
        if command.check and r.returncode != 0:
            msg = f"Command failed (exit code: {r.returncode}): {text[:2000]}"
            if command.error_msg:
                msg = f"{command.error_msg}: {msg}"
            raise NonZeroExitCodeError(msg)
        return CommandResponse(stdout=text, stderr="", exit_code=r.returncode)

    async def cancel_last(self) -> CancelResponse:
        return CancelResponse(ok=False, message="not supported on E2B shim")

    async def read_file(self, request: ReadFileRequest) -> ReadFileResponse:
        content = await self._env.read_file(request.path, encoding=request.encoding, errors=request.errors)
        return ReadFileResponse(content=content)

    async def write_file(self, request: WriteFileRequest) -> WriteFileResponse:
        await self._env.write_file(request.path, request.content)
        return WriteFileResponse()

    async def upload(self, request: UploadRequest) -> UploadResponse:
        await self._env._ensure_resumed()
        self._env._inflight += 1
        try:
            await self._env._e2b.copy(request.source_path, request.target_path, upload=True, timeout=600.0)
        finally:
            self._env._inflight -= 1
        return UploadResponse()

    async def close(self) -> CloseResponse:
        return CloseResponse()


class _E2BDeploymentShim:
    """Stand-in for ``SWEEnv.deployment``: ``runtime`` (SWE-ReX API) + lifecycle stubs."""

    def __init__(self, env_shim: "E2BSWEEnvShim"):
        self._env = env_shim
        self._runtime_cache: _E2BRuntimeShim | None = None

    @property
    def runtime(self) -> _E2BRuntimeShim:
        """Lazily construct Rex-compatible runtime (upstream ``ToolHandler`` reads this)."""
        if self._runtime_cache is None:
            self._runtime_cache = _E2BRuntimeShim(self._env)
        return self._runtime_cache

    def add_hook(self, hook: Any) -> None:
        del hook

    async def start(self, *args: Any, **kwargs: Any) -> None:
        """Sandbox is already running when the shim is constructed."""
        del args, kwargs

    async def stop(self, *args: Any, **kwargs: Any) -> None:
        """Actual teardown is ``E2BEnv.cleanup`` / delete sandbox, not here."""
        del args, kwargs

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        del timeout
        try:
            alive = bool(self._env._e2b.alive)
            return IsAliveResponse(is_alive=alive, message="" if alive else "sandbox closed")
        except Exception as e:
            return IsAliveResponse(is_alive=False, message=str(e))


class E2BSWEEnvShim:
    """Duck-typed subset of ``SWEEnv`` backed by ``E2BEnv``."""

    def __init__(self, e2b: E2BEnv, *, repo: RepoConfig | None, initial_cwd: str = "/testbed"):
        self._e2b = e2b
        self.repo = repo
        self.name = "main"
        self._cwd = initial_cwd
        self._exports: dict[str, str] = {"ROOT": initial_cwd}
        if e2b.default_env.get("PATH"):
            self._exports["PATH"] = e2b.default_env["PATH"]
        self._step_pause_enabled = False
        self._step_pause_active = False
        self._inflight = 0
        self._resume_lock = asyncio.Lock()
        self.deployment = _E2BDeploymentShim(self)

    @property
    def alive(self) -> bool:
        return self._e2b.alive

    def enable_step_pause(self) -> None:
        """Enable per-step sandbox pause/resume. Call after bootstrap completes."""
        self._step_pause_enabled = True
        _log.info("[E2BSWEEnvShim] Per-step sandbox pause/resume ENABLED")

    def disable_step_pause(self) -> None:
        self._step_pause_enabled = False
        self._step_pause_active = False

    async def _ensure_resumed(self) -> None:
        if not self._step_pause_active or not self._e2b.is_paused:
            return
        async with self._resume_lock:
            if self._e2b.is_paused:
                _log.debug("[E2BSWEEnvShim] Resuming sandbox before command execution")
                await self._e2b.resume_sandbox()

    async def _maybe_pause(self) -> None:
        if not self._step_pause_active or self._e2b.is_paused:
            return
        if self._inflight > 0:
            return
        try:
            await self._e2b.pause_sandbox()
        except Exception as exc:
            _log.warning(f"[E2BSWEEnvShim] pause_sandbox() failed ({exc}); disabling step-pause")
            self._step_pause_enabled = False
            self._step_pause_active = False

    def _bash_one_shot(self, inner: str) -> str:
        export_prefix = ""
        if self._exports:
            parts = [f"export {k}={shlex.quote(str(v))}" for k, v in self._exports.items()]
            export_prefix = " && ".join(parts) + " && "
        # SWE-agent ToolHandler.reset uses communicate(" && ".join(_reset_commands)) which is
        # "" when there are no reset commands — must not emit "cd ... && ;" (bash syntax error).
        body = (inner or "").strip() if isinstance(inner, str) else ""
        if not body:
            body = ":"
        # Track cwd across calls (stateless E2B execute).
        inner_wrapped = f"{export_prefix}cd {shlex.quote(self._cwd)} && {body}; __rc=$?; pwd > /tmp/.swe_e2b_pwd; exit $__rc"
        return f"bash -lc {shlex.quote(inner_wrapped)}"

    async def _sync_cwd(self) -> None:
        try:
            out = await self._e2b.execute("cat /tmp/.swe_e2b_pwd 2>/dev/null || true", check=False, timeout=30.0)
            p = out.output.decode("utf-8", errors="replace").strip()
            if p:
                self._cwd = p
        except Exception:
            pass

    async def communicate(
        self,
        input: str,
        timeout: int | float = 25,
        *,
        check: Literal["warn", "ignore", "raise"] = "ignore",
        error_msg: str = "Command failed",
    ) -> str:
        if input is not None and ("git log" in input or "git diff" in input or "git show" in input):
            return "Illegal actions: `git log`, `git diff`, and `git show` are not allowed."

        await self._ensure_resumed()
        self._inflight += 1
        try:
            script = self._bash_one_shot(input)
            out = await self._e2b.execute(script, check=False, timeout=float(timeout))
            await self._sync_cwd()
            text = out.output.decode("utf-8", errors="replace")
            if check != "ignore" and out.returncode != 0:
                _log.error("%s:\n%s", error_msg, text[:2000])
                msg = f"Command {input!r} failed (exit_code={out.returncode}): {error_msg}"
                if check == "raise":
                    raise RuntimeError(msg)
        finally:
            self._inflight -= 1
        await self._maybe_pause()
        return text

    async def set_env_variables(self, env_variables: dict[str, str]) -> None:
        if not env_variables:
            return
        self._exports.update({str(k): str(v) for k, v in env_variables.items()})

    async def read_file(self, path: str | PurePath, encoding: str | None = None, errors: str | None = None) -> str:
        await self._ensure_resumed()
        self._inflight += 1
        try:
            enc = encoding or "utf-8"
            err = errors or "strict"
            return await self._e2b.read_file(str(path), encoding=enc, errors=err)
        finally:
            self._inflight -= 1

    async def write_file(self, path: str | PurePath, content: str) -> None:
        await self._ensure_resumed()
        self._inflight += 1
        try:
            await self._e2b.write_file(str(path), content)
        finally:
            self._inflight -= 1

    async def upload(self, *, source_path: str, target_path: str, timeout: float = 600.0) -> None:
        """Host → sandbox copy; used by ``SiiToolHandler._upload_bundles``."""
        await self._ensure_resumed()
        self._inflight += 1
        try:
            await self._e2b.copy(source_path, target_path, upload=True, timeout=timeout)
        finally:
            self._inflight -= 1

    async def execute(
        self,
        cmd: str,
        stdin: BinaryIO | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        forward_env: list[str] | None = None,
        timeout: float = 180.0,
        check: bool = True,
    ) -> ContainerOutput:
        """One-shot command like ``E2BEnv.execute``; used by ``SiiToolHandler._is_command_available``."""
        await self._ensure_resumed()
        self._inflight += 1
        try:
            return await self._e2b.execute(
                cmd, stdin=stdin, cwd=cwd, env=env, forward_env=forward_env, timeout=timeout, check=check
            )
        finally:
            self._inflight -= 1

    async def execute_command(
        self,
        command: str,
        shell: bool = True,
        check: bool = False,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        merged = dict(env or {})
        inner = command if shell else f"exec {command}"
        script = inner
        if cwd:
            script = f"cd {shlex.quote(cwd)} && {inner}"
        script = self._bash_one_shot(script)
        await self._e2b.execute(script, cwd=None, env=merged, check=check, timeout=1800.0)

    async def interrupt_session(self) -> None:
        _log.warning("[E2BSWEEnvShim] interrupt_session: no-op (E2B has no Rex-style session interrupt)")

    async def hard_reset(self) -> None:
        """Approximate ``SWEEnv.hard_reset`` without recreating the sandbox."""
        self._cwd = "/"
        await self.communicate("cd /", check="raise", timeout=60.0)
        if self.repo is not None:
            rn = self.repo.repo_name
            cmds = [f"cd /{rn}", "export ROOT=$(pwd -P)", *self.repo.get_reset_commands()]
            await self.communicate(" && ".join(cmds), check="raise", timeout=180.0, error_msg="Failed to clean repository")
            self._cwd = f"/{rn}"
            self._exports["ROOT"] = f"/{rn}"
        else:
            self._cwd = "/testbed"
            self._exports["ROOT"] = "/testbed"


class E2BRLContainerEnv(ContainerEnv):
    """``ContainerEnv`` for eval/reward + ``_env`` shim for RLTokenAgent / swe-agent."""

    def __init__(self, inner: E2BEnv, *, sample: dict, initial_cwd: str | None = None):
        self._inner = inner
        base_commit = str(sample.get("base_commit") or "HEAD")
        ds = (sample.get("data_source") or "").lower()
        skip_fetch = "swefactory" in ds
        ic = initial_cwd if (initial_cwd and str(initial_cwd).strip()) else "/testbed"
        self._shim = E2BSWEEnvShim(
            inner,
            repo=PreExistingRepoConfig(repo_name="testbed", base_commit=base_commit, skip_fetch=skip_fetch),
            initial_cwd=ic,
        )

    @property
    def _env(self) -> E2BSWEEnvShim:
        return self._shim

    async def execute(self, cmd: str, stdin=None, cwd=None, env=None, forward_env=None, timeout=180.0, check=True):
        return await self._inner.execute(cmd, stdin=stdin, cwd=cwd, env=env, forward_env=forward_env, timeout=timeout, check=check)

    async def popen(self, cmd: str, cwd=None, env=None, forward_env=None, timeout=180.0):
        return await self._inner.execute(cmd, cwd=cwd, env=env, forward_env=forward_env, timeout=timeout, check=False)

    async def read_file(self, path: str, encoding: str = "utf-8", errors: str = "strict") -> str:
        return await self._inner.read_file(path, encoding=encoding, errors=errors)

    async def write_file(self, path: str, content: str) -> None:
        await self._inner.write_file(path, content)

    async def copy(self, src: str, dst: str, upload: bool = True, cwd=None, timeout=180.0):
        return await self._inner.copy(src, dst, upload=upload, cwd=cwd, timeout=timeout)

    def get_handle(self) -> dict[str, Any] | None:
        """Partial-rollout: sandbox id + tracked shell cwd for ``E2BEnvBuilder.start`` resume."""
        base = self._inner.get_handle()
        if base is None:
            return None
        out = dict(base)
        out["swe_cwd"] = self._shim._cwd
        return out

    async def detach(self) -> None:
        await self._inner.detach()

    async def cleanup(self) -> None:
        await self._inner.cleanup()

    def enable_step_pause(self) -> None:
        self._shim.enable_step_pause()

    def disable_step_pause(self) -> None:
        self._shim.disable_step_pause()

    @property
    def alive(self) -> bool:
        return self._inner.alive
