"""
SWE-agent / RLTokenAgent expect a SWEEnv-like object (communicate, repo, deployment, …).

E2B sandboxes are driven via ``E2BEnv.execute`` (stateless shells). This module wraps
``E2BEnv`` in a thin shim so ``RLTokenAgentWrapper`` can pass ``env._env`` to upstream
swe-agent while rollout/reward still use ``ContainerEnv`` (execute/read_file/write_file)
on the outer wrapper.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import shlex
import time
from pathlib import PurePath
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
)
from swerex.runtime.abstract import Command as RexCommand
from swerex.runtime.abstract import (
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

from loguru import logger as _loguru

from .base import ContainerEnv, ContainerOutput
from .e2b import E2BEnv

_log = logging.getLogger(__name__)

# Matches ANSI escape sequences (same pattern as K8s's _strip_control_chars in BashSession).
_ANSI_ESCAPE_RE = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


class E2BPersistentSession:
    """Persistent bash session over E2B commands API with stdin=True.

    Uses ``sandbox.commands.run("bash -l", background=True, stdin=True)``
    + ``sandbox.commands.send_stdin(pid, data)`` to maintain a long-lived
    bash process with stdin kept open.

    Analogous to SWE-ReX's BashSession (pexpect-based) used by K8s.
    """

    SENTINEL_PREFIX = "__E2B_DONE_"
    SENTINEL_SUFFIX = "__"

    def __init__(self, e2b_env: E2BEnv):
        self._e2b = e2b_env
        self._pid: int | None = None
        self._handle: Any = None
        self._buffer: list[str] = []
        self._event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._started = False
        self._dead = False
        self._counter = 0

    async def start(self, env_vars: dict[str, str] | None = None, cwd: str = "/") -> None:
        """Start a persistent bash -l process with stdin open."""
        sandbox = self._e2b.sandbox
        if sandbox is None:
            raise RuntimeError("E2B sandbox is not available")

        self._buffer.clear()
        self._event.clear()
        self._dead = False

        envs = dict(env_vars or {})
        envs.setdefault("TERM", "dumb")
        envs.setdefault("LANG", "C.UTF-8")
        envs.setdefault("LC_ALL", "C.UTF-8")

        _loguru.info(f"[E2BPersistentSession] Starting bash (cwd={cwd}, stdin=True, background=True)...")
        handle = await sandbox.commands.run(
            "bash -l",
            background=True,
            stdin=True,
            on_stdout=self._on_stdout,
            on_stderr=self._on_stderr,
            user="root",
            cwd=cwd,
            envs=envs,
            timeout=86400,
        )
        self._handle = handle
        self._pid = handle.pid
        _loguru.info(f"[E2BPersistentSession] Bash started (pid={self._pid}). Sending init commands...")

        await asyncio.sleep(0.3)

        init_cmds = "export PS1='' PS2='' PS0=''; set +o history\n"
        await self._raw_send(init_cmds)
        await asyncio.sleep(0.3)
        _loguru.info(f"[E2BPersistentSession] Init done. Buffer has {len(self._buffer)} chunks: {''.join(self._buffer)[:200]!r}")
        self._buffer.clear()
        self._event.clear()
        self._started = True
        _loguru.info(f"[E2BPersistentSession] Session ready (pid={self._pid}, cwd={cwd})")

    def _on_stdout(self, data: Any) -> None:
        text = data if isinstance(data, str) else (data.decode("utf-8", errors="replace") if isinstance(data, bytes) else str(data))
        self._buffer.append(text)
        self._event.set()

    def _on_stderr(self, data: Any) -> None:
        text = data if isinstance(data, str) else (data.decode("utf-8", errors="replace") if isinstance(data, bytes) else str(data))
        self._buffer.append(text)
        self._event.set()

    async def _raw_send(self, text: str) -> None:
        if self._pid is None:
            raise RuntimeError("Session not started")
        try:
            await self._e2b.sandbox.commands.send_stdin(self._pid, text)
        except Exception:
            self._dead = True
            raise

    async def run(self, cmd: str, timeout: float = 180.0) -> tuple[str, int]:
        """Execute command in the persistent session. Returns (output, exit_code)."""
        if self._dead:
            raise RuntimeError("E2B persistent session is no longer alive")
        async with self._lock:
            return await self._run_locked(cmd, timeout)

    async def _run_locked(self, cmd: str, timeout: float) -> tuple[str, int]:
        self._counter += 1
        sentinel = f"{self.SENTINEL_PREFIX}{self._counter}_{int(time.time() * 1000)}{self.SENTINEL_SUFFIX}"
        sentinel_re = re.compile(rf"{re.escape(sentinel)}(\d+)")

        self._buffer.clear()
        self._event.clear()

        payload = f"{cmd}\nprintf '{sentinel}%d\\n' $?\n"
        _loguru.info(f"[E2BPersistentSession] run #{self._counter} cmd={cmd[:80]!r} (timeout={timeout:.1f})")
        try:
            await self._e2b.sandbox.commands.send_stdin(self._pid, payload)
        except Exception as exc:
            self._dead = True
            raise RuntimeError(f"E2B persistent session stdin failed: {exc}") from exc

        deadline = asyncio.get_event_loop().time() + timeout
        _first_data_logged = False
        while True:
            combined = "".join(self._buffer)
            if not _first_data_logged and combined:
                _loguru.info(f"[E2BPersistentSession] run #{self._counter} first data received ({len(combined)} bytes): {combined[:200]!r}")
                _first_data_logged = True
            m = sentinel_re.search(combined)
            if m is not None:
                output_part = combined[: m.start()].rstrip("\n")
                exit_code = int(m.group(1))
                _loguru.info(f"[E2BPersistentSession] run #{self._counter} done exit_code={exit_code} output_len={len(output_part)}")
                return output_part, exit_code

            if self._dead:
                partial = combined[:500]
                raise RuntimeError(f"E2B persistent session died during command. Partial output: {partial}")

            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                partial = combined[:500]
                _loguru.error(f"[E2BPersistentSession] run #{self._counter} TIMEOUT. buffer_len={len(combined)} partial={partial!r}")
                raise TimeoutError(f"E2B persistent session: command timed out after {timeout}s. Partial output: {partial}")
            self._event.clear()
            try:
                await asyncio.wait_for(self._event.wait(), timeout=min(remaining, 5.0))
            except asyncio.TimeoutError:
                continue

    async def export(self, env_vars: dict[str, str]) -> None:
        """Inject env vars into the already-running session via `export`."""
        if not env_vars or not self.alive:
            return
        parts = [f"export {shlex.quote(str(k))}={shlex.quote(str(v))}" for k, v in env_vars.items()]
        async with self._lock:
            await self._run_locked("; ".join(parts), timeout=30.0)

    async def close(self) -> None:
        if self._pid is not None and self._e2b.sandbox is not None:
            with contextlib.suppress(Exception):
                await self._e2b.sandbox.commands.kill(self._pid)
            self._pid = None
        self._handle = None
        self._started = False
        self._dead = True
        _loguru.info("[E2BPersistentSession] Session closed")

    @property
    def alive(self) -> bool:
        return self._started and self._pid is not None and not self._dead


class _E2BRuntimeShim(AbstractRuntime):
    """Maps SWE-ReX ``AbstractRuntime`` calls onto ``E2BEnv`` (stateless shell + file APIs)."""

    def __init__(self, env_shim: E2BSWEEnvShim):
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
        if isinstance(action, BashInterruptAction):
            await self._env.interrupt_session()
            return BashObservation(output="", exit_code=0, session_type="bash")
        if not isinstance(action, BashAction):
            msg = f"Unsupported session action: {type(action).__name__}"
            raise TypeError(msg)

        tout = float(action.timeout) if action.timeout is not None else self._env._e2b.runtime_timeout

        # Persistent session path: all commands share the same bash process
        if self._env._use_persistent_session:
            return await self._run_in_persistent_session(action, tout)

        # Stateless one-shot path (original behavior)
        if action.is_interactive_command or action.is_interactive_quit:
            out = await self._env.communicate(
                action.command,
                timeout=tout,
                check="ignore",
            )
            return BashObservation(output=out, exit_code=0, session_type="bash")

        inner = action.command
        script = self._env._bash_one_shot(inner)
        r = await self._env._e2b.execute(script, check=False, timeout=tout)
        await self._env._sync_cwd()
        text = _strip_ansi(r.output.decode("utf-8", errors="replace"))
        if action.check == "ignore":
            return BashObservation(output=text, exit_code=None, session_type="bash")
        exit_code = int(r.returncode)
        if action.check == "raise" and exit_code != 0:
            msg = f"Command {inner!r} failed with exit code {exit_code}. Output:\n{text!r}"
            if action.error_msg:
                msg = f"{action.error_msg}: {msg}"
            raise NonZeroExitCodeError(msg)
        return BashObservation(output=text, exit_code=exit_code, session_type="bash")

    async def _run_in_persistent_session(self, action: BashAction, timeout: float) -> BashObservation:
        """Route action through the persistent PTY bash session."""
        session = await self._env._ensure_session()
        cmd = (action.command or "").strip()
        if not cmd:
            return BashObservation(output="", exit_code=0, session_type="bash")
        output, exit_code = await session.run(cmd, timeout=timeout)
        output = _strip_ansi(output)
        if action.check == "ignore":
            return BashObservation(output=output, exit_code=None, session_type="bash")
        if action.check == "raise" and exit_code != 0:
            msg = f"Command {cmd!r} failed with exit code {exit_code}. Output:\n{output!r}"
            if action.error_msg:
                msg = f"{action.error_msg}: {msg}"
            raise NonZeroExitCodeError(msg)
        return BashObservation(output=output, exit_code=exit_code, session_type="bash")

    async def close_session(self, request: CloseSessionRequest) -> CloseSessionResponse:
        del request
        return CloseBashSessionResponse()

    async def execute(self, command: RexCommand) -> CommandResponse:
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
        tout = float(command.timeout) if command.timeout is not None else self._env._e2b.runtime_timeout
        r = await self._env._e2b.execute(script, cwd=None, env={}, check=False, timeout=tout)
        text = r.output.decode("utf-8", errors="replace")
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
        await self._env._e2b.copy(request.source_path, request.target_path, upload=True, timeout=600.0)
        return UploadResponse()

    async def close(self) -> CloseResponse:
        return CloseResponse()


class _E2BDeploymentShim:
    """Stand-in for ``SWEEnv.deployment``: ``runtime`` (SWE-ReX API) + lifecycle stubs."""

    def __init__(self, env_shim: E2BSWEEnvShim):
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

    def __init__(self, e2b: E2BEnv, *, repo: RepoConfig | None, initial_cwd: str = "/testbed", use_persistent_session: bool = False):
        self._e2b = e2b
        self.repo = repo
        self.name = "main"
        self._cwd = initial_cwd
        self._exports: dict[str, str] = {"ROOT": initial_cwd}
        if e2b.default_env.get("PATH"):
            self._exports["PATH"] = e2b.default_env["PATH"]
        self._step_pause_enabled = False
        self._resume_lock = asyncio.Lock()
        self._use_persistent_session = use_persistent_session
        self._session: E2BPersistentSession | None = None
        self.deployment = _E2BDeploymentShim(self)

    @property
    def alive(self) -> bool:
        return self._e2b.alive

    def enable_step_pause(self) -> None:
        """Enable per-step sandbox pause/resume."""
        self._step_pause_enabled = True
        _log.info("[E2BSWEEnvShim] Per-step sandbox pause/resume ENABLED")

    def disable_step_pause(self) -> None:
        self._step_pause_enabled = False

    async def pause_sandbox(self) -> None:
        """Explicitly pause sandbox (called by agent before LLM inference)."""
        if not self._step_pause_enabled or self._e2b.is_paused:
            return
        try:
            await self._e2b.pause_sandbox()
            _log.debug("[E2BSWEEnvShim] Sandbox paused for inference")
        except Exception as exc:
            _log.warning(f"[E2BSWEEnvShim] pause_sandbox() failed ({exc}); disabling step-pause")
            self._step_pause_enabled = False

    async def resume_sandbox(self) -> None:
        """Explicitly resume sandbox (called by agent before command execution)."""
        if not self._step_pause_enabled or not self._e2b.is_paused:
            return
        async with self._resume_lock:
            if self._e2b.is_paused:
                await self._e2b.resume_sandbox()
                _log.debug("[E2BSWEEnvShim] Sandbox resumed for execution")

    def _bash_one_shot(self, inner: str) -> str:
        export_prefix = ""
        parts: list[str] = []
        # Re-export default_env (PATH, PYTHONPATH, etc.) inside the login shell,
        # because bash -l reads /etc/profile which may reset PATH to system default.
        if self._e2b.default_env:
            parts.extend(
                f"export {k}={shlex.quote(str(v))}"
                for k, v in self._e2b.default_env.items()
            )
        if self._exports:
            parts.extend(f"export {k}={shlex.quote(str(v))}" for k, v in self._exports.items())
        if parts:
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

    async def _ensure_session(self) -> E2BPersistentSession:
        """Lazily start the persistent PTY bash session."""
        if self._session is not None and self._session.alive:
            return self._session
        _loguru.info(f"[E2BSWEEnvShim] _ensure_session: creating new PTY session (cwd={self._cwd})")
        session = E2BPersistentSession(self._e2b)
        env_vars = dict(self._exports)
        if self._e2b.default_env:
            env_vars.update(self._e2b.default_env)
        await session.start(env_vars=env_vars, cwd=self._cwd)
        self._session = session
        return self._session

    async def communicate(
        self,
        input: str,
        timeout: int | float = 25,
        *,
        check: Literal["warn", "ignore", "raise"] = "ignore",
        error_msg: str = "Command failed",
    ) -> str:
        if self._use_persistent_session:
            return await self._communicate_session(input, timeout, check=check, error_msg=error_msg)
        return await self._communicate_stateless(input, timeout, check=check, error_msg=error_msg)

    async def _communicate_stateless(self, input: str, timeout: int | float, *, check: str, error_msg: str) -> str:
        """Stateless one-shot path (step_pause: true)."""
        try:
            script = self._bash_one_shot(input)
            out = await self._e2b.execute(script, check=False, timeout=float(timeout))
            await self._sync_cwd()
            text = _strip_ansi(out.output.decode("utf-8", errors="replace"))
            if check != "ignore" and out.returncode != 0:
                _log.error("%s:\n%s", error_msg, text[:2000])
                msg = f"Command {input!r} failed (exit_code={out.returncode}): {error_msg}"
                if check == "raise":
                    raise RuntimeError(msg)
            return text
        except Exception as e:
            if check == "raise":
                raise RuntimeError(f"{error_msg}: {e}") from e
            if check == "warn":
                _log.error("%s: %s", error_msg, e)
            return str(e)

    async def _communicate_session(self, input: str, timeout: int | float, *, check: str, error_msg: str) -> str:
        """Persistent session path (step_pause: false) — analogous to K8s SWEEnv.communicate."""
        try:
            session = await self._ensure_session()
            cmd = (input or "").strip()
            if not cmd:
                return ""
            output, exit_code = await session.run(cmd, timeout=float(timeout))
            output = _strip_ansi(output)
            if check != "ignore" and exit_code != 0:
                _log.error("%s:\n%s", error_msg, output[:2000])
                msg = f"Command {input!r} failed (exit_code={exit_code}): {error_msg}"
                if check == "raise":
                    raise RuntimeError(msg)
            return output
        except Exception as e:
            if check == "raise":
                raise RuntimeError(f"{error_msg}: {e}") from e
            if check == "warn":
                _log.error("%s: %s", error_msg, e)
            return str(e)

    async def set_env_variables(self, env_variables: dict[str, str]) -> None:
        if not env_variables:
            return
        normalized = {str(k): str(v) for k, v in env_variables.items()}
        self._exports.update(normalized)
        # Sync PATH/PYTHONPATH to E2BEnv.default_env so that calls bypassing the shim
        # (e.g. E2BRLContainerEnv.execute → E2BEnv.execute) also see the latest values.
        for key in ("PATH", "PYTHONPATH"):
            if key in normalized:
                self._e2b.default_env[key] = normalized[key]
        # Propagate into the running persistent bash so later in-session commands
        # see the new vars (the bash process's env was frozen at start()).
        if self._session is not None and self._session.alive:
            await self._session.export(normalized)

    async def read_file(self, path: str | PurePath, encoding: str | None = None, errors: str | None = None) -> str:
        enc = encoding or "utf-8"
        err = errors or "strict"
        return await self._e2b.read_file(str(path), encoding=enc, errors=err)

    async def write_file(self, path: str | PurePath, content: str) -> None:
        await self._e2b.write_file(str(path), content)

    async def upload(self, *, source_path: str, target_path: str, timeout: float = 600.0) -> None:
        """Host → sandbox copy; used by ``SiiToolHandler._upload_bundles``."""
        await self._e2b.copy(source_path, target_path, upload=True, timeout=timeout)

    async def execute(
        self,
        cmd: str,
        stdin: BinaryIO | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        forward_env: list[str] | None = None,
        timeout: float = 900.0,
        check: bool = True,
    ) -> ContainerOutput:
        """One-shot command like ``E2BEnv.execute``; used by ``SiiToolHandler._is_command_available``."""
        return await self._e2b.execute(cmd, stdin=stdin, cwd=cwd, env=env, forward_env=forward_env, timeout=timeout, check=check)

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
        await self._e2b.execute(script, cwd=None, env=merged, check=check, timeout=self._e2b.runtime_timeout)

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

    def __init__(
        self,
        inner: E2BEnv,
        *,
        sample: dict,
        initial_cwd: str | None = None,
        repo_name: str = "testbed",
        use_persistent_session: bool = False,
    ):
        self._inner = inner
        base_commit = str(sample.get("base_commit") or "HEAD")
        ds = (sample.get("data_source") or "").lower()
        skip_fetch = "swefactory" in ds
        ic = initial_cwd if (initial_cwd and str(initial_cwd).strip()) else f"/{repo_name}"
        self._shim = E2BSWEEnvShim(
            inner,
            repo=PreExistingRepoConfig(repo_name=repo_name, base_commit=base_commit, skip_fetch=skip_fetch),
            initial_cwd=ic,
            use_persistent_session=use_persistent_session,
        )

    @property
    def _env(self) -> E2BSWEEnvShim:
        return self._shim

    async def execute(self, cmd: str, stdin=None, cwd=None, env=None, forward_env=None, timeout=900.0, check=True):
        return await self._inner.execute(cmd, stdin=stdin, cwd=cwd, env=env, forward_env=forward_env, timeout=timeout, check=check)

    async def popen(self, cmd: str, cwd=None, env=None, forward_env=None, timeout=900.0):
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
        if self._shim._session:
            await self._shim._session.close()
        await self._inner.detach()

    async def cleanup(self) -> None:
        if self._shim._session:
            await self._shim._session.close()
        await self._inner.cleanup()

    def enable_step_pause(self) -> None:
        self._shim.enable_step_pause()

    def disable_step_pause(self) -> None:
        self._shim.disable_step_pause()

    async def pause_sandbox(self) -> None:
        await self._shim.pause_sandbox()

    async def resume_sandbox(self) -> None:
        await self._shim.resume_sandbox()

    @property
    def alive(self) -> bool:
        return self._inner.alive
