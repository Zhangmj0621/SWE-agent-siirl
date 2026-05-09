import asyncio
import base64
import contextlib
import json
import logging
import os
import shlex
from pathlib import Path
from typing import Any, BinaryIO

from loguru import logger

from siirl.execution.rollout.utils import EnvCreateError

from .base import ContainerBuildArgs, ContainerEnv, ContainerEnvBuilder, ContainerOutput, ContainerStartArgs
from .e2b_http200_compat import apply_e2b_http200_sdk_patch

_log = logging.getLogger(__name__)


def _patch_e2b_sdk_parse_http200_create_sandbox() -> None:
    apply_e2b_http200_sdk_patch(_log)


def _e2b_template_alias_for_docker_image(image_name: str) -> str:
    """Sanitize docker image ref to E2B template alias.

    Keep in sync with ``alias_for_image`` in
    ``scripts/build_swesmith_templates_ready.py`` (siirl-agentic_e2b).
    """
    alias = (
        image_name.replace("/", "-")
        .replace(".", "-")
        .replace("_", "-")
        .replace(":", "-")
        .lower()
    )
    if len(alias) > 64:
        alias = alias[:64]
    return alias


def _to_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return str(output)


def _find_swe_agent_tools_host_dir(explicit: str | None = None) -> Path | None:
    """Locate SWE-agent's `tools/` directory on the host filesystem."""
    if explicit:
        p = Path(explicit).expanduser().resolve()
        return p if p.exists() else None

    here = Path(__file__).resolve()
    suffixes = (
        ("SWE-agent", "tools"),
        ("swe-agent", "tools"),
        ("3rdparty", "swe-agent", "tools"),
    )
    for parent in [here, *here.parents]:
        for parts in suffixes:
            candidate = parent.joinpath(*parts)
            if candidate.is_dir():
                return candidate
    return None


def _extract_resume_handle(runtime_meta: Any) -> dict | None:
    """Same key as ``k8s_adapter._extract_resume_handle`` — set by ``SWEAgentFlow`` on resume."""
    if runtime_meta is None:
        return None
    if isinstance(runtime_meta, dict):
        return runtime_meta.get("_resume_handle")
    return getattr(runtime_meta, "_resume_handle", None)


def _extract_sample_and_eval_flag(runtime_meta: Any) -> tuple[dict, bool]:
    """Same contract as k8s_adapter (duplicated to avoid importing SWEEnv stack on e2b-only installs)."""
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


def _resolve_image_name(sample: dict) -> str:
    """Prefer dataset ``image_name``; else derive from SWE-bench ``make_test_spec``."""
    from swebench.harness.test_spec.test_spec import make_test_spec

    s = dict(sample)
    if "extra_info" in s and isinstance(s["extra_info"], dict):
        s = {**s, **s["extra_info"]}
    if "image_name" in s and not s["image_name"]:
        del s["image_name"]
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        if key in s and isinstance(s[key], str):
            with contextlib.suppress(json.JSONDecodeError):
                s[key] = json.loads(s[key])

    if image_name := s.get("image_name"):
        _log.info(f"[E2BEnvBuilder] Using dataset image_name: {image_name}")
        return image_name
    spec = make_test_spec(s, namespace="swebench")
    _log.info(f"[E2BEnvBuilder] Fallback to TestSpec, image_name: {spec.instance_image_key}")
    return spec.instance_image_key


async def _upload_dir(
    env: "E2BEnv", host_dir: Path, container_dir: str, *, per_file_timeout: float = 180.0
) -> None:
    """Upload a host directory into the sandbox by copying files one-by-one."""
    host_dir = host_dir.resolve()
    if not host_dir.exists() or not host_dir.is_dir():
        raise FileNotFoundError(f"Host bundle dir not found: {host_dir}")

    await env.execute(f"mkdir -p {shlex.quote(container_dir)}", timeout=60.0, check=True)

    for p in host_dir.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(host_dir).as_posix()
        dst = f"{container_dir.rstrip('/')}/{rel}"
        await env.copy(str(p), dst, upload=True, timeout=per_file_timeout)


async def _install_sweagent_tool_bundles(env: "E2BEnv", conf: dict) -> None:
    """Install SWE-agent tool bundles into the sandbox and add them to PATH."""
    if not bool(conf.get("install_sweagent_tools", False)):
        return

    tools_host_dir = _find_swe_agent_tools_host_dir(conf.get("sweagent_tools_host_path"))
    if tools_host_dir is None:
        raise FileNotFoundError(
            "install_sweagent_tools=true but could not locate SWE-agent tools directory. "
            "Set `sweagent_tools_host_path` to an absolute host path like .../swe-agent/tools."
        )

    bundle_names = conf.get(
        "sweagent_bundles",
        [
            "registry",
            "windowed",
            "search",
            "submit",
            "diff_state",
        ],
    )
    bundle_names = [str(x) for x in (bundle_names or [])]
    if not bundle_names:
        return

    logger.info(f"[E2BEnvBuilder] Installing SWE-agent bundles: {bundle_names}")

    base_path = (await env.execute("bash -lc 'echo -n \"$PATH\"'", timeout=60.0, check=True)).output.decode(
        "utf-8", errors="replace"
    )
    base_path = base_path.strip() or "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

    container_tools_root = str(conf.get("sweagent_tools_container_root", "/root/tools"))
    bin_prefixes: list[str] = []

    for name in bundle_names:
        host_bundle_dir = (tools_host_dir / name).resolve()
        if not host_bundle_dir.exists():
            raise FileNotFoundError(f"SWE-agent bundle not found on host: {host_bundle_dir}")

        container_bundle_dir = f"{container_tools_root.rstrip('/')}/{host_bundle_dir.name}"
        await _upload_dir(env, host_bundle_dir, container_bundle_dir)

        cmds: list[str] = [f"chmod +x {shlex.quote(container_bundle_dir)}/bin/* 2>/dev/null || true"]
        if (host_bundle_dir / "install.sh").exists():
            cmds.append(f"cd {shlex.quote(container_bundle_dir)} && bash -lc 'source install.sh'")
        await env.execute(" && ".join(cmds), timeout=300.0, check=False)

        bin_prefixes.append(f"{container_bundle_dir}/bin")

    await env.execute("bash -lc 'mkdir -p /root && printf \"%s\" \"{}\" > /root/state.json'", timeout=60.0, check=True)
    await env.execute(
        "bash -lc 'mkdir -p /root && printf \"%s\" \"{}\" > /root/.swe-agent-env'", timeout=60.0, check=True
    )

    env.default_env["PATH"] = ":".join(bin_prefixes + [base_path])
    python_paths = [f"{container_tools_root.rstrip('/')}/{name}" for name in bundle_names]
    env.default_env["PYTHONPATH"] = ":".join(python_paths)
    env.default_env.setdefault("PAGER", "cat")
    env.default_env.setdefault("MANPAGER", "cat")
    env.default_env.setdefault("GIT_PAGER", "cat")
    env.default_env.setdefault("PIP_PROGRESS_BAR", "off")
    env.default_env.setdefault("TQDM_DISABLE", "1")


async def _install_sweagent_tools_adapter(env: "E2BEnv", conf: dict) -> None:
    """Install lightweight SWE-agent-like tools implemented in siirl-agentic."""
    if not bool(conf.get("install_sweagent_tools_adapter", False)):
        return

    await env.execute("bash -lc 'mkdir -p /root && printf \"%s\" \"{}\" > /root/state.json'", timeout=60.0, check=True)

    adapter_host = Path(__file__).resolve().parents[1] / "tools" / "swe_tools.py"
    if not adapter_host.exists():
        raise FileNotFoundError(f"swe_tools adapter script not found on host: {adapter_host}")

    adapter_container = str(conf.get("sweagent_tools_adapter_path", "/root/swe_tools.py"))
    await env.copy(str(adapter_host), adapter_container, upload=True, timeout=180.0)
    await env.execute(f"chmod +x {shlex.quote(adapter_container)}", timeout=60.0, check=True)

    bin_dir = str(conf.get("sweagent_tools_adapter_bin_dir", "/usr/local/bin"))
    await env.execute(f"mkdir -p {shlex.quote(bin_dir)}", timeout=60.0, check=True)

    tool_names = [
        "open",
        "goto",
        "scroll_up",
        "scroll_down",
        "search_dir",
        "search_file",
        "find_file",
        "create",
        "edit",
        "insert",
        "replace_in_window",
        "str_replace_editor",
        "submit",
        "swe_tool",
    ]
    wrapper = (
        "#!/bin/sh\n"
        "cmd=\"$0\"; name=\"${cmd##*/}\"\n"
        f"exec python3 {adapter_container} \"$name\" \"$@\"\n"
    ).encode("utf-8")
    wrapper_b64 = base64.b64encode(wrapper).decode("ascii")
    for name in tool_names:
        dst = f"{bin_dir.rstrip('/')}/{name}"
        cmd = (
            f"base64 -d <<'__SWE_TOOL__' > {shlex.quote(dst)}\n"
            f"{wrapper_b64}\n"
            "__SWE_TOOL__\n"
            f"chmod +x {shlex.quote(dst)}"
        )
        await env.execute(cmd, timeout=60.0, check=True)

    base_path = (await env.execute("bash -lc 'echo -n \"$PATH\"'", timeout=60.0, check=True)).output.decode(
        "utf-8", errors="replace"
    )
    base_path = base_path.strip() or "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    if bin_dir not in base_path.split(":"):
        env.default_env["PATH"] = f"{bin_dir}:{base_path}"
    env.default_env.setdefault("PAGER", "cat")
    env.default_env.setdefault("MANPAGER", "cat")
    env.default_env.setdefault("GIT_PAGER", "cat")
    env.default_env.setdefault("PIP_PROGRESS_BAR", "off")
    env.default_env.setdefault("TQDM_DISABLE", "1")


class E2BEnv(ContainerEnv):
    """E2B sandbox-backed environment."""

    def __init__(
        self,
        sandbox: Any,
        *,
        default_cwd: str | None = None,
        default_env: dict[str, str] | None = None,
        default_forward_env: list[str] | None = None,
        request_timeout: float | None = None,
    ):
        self.sandbox = sandbox
        self.default_cwd = default_cwd
        self.default_env = default_env or {}
        self.default_forward_env = default_forward_env or []
        self.request_timeout = request_timeout
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

    async def popen(
        self,
        cmd: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        forward_env: list[str] | None = None,
        timeout: float = 180.0,
    ) -> ContainerOutput:
        raise NotImplementedError

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
        if self._closed:
            raise RuntimeError("E2B sandbox is closed")

        env = env or {}
        forward_env = forward_env or []
        merged_env = self._merge_env(env, forward_env)
        effective_cwd = cwd or self.default_cwd

        script = cmd
        if stdin is not None:
            payload = stdin.read()
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
            b64 = base64.b64encode(payload).decode("ascii")
            script = f"base64 -d <<'__E2B_STDIN__' | /bin/sh -c {shlex.quote(cmd)}\n{b64}\n__E2B_STDIN__"

        if effective_cwd:
            script = f"cd {shlex.quote(effective_cwd)} && {script}"

        if merged_env:
            exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in merged_env.items())
            script = f"env {exports} /bin/sh -c {shlex.quote(script)}"

        cmd_preview = cmd if len(cmd) <= 240 else cmd[:240] + "...<truncated>"
        logger.debug(
            f"[E2BEnv] running command in sandbox cwd={effective_cwd or '<default>'} timeout={timeout}s cmd={cmd_preview}"
        )

        def _run() -> Any:
            kwargs: dict[str, Any] = {"timeout": int(timeout), "user": "root"}
            if self.request_timeout is not None:
                kwargs["request_timeout"] = self.request_timeout
            try:
                return self.sandbox.commands.run(script, **kwargs)
            except TypeError:
                kwargs.pop("request_timeout", None)
                try:
                    return self.sandbox.commands.run(script, **kwargs)
                except TypeError:
                    kwargs.pop("user", None)
                    return self.sandbox.commands.run(script, **kwargs)

        result = await asyncio.to_thread(_run)
        stdout = _to_text(getattr(result, "stdout", ""))
        stderr = _to_text(getattr(result, "stderr", ""))
        output = (stdout + stderr).encode("utf-8", errors="replace")
        returncode = int(getattr(result, "exit_code", 0) or 0)
        logger.debug(f"[E2BEnv] command finished returncode={returncode}")

        if check and returncode != 0:
            raise Exception(f"e2b command failed (exit code {returncode}): {_to_text(output)[:200]}")

        return ContainerOutput(output=output, returncode=returncode)

    def get_handle(self) -> dict[str, Any] | None:
        """Return sandbox identity for partial-rollout resume (``Sandbox.connect``).

        Must be called before ``detach`` while ``self.sandbox`` is still held.
        """
        if self._closed or self.sandbox is None:
            return None
        sid = getattr(self.sandbox, "sandbox_id", None)
        if not sid:
            _log.warning("[E2BEnv] get_handle: sandbox has no sandbox_id; partial resume unavailable")
            return None
        return {"sandbox_id": str(sid)}

    async def detach(self) -> None:
        """Release the local SDK handle without ``kill()`` — remote sandbox keeps running.

        Pairs with ``get_handle`` on the ``SglangGenerationAborted`` path in ``SWEAgentFlow``.
        """
        if self._closed:
            return
        self._closed = True
        sb = self.sandbox
        self.sandbox = None
        if sb is None:
            return

        def _release_local_client() -> None:
            for name in ("close", "aclose", "_close"):
                fn = getattr(sb, name, None)
                if callable(fn):
                    with contextlib.suppress(Exception):
                        fn()
                    return

        await asyncio.to_thread(_release_local_client)

    async def read_file(self, path: str, encoding: str = "utf-8", errors: str = "strict") -> str:
        out = await self.execute(f"cat {shlex.quote(path)}", check=True, timeout=120.0)
        return out.output.decode(encoding, errors=errors)

    async def write_file(self, path: str, content: str) -> None:
        data = content.encode("utf-8")
        b64 = base64.b64encode(data).decode("ascii")
        parent = str(Path(path).parent)
        await self.execute(
            f"mkdir -p {shlex.quote(parent)} && "
            f"base64 -d <<'__E2B_WF__' > {shlex.quote(path)}\n{b64}\n__E2B_WF__",
            check=True,
            timeout=180.0,
        )

    async def copy(
        self,
        src: str,
        dst: str,
        upload: bool = True,
        cwd: str | None = None,
        timeout: float = 180.0,
    ):
        if self._closed:
            raise RuntimeError("E2B sandbox is closed")

        effective_cwd = cwd or self.default_cwd or "/"
        src_path = src if src.startswith("/") else str(Path(effective_cwd) / src)
        dst_path = dst if dst.startswith("/") else str(Path(effective_cwd) / dst)

        if upload:
            local_path = Path(src_path)
            if not local_path.exists():
                raise FileNotFoundError(f"Source file does not exist: {src}")
            if local_path.is_dir():
                await _upload_dir(self, local_path, dst_path, per_file_timeout=timeout)
                return
            payload = local_path.read_bytes()
            b64 = base64.b64encode(payload).decode("ascii")
            parent = str(Path(dst_path).parent)
            cmd = (
                f"mkdir -p {shlex.quote(parent)} && "
                f"base64 -d <<'__E2B_COPY__' > {shlex.quote(dst_path)}\n{b64}\n__E2B_COPY__"
            )
            await self.execute(cmd, timeout=timeout, check=True)
            return

        cmd = f"base64 {shlex.quote(src_path)}"
        out = await self.execute(cmd, timeout=timeout, check=True)
        data = base64.b64decode(out.output)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(data)

    async def cleanup(self):
        if self._closed:
            return
        try:

            def _kill() -> Any:
                try:
                    return self.sandbox.kill()
                except Exception:
                    return None

            await asyncio.to_thread(_kill)
        finally:
            self._closed = True

    @property
    def alive(self) -> bool:
        return not self._closed


class E2BEnvBuilder(ContainerEnvBuilder):
    """Builder for E2B-backed container environments."""

    def __init__(self, conf: dict):
        self.config = conf
        self.api_key = conf.get("api_key") or os.environ.get("E2B_API_KEY")
        self.api_url = conf.get("api_url") or os.environ.get("E2B_API_URL")
        self.force_http = bool(conf.get("force_http", False))
        self.request_timeout = conf.get("request_timeout")
        self.timeout = int(conf.get("timeout", 3600))
        self.allow_internet_access = bool(conf.get("allow_internet_access", True))
        self.auto_pause = bool(conf.get("auto_pause", False))
        self.template_map = conf.get("template_map", {})
        self.enable_build = bool(conf.get("enable_build", False))

    def _resolve_template(self, image_or_template: str) -> str:
        if image_or_template in self.template_map:
            return str(self.template_map[image_or_template])
        return _e2b_template_alias_for_docker_image(image_or_template)

    def _connect_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_url:
            kwargs["api_url"] = self.api_url
        if self.force_http:
            kwargs["force_http"] = True
        return kwargs

    async def build(self, args: ContainerBuildArgs):
        if not self.enable_build:
            logger.info("[E2BEnvBuilder] build skipped (enable_build=false)")
            return {"skipped": True}

        from e2b import AsyncTemplate

        template_obj = AsyncTemplate().from_image(image=args.tag)
        kwargs: dict[str, Any] = {
            "template": template_obj,
            "alias": args.tag,
            "skip_cache": args.nocache,
        }
        if args.resource_limits and "memory" in args.resource_limits:
            memory = args.resource_limits["memory"]
            if isinstance(memory, str) and memory.lower().endswith("g"):
                kwargs["memory_mb"] = int(memory[:-1]) * 1024
            elif isinstance(memory, int):
                kwargs["memory_mb"] = int(memory)

        if args.resource_limits and "cpus" in args.resource_limits:
            kwargs["cpu_count"] = int(args.resource_limits["cpus"])

        try:
            await AsyncTemplate.build(**kwargs)
        except TypeError:
            kwargs.pop("cpu_count", None)
            await AsyncTemplate.build(**kwargs)
        return {"built": True, "alias": args.tag}

    async def start(self, args: ContainerStartArgs | None = None, runtime_meta: Any = None) -> ContainerEnv:
        """Start sandbox. When ``args`` is None (SWE RL flow), resolve image from ``runtime_meta`` like K8s adapter.

        Partial-rollout resume: when ``runtime_meta`` carries ``_resume_handle`` with
        ``sandbox_id`` (from ``E2BEnv.get_handle``), reconnect via ``Sandbox.connect`` instead
        of ``beta_create``, skip tool installs, and restore ``swe_cwd`` on the shim.
        """
        from e2b import Sandbox

        resume_handle = _extract_resume_handle(runtime_meta) if runtime_meta is not None else None
        sandbox_id: str | None = None
        resume_cwd: str | None = None
        if isinstance(resume_handle, dict):
            sid = resume_handle.get("sandbox_id")
            if sid:
                sandbox_id = str(sid)
            sc = resume_handle.get("swe_cwd")
            if sc:
                resume_cwd = str(sc)

        if args is None:
            if runtime_meta is None:
                raise ValueError("E2BEnvBuilder.start requires ContainerStartArgs or runtime_meta")
            sample, _is_eval = _extract_sample_and_eval_flag(runtime_meta)
            image_name = _resolve_image_name(sample)
            args = ContainerStartArgs(
                image=image_name,
                cwd="/testbed",
                startup_timeout=float(self.config.get("startup_timeout", 1800.0)),
            )

        def _connect_existing(sid: str) -> Any:
            _patch_e2b_sdk_parse_http200_create_sandbox()
            kwargs: dict[str, Any] = {**self._connect_kwargs(), "timeout": self.timeout}
            if self.request_timeout is not None:
                kwargs["request_timeout"] = self.request_timeout
            connect = getattr(Sandbox, "connect", None)
            if connect is None:
                raise EnvCreateError("E2B SDK Sandbox has no connect(); cannot resume partial rollout")
            try:
                return connect(sid, **kwargs)
            except TypeError:
                for key in ("force_http", "request_timeout", "timeout"):
                    kwargs.pop(key, None)
                try:
                    return connect(sid, **kwargs)
                except TypeError:
                    return connect(sid, **self._connect_kwargs())

        template = self._resolve_template(args.image)
        create_kwargs = {
            **self._connect_kwargs(),
            "template": template,
            "timeout": self.timeout,
            "auto_pause": self.auto_pause,
            "allow_internet_access": self.allow_internet_access,
        }

        def _create() -> Any:
            _patch_e2b_sdk_parse_http200_create_sandbox()
            kwargs = dict(create_kwargs)
            if self.request_timeout is not None:
                kwargs["request_timeout"] = self.request_timeout
            try:
                return Sandbox.beta_create(**kwargs)
            except TypeError:
                for key in ("force_http", "request_timeout", "auto_pause", "allow_internet_access"):
                    kwargs.pop(key, None)
                return Sandbox.beta_create(**kwargs)

        skip_tool_install = False
        try:
            if sandbox_id:
                logger.info(f"[E2BEnvBuilder] Partial-rollout resume: Sandbox.connect({sandbox_id!r})")
                sandbox = await asyncio.to_thread(_connect_existing, sandbox_id)
                skip_tool_install = True
            else:
                sandbox = await asyncio.to_thread(_create)
        except Exception as e:
            err = str(e).lower()
            if (
                "no address associated with hostname" in err
                or "name or service not known" in err
                or type(e).__name__ == "gaierror"
            ):
                raise EnvCreateError(
                    "E2B: cannot resolve the sandbox envd hostname (DNS). "
                    "E2B_API_URL only reaches the control plane; after create, the SDK connects to "
                    "`<port>-<sandbox_id>.<sandbox_domain>`. Ensure this machine resolves that host "
                    "(internal DNS, split-horizon, /etc/hosts, or ops-provided edge — see "
                    "examples/swe_e2b_smoke/raw_post_sandbox.py header for E2B_DOMAIN / envd notes)."
                ) from e
            raise
        env = E2BEnv(
            sandbox,
            default_cwd=args.cwd,
            default_env=args.env,
            default_forward_env=args.forward_env,
            request_timeout=self.request_timeout,
        )

        if skip_tool_install and not resume_cwd:
            try:
                ro = await env.execute("bash -lc 'pwd'", check=False, timeout=30.0)
                lines = ro.output.decode("utf-8", errors="replace").strip().splitlines()
                candidate = lines[-1] if lines else ""
                if candidate.startswith("/"):
                    resume_cwd = candidate
            except Exception:
                pass

        if not skip_tool_install:
            await env.execute(
                "git config --global --add safe.directory /testbed",
                check=False, timeout=30.0,
            )

        if args.cmd:
            await env.execute(args.cmd, cwd=args.cwd, env=args.env, forward_env=args.forward_env, timeout=120.0)
        if not skip_tool_install:
            await _install_sweagent_tool_bundles(env, self.config)
            await _install_sweagent_tools_adapter(env, self.config)

        if runtime_meta is not None:
            sample, _ = _extract_sample_and_eval_flag(runtime_meta)
        else:
            sample = {"base_commit": "HEAD"}

        from .e2b_swe_shim import E2BRLContainerEnv

        return E2BRLContainerEnv(
            env,
            sample=sample,
            initial_cwd=resume_cwd if sandbox_id else None,
        )
