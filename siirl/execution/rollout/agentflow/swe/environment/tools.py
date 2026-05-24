import asyncio
import json
import os

from sweagent.tools.tools import ToolConfig, ToolHandler

from .base import ContainerEnv

_PATH_SENTINEL = "__SII_PATH__"


# patch ToolHandler, modify env call func
class SiiToolHandler(ToolHandler):
    def __init__(self, tools: ToolConfig):
        super().__init__(tools)

    async def install(self, env: ContainerEnv) -> None:
        try:
            await self._install_commands(env)
        except Exception as e:
            self.logger.warning(f"Tools _install_commands failed (will try to continue): {e}")
        await self.reset(env)

    async def reset(self, env: ContainerEnv) -> None:
        self.logger.info("Resetting tools")
        env_variables = self.config.env_variables.copy() | {var: os.getenv(var) for var in self.config.propagate_env_variables}
        await env.set_env_variables(env_variables)
        await env.write_file("/root/.swe-agent-env", json.dumps(self.config.registry_variables))
        await env.write_file("/root/state.json", "{}")
        if self._reset_commands:
            try:
                await env.communicate(" && ".join(self._reset_commands), check="raise", timeout=self.config.install_timeout)
            except Exception as e:
                self.logger.warning(f"Tools reset commands failed (will try to continue): {e}")

    async def _upload_bundles(self, env: ContainerEnv) -> None:
        await asyncio.gather(
            *(
                env.upload(source_path=bundle.path.as_posix(), target_path=f"/root/tools/{bundle.path.name}")
                for bundle in self.config.bundles
            )
        )

    async def _install_commands(self, env: ContainerEnv) -> None:
        """Install tools with minimal HTTP round-trips.

        Combines all bundle install steps into ONE communicate call
        instead of separate calls per bundle + verification + PATH check.
        """
        await env.set_env_variables(self.config.env_variables)

        # Upload bundles (parallel HTTP calls — unavoidable)
        await self._upload_bundles(env)

        # Build a single script that does everything:
        # chmod + install.sh for all bundles + print PATH at the end
        parts = []
        for bundle in self.config.bundles:
            bin_path = f"/root/tools/{bundle.path.name}/bin"
            parts.append(f"chmod +x {bin_path}/* 2>/dev/null")
            if (bundle.path / "install.sh").exists():
                parts.append(f"cd /root/tools/{bundle.path.name} && . ./install.sh")
            parts.append(f"chmod +x {bin_path}/* 2>/dev/null")
        parts.append(f"echo {_PATH_SENTINEL}$PATH")

        combined_script = " && ".join(parts)

        try:
            output = await env.communicate(combined_script, check="raise", timeout=self.config.install_timeout)
        except Exception as e:
            self.logger.warning(f"Tools install script failed (will try to continue): {e}")
            output = ""

        # Extract PATH from output and persist it
        base_path = ""
        if _PATH_SENTINEL in output:
            base_path = output.split(_PATH_SENTINEL)[-1].strip().split("\n")[0]

        if not base_path:
            try:
                base_path = (await env.communicate("echo $PATH", check="raise")).strip()
            except Exception:
                base_path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

        tool_bin_dirs = [f"/root/tools/{bundle.path.name}/bin" for bundle in self.config.bundles]
        for d in tool_bin_dirs:
            if d not in base_path:
                base_path = f"{d}:{base_path}"
        # In E2B stateless mode, the adapter at /usr/local/bin/submit handles
        # submission correctly (emits <<SWE_AGENT_SUBMISSION>> + writes model.patch).
        # Ensure it takes PATH priority over bundle scripts which depend on
        # EnvRegistry (ROOT variable) that may not be set up in E2B.
        if hasattr(env, "_cwd") and "/usr/local/bin" in base_path:
            parts = base_path.split(":")
            parts = [p for p in parts if p != "/usr/local/bin"]
            base_path = "/usr/local/bin:" + ":".join(parts)
        await env.set_env_variables({"PATH": base_path})

    # Getting state
    # -------------

    async def _get_state(self, env: ContainerEnv) -> dict[str, str]:
        """Retrieve the state from the environment"""
        try:
            state_str = await env.read_file("/root/state.json")
        except FileNotFoundError:
            self.logger.warning("State file not found, returning empty state")
            return {}
        if not state_str.strip():
            self.logger.warning("State file is empty, returning empty state")
            return {}
        try:
            state = json.loads(state_str)
        except json.JSONDecodeError as e:
            msg = f"State {state_str!r} is not valid json. This is an internal error, please report it."
            raise ValueError(msg) from e
        if not isinstance(state, dict):
            msg = f"State commands must return a dictionary. Got {state!r} instead."
            raise ValueError(msg)
        return state

    async def get_state(self, env: ContainerEnv) -> dict[str, str]:
        """Execute state commands from all bundles and combine their results.

        Optimization: in stateless E2B mode, _bash_one_shot already writes
        state.json on every command execution and tracks cwd in-memory.
        Skip the extra communicate + read_file round-trips (saves 3 HTTP calls/step).
        """
        if self.mock_state is not None:
            return self.mock_state

        # Fast path: E2B shim tracks cwd in-memory, state.json already written by _bash_one_shot
        if hasattr(env, "_cwd"):
            return {"working_dir": env._cwd}

        for _, state_command in enumerate(self.config.state_commands):
            await env.communicate(state_command, check="warn")

        combined_state = await self._get_state(env)
        return combined_state
