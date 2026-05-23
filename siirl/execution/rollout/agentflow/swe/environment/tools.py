import asyncio
import json
import os

from sweagent.tools.tools import ToolConfig, ToolHandler

from .base import ContainerEnv


# patch ToolHandler, modify env call func
class SiiToolHandler(ToolHandler):
    def __init__(self, tools: ToolConfig):
        super().__init__(tools)

    async def install(self, env: ContainerEnv) -> None:
        await self._install_commands(env)
        await self.reset(env)

    async def reset(self, env: ContainerEnv) -> None:
        self.logger.info("Resetting tools")
        env_variables = self.config.env_variables.copy() | {var: os.getenv(var) for var in self.config.propagate_env_variables}
        await env.set_env_variables(env_variables)
        await env.write_file("/root/.swe-agent-env", json.dumps(self.config.registry_variables))
        await env.write_file("/root/state.json", "{}")
        await env.communicate(" && ".join(self._reset_commands), check="raise", timeout=self.config.install_timeout)

    async def _upload_bundles(self, env: ContainerEnv) -> None:
        await asyncio.gather(
            *(
                env.upload(source_path=bundle.path.as_posix(), target_path=f"/root/tools/{bundle.path.name}")
                for bundle in self.config.bundles
            )
        )

    async def _is_command_available(self, env: ContainerEnv, command: str, env_vars: dict[str, str]) -> None:
        if command == "bash":
            return
        try:
            # Use execute instead of communicate to match original SWE-agent behavior
            # execute runs in a subprocess with env vars, not in the bash session
            result = await env.execute(f"which {command}", env=env_vars, timeout=30.0, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"Command which {command} failed with exit code {result.returncode}")

        except Exception:
            msg = f"Tool {command} is not available in the container."
            raise RuntimeError(msg) from None

    async def _check_available_commands(self, env: ContainerEnv, env_vars: dict[str, str]) -> None:
        await asyncio.gather(*(self._is_command_available(env, command.name, env_vars) for command in self.config.commands))

    async def _install_commands(self, env: ContainerEnv) -> None:
        """Make sure all commands are available in the container"""
        await env.set_env_variables(self.config.env_variables)
        cwd = (await env.communicate("pwd", check="raise")).strip()

        # Test communicate method with simple commands (allow tools directory to not exist yet)
        # test_pwd = await env.communicate(
        #     "pwd && echo 'TEST_SUCCESS' && ls /root/tools || echo 'tools directory not created yet'",
        #     check="raise",
        #     timeout=10.0,
        # )
        # self.logger.info(f"pwd test result:\n{test_pwd}")

        await self._upload_bundles(env)

        # Verify upload succeeded - check if files exist in container
        # self.logger.info("Checking if bundles were uploaded successfully...")
        # for bundle in self.config.bundles:
        #     target_path = f"/root/tools/{bundle.path.name}"
        #     try:
        #         # Check if directory exists
        #         ls_result = await env.communicate(f"ls -la {target_path}", check="warn", timeout=10.0)
        #         self.logger.info(f"ls -la {target_path}:\n{ls_result}")

        #         # Check if bin directory exists
        #         bin_path = f"{target_path}/bin"
        #         bin_ls = await env.communicate(f"ls -la {bin_path} 2>&1", check="warn", timeout=10.0)
        #         self.logger.info(f"ls -la {bin_path}:\n{bin_ls}")

        #         # Check if str_replace_editor exists in this bundle
        #         if bundle.path.name == "edit_anthropic":
        #             str_replace_check = await env.communicate(f"ls -la {bin_path}/str_replace_editor 2>&1", check="warn", timeout=10.0)
        #             self.logger.info(f"str_replace_editor check:\n{str_replace_check}")
        #     except Exception as e:
        #         self.logger.error(f"Failed to verify bundle {bundle.path.name}: {e}")

        for bundle in self.config.bundles:
            bin_path = f"/root/tools/{bundle.path.name}/bin"

            cmds = [
                f"export PATH=/root/tools/{bundle.path.name}/bin:$PATH",
                f"chmod +x /root/tools/{bundle.path.name}/bin/* 2>&1 || echo 'CHMOD_FAILED'",
            ]
            if (bundle.path / "install.sh").exists():
                cmds.append(f"cd /root/tools/{bundle.path.name} && . ./install.sh")
            cmds.append(f"chmod +x /root/tools/{bundle.path.name}/bin/* 2>&1 || echo 'CHMOD_FAILED'")
            cmds.append(f"ls -la {bin_path} 2>&1 || echo 'LS_FAILED'")

            cmd_str = " && ".join(cmds)

            try:
                await env.communicate(
                    cmd_str,
                    check="raise",
                    timeout=self.config.install_timeout,
                )
            except Exception as e:
                self.logger.error(f"Commands FAILED for {bundle.path.name}: {e}")
                raise

            await env.communicate(f"ls -la {bin_path} 2>&1 || echo 'DIRECTORY_NOT_FOUND'", check="warn", timeout=10.0)
        await env.communicate(f"cd {cwd}", check="raise")

        # In stateless mode (E2B), each communicate() is an independent bash -lc
        # process — PATH exports within one call don't persist to the next.
        # Explicitly prepend all tool bin directories to the PATH.
        base_path = (await env.communicate("echo $PATH", check="raise")).strip()
        tool_bin_dirs = [f"/root/tools/{bundle.path.name}/bin" for bundle in self.config.bundles]
        for d in tool_bin_dirs:
            if d not in base_path:
                base_path = f"{d}:{base_path}"
        await env.set_env_variables({"PATH": base_path})
        await self._check_available_commands(env, {"PATH": base_path})

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
        This can be used to extract environment variables etc. from the environment.
        """
        if self.mock_state is not None:
            return self.mock_state

        for _, state_command in enumerate(self.config.state_commands):
            await env.communicate(state_command, check="warn")

        combined_state = await self._get_state(env)
        return combined_state
