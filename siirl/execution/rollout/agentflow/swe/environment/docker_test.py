import io

import pytest

from .base import ContainerStartArgs
from .docker import DockerEnvBuilder


@pytest.mark.asyncio
async def test_docker_env_minimal():
    # Pull alpine:latest (docker will pull if not present)
    builder = DockerEnvBuilder(conf={})
    args = ContainerStartArgs(
        image="alpine:latest",
        cmd=None,
        cwd="/",
        env={},
        forward_env=[],
        container_timeout="60",  # 1 minute
        startup_timeout=60.0,
    )
    env = await builder.start(args)
    try:
        # 1. Run a simple command
        out = await env.execute("echo hello")
        assert out.returncode == 0
        assert b"hello" in out.output

        # 2. Run a command that reads from stdin and writes to stdout
        stdin_data = b"testinput\n"
        out2 = await env.execute("cat", stdin=io.BytesIO(stdin_data))
        assert out2.returncode == 0
        assert b"testinput" in out2.output

        # 3. Timeout handling: run a sleep command with a short timeout
        with pytest.raises(TimeoutError):
            await env.execute("sleep 10", timeout=0.5)

        # 4. Check that nonzero exit code raises if check=True
        with pytest.raises(Exception):
            await env.execute("sh -c 'exit 42'", check=True)

        # 5. Check that nonzero exit code does not raise if check=False
        out3 = await env.execute("sh -c 'exit 42'", check=False)
        assert out3.returncode == 42

        input("Press enter to continue")

    finally:
        await env.cleanup()
