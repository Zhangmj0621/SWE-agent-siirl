# Support dataset constructed from swefactory
# Eval logic is simplified to eval script return code.

from pydantic import BaseModel, Field

from ..base import SWESample
from ..environment import ContainerEnv, ContainerStartArgs
from .base import Runtime, RuntimeBuilder, SWESampleData


class SFSample(BaseModel):
    instance_id: str
    problem_statement: str
    base_commit: str
    image_name: str
    repo: str = Field(default="")
    patch: str = Field(default="")
    test_patch: str = Field(default="")
    hints_test: str = Field(default="")
    eval_script: str = Field(default="")


class SWEFactoryRuntime(Runtime):
    def __init__(self, sample: SWESample):
        self.sample = sample
        self.m = sample.m

    async def bootstrap(self, env: ContainerEnv):
        await env.execute(f"git checkout {self.sfsample.base_commit}")
        # Note: we avoid stdin=BytesIO(...) because K8sEnvAdapter.execute does not
        # support stdin; use write_file + git apply <file> instead.
        await env.write_file("/tmp/test.patch", self.sfsample.test_patch)
        await env.execute("git apply --verbose --reject /tmp/test.patch", check=False)

    async def diff(self, env: ContainerEnv):
        output = await env.execute("git add -A && git diff --cached")
        self.m.rollout.patch = output.output

    async def eval(self, env: ContainerEnv):
        # apply patch
        if self.m.rollout.patch is None:
            raise RuntimeError("must run diff before patch")
        await env.execute(f"git checkout {self.sfsample.base_commit}")
        patch_str = (
            self.m.rollout.patch.decode("utf-8", errors="replace")
            if isinstance(self.m.rollout.patch, bytes)
            else self.m.rollout.patch
        )
        await env.write_file("/tmp/model.patch", patch_str)
        await env.execute("git apply --verbose --reject /tmp/model.patch", check=False)

        # run eval script
        await env.write_file("/eval.sh", self.sfsample.eval_script)
        output = await env.execute("bash /eval.sh", check=False)
        # naive reward
        if output.returncode == 0:
            self.sample.reward = 1.0
        else:
            self.sample.reward = 0.0

    @property
    def sfsample(self) -> SFSample:
        return self.m.data.runtime_meta


class SWEFactoryBuiler(RuntimeBuilder):
    def __init__(self, config: dict):
        pass

    def parse_sampledata(self, sample: dict) -> SWESampleData:
        s = SFSample.model_validate(sample)
        container = ContainerStartArgs(image=s.image_name, cwd="/testbed")
        return SWESampleData(
            container_args=container,
            problem_statement=s.problem_statement,
            runtime_meta=s,
        )

    def build(self, sample: SWESample) -> Runtime:
        return SWEFactoryRuntime(sample)
