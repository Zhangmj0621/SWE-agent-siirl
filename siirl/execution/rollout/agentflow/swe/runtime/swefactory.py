# Support dataset constructed from swefactory
# Eval logic is simplified to eval script return code.

from io import BytesIO
from pydantic import BaseModel, Field

from .base import SWESampleData, Runtime, RuntimeBuilder
from ..environment import ContainerStartArgs, ContainerEnv
from ..base import SWESample


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
        stdin = BytesIO(self.sfsample.test_patch.encode("utf-8"))
        await env.execute("git apply --verbose --reject -", stdin=stdin)

    async def diff(self, env: ContainerEnv):
        output = await env.execute("git add -A && git diff --cached")
        self.m.rollout.patch = output.output

    async def eval(self, env: ContainerEnv):
        # apply patch
        if self.m.rollout.patch is None:
            raise RuntimeError("must run diff before patch")
        await env.execute(f"git checkout {self.sfsample.base_commit}")
        stdin = BytesIO(self.m.rollout.patch)
        await env.execute("git apply --verbose --reject -", stdin=stdin)

        # run eval script
        stdin = BytesIO(self.sfsample.eval_script.encode("utf-8"))
        await env.execute("cat > /eval.sh", stdin=stdin)
        output = await env.execute("bash /eval.sh", check=False)
        # naive reward
        if output.returncode == 0:
            self.sample.reward = 1.0
        else:
            self.sample.reward = 0.0
    
    @property
    def sfsample(self)->SFSample:
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
