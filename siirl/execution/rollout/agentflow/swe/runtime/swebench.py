# Support swebench dataset
# must have run build_env_images on dataset before run.

import tempfile
from dataclasses import dataclass
from io import BytesIO
from typing import cast

from pydantic import BaseModel, Field
from swebench.harness.constants import LATEST, SWEbenchInstance
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec

from ..base import SWESample
from ..environment import ContainerEnv, ContainerStartArgs
from .base import Runtime, RuntimeBuilder, SWESampleData


class SWEBenchConfig(BaseModel):
    namespace: str | None = Field(default=None)
    base_image_tag: str = LATEST
    env_image_tag: str = LATEST
    instance_image_tag: str = LATEST
    arch: str = "x86_64"


@dataclass(frozen=True)
class SBSample:
    instance: SWEbenchInstance
    spec: TestSpec


class SWEBenchRuntime(Runtime):
    """See swebench.harness.run_evaluation:run_instance"""

    def __init__(self, sample: SWESample):
        self.sample = sample
        self.m = sample.m

    async def bootstrap(self, env: ContainerEnv):
        await self._bootstrap_container(env)

    async def diff(self, env: ContainerEnv):
        output = await env.execute("git add -A && git diff --cached")
        self.m.rollout.patch = output.output

    async def eval(self, env: ContainerEnv):
        await self._bootstrap_container(env)
        # apply patch
        if self.m.rollout.patch is None:
            raise RuntimeError("must run diff before patch")
        await env.execute(f"git checkout {self.instance['base_commit']}")
        stdin = BytesIO(self.m.rollout.patch)
        await env.execute("git apply --verbose --reject -", stdin=stdin)

        # run eval script
        stdin = BytesIO(self.spec.eval_script.encode("utf-8"))
        await env.execute("cat > /eval.sh", stdin=stdin)
        output = await env.execute("bash /eval.sh", check=False)

        with tempfile.NamedTemporaryFile() as f:
            prediction = {
                "instance_id": self.spec.instance_id,
                "model_patch": "",  # placeholder
            }
            f.write(output.output)
            report = get_eval_report(self.spec, prediction, f.name, True)
            report = report[self.spec.instance_id]
            """
            example schema:
            {
                "patch_is_None": False,
                "patch_exists": True,
                "patch_successfully_applied": True,
                "resolved": True,
                "tests_status": {
                    "FAIL_TO_PASS": {
                        "success": ["test_case_1", "test_case_3"],
                        "failure": ["test_case_2"],
                    },
                    "PASS_TO_PASS": {"success": ["test_case_4", "test_case_5"], "failure": []},
                    "FAIL_TO_FAIL": {"success": [], "failure": []}, # or None
                    "PASS_TO_FAIL": {"success": [], "failure": []}, # or None
                },
            }
            """
        # naive reward
        if report["resolved"]:
            self.sample.reward = 1.0
        else:
            self.sample.reward = 0.0

    @property
    def spec(self) -> TestSpec:
        return self.m.data.runtime_meta.spec

    @property
    def instance(self) -> SWEbenchInstance:
        return self.m.data.runtime_meta.instance

    async def _bootstrap_container(self, env: ContainerEnv):
        # rewrite of swebench.harness.docker_build:build_instance_image
        # may optimize if env provide image build interface
        stdin = BytesIO(self.spec.setup_env_script.encode("utf-8"))
        await env.execute("cat - | bash", stdin=stdin)
        stdin = BytesIO(self.spec.install_repo_script.encode("utf-8"))
        await env.execute("cat - | bash", stdin=stdin)

        # apply test patch
        await env.execute(f"git checkout {self.instance['base_commit']}")
        stdin = BytesIO(self.instance["test_patch"].encode("utf-8"))
        await env.execute("git apply --verbose --reject -", stdin=stdin)

    def bootstrap_sync(self, env: ContainerEnv):
        """Synchronous version of bootstrap() - for use in thread pool."""
        return self._bootstrap_container_sync(env)

    def _bootstrap_container_sync(self, env: ContainerEnv):
        """Synchronous version of _bootstrap_container()."""
        # rewrite of swebench.harness.docker_build:build_instance_image
        # may optimize if env provide image build interface
        # Note: K8s adapter doesn't support stdin, so write to file first
        env.write_file_sync("/tmp/setup_env.sh", self.spec.setup_env_script)
        env.execute_sync("cat /tmp/setup_env.sh | bash")
        env.write_file_sync("/tmp/install_repo.sh", self.spec.install_repo_script)
        env.execute_sync("cat /tmp/install_repo.sh | bash")

        # apply test patch
        env.execute_sync(f"git checkout {self.instance['base_commit']}")
        env.write_file_sync("/tmp/test.patch", self.instance["test_patch"])
        env.execute_sync("git apply --verbose --reject - < /tmp/test_patch")

    def diff_sync(self, env: ContainerEnv):
        """Synchronous version of diff() - for use in thread pool."""
        output = env.execute_sync("git add -A && git diff --cached")
        self.m.rollout.patch = output.output

    def eval_sync(self, env: ContainerEnv):
        """Synchronous version of eval() - for use in thread pool."""
        self._bootstrap_container_sync(env)
        # apply patch
        if self.m.rollout.patch is None:
            raise RuntimeError("must run diff before patch")
        env.execute_sync(f"git checkout {self.instance['base_commit']}")
        # Note: write_file_sync doesn't support stdin, so write to file
        env.write_file_sync("/tmp/model.patch", self.m.rollout.patch)
        env.execute_sync("git apply --verbose --reject - < /tmp/model.patch")

        # run eval script
        env.write_file_sync("/eval.sh", self.spec.eval_script)
        output = env.execute_sync("bash /eval.sh", check=False)

        with tempfile.NamedTemporaryFile() as f:
            prediction = {
                "instance_id": self.spec.instance_id,
                "model_patch": "",  # placeholder
            }
            f.write(output.output)
            report = get_eval_report(self.spec, prediction, f.name, True)
            report = report[self.spec.instance_id]

        # naive reward
        if report["resolved"]:
            self.sample.reward = 1.0
        else:
            self.sample.reward = 0.0


class SWEBenchBuiler(RuntimeBuilder):
    def __init__(self, config: dict):
        conf = SWEBenchConfig.model_validate(config)
        self.config = conf.model_dump()

    def parse_sampledata(self, sample: dict) -> SWESampleData:
        # Extract fields from extra_info if present (parquet format)
        if "extra_info" in sample and isinstance(sample["extra_info"], dict):
            sample = {**sample, **sample["extra_info"]}

        s = cast(SWEbenchInstance, sample)
        spec = make_test_spec(s, **self.config)

        container = ContainerStartArgs(
            image=spec.base_image_key,
            cwd="/testbed",
        )

        return SWESampleData(
            container_args=container,
            problem_statement=s["problem_statement"],
            runtime_meta=SBSample(s, spec),
        )

    def build(self, sample: SWESample) -> Runtime:
        return SWEBenchRuntime(sample)
