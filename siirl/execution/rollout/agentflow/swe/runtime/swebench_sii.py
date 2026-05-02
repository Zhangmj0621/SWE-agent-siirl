# Support swebench verified dataset, using sii configs (with prebuilt images)

import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel
from swebench.harness.constants import SWEbenchInstance
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec

from ..environment import ContainerEnv, ContainerStartArgs
from .base import Runtime, RuntimeBuilder, SWESampleData

# Avoid circular import
if TYPE_CHECKING:
    from ..base import SWESample


@dataclass(frozen=True)
class SBSample:
    instance: SWEbenchInstance
    spec: TestSpec


class SWEBenchConfig(BaseModel):
    pass


class SWEBenchRuntime(Runtime):
    """See swebench.harness.run_evaluation:run_instance"""

    def __init__(self, sample: "SWESample"):
        self.sample = sample
        self.m = sample.m

    async def bootstrap(self, env: ContainerEnv):
        pass

    async def diff(self, env: ContainerEnv):
        output = await env.execute("git add -A && git diff --cached")
        self.m.rollout.patch = output.output

    async def eval(self, env: ContainerEnv):
        # apply patch
        if self.m.rollout.patch is None:
            raise RuntimeError("must run diff before patch")
        patch_str = (
            self.m.rollout.patch.decode("utf-8", errors="replace")
            if isinstance(self.m.rollout.patch, bytes)
            else self.m.rollout.patch
        )
        # Note: we avoid stdin=BytesIO(...) because K8sEnvAdapter.execute does not
        # support stdin; use write_file + bash/git <file> instead.
        await env.write_file("/tmp/model.patch", patch_str)
        await env.execute("git apply --verbose --reject /tmp/model.patch", check=False)

        # run eval script
        await env.write_file("/eval.sh", self.spec.eval_script)
        output = await env.execute("bash /eval.sh", check=False)

        with tempfile.NamedTemporaryFile() as f:
            prediction = {
                "instance_id": self.spec.instance_id,
                "model_patch": self.m.rollout.patch,  # placeholder
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


def get_swebench_docker_image_name(iid: str, data_source: str = "swerebench") -> str:
    """Get the image name for a SWEBench instance.

    Args:
        iid: Instance ID (e.g., "django__django-12345")
        data_source: Data source type, "swefactory" or "swerebench"
    """
    # swefactory uses "swefactory" namespace, swerebench uses "swebench" namespace
    namespace = "swefactory" if data_source == "swefactory" else "swebench"
    # First lowercase the instance_id, then replace __ with _1776_ (must be after lower())
    iid_lower = iid.lower()
    id_docker_compatible = iid_lower.replace("__", "_1776_")
    return f"docker.io/{namespace}/sweb.eval.x86_64.{id_docker_compatible}:latest"


class SWEBenchBuiler(RuntimeBuilder):
    def __init__(self, config: dict):
        self.config = SWEBenchConfig.model_validate(config)

    def parse_sampledata(self, sample: dict) -> SWESampleData:
        # Extract data_source before merging extra_info
        data_source = sample.get("data_source", "swerebench")

        # Extract fields from extra_info if present (parquet format)
        if "extra_info" in sample and isinstance(sample["extra_info"], dict):
            sample = {**sample, **sample["extra_info"]}

        s = cast(SWEbenchInstance, sample)
        spec = make_test_spec(s)
        _ = spec.install_repo_script  # Trigger property initialization
        image = get_swebench_docker_image_name(s["instance_id"], data_source)
        # Note: Don't map to ACR here - K8sDeployment.start() will handle it if use_acr=True
        # TODO: use prebuilt image by now
        container = ContainerStartArgs(
            image=image,
            cwd="/testbed",
        )
        return SWESampleData(
            container_args=container,
            problem_statement=s["problem_statement"],
            runtime_meta=SBSample(s, spec),
            issue_images=s.get("issue_images"),
        )

    def build(self, sample: "SWESample") -> Runtime:
        return SWEBenchRuntime(sample)
