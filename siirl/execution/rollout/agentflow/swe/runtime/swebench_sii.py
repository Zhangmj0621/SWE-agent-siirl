# Support swebench verified dataset, using sii configs (with prebuilt images)

import os
import tempfile
from dataclasses import dataclass
from io import BytesIO
from typing import cast

from pydantic import BaseModel
from swebench.harness.constants import SWEbenchInstance
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec

from ..base import SWESample
from ..environment import ContainerEnv, ContainerStartArgs
from .base import Runtime, RuntimeBuilder, SWESampleData


@dataclass(frozen=True)
class SBSample:
    instance: SWEbenchInstance
    spec: TestSpec


class SWEBenchConfig(BaseModel):
    pass


class SWEBenchRuntime(Runtime):
    """See swebench.harness.run_evaluation:run_instance"""

    def __init__(self, sample: SWESample):
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
        stdin = BytesIO(self.m.rollout.patch)
        await env.execute("git apply --verbose --reject -", stdin=stdin)

        # run eval script
        stdin = BytesIO(self.spec.eval_script.encode("utf-8"))
        await env.execute("cat > /eval.sh", stdin=stdin)
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


# using aliyun container registry
ACR_REGISTRY = os.environ["ACR_REGISTRY"]
ACR_NAMESPACE = os.environ["ACR_NAMESPACE"]


def map_image_to_acr(image: str) -> str:
    # Ensure image has a tag
    if ":" in image:
        image_part, tag = image.rsplit(":", 1)
    else:
        image_part, tag = image, "latest"

    # Strip registry prefix if present (e.g., docker.io/, ghcr.io/, quay.io/, localhost:5000/)
    segments = image_part.split("/")
    if len(segments) > 1 and ("." in segments[0] or ":" in segments[0] or segments[0] == "localhost"):
        image_part = "/".join(segments[1:])

    # Replace "/" with "--" to flatten into a single repo tag
    if tag in image:
        acr_tag = image_part.replace("/", "--") + f"--{tag}"
    else:
        acr_tag = image_part.replace("/", "--")
    return f"{ACR_REGISTRY}/{ACR_NAMESPACE}:{acr_tag}"


def get_swebench_docker_image_name(iid: str) -> str:
    """Get the image name for a SWEBench instance."""
    id_docker_compatible = iid.replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()


class SWEBenchBuiler(RuntimeBuilder):
    def __init__(self, config: dict):
        self.config = SWEBenchConfig.model_validate(config)

    def parse_sampledata(self, sample: dict) -> SWESampleData:
        s = cast(SWEbenchInstance, sample)
        spec = make_test_spec(s)
        spec.install_repo_script
        image = get_swebench_docker_image_name(s["instance_id"])
        image = map_image_to_acr(image)
        # TODO: use prebuilt image by now
        container = ContainerStartArgs(
            image=image,
            cwd="/testbed",
        )

        return SWESampleData(
            container_args=container,
            problem_statement=s["problem_statement"],
            runtime_meta=SBSample(s, spec),
        )

    def build(self, sample: SWESample) -> Runtime:
        return SWEBenchRuntime(sample)
