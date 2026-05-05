# SWE-bench evaluation using dedicated K8s pods for RL training
# This implementation isolates evaluation from agent execution using separate pods

import asyncio
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from loguru import logger
from pydantic import BaseModel
from swebench.harness.constants import LOG_REPORT, LOG_TEST_OUTPUT, MAP_REPO_VERSION_TO_SPECS, RUN_EVALUATION_LOG_DIR, SWEbenchInstance
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec
from swebench.harness.utils import ensure_trailing_newline

from ..environment import ContainerEnv
from .base import Runtime, RuntimeBuilder, SWESampleData

# Import run_instance_k8s_for_rl from SWE-bench
try:
    from swebench.harness.run_evaluation_k8s import run_instance_k8s_for_rl
except ImportError:
    logger.warning("[K8S_EVAL] Failed to import run_instance_k8s_for_rl, fallback to local implementation")
    run_instance_k8s_for_rl = None

# Avoid circular import
if TYPE_CHECKING:
    from ..base import SWESample


def _sanitize_run_id_part(value: object, max_len: int) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value)).strip("-")
    return (safe or "unknown")[:max_len].strip("-") or "unknown"


def _make_trajectory_run_id(base_run_id: str, trajectory_id: str) -> str:
    base = _sanitize_run_id_part(base_run_id, 40)
    traj = _sanitize_run_id_part(trajectory_id, 24)
    return f"{base}-t-{traj}"


async def check_git_repo_status(env: ContainerEnv, timeout: int = 10) -> bool:
    """校验 git 仓库状态，返回 True=有效，False=无效"""
    try:
        # 检查是否在 git 仓库内（最快的校验方式）
        await env.execute("git rev-parse --is-inside-work-tree", check=True, timeout=timeout)  # 非 git 仓库会返回非0退出码，触发异常
        # 检查 git 仓库是否被锁定（避免 git add 卡住）
        lock_check = await env.execute("bash -c 'test -f .git/index.lock && echo locked || echo ok'", check=False, timeout=timeout)
        if "locked" in lock_check.output.decode("utf-8"):
            logger.warning("git 仓库被锁定（.git/index.lock 存在），清理锁文件")
            # 清理锁文件（避免 git 操作卡住）
            await env.execute("rm -f .git/index.lock", check=False, timeout=timeout)
        return True
    except Exception as e:
        logger.error(f"git 仓库无效: {e}")
        return False


async def has_git_changes(env: ContainerEnv, timeout: int = 10) -> bool:
    """检查工作区是否有 git 可追踪的修改"""
    try:
        # git diff --quiet：有修改则退出码1，无修改则0；--exit-code 等价于 --quiet
        diff_check = await env.execute("git diff --quiet --exit-code", check=False, timeout=timeout)
        # 退出码 1 = 有修改，0 = 无修改
        return diff_check.returncode != 0
    except Exception as e:
        logger.warning(f"检查 git 修改失败，默认认为有修改: {e}")
        return True


async def extract_git_patch_safely(env: ContainerEnv, timeout: int = 30) -> bytes:
    if not await check_git_repo_status(env, timeout=timeout):
        return b""
    if not await has_git_changes(env, timeout=timeout):
        logger.info("无 git 修改，返回空补丁")
        return b""
    git_common_opts = "-c core.askpass=false -c user.name='temp' -c user.email='temp@example.com'"
    try:
        await env.execute(f"git {git_common_opts} add -A", check=False, timeout=timeout)
        diff_output = await env.execute(f"git {git_common_opts} diff --cached", check=False, timeout=timeout)
        return diff_output.output
    except Exception as e:
        logger.error(f"提取补丁失败: {e}")
        return b""


@dataclass(frozen=True)
class SBSample:
    instance: dict  # SWEbenchInstance
    spec: TestSpec  # Environment builder will use this to create ContainerStartArgs


class SWEBenchConfig(BaseModel):
    allow_partial_patch: bool = True  # Allow evaluation to continue even if patch doesn't apply cleanly
    run_id: str = "default"  # Run ID for logging directory structure
    model_name: str = "swe-agent-rl"  # Model name for logging directory structure
    save_eval_logs: bool = True  # Whether to save evaluation logs (patch.diff, eval.sh, test_output.txt, report.json)
    use_run_instance_k8s_for_rl: bool | None = (
        None  # Whether to use run_instance_k8s_for_rl from SWE-bench.
        # None=auto-detect, True=force use, False=force local implementation
    )


class SWEBenchK8sEvalRuntime(Runtime):
    """
    SWE-bench Runtime with K8s-based isolated evaluation.

    Key difference from SWEBenchRuntime:
    - Agent runs in one pod (during generate phase)
    - Evaluation runs in a SEPARATE pod (during reward phase)
    - The env passed to eval() is the isolated evaluation pod

    When using run_instance_k8s_for_rl:
    - No external env needs to be created (run_instance_k8s_for_rl creates its own pod)
    - The reward() phase can skip pod creation entirely
    """

    def __init__(self, sample: "SWESample", config: SWEBenchConfig, full_config: dict | None = None):
        self.sample = sample
        self.m = sample.m
        self.config = config
        self._config = full_config or {}  # Full config dict (for accessing k8s_namespace, etc.)
        # spec is accessed via @property that reads from runtime_meta

    @property
    def _use_run_instance_k8s_for_rl(self) -> bool:
        """
        Determine whether to use run_instance_k8s_for_rl from SWE-bench.

        Returns:
            bool: True if run_instance_k8s_for_rl should be used, False otherwise
        """
        config_value = self.config.use_run_instance_k8s_for_rl

        if config_value is True:
            # Force use run_instance_k8s_for_rl
            if run_instance_k8s_for_rl is None:
                logger.warning(
                    "[K8S_EVAL] Config forces use_run_instance_k8s_for_rl=True "
                    "but function is not available, falling back to local implementation"
                )
                return False
            return True
        elif config_value is False:
            # Force use local implementation
            return False
        else:
            # Auto-detect: use run_instance_k8s_for_rl if available
            return run_instance_k8s_for_rl is not None

    @property
    def needs_external_env(self) -> bool:
        """
        Check if the eval method needs an external env to be created.

        Returns:
            bool: True if external env is needed, False if run_instance_k8s_for_rl will handle pod creation
        """
        return not self._use_run_instance_k8s_for_rl

    def _get_trajectory_id(self) -> str:
        """
        Get a unique identifier for this trajectory to avoid file conflicts in GRPO training.

        In GRPO, multiple trajectories (default n=8) are generated for the same instance_id.
        Each trajectory needs its own log directory to avoid patch/report/test_output conflicts.

        Priority:
        1. uid from rollout data (set by GRPO data preprocessing)
        2. Python object ID as fallback

        Returns:
            str: Unique trajectory identifier
        """
        # Try to get uid from sample (GRPO sets this in preprocess_dataloader)
        # The uid is an integer (0, 1, 2, ...) for each trajectory in a GRPO group
        uid = getattr(self.sample, "uid", None)

        if uid is not None:
            return str(uid)

        # Fallback: use Python object ID (unique but not predictable)
        # This shouldn't happen in normal GRPO training, but provides safety
        return f"obj_{id(self.sample)}"

    async def _do_bootstrap(self, env: ContainerEnv):
        """Bootstrap environment - minimal to avoid resource exhaustion.

        IMPORTANT: With 64 concurrent samples, executing git commands here creates
        64+ temporary threads that exhaust system resources. The start() method
        already handles all initialization including reset, so we skip verification.
        """
        # Skip all git operations to avoid:
        # - Creating 64+ concurrent threads for git commands
        # - Network bottlenecks from simultaneous kubectl/git commands
        # - Git repository lock contention

        # The start() method already handled:
        # - Deployment initialization
        # - Bash session creation with env variables
        # - reset() which did cd /, _copy_repo(), _reset_repository()

        logger.debug("[K8S_BOOTSTRAP] Skipped verification (start() already initialized)")
        return

    async def diff(self, env: ContainerEnv):
        """
        Extract patch from git.

        This runs in the AGENT pod after agent completion.
        """
        if not self.m.rollout.patch:
            # Use check=False to avoid exception when git diff --cached returns empty (no modifications)
            output = await env.execute("git add -A && git diff --cached", check=False)
            logger.info(f"[K8S_EVAL_DIFF] Extracted patch: {len(output.output)} bytes, exit_code={output.returncode}")
            if len(output.output) == 0:
                logger.warning("[K8S_EVAL_DIFF] No patch extracted! Git diff returned empty.")
                # Try to get git status for debugging
                try:
                    status_output = await env.execute("git status", check=False)
                    logger.debug(f"[K8S_EVAL_DIFF] Git status:\n{status_output.output.decode('utf-8', errors='replace')}")
                except Exception as e:
                    logger.warning(f"[K8S_EVAL_DIFF] Failed to get git status: {e}")
            else:
                logger.debug(f"[K8S_EVAL_DIFF] Patch preview:\n{output.output[:500].decode('utf-8', errors='replace')}")
            self.m.rollout.patch = output.output

    async def eval(self, env: ContainerEnv | None):
        """
        Evaluate patch using the provided evaluation pod (fully async).

        IMPORTANT: The env parameter is a SEPARATE pod started in reward() phase.
        This ensures isolation between agent and evaluation.

        When using run_instance_k8s_for_rl:
        - env can be None (run_instance_k8s_for_rl will create its own pod)
        """
        instance_id = self.spec.instance_id
        logger.info(f"[K8S_EVAL] Starting evaluation for {instance_id}")

        # Check patch status but continue with evaluation even if empty
        if self.m.rollout.patch is None:
            logger.warning("[K8S_EVAL] Patch is None, will evaluate base state")
            self.m.rollout.patch = b""

        patch_size = len(self.m.rollout.patch) if isinstance(self.m.rollout.patch, bytes | str) else 0
        if patch_size == 0:
            logger.warning(
                f"[K8S_EVAL] Patch is empty ({type(self.m.rollout.patch).__name__}), "
                f"will evaluate base state (no changes applied)"
            )
        else:
            logger.info(f"[K8S_EVAL] Patch size: {patch_size} bytes, will apply and evaluate")

        if self._use_run_instance_k8s_for_rl:
            logger.info("[K8S_EVAL] Using run_instance_k8s_for_rl from SWE-bench (will create its own pod)")
            await self._eval_with_run_instance_k8s()
            return
        logger.info("[K8S_EVAL] Using local implementation (requires external env)")
        if env is None:
            logger.error("[K8S_EVAL] Local implementation requires env but got None!")
            self.sample.reward = 0.0
            return
        await self._eval_local(env)

    # Patch application strategies (same as run_instance_k8s_for_rl)
    GIT_APPLY_CMDS = [
        "git apply --verbose",
        "git apply --verbose --reject",
        "patch --forward --batch --fuzz=5 -p1 -i",
    ]

    async def _eval_with_run_instance_k8s(self):
        """
        Use run_instance_k8s_for_rl from SWE-bench to evaluate the patch.

        This function creates a separate K8s pod for evaluation, ensuring
        complete isolation from the agent execution environment.
        """
        instance_id = self.spec.instance_id

        trajectory_id = self._get_trajectory_id()
        run_id = f"{self.spec.instance_id}/{trajectory_id}"
        run_id = run_id.replace("/", "-").replace("_", "-")  # K8s-safe
        # Prepare prediction dict for run_instance_k8s_for_rl
        pred = {
            "instance_id": instance_id,
            "model_name_or_path": self.config.model_name,
            "model_patch": self.m.rollout.patch,
        }

        # Prepare logger for run_instance_k8s_for_rl
        # run_instance_k8s_for_rl expects a standard logging.Logger, not loguru logger
        # Create a standard logger with appropriate name
        eval_logger = logging.getLogger(f"swebench.k8s_eval.{instance_id}")
        eval_logger.setLevel(logging.INFO)

        try:
            # Call run_instance_k8s_for_rl
            # Get namespace from environment config (same as K8sEnvAdapter)
            namespace = self._config.get("environment", {}).get("namespace", "swe-siirl")
            # run_instance_k8s_for_rl is a blocking SWE-bench call that talks to the
            # K8s API for minutes; offload to a worker thread so the event loop stays
            # free for other rollouts.
            result = await asyncio.to_thread(
                run_instance_k8s_for_rl,
                test_spec=self.spec,
                pred=pred,
                run_id=run_id,
                namespace=namespace,
                timeout=1800,
                dataset_image_name=None,
                logger=eval_logger,
                allow_partial_patch=self.config.allow_partial_patch,
            )

            # Extract result and set reward
            # run_instance_k8s_for_rl returns: {"completed": bool, "resolved": bool, "error": str}
            completed = result.get("completed", False)
            resolved = result.get("resolved", False)
            error = result.get("error", "")

            # Set reward based on completion status and resolution
            # Only reward=1.0 if evaluation completed successfully AND tests passed
            if completed and resolved:
                self.sample.reward = 1.0
                logger.info(f"[K8S_EVAL] {instance_id} ✓ RESOLVED")
            elif completed and not resolved:
                self.sample.reward = 0.0
                logger.info(f"[K8S_EVAL] {instance_id} ✗ NOT RESOLVED (tests failed)")
            elif not completed:
                self.sample.reward = 0.0
                logger.warning(f"[K8S_EVAL] {instance_id} ✗ EVALUATION FAILED (completed={completed}, resolved={resolved})")
                if error:
                    logger.warning(f"[K8S_EVAL] Error: {error}")
            else:
                # Should not reach here, but handle gracefully
                self.sample.reward = 0.0
                logger.warning(f"[K8S_EVAL] {instance_id} ✗ UNEXPECTED STATE (completed={completed}, resolved={resolved})")

            # Store metadata
            if not hasattr(self.m, "eval_metadata"):
                self.m.eval_metadata = {}

            self.m.eval_metadata.update(
                {
                    "k8s_eval": True,
                    "isolated_pod": True,
                    "resolved": resolved,
                    "completed": completed,
                    "using_run_instance_k8s_for_rl": True,
                    "error": error if error else None,
                }
            )

        except Exception as e:
            logger.error(f"[K8S_EVAL] {instance_id} evaluation failed with run_instance_k8s_for_rl: {e}")
            logger.exception("[K8S_EVAL] Error details:")
            self.sample.reward = 0.0

            if not hasattr(self.m, "eval_metadata"):
                self.m.eval_metadata = {}

            self.m.eval_metadata.update(
                {
                    "k8s_eval": True,
                    "isolated_pod": True,
                    "resolved": False,
                    "error": str(e),
                    "exception": type(e).__name__,
                    "using_run_instance_k8s_for_rl": True,
                }
            )

    async def _eval_local(self, env: ContainerEnv):
        """
        Local implementation of evaluation logic (fallback when run_instance_k8s_for_rl is not available).

        This is different from the original swebench_sii.py which evaluates
        in the same pod as the agent. Here we use the isolated evaluation pod.
        """
        instance_id = self.spec.instance_id

        # Set up logging directory (same as run_instance_k8s_for_rl)
        model_name_or_path = self.config.model_name.replace("/", "__")

        # Get trajectory unique identifier to avoid file conflicts in GRPO training
        # When GRPO generates multiple trajectories for the same instance_id, each needs a separate log directory
        trajectory_id = self._get_trajectory_id()

        # Get weight_version (training step) from sample metadata (set in preprocess)
        weight_version = getattr(self.sample.m, "weight_version", None) if hasattr(self.sample, "m") else None
        weight_version_str = f"step_{weight_version}" if weight_version is not None else "step_unknown"

        # Construct log directory with step information
        log_dir = RUN_EVALUATION_LOG_DIR / self.config.run_id / model_name_or_path / weight_version_str / instance_id / trajectory_id

        if self.config.save_eval_logs:
            log_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[K8S_EVAL] Log directory: {log_dir}")
            logger.debug(f"[K8S_EVAL] Trajectory ID: {trajectory_id} (for GRPO multi-trajectory isolation)")

            # Set up file paths (same as run_instance_k8s_for_rl)
            report_path = log_dir / LOG_REPORT
            test_output_path = log_dir / LOG_TEST_OUTPUT
            patch_file = log_dir / "patch.diff"
            eval_file = log_dir / "eval.sh"
        else:
            logger.info("[K8S_EVAL] Skipping log file creation (save_eval_logs=False)")
            report_path = None
            test_output_path = None
            patch_file = None
            eval_file = None

        try:
            # Apply patch to the evaluation pod (using the provided env)
            logger.info("[K8S_EVAL] Applying patch...")
            patch_bytes = self.m.rollout.patch if isinstance(self.m.rollout.patch, bytes) else self.m.rollout.patch.encode("utf-8")

            # Log patch size and first few chars for debugging
            logger.info(f"[K8S_EVAL] Patch size: {len(patch_bytes)} bytes")
            if len(patch_bytes) == 0:
                logger.warning("[K8S_EVAL] Patch is empty!")
            else:
                logger.debug(f"[K8S_EVAL] Patch starts with: {patch_bytes[:100]}")

            # K8s adapter doesn't support stdin, so write patch to file first
            patch_str = patch_bytes.decode("utf-8", errors="replace")

            # Save patch to file (same as run_instance_k8s_for_rl)
            if self.config.save_eval_logs and patch_file:
                patch_file.write_text(ensure_trailing_newline(patch_str))
                logger.info(f"[K8S_EVAL] Patch written to {patch_file}")

            await env.write_file("/tmp/patch.patch", ensure_trailing_newline(patch_str))

            # Try multiple patch application strategies (same as run_instance_k8s_for_rl)
            applied_patch = False
            for i, git_apply_cmd in enumerate(self.GIT_APPLY_CMDS):
                cmd_str = f"{git_apply_cmd} /tmp/patch.patch"
                logger.info(f"[K8S_EVAL] Attempting patch application (strategy {i+1}/{len(self.GIT_APPLY_CMDS)}): {cmd_str}")
                result = await env.execute(f"bash -c '{cmd_str} 2>&1'", check=False)

                if result.returncode == 0:
                    output = result.output.decode("utf-8", errors="replace")
                    logger.info(f"[K8S_EVAL] Patch applied successfully with strategy {i+1}:\n{output}")
                    applied_patch = True
                    break
                else:
                    output = result.output.decode("utf-8", errors="replace")
                    logger.info(f"[K8S_EVAL] Patch attempt {i+1} failed (exit code {result.returncode}):\n{output}")

            if not applied_patch:
                logger.warning(f"[K8S_EVAL] All {len(self.GIT_APPLY_CMDS)} patch application strategies failed")
                # Log the full patch content for debugging
                logger.warning(f"[K8S_EVAL] Patch content:\n{patch_str}")

                if self.config.allow_partial_patch:
                    logger.warning("[K8S_EVAL] Continuing evaluation because allow_partial_patch=True")
                else:
                    logger.warning("[K8S_EVAL] Patch cannot be applied and allow_partial_patch=False, setting reward=0")
                    self.sample.reward = 0.0
                    return

            # Get git diff BEFORE running eval script (for validation)
            logger.info("[K8S_EVAL] Capturing git diff before running tests...")
            git_diff_before_result = await env.execute("git --no-pager -c core.fileMode=false diff", check=False, timeout=120)
            git_diff_before = git_diff_before_result.output.decode("utf-8", errors="replace")
            logger.debug(f"[K8S_EVAL] Git diff before:\n{git_diff_before}")

            # Run eval script in the evaluation pod
            logger.info("[K8S_EVAL] Running tests...")

            # Save eval script to file (same as run_instance_k8s_for_rl)
            if self.config.save_eval_logs and eval_file:
                eval_file.write_text(self.spec.eval_script)
                logger.info(f"[K8S_EVAL] Eval script written to {eval_file}")

            # K8s adapter doesn't support stdin, so write script directly
            await env.write_file("/eval.sh", self.spec.eval_script)
            # Increase timeout for test execution (1800 seconds = 30 minutes)
            # Also set GIT_PAGER to avoid terminal hanging issues
            output = await env.execute("bash -c 'GIT_PAGER=cat bash /eval.sh'", check=False, timeout=1800)

            # Get test output
            test_output = output.output

            # Get git diff after running eval script
            git_diff_after_result = await env.execute("git --no-pager -c core.fileMode=false diff", check=False, timeout=120)
            git_diff_after = git_diff_after_result.output.decode("utf-8", errors="replace")
            logger.info(f"[K8S_EVAL] Git diff after:\n{git_diff_after}")

            if git_diff_before != git_diff_after:
                logger.info("[K8S_EVAL] Git diff changed after running eval script")

            # Save test output to file for grading (same as run_instance_k8s_for_rl)
            # output.output is bytes, so decode it for text mode file
            test_output_str = test_output.decode("utf-8", errors="replace") if isinstance(test_output, bytes) else test_output
            if self.config.save_eval_logs and test_output_path:
                with open(test_output_path, "w") as f:
                    f.write(test_output_str)
                    logger.info(f"[K8S_EVAL] Test output written to {test_output_path}")

            # Grade the result
            pred = {
                "instance_id": instance_id,
                "model_name_or_path": self.config.model_name,
                "model_patch": self.m.rollout.patch,
            }

            # For grading, we need a test log path. If not saving logs, use a temp file.
            temp_file_created = False
            if self.config.save_eval_logs and test_output_path:
                test_log_path = Path(test_output_path)
            else:
                # Create a temporary file for grading
                with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp_f:
                    tmp_f.write(test_output_str)
                    test_log_path = Path(tmp_f.name)
                    temp_file_created = True
                logger.debug(f"[K8S_EVAL] Using temp file for grading: {test_log_path}")

            try:
                report = get_eval_report(
                    test_spec=self.spec,
                    prediction=pred,
                    test_log_path=test_log_path,
                    include_tests_status=True,
                )
            finally:
                # Clean up temp file if it was created (use finally to ensure cleanup even on exception)
                if temp_file_created:
                    try:
                        test_log_path.unlink()
                        logger.debug(f"[K8S_EVAL] Cleaned up temp file: {test_log_path}")
                    except Exception as e:
                        logger.warning(f"[K8S_EVAL] Failed to clean up temp file {test_log_path}: {e}")

            # Write report to report.json (same as run_instance_k8s_for_rl)
            if self.config.save_eval_logs and report_path:
                with open(report_path, "w") as f:
                    f.write(json.dumps(report, indent=4))
                    logger.info(f"[K8S_EVAL] Report written to {report_path}")

            # Set reward
            resolved = report[instance_id]["resolved"]
            if resolved:
                self.sample.reward = 1.0
                logger.info(f"[K8S_EVAL] {instance_id} ✓ RESOLVED")
            else:
                self.sample.reward = 0.0
                logger.info(f"[K8S_EVAL] {instance_id} ✗ NOT RESOLVED")

            # Store metadata
            if not hasattr(self.m, "eval_metadata"):
                self.m.eval_metadata = {}

            self.m.eval_metadata.update(
                {
                    "k8s_eval": True,
                    "isolated_pod": True,
                    "resolved": resolved,
                    "using_local_implementation": True,
                }
            )

        except Exception as e:
            logger.error(f"[K8S_EVAL] {instance_id} evaluation failed: {e}")
            logger.exception("[K8S_EVAL] Error details:")
            self.sample.reward = 0.0

            if not hasattr(self.m, "eval_metadata"):
                self.m.eval_metadata = {}

            self.m.eval_metadata.update(
                {
                    "k8s_eval": True,
                    "isolated_pod": True,
                    "resolved": False,
                    "error": str(e),
                    "exception": type(e).__name__,
                    "using_local_implementation": True,
                }
            )

    async def _reapply_pre_install(self, env: ContainerEnv):
        """Re-apply pre_install commands lost by _reset_repository().

        During Docker image build, pre_install commands (e.g., adding ``-rA``
        to tox.ini for pytest) are executed and committed as part of the
        SWE-bench setup commit.  When SWEEnv resets the repo to base_commit,
        those changes are reverted.  This method re-runs the same pre_install
        commands so the eval environment matches what the log parser expects.
        """
        try:
            if self.spec.install_config is not None:
                specs = self.spec.install_config
            else:
                specs = MAP_REPO_VERSION_TO_SPECS[self.spec.repo][self.spec.version]

            pre_install_cmds = specs.get("pre_install", [])
            if not pre_install_cmds:
                return

            for cmd in pre_install_cmds:
                logger.info(f"[K8S_EVAL] Re-applying pre_install: {cmd}")
                script = f"#!/bin/bash\ncd /testbed && {cmd}\n"
                await env.write_file("/tmp/_pre_install.sh", script)
                await env.execute("bash /tmp/_pre_install.sh", check=False, timeout=60)
        except Exception as e:
            logger.warning(f"[K8S_EVAL] Failed to re-apply pre_install (non-fatal): {e}")

    @property
    def spec(self) -> TestSpec:
        return self.m.data.runtime_meta.spec

    @property
    def instance(self) -> dict:
        return self.m.data.runtime_meta.instance


class SWEBenchK8sEvalBuilder(RuntimeBuilder):
    """Builder for K8s-based isolated evaluation runtime."""

    def __init__(self, config: dict):
        self.config = SWEBenchConfig.model_validate(config)
        # 保存完整配置（包含 environment 配置）
        self.full_config = config

        # 调试日志：打印配置结构
        logger.info(f"[SWEBenchK8sEvalBuilder.__init__] Received config with keys: {list(config.keys())}")
        logger.info(f"[SWEBenchK8sEvalBuilder.__init__] Has 'environment': {'environment' in config}")
        if "environment" in config:
            logger.info(f"[SWEBenchK8sEvalBuilder.__init__] environment.namespace: {config['environment'].get('namespace')}")

    def parse_sampledata(self, sample: dict) -> SWESampleData:
        """
        Parse SWE-bench sample into SampleData.

        Creates TestSpec for Runtime use.
        Environment builder will also create its own TestSpec for container setup.
        """
        # === 创建 TestSpec（Runtime 需要 spec 信息）===

        # Extract fields from extra_info if present (parquet format)
        if "extra_info" in sample and isinstance(sample["extra_info"], dict):
            sample = {**sample, **sample["extra_info"]}

        # Fix empty image_name
        if "image_name" in sample and not sample["image_name"]:
            del sample["image_name"]

        # Parse JSON fields
        import contextlib

        s = cast(SWEbenchInstance, sample)
        for key in ["FAIL_TO_PASS", "PASS_TO_PASS"]:
            if key in s and isinstance(s[key], str):
                with contextlib.suppress(json.JSONDecodeError):
                    s[key] = json.loads(s[key])

        # Create TestSpec
        # 按 data_source 选 namespace，swe-bench 的 run_evaluation_k8s 内部会按
        # test_spec.namespace 分支选 map_image_to_acr / map_image_to_acr_swefactory。
        ds = (sample.get("data_source") or "").lower()
        test_spec_namespace = "swefactory" if "swefactory" in ds else "swebench"
        spec = make_test_spec(sample, namespace=test_spec_namespace)
        logger.info(
            f"[SWEBenchK8sEvalBuilder] Created TestSpec, instance_id={spec.instance_id}, "
            f"data_source={ds!r}, namespace={test_spec_namespace!r}"
        )

        return SWESampleData(
            container_args=None,  # Environment builder 会创建
            problem_statement=sample.get("problem_statement", ""),
            runtime_meta=SBSample(instance=sample, spec=spec),
        )

    def build(self, sample: "SWESample") -> Runtime:
        """Build K8s evaluation runtime."""
        return SWEBenchK8sEvalRuntime(sample, self.config, self.full_config)
