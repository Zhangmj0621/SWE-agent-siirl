from __future__ import annotations

# noqa: E402 - Imports must come after path setup
import asyncio
import base64
import contextlib
import copy
import json

# ============================================================================
# CRITICAL: Setup 3rdparty paths BEFORE importing anything from sweagent/swerex
# ============================================================================
import time
import uuid
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from jinja2 import Template
from pydantic import BaseModel
from sweagent import __version__, get_agent_commit_hash, get_rex_commit_hash, get_rex_version
from sweagent.agent.action_sampler import AbstractActionSampler, ActionSamplerConfig

# Import original RLTokenAgent from Agentic_RL
from sweagent.agent.agents import DefaultAgentConfig  # 导入原始的 DefaultAgentConfig（包含 model 字段）
from sweagent.agent.agents import EXIT_FORFEIT_TOKEN, RETRY_WITH_OUTPUT_TOKEN, RETRY_WITHOUT_OUTPUT_TOKEN, AbstractAgent, DefaultAgent
from sweagent.agent.agents import RLTokenAgent as OriginalRLTokenAgent
from sweagent.agent.agents import (
    TemplateConfig,
    _BlockedActionError,
    _ExitForfeit,
    _RetryWithoutOutput,
    _RetryWithOutput,
    _TotalExecutionTimeExceeded,
)
from sweagent.agent.history_processors import DefaultHistoryProcessor, HistoryProcessor
from sweagent.agent.hooks.abstract import AbstractAgentHook, CombinedAgentHook
from sweagent.agent.models import GenericAPIModelConfig, LiteLLMModel
from sweagent.agent.problem_statement import (
    ProblemStatement,
    ProblemStatementConfig,
    SWEBenchMultimodalProblemStatement,
    TextProblemStatement,
)

# load origin sweagent
from sweagent.run.run_single import RunSingleConfig
from sweagent.tools.tools import ToolConfig, ToolHandler
from sweagent.types import AgentInfo, AgentRunResult, StepOutput, Trajectory, TrajectoryStep
from sweagent.utils.config import _strip_abspath_from_dict
from sweagent.utils.log import get_logger
from sweagent.utils.patch_formatter import PatchFormatter
from swerex.exceptions import BashIncorrectSyntaxError, CommandTimeoutError, SwerexException
from tenacity import RetryError
from typing_extensions import Self
from unidiff import UnidiffParseError

from siirl.execution.rollout.utils import (
    ContentPolicyViolationError,
    ContextWindowExceededError,
    CostLimitExceededError,
    FormatError,
    SglangGenerationAborted,
    TotalCostLimitExceededError,
)

from ...base import Model
from ..base import SWESample
from ..environment import ContainerEnv
from ..environment.tools import SiiToolHandler
from .base import Agent, AgentBuilder

# ==============================================================================
# NOTE: No need to patch AbstractAgentHook, CombinedAgentHook, or StepOutput
# The original SWE-agent library already supports:
# - rollout_log_probs
# - rollout_routed_experts
# - reasoning_content
# - output_tokens
# - thinking_blocks
# ==============================================================================


# ==============================================================================
# Wrapper for Original RLTokenAgent from /mnt/workspace/hujr/swe-agent
# ==============================================================================


class StateWithTokenizer:
    """State object that provides tokenizer to original RLTokenAgent."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class RLTokenAgentWrapper(AbstractAgent):
    """
    Wrapper that uses the original RLTokenAgent from Agentic_RL.

    This wrapper:
    1. Wraps the original RLTokenAgent
    2. Provides async interface (original run() is sync)
    3. Properly assigns token data to sample for RL training
    4. Handles ToolHandler adaptation
    """

    def __init__(
        self,
        *,
        templates: TemplateConfig,
        tools: ToolHandler,
        history_processors: list[HistoryProcessor],
        model: Model,
        max_requeries: int = 3,
        name: str = "main",
        problem_statement: ProblemStatement | ProblemStatementConfig,
        sample: SWESample,
        _catch_errors: bool = True,
        _always_require_zero_exit_code: bool = False,
        action_sampler_config: ActionSamplerConfig | None = None,
    ):
        """Initialize the wrapper."""
        self.sample = sample
        self.problem_statement = problem_statement
        self.model = model

        # Create state with tokenizer for original RLTokenAgent
        if hasattr(model, "tokenizer"):
            state = StateWithTokenizer(model.tokenizer)
        else:
            raise ValueError(f"Model {type(model).__name__} must have 'tokenizer' attribute for RLTokenAgent")

        tool_handler = tools

        # ``model`` is a ``SweSglangModel`` — a faithful async port of the
        # pre-6564c63 ``query_for_swe`` wrapper. Upstream ``RLTokenAgent``
        # reads ``output.get("new_prompt_token_ids", [])`` etc. with defaults,
        # so missing keys (we don't set them) degrade cleanly to empty
        # ``sample.tokens / loss_mask``. Enabling proper RL token tracking is
        # a separate workstream — do NOT bolt it onto this wrapper, the
        # incremental-tokenise attempt is what caused the reward regression.
        self._agent = OriginalRLTokenAgent(
            templates=templates,
            tools=tool_handler,
            history_processors=history_processors,
            model=model,
            max_requeries=max_requeries,
            name=name,
            _catch_errors=_catch_errors,
            _always_require_zero_exit_code=_always_require_zero_exit_code,
            action_sampler_config=action_sampler_config,
            state=state,  # Critical: provide state with tokenizer - must be last
        )

        # Expose key attributes
        self.name = self._agent.name
        self.tools = tools
        self.history = self._agent.history
        self.trajectory = self._agent.trajectory
        self.info = self._agent.info

        # Populated by ``run`` / ``resume`` when SglangGenerationAborted
        # propagates — SWEAgentFlow reads this in the abort branch to build
        # ``sample.partial_agent_data`` and delay env cleanup.
        self.partial_state: dict[str, Any] | None = None

    @property
    def templates(self):
        """Access to agent's template configuration."""
        return self._agent.templates

    def add_hook(self, hook):
        """Add hook to the underlying agent."""
        self._agent.add_hook(hook)

    async def setup(
        self,
        env: ContainerEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        output_dir: Path = Path("."),
    ) -> None:
        """Setup the agent for a new instance."""
        swe_env = env._env if hasattr(env, "_env") else env
        await self._agent.setup(env=swe_env, problem_statement=problem_statement, output_dir=output_dir)

    async def run(
        self,
        env: ContainerEnv,
        output_dir: Path = Path("."),
    ) -> AgentRunResult:
        """Run the agent on a problem instance (async).

        Awaits upstream ``RLTokenAgent.run`` directly, so step-level concurrency
        between rollouts is preserved at every ``await`` point. If rollout is
        aborted mid-step by ``SglangGenerationAborted``, snapshot agent state,
        pause the sandbox (E2B snapshot), and re-raise so upper layers can hand
        the sample back to the shared partial queue.
        """
        swe_env = env._env if hasattr(env, "_env") else env

        try:
            result = await self._agent.run(
                env=swe_env, problem_statement=self.problem_statement, output_dir=output_dir
            )
        except SglangGenerationAborted as abort:
            self._save_abort_partial(abort)
            self.partial_state = self._snapshot()
            self.partial_state["swe_env_handle"] = env.get_handle()
            await env.detach()
            raise

        self._backfill_sample(result)
        return result

    async def resume(
        self,
        env: ContainerEnv,
        partial: dict[str, Any],
        output_dir: Path = Path("."),
    ) -> AgentRunResult:
        """Partial-rollout entry point. Pairs with ``run``'s snapshot path.

        Call order: this method ``setup``s the upstream agent, restores
        model-side TokenManager state from ``partial``, restores upstream
        agent state, then calls ``upstream_agent.resume`` which skips
        ``setup``/``on_run_start`` and jumps straight into the step loop.
        """
        swe_env = env._env if hasattr(env, "_env") else env

        # Preflight (setup + restore). Wrapped only for diagnostics — the
        # re-raise keeps original semantics. If this phase fails silently the
        # exception reaches AgentFlowCallable's generic handler and the sample
        # enters the batch as prompt=1/response=0; this log makes that
        # situation visible.
        try:
            # 1) setup is still required — it binds env/problem_statement/tools on
            #    the upstream agent, installs tools on the pod, and seeds
            #    info/history. restore_state below overwrites the bits that
            #    setup would otherwise discard (history, trajectory, info, ...).
            await self._agent.setup(
                env=swe_env, problem_statement=self.problem_statement, output_dir=output_dir
            )

            # 2) model side first: TokenManager / _rid / stats / _processed_message_count
            self.model.restore_state(partial)

            # 3) upstream agent state
            self._agent.restore_state(partial["swe_agent_state"])
        except Exception as e:
            self._agent.logger.exception(
                "[RLTokenAgentWrapper.resume] PREFLIGHT FAILED "
                "exc_type=%s rid=%s assistant_turns=%s env_turns=%s has_env_handle=%s "
                "prompts_ids_len=%s swe_processed_message_count=%s init_input_ids_len=%s: %s",
                type(e).__name__,
                partial.get("rid", "?"),
                partial.get("assistant_turns", "?"),
                partial.get("env_turns", "?"),
                partial.get("swe_env_handle") is not None,
                len(partial.get("prompts_ids", []) or []),
                partial.get("swe_processed_message_count", "?"),
                len((partial.get("swe_agent_state") or {}).get("init_input_ids", []) or []),
                e,
            )
            raise

        try:
            result = await self._agent.resume(
                env=swe_env, problem_statement=self.problem_statement, output_dir=output_dir
            )
        except SglangGenerationAborted as abort:
            self._save_abort_partial(abort)
            self.partial_state = self._snapshot()
            self.partial_state["swe_env_handle"] = env.get_handle()
            await env.detach()
            raise

        self._backfill_sample(result)
        return result

    def _save_abort_partial(self, abort: SglangGenerationAborted) -> None:
        carried_resp = list(getattr(self.model, "_pending_partial_resp", None) or [])
        carried_lp = list(getattr(self.model, "_pending_partial_lp", None) or [])

        new_resp = list(getattr(abort, "responses", None) or [])
        new_lp = list(getattr(abort, "rollout_log_prob", None) or [])

        merged_resp = carried_resp + new_resp
        merged_lp = carried_lp + new_lp

        siirl_sample = getattr(self.model, "_current_sample", None)
        if siirl_sample is not None:
            siirl_sample.partial_response_ids = merged_resp
            siirl_sample.partial_rollout_log_prob = merged_lp
            siirl_sample.partial_loss_mask = [1] * len(merged_resp)

        # Clear model-side stash — it's ephemeral (one-_single_query-call)
        # state and must not leak into a later unrelated generate.
        self.model._pending_partial_resp = []
        self.model._pending_partial_lp = []

    def _backfill_sample(self, result: AgentRunResult) -> None:
        """Write rollout outputs back onto the siirl Sample.

        Shared by both normal-completion paths in ``run`` and ``resume``.
        """
        # CRITICAL: Assign token tracking data to sample for RL training
        self.sample.prompts = self._agent.init_input_ids
        self.sample.tokens = self._agent.input_ids[len(self._agent.init_input_ids) :]
        self.sample.loss_mask = self._agent.loss_mask[len(self._agent.init_input_ids) :]
        self.sample.rollout_log_probs = getattr(self._agent, "rollout_log_probs", [])
        # Upstream now also surfaces base64-encoded routed experts; forward it
        # untouched so the training end can reshape it.
        self.sample.rollout_routed_experts = getattr(self._agent, "routed_experts_raw", "") or ""

        # Store patch / exit_status in rollout for evaluation.
        # 优先读 self._agent.info（写 traj 的同一个 dict），落回 result.info 兜底。
        # Train 阶段只对 "submitted" 做 eval；validate 阶段对 "submitted*" 都做 eval。
        agent_info = getattr(self._agent, "info", {}) or {}
        info = agent_info if agent_info else (result.info or {})
        self.sample.m.rollout.patch = info.get("submission", None)
        self.sample.m.rollout.exit_status = info.get("exit_status", None)

    def _snapshot(self) -> dict[str, Any]:
        """Build the partial_agent_data dict (naive_flow-aligned shape).

        naive_flow key convention (see naive_flow._save_partial_agent_data):
            rid / messages / prompts_ids / response_ids / response_mask /
            rollout_log_prob / assistant_turns / env_turns /
            env_rewards / env_kwargs / routed_experts
        SWE-specific additions: swe_agent_state, swe_model_stats,
        swe_processed_message_count. ``swe_env_handle`` and
        ``swe_runtime_bootstrapped`` are filled in by SWEAgentFlow.
        """
        model_state = self.model.dump_state()
        upstream_trajectory = getattr(self._agent, "_trajectory", None) or list(self._agent.trajectory)
        env_turns = sum(
            1 for step in upstream_trajectory if step.get("tool_calls")
        )
        return {
            # naive_flow-aligned
            "rid": model_state["rid"],
            "messages": copy.deepcopy(self._agent.history),
            "prompts_ids": model_state["prompts_ids"],
            "response_ids": model_state["response_ids"],
            "response_mask": model_state["response_mask"],
            "rollout_log_prob": model_state["rollout_log_prob"],
            "assistant_turns": len(upstream_trajectory),
            "env_turns": env_turns,
            "env_rewards": [],
            "env_kwargs": {},
            "routed_experts": getattr(self._agent, "routed_experts_raw", "") or "",
            # SWE-specific
            "swe_agent_state": self._agent.dump_state(),
            "swe_model_stats": model_state["swe_model_stats"],
            "swe_processed_message_count": model_state["swe_processed_message_count"],
        }


class RLTokenAgentBuilder(AgentBuilder):
    def __init__(self, config: dict):
        config.pop("name")
        # convert config to DefaultAgentConfig
        # Note: model field is required by DefaultAgentConfig but not used in training (sample.model is used instead)

        # Get model config from agent configuration
        model_config = config.get("model", {})

        # Filter out fields not supported by GenericAPIModelConfig
        # max_workers and tool_parser are SGLang-specific fields that GenericAPIModelConfig doesn't support
        # These fields are still used by build_agentflow() for SGLangModelConfig, just not here
        unsupported_fields = {"max_workers", "tool_parser"}
        clean_model_config = {k: v for k, v in model_config.items() if k not in unsupported_fields}

        # Ensure name field exists (required by GenericAPIModelConfig)
        # The actual model used in training comes from sample.model (SGLang), not this config
        if "name" not in clean_model_config:
            clean_model_config["name"] = "dummy"  # Placeholder, not used in training

        self.config = DefaultAgentConfig(
            name=config.get("name", "main"),
            templates=TemplateConfig.model_validate(config.get("templates", {})),
            tools=ToolConfig.model_validate(config.get("tools", {})),
            history_processors=config.get("history_processors", [DefaultHistoryProcessor()]),
            max_requeries=config.get("max_requeries", 3),
            action_sampler=config.get("action_sampler"),
            model=clean_model_config,
        )

    def build(self, sample: SWESample) -> Agent:
        # Pop problem_statement from instance to avoid passing it twice
        instance = sample.m.data.runtime_meta.instance.copy()
        if "problem_statement" in instance:
            instance.pop("problem_statement")
        # Pop repo to avoid duplicate keyword error in _get_format_dict
        # The original SWE-agent code computes repo from self._env.repo
        if "repo" in instance:
            instance.pop("repo")

        # Handle multimodal (with images) vs text-only
        if sample.m.data.issue_images:
            instance.pop("issue_images", None)
            problem_statement = SWEBenchMultimodalProblemStatement(
                text=sample.m.data.problem_statement,
                issue_images=sample.m.data.issue_images,
                id=instance["instance_id"],
                extra_fields=instance,
            )
        else:
            problem_statement = TextProblemStatement(
                text=sample.m.data.problem_statement,
                id=instance["instance_id"],
                extra_fields=instance,
            )

        # Use original RLTokenAgent via wrapper
        return RLTokenAgentWrapper(
            templates=self.config.templates,
            tools=SiiToolHandler(self.config.tools),
            history_processors=self.config.history_processors,
            model=sample.model,
            max_requeries=self.config.max_requeries,
            name=self.config.name,
            action_sampler_config=self.config.action_sampler,
            problem_statement=problem_statement,
            sample=sample,
        )


class RLTokenAgent(AbstractAgent):
    def __init__(
        self,
        *,
        templates: TemplateConfig,
        tools: SiiToolHandler,
        history_processors: list[HistoryProcessor],
        model: Model,
        max_requeries: int = 3,
        name: str = "main",
        problem_statement: ProblemStatement | ProblemStatementConfig,
        sample: SWESample,
        _catch_errors: bool = True,
        _always_require_zero_exit_code: bool = False,
        action_sampler_config: ActionSamplerConfig | None = None,
    ):
        """The agent handles the behaviour of the model and how it interacts with the environment.

        To run the agent, either call `self.run` or `self.setup` and then `self.step` in a loop.
        """

        self.input_ids: list[int] = []
        self.loss_mask: list[int] = []
        self.rollout_log_probs: list[float] = []  # Log probabilities for RL training
        self.init_input_ids: list[int] = []
        self.system_prompt_prefix: list[int] = []
        self.generate_prompt_suffix: list[int] = []
        self.user_input_flag: bool = True
        self.tokenizer = model.tokenizer
        # normal setting
        self._catch_errors = _catch_errors
        self._always_require_zero_exit_code = _always_require_zero_exit_code
        self.name = name
        self.model = model
        self.templates = templates
        self.tools = tools
        # sglangModel use Default parse_function:FunctionCallingParser
        # todo: Maybe need config to control
        # if isinstance(self.model, HumanThoughtModel):
        #     self.tools.config.parse_function = ThoughtActionParser()
        # elif isinstance(self.model, HumanModel):
        #     self.tools.config.parse_function = ActionOnlyParser()
        self.history_processors = history_processors
        self.max_requeries = max_requeries
        self.logger = get_logger("swea-agent", emoji="🤠")
        # Set in run method
        self._env: ContainerEnv | None = None
        self._problem_statement: ProblemStatement | ProblemStatementConfig | None = None
        self.traj_path: Path | None = None

        #: The following three attributes collect the information about how the agent
        #: solved the problem.
        self.sample = sample
        self.problem_statement = problem_statement
        self.history = []
        self._trajectory = []
        self.info = AgentInfo()

        self._chook = CombinedAgentHook()

        self._replay_config: BaseModel | None = None
        """This can be set to a RunSingleConfig from the Run instance whenever possible.
        It can be used to replay the agent's trajectory in an environment.
        """

        self._action_sampler: AbstractActionSampler | None = None
        if action_sampler_config is not None:
            self._action_sampler = action_sampler_config.get(self.model, self.tools)

        #: Count how many timeout errors have occurred consecutively. Kills agent
        #: after 5 of them.
        self._n_consecutive_timeouts = 0
        self._total_execution_time = 0.0
        self.uuid = uuid.uuid4()

        # Save/reset some attributes
        def get_short_uuid3(uid, length=16):
            # 先生成22位Base64编码的UUID（方案2）

            b64_str = base64.b64encode(uid.bytes).decode("utf-8").replace("+", "_").replace("/", "-").rstrip("=")
            # 截取前N位（确保长度≥16，避免重复概率升高）
            if length < 16:
                length = 16  # 强制最低16位
            return b64_str[:length]

        self.uuid = get_short_uuid3(self.uuid)

    @classmethod
    def from_config(
        cls, config: dict, model: Model, problem_statement: ProblemStatement | ProblemStatementConfig, sample: SWESample
    ) -> Self:
        # To ensure that all models stay completely independent, we deepcopy the
        # model config, because it lives on as a property in the model, tools, etc.
        config = config.model_copy(deep=True)
        return cls(
            templates=config.templates,
            tools=SiiToolHandler(config.tools),
            history_processors=config.history_processors,
            model=model,
            max_requeries=config.max_requeries,
            action_sampler_config=config.action_sampler,
            problem_statement=problem_statement,
            sample=sample,
        )

    def add_hook(self, hook: AbstractAgentHook) -> None:
        """Add hook to agent"""
        hook.on_init(agent=self)
        self._chook.add_hook(hook)

    # Properties
    # ----------

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @property
    def replay_config(self) -> BaseModel | None:
        return self._replay_config

    @replay_config.setter
    def replay_config(self, value: BaseModel):
        # Do import here to avoid circular dependency

        self._replay_config = RunSingleConfig.model_validate(_strip_abspath_from_dict(value.model_dump()))

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Return the history of the agent for this attempt since the last reset,
        processed through all history processors.
        """
        filtered_history = [entry for entry in self.history if entry["agent"] == self.name]  # type: ignore

        # Chain the history processors
        messages = filtered_history
        for processor in self.history_processors:
            messages = processor(messages)

        return messages  # type: ignore

    # Methods
    # -------

    def _append_history(self, item: dict[str, Any]) -> None:
        """Adds an item to the history."""
        self.logger.info("[_append_history] Starting, role: %s", item.get("role"))
        self._chook.on_query_message_added(**item)
        self.history.append(item)  # type: ignore

        # Token tracking is optional but required for RL token-in/token-out mode.

        tools_schema = getattr(self.tools.config, "tools", None)

        if item["role"] == "system":
            self.logger.info("[_append_history] Processing system message")
            # TODO: Add try catch
            self.init_input_ids = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": item["content"]}],
                tools=tools_schema,
                add_generation_prompt=False,
                tokenize=True,
            )
            self.input_ids.extend(self.init_input_ids)
            self.loss_mask.extend([0] * len(self.init_input_ids))
            self.logger.info("[_append_history] System message processed")

        elif item["role"] == "assistant":
            output_tokens = item.get("output_tokens")
            if output_tokens is None or not isinstance(output_tokens, list):
                output_tokens = []
            # TODO: 这个 [198] 是 qwen tokenizer的，要改成通用的
            content_token_ids = self.generate_prompt_suffix + output_tokens + [198]  # '\n'
            self.input_ids.extend(content_token_ids)
            self.loss_mask.extend(
                [0] * len(self.generate_prompt_suffix) + [1] * (len(content_token_ids) - len(self.generate_prompt_suffix))
            )
            # self.input_ids.extend(content_token_ids)

            # # Build loss_mask for output_tokens, masking </think> (151668) and following \n (271)
            # THINK_CLOSE_TOKEN = 151668  # </think>
            # NEWLINE_TOKEN = 271  # \n
            # output_mask = []
            # for i, token_id in enumerate(output_tokens):
            #     if token_id == THINK_CLOSE_TOKEN:
            #         # Mask </think> token
            #         output_mask.append(0)
            #     elif i > 0 and output_tokens[i - 1] == THINK_CLOSE_TOKEN and token_id == NEWLINE_TOKEN:
            #         # Mask \n token following </think>
            #         output_mask.append(0)
            #     else:
            #         output_mask.append(1)
            # # Add mask for trailing \n [198]
            # output_mask.append(1)
            # self.loss_mask.extend([0] * len(self.generate_prompt_suffix) + output_mask)
        elif item["role"] == "user":
            content_token_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": item["content"]}],
                add_generation_prompt=True,
                tokenize=True,
            )
            # print(f"content_token_ids: {content_token_ids}")
            # print(f"system_prompt_prefix: {self.system_prompt_prefix}")
            content_token_ids = content_token_ids[len(self.system_prompt_prefix) :]
            # self.logger.debug(f"content: {item['content']}")
            # self.logger.debug(f"instance template: {Template(self.templates.instance_template).render(**self._get_format_dict())}")
            # self.logger.debug(
            #     f"item['content'] == Template(self.templates.instance_template).render(**self._get_format_dict()): "
            #     f"{item['content'] == Template(self.templates.instance_template).render(**self._get_format_dict())}"
            # )
            if self.user_input_flag:
                self.init_input_ids.extend(content_token_ids)
                self.user_input_flag = False
                self.logger.debug("user instance template matched")
            self.input_ids.extend(content_token_ids)
            self.loss_mask.extend([0] * len(content_token_ids))
            # import pdb; pdb.set_trace()
        elif item["role"] == "tool":
            tool_msg: dict[str, Any] = {"role": "tool", "content": item["content"]}
            content_token_ids = self.tokenizer.apply_chat_template(
                [tool_msg],
                add_generation_prompt=True,
                tokenize=True,
            )
            content_token_ids = content_token_ids[len(self.system_prompt_prefix) :]
            self.input_ids.extend(content_token_ids)
            self.loss_mask.extend([0] * len(content_token_ids))

    async def setup(
        self,
        env: ContainerEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        output_dir: Path = Path("."),
    ) -> None:
        """Setup the agent for a new instance. This includes
        formatting the system message and adding demonstrations to the history.

        This method is called by `self.run`.
        """

        # Initialize tokenizer templates for tracking
        if self is None or not hasattr(self, "tokenizer"):
            raise ValueError("RLTokenAgent requires `state.tokenizer` for token tracking (token-in/token-out).")

        try:
            self.system_prompt_prefix = self.tokenizer.apply_chat_template(
                [{}],
                add_generation_prompt=False,
                tokenize=True,
            )
        except Exception:
            self.system_prompt_prefix = []

        # Get the suffix that's added when generation prompt is enabled
        self.generate_prompt_suffix = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            add_generation_prompt=True,
            tokenize=True,
        )[
            len(
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": ""}],
                    add_generation_prompt=False,
                    tokenize=True,
                )
            ) :
        ]

        self.input_ids = []
        self.loss_mask = []

        output_dir.mkdir(parents=True, exist_ok=True)

        # apply template configuration to multimodal problem statements
        if (
            hasattr(problem_statement, "type")
            and problem_statement.type == "swe_bench_multimodal"
            and isinstance(problem_statement, SWEBenchMultimodalProblemStatement)
            and not problem_statement.disable_image_processing
            and self.templates.disable_image_processing
        ):
            problem_statement.disable_image_processing = True

        self._problem_statement = problem_statement
        self._env = env
        iid = self._problem_statement.id
        self.logger.info("Setting up agent for instance %s", iid)

        self.traj_path = output_dir / (self._problem_statement.id + "_" + self.uuid + ".traj")
        self.logger.info("Trajectory will be saved to %s", self.traj_path)

        self._chook.on_tools_installation_started()
        await self.tools.install(self._env)
        self._chook.on_setup_attempt()
        self.info = AgentInfo()
        self.info["swe_agent_hash"] = get_agent_commit_hash()
        self.info["swe_agent_version"] = __version__
        self.info["swe_rex_version"] = get_rex_version()
        self.info["swe_rex_hash"] = get_rex_commit_hash()
        assert self._env is not None
        assert self._problem_statement is not None
        await self._env.set_env_variables({"PROBLEM_STATEMENT": self._problem_statement.get_problem_statement_for_env()})
        self.add_system_message_to_history()
        self.add_demonstrations_to_history()
        self.logger.info("[setup] About to call add_instance_template_to_history...")
        self.add_instance_template_to_history(state=await self.tools.get_state(self._env))
        self.logger.info("[setup] add_instance_template_to_history completed, calling on_setup_done...")
        self._chook.on_setup_done()
        self.logger.info("[setup] on_setup_done completed")

    def add_system_message_to_history(self) -> None:
        """Add system message to history"""
        assert self._problem_statement is not None
        system_msg = Template(self.templates.system_template).render(**self._get_format_dict())
        # self.logger.info(f"SYSTEM ({self.name})\n{system_msg}")
        self._append_history({"role": "system", "content": system_msg, "agent": self.name, "message_type": "system_prompt"})

    def add_demonstrations_to_history(self) -> None:
        """Add demonstrations to history"""
        for demonstration_path in self.templates.demonstrations:
            self._add_demonstration_to_history(demonstration_path)

    def _add_demonstration_to_history(self, demonstration_path: Path) -> None:
        """Load demonstration from disk and add to history"""
        if self.templates.demonstration_template is None and not self.templates.put_demos_in_history:
            msg = "Cannot use demonstrations without a demonstration template or put_demos_in_history=True"
            raise ValueError(msg)

        # Load history
        self.logger.info(f"DEMONSTRATION: {demonstration_path}")
        _demo_text = Path(demonstration_path).read_text()
        demo_history = yaml.safe_load(_demo_text)["history"] if demonstration_path.suffix == ".yaml" else json.loads(_demo_text)["history"]

        if self.templates.put_demos_in_history:
            # Add demonstrations to history step-by-step
            for entry in demo_history:
                if entry["role"] != "system":
                    entry["is_demo"] = True
                    self._append_history(entry)
        else:
            # Add demonstration as single message to history
            demo_history = [entry for entry in demo_history if entry["role"] != "system"]
            demo_message = "\n".join([entry["content"] for entry in demo_history])
            assert self.templates.demonstration_template is not None
            demonstration = Template(self.templates.demonstration_template).render(demonstration=demo_message)
            self._append_history(
                {
                    "agent": self.name,
                    "content": demonstration,
                    "is_demo": True,
                    "role": "user",
                    "message_type": "demonstration",
                },
            )

    def _get_format_dict(self, **kwargs) -> dict[str, Any]:
        """Get the dictionary of key value pairs used to format the templates

        Args:
            **kwargs: additional keyword arguments to be added to the format dictionary
        """
        assert self._problem_statement is not None
        assert self._env is not None

        return dict(
            command_docs=self.tools.config.command_docs,
            **self.tools.config.env_variables,
            **kwargs,
            problem_statement=self._problem_statement.get_problem_statement(),
            **self._problem_statement.get_extra_fields(),
        )

    def _add_templated_messages_to_history(self, templates: list[str], tool_call_ids: list[str] | None = None, **kwargs: Any) -> None:
        """Populate selected template(s) with information (e.g., issue, arguments, state)
        and add to history.

        Args:
            templates: templates to populate and add to history
            tool_call_ids: tool call ids to be added to the history
            **kwargs: keyword arguments to be passed to the templates (in addition to the
                ones in `self._get_format_dict`)
        """
        self.logger.info("[_add_templated_messages_to_history] Starting with %d templates", len(templates))
        messages = []

        self.logger.info("[_add_templated_messages_to_history] Getting format dict...")
        format_dict = self._get_format_dict(**kwargs)
        self.logger.info("[_add_templated_messages_to_history] Format dict keys: %s", list(format_dict.keys()))

        for i, template in enumerate(templates):
            self.logger.info("[_add_templated_messages_to_history] Rendering template %d/%d", i + 1, len(templates))
            try:
                messages.append(Template(template).render(**format_dict))
            except KeyError:
                self.logger.debug("The following keys are available: %s", format_dict.keys())
                raise
        message = "\n".join(messages)
        self.logger.info("[_add_templated_messages_to_history] Message rendered, length: %d", len(message))

        # We disable syntax highlighting here, because some inputs can lead to a complete cross-thread
        # freeze in the agent. See https://github.com/SWE-agent/SWE-agent/issues/901 .
        # self.logger.info(f"🤖 MODEL INPUT\n{message}", extra={"highlighter": None})
        history_item: dict[str, Any] = {
            "role": "user",
            "content": message,
            "agent": self.name,
            "message_type": "observation",
        }
        # import pdb; pdb.set_trace()
        if tool_call_ids:
            assert len(tool_call_ids) == 1, "This should be ensured by the FunctionCalling parse method"
            history_item["role"] = "tool"
            history_item["tool_call_ids"] = tool_call_ids
            history_item["tool_call_id"] = tool_call_ids[0]
            history_item["name"] = kwargs["tool_calls"][0]["function"]["name"]

            # 为了兼容models.py中的格式，也添加tool_call_id（单数）
        self.logger.info("[_add_templated_messages_to_history] About to call _append_history...")
        self._append_history(history_item)
        self.logger.info("[_add_templated_messages_to_history] _append_history completed")

    def add_step_to_history(self, step: StepOutput) -> None:
        """Adds a step (command that was run and output) to the model history"""
        assistant_message = {
            "role": "assistant",
            "content": step.output if not step.tool_calls else (step.thought or step.output),
            "thought": step.thought,
            "action": step.action,
            "agent": self.name,
            "message_type": "action",
            "thinking_blocks": step.thinking_blocks,
            "reasoning_content": step.reasoning_content,
            "output_tokens": step.output_tokens,
        }
        # 如果有tool_calls，添加到assistant消息中
        if step.tool_calls:
            assistant_message["tool_calls"] = step.tool_calls
        self._append_history(assistant_message)

        elided_chars = 0
        if step.observation.strip() == "":
            # Show no output template if observation content was empty
            templates = [self.templates.next_step_no_output_template]
        elif len(step.observation) > self.templates.max_observation_length:
            templates = [self.templates.next_step_truncated_observation_template]
            elided_chars = len(step.observation) - self.templates.max_observation_length
        else:
            # Show standard output template if there is observation content
            templates = [self.templates.next_step_template]
        self._add_templated_messages_to_history(
            templates,
            observation=step.observation,
            elided_chars=elided_chars,
            max_observation_length=self.templates.max_observation_length,
            tool_call_ids=step.tool_call_ids,
            tool_calls=step.tool_calls,
            **step.state,
        )

    def add_instance_template_to_history(self, state: dict[str, str]) -> None:
        """Add observation to history, as well as the instance template or demonstrations if we're
        at the start of a new attempt.
        """
        self.logger.info("[add_instance_template_to_history] Starting, state: %s", state)
        templates: list[str] = []
        # Determine observation template based on what prior observation was
        assert self.history[-1]["role"] == "system" or self.history[-1].get("is_demo", False)
        # Show instance template if prev. obs. was initial system message
        templates = [self.templates.instance_template]
        if self.templates.strategy_template is not None:
            templates.append(self.templates.strategy_template)

        self.logger.info("[add_instance_template_to_history] Calling _add_templated_messages_to_history with %d templates", len(templates))
        self._add_templated_messages_to_history(templates, **state)  # type: ignore
        self.logger.info("[add_instance_template_to_history] Completed")

    def get_trajectory_data(self) -> dict[str, Any]:
        """Get all data that we save in .traj files."""

        assert self._env is not None
        # The deepcopy here is important because else the
        # data["info"]["model_stats"] update will create havoc!
        attempt_data = copy.deepcopy(
            {
                "trajectory": self.trajectory,
                "history": self.history,
                "info": self.info,
            }
        )
        attempt_data["replay_config"] = self.replay_config.model_dump_json() if self.replay_config is not None else None
        attempt_data["environment"] = self._env.name
        return attempt_data

    def save_trajectory(
        self,
    ) -> None:
        """Save the trajectory to disk.
        This includes the history, the environment state, and the model stats.
        """
        data = self.get_trajectory_data()
        assert self.traj_path is not None
        self.traj_path.write_text(json.dumps(data, indent=2))

    def get_model_requery_history(self, error_template: str, *, output: str, **kwargs: str | int | float | bool | None) -> list[int]:
        """Ask the model to correct after a hitting one of the following errors:

        1. Malformatted output (could not parse action)
        2. Blocked action (command is on the blocklist)
        3. Bash command syntax error

        At the time this function is called, the proposed action and observation are not part of the history
        yet.

        This function adds temporary history based on the error template and queries the model.
        If the model is able to correct itself, the records of the mistakes will not be part of the history
        (but they are saved in the trajectory).

        Args:
            error_template: error template
            output: model output
            **kwargs: keyword arguments to be passed to the error template

        Returns:
            model output after requery
        """
        format_dict = {**kwargs, **self._get_format_dict()}
        error_template = Template(error_template).render(**format_dict)

        self.logger.warning(f"{error_template}")
        # self.logger.debug(f"**kwargs: {kwargs}")
        input_ids = copy.deepcopy(self.input_ids)
        return input_ids

    async def attempt_autosubmission_after_error(self, step: StepOutput) -> StepOutput:
        """For most exceptions, we attempt to still extract the patch and submit that.
        This means we send the `submit` command to the runtime and parse the output.
        """
        self.logger.warning("Attempting autosubmission after error")
        step = step.model_copy(deep=True)
        step.done = True
        assert self._env is not None
        if not self._env.alive:
            # The agent is dead. This is very bad. Maybe we can take a 'diff' that was saved
            # for a previous step? (if running with diff in tools)
            self.logger.error("Runtime is no longer alive")
            try:
                last_trajectory_step = self.trajectory[-1]
            except IndexError:
                self.logger.info("No last trajectory step to extract patch from")
                return step
            if "diff" not in last_trajectory_step["state"]:
                self.logger.info("No diff in last trajectory step state, cannot autosubmit")
                return step
            diff = last_trajectory_step["state"]["diff"]
            self.logger.info("Using diff from last trajectory step to autosubmit")
            step.submission = diff
            if step.submission:
                step.observation = "Environment died unexpectedly. Exited (autosubmitted)"
                step.exit_status = f"submitted ({step.exit_status})"
            else:
                self.logger.info("Diff from last traj step empty.")
            return step
        # Let us manually run the submission command and collect the output
        repo_name = "/"
        if self._env.repo is not None:
            repo_name = f"/{self._env.repo.repo_name}"
        submission_command = "git add -A && git diff --cached > /root/model.patch"
        self.logger.info("Executing submission command %s in %s", submission_command, repo_name)
        try:
            await self._env.execute_command(submission_command, check=True, cwd=repo_name)
        except Exception as e:
            self.logger.error("Failed to execute submission command, got %s", e)

        # Check git status for debugging
        try:
            status_output = await self._env.execute("git status --short", check=False, cwd=repo_name)
            status_str = status_output.output.decode("utf-8", errors="replace")
            self.logger.debug(f"Git status after autosubmission command:\n{status_str}")
        except Exception as e:
            self.logger.warning(f"Failed to get git status: {e}")

        # There's still hope for the submission, because the `/root/model.patch` file might have been
        # generated by the state command
        step = await self.handle_submission(step, observation="", force_submission=True)
        if step.submission:
            self.logger.info("Exiting with autosubmission")
            step.observation = "Exited (autosubmitted)"
        else:
            self.logger.warning("Autosubmission failed: submission is empty after git diff")
        return step

    async def handle_submission(self, step: StepOutput, *, observation="", force_submission: bool = False) -> StepOutput:
        """Check if there was a submission in the observation and handle it.

        Args:
            step:
            observation: If specified, will use this rather than stepobservation
            force_submission: If True, will always submit even if no submission is found

        Returns:
            step: step with submission and observation updated (if submission was found)
        """
        self.logger.info("[handle_submission] Starting, is_submission check...")
        step = step.model_copy(deep=True)
        assert self.tools is not None
        is_submission = self.tools.check_for_submission_cmd(observation or step.observation)
        self.logger.info(f"[handle_submission] is_submission={is_submission}, force_submission={force_submission}")
        if is_submission or force_submission:
            assert self._env is not None
            self.logger.info("[handle_submission] About to read /root/model.patch...")
            try:
                submission = await self._env.read_file("/root/model.patch", encoding="utf-8", errors="backslashreplace")
                self.logger.info(f"[handle_submission] Read {len(submission)} bytes from model.patch")
            except FileNotFoundError:
                self.logger.warning("Submission file not found, no submission was made")
                return step
            except Exception as e:
                self.logger.exception("Failed to read submission file, got %s", e)
                return step
            if submission.strip() != "":
                step.submission = submission
            else:
                step.submission = None
            step.observation = submission
            if not step.exit_status:
                step.exit_status = "submitted"
            elif step.submission:
                step.exit_status = f"submitted ({step.exit_status})"
            step.done = True
            self.logger.info(f"Found submission: {submission}")
        self.logger.info("[handle_submission] Completed")
        return step

    async def _async_read_file(self, path: str | PurePosixPath) -> str:
        """Async wrapper for reading files.

        Now that K8sEnvAdapter.read_file is truly async, we can directly call it
        instead of using asyncio.to_thread which would create unnecessary threads.
        """
        # Directly call the async read_file method
        return await self._env.read_file(str(path), errors="replace")  # type: ignore[attr-defined]

    async def _get_edited_files_with_context(self, patch: str) -> dict[str, str]:
        """Get the edited files with context from the patch (async version with concurrent file reads)."""
        assert self._env is not None
        try:
            if self._env.repo is None:
                pf = None
            else:
                # Parse patch to get list of files that need to be read
                from unidiff import PatchSet

                parsed_patch = PatchSet(patch)

                # Build a dict of file content cache
                file_cache: dict[str, str] = {}

                # Concurrently read all modified files using asyncio.gather
                read_tasks = []
                file_paths = []

                for patch_item in parsed_patch:
                    if not patch_item.is_modified_file:
                        continue
                    file_path = patch_item.path
                    file_paths.append(file_path)
                    full_path = PurePosixPath("/") / self._env.repo.repo_name / file_path  # type: ignore[attr-defined]
                    read_tasks.append(self._async_read_file(full_path))

                # Wait for all file reads to complete concurrently
                if read_tasks:
                    file_contents = await asyncio.gather(*read_tasks, return_exceptions=True)
                    for file_path, content in zip(file_paths, file_contents, strict=False):
                        if isinstance(content, Exception):
                            self.logger.warning(f"Failed to read {file_path}: {content}")
                            file_cache[file_path] = f"# Error reading file: {content}"
                        else:
                            file_cache[file_path] = content

                # Create PatchFormatter with cached read_method
                pf = (
                    PatchFormatter(
                        patch,
                        read_method=lambda path: file_cache.get(path, "# File not in cache"),
                    )
                    if patch
                    else None
                )
        except UnidiffParseError:
            self.logger.error("Failed to parse patch with unidiff. Some variables will be empty.")
            pf = None
            # We still need to populate the variables
        out = {}
        for context_length in [30, 50, 70]:
            value = "Empty. No edited files found."
            if pf is not None:
                value = pf.get_files_str(original=False, context_length=context_length)
            out[f"edited_files{context_length}"] = value
        return out

    async def handle_action(self, step: StepOutput) -> StepOutput:
        """Runs an action proposed by the agent in the environment and returns the corresponding output.

        Args:
            action: command to run in bash shell
            output: output from model (only used for error handling)

        Returns:
            action_execution_output: action execution output
        """
        if self.tools.should_block_action(step.action):
            raise _BlockedActionError()

        if step.action.strip() == "exit":
            self.logger.info("Exiting agent")
            step.done = True
            step.observation = "Exited"
            step.exit_status = "exit_command"
            assert self._env is not None
            step.state = await self.tools.get_state(env=self._env)  # for history
            return step

        assert self._env is not None
        self._chook.on_action_started(step=step)
        execution_t0 = time.perf_counter()
        run_action: str = self.tools.guard_multiline_input(step.action).strip()
        try:
            step.observation = await self._env.communicate(
                input=run_action,
                timeout=self.tools.config.execution_timeout,
                check="raise" if self._always_require_zero_exit_code else "ignore",
            )
        except CommandTimeoutError:
            self._n_consecutive_timeouts += 1
            if self._n_consecutive_timeouts >= self.tools.config.max_consecutive_execution_timeouts:
                msg = "Exiting agent due to too many consecutive execution timeouts"
                self.logger.critical(msg)
                step.execution_time = time.perf_counter() - execution_t0
                self._total_execution_time += step.execution_time
                raise
            try:
                self._env.interrupt_session()
            except Exception as f:
                self.logger.exception("Failed to interrupt session after command timeout: %s", f)
                step.execution_time = time.perf_counter() - execution_t0
                self._total_execution_time += step.execution_time
                raise
            step.observation = Template(self.templates.command_cancelled_timeout_template).render(
                **self._get_format_dict(),
                timeout=self.tools.config.execution_timeout,
                command=run_action,
            )
        else:
            self._n_consecutive_timeouts = 0
        # self.logger.debug(f"step.observation: {step.observation}")
        step.execution_time = time.perf_counter() - execution_t0
        self._total_execution_time += step.execution_time
        self._chook.on_action_executed(step=step)
        step.state = await self.tools.get_state(env=self._env)

        if RETRY_WITH_OUTPUT_TOKEN in step.observation:
            step.observation = step.observation.replace(RETRY_WITH_OUTPUT_TOKEN, "")
            raise _RetryWithOutput()
        elif RETRY_WITHOUT_OUTPUT_TOKEN in step.observation:
            step.observation = step.observation.replace(RETRY_WITHOUT_OUTPUT_TOKEN, "")
            raise _RetryWithoutOutput()
        elif EXIT_FORFEIT_TOKEN in step.observation:
            raise _ExitForfeit()

        return await self.handle_submission(step)

    async def forward(self, history: list[int]) -> StepOutput:
        """Forward the model without handling errors.

        All exceptions raised will contain the `StepOutput` object
        with some of the attributes set.

        Args:
            history: history to query the model with

        Returns:
            step_output: step output
        """
        if self._total_execution_time > self.tools.config.total_execution_timeout:
            raise _TotalExecutionTimeExceeded()

        # we continuously add actions, output etc. to the step object
        # because some of the specific exception handling requires some of these
        # attributes (e.g., if we want to requery the model for a bash syntax error, we
        # need to have the previous model output to format the requery template)
        step = StepOutput()
        # Note: query expects list[dict] (messages), but we use token ids (list[int])
        # To avoid Pydantic warnings, we store messages instead
        step.query = copy.deepcopy(self.messages)  # Store messages, not token ids
        try:
            # Forward model and get actions
            # Hooks/inspectors expect message-shaped history; token ids are the model input.
            self._chook.on_model_query(messages=self.messages, agent=self.name)

            # todo: Add all options to the extra info
            if self._action_sampler is not None:
                assert self._problem_statement is not None
                best = self._action_sampler.get_action(
                    problem_statement=self._problem_statement,
                    trajectory=self.trajectory,
                    history=self.messages,  # action sampler operates on message history
                )
                output = best.completion
                # todo: Handle history and trajectory
                step.extra_info.update(best.extra_info)
            else:
                output = await self.model.query(history)  # type: ignore
                # convert modelresponse to dict sweagent needed
                output = asdict(output)
                output["message"] = output.pop("output")

            step.output = output["message"]
            # todo: Can't I override the parser in __init__?
            step.thought, step.action = self.tools.parse_actions(output)
            step.thinking_blocks = output.get("thinking_blocks", [])
            step.reasoning_content = output.get("reasoning_content")
            step.output_tokens = output.get("output_tokens", [])
            if output.get("tool_calls") is not None:
                step.tool_call_ids = [call["id"] for call in output["tool_calls"]]
                step.tool_calls = output["tool_calls"]
            self._chook.on_actions_generated(step=step)

            # Resume sandbox right before env interaction
            if hasattr(self._env, "resume_sandbox"):
                await self._env.resume_sandbox()

            result = await self.handle_action(step)

            # Pause sandbox immediately after env interaction
            if hasattr(self._env, "pause_sandbox"):
                await self._env.pause_sandbox()

            return result
        except Exception as e:
            # Make sure sandbox is resumed on error path (for autosubmission)
            if hasattr(self._env, "resume_sandbox"):
                try:
                    await self._env.resume_sandbox()
                except Exception:
                    pass
            if step.action == step.thought == "":
                # Probably the parsing failed/no action included. Let's still fill in thought
                # so that trajectory viewers have something to show us for this step.
                step.thought = step.output
            # Attach the step object to the exception
            e.step = step  # type: ignore
            raise

    async def forward_with_handling(self, history: list[int]) -> StepOutput:
        """Forward the model and handle errors, requerying the model if we can.
        For example, if the model outputs a bash command that has syntax errors,
        we will not execute it but requery the model for a corrected command.

        Note: This will update the trajectory, but not the history.

        Args:
            history: history to forward

        Returns:
            step_output: step output
        """

        async def handle_error_with_autosubmission(exit_status: str, message: str) -> StepOutput:
            """Attempts to autosubmit (extract patch from the environment) and stops the loop."""
            self.logger.warning(message)
            if hasattr(self._env, "resume_sandbox"):
                try:
                    await self._env.resume_sandbox()
                except Exception:
                    pass
            return await self.attempt_autosubmission_after_error(
                StepOutput(
                    thought=message,
                    exit_status=exit_status,
                    output=message,
                    output_tokens=[],
                    done=True,
                )
            )

        def handle_error_with_retry(exception: Exception, template: str, n_requeries: int) -> list[int]:
            """Requeries the model if the error is a format/blocklist/bash syntax error."""
            self.logger.warning("Requerying model after %s (%dth requery)", type(exception).__name__, n_requeries)
            step: StepOutput = getattr(exception, "step", StepOutput())
            self.add_step_to_trajectory(step)
            exception_message = getattr(exception, "message", "")
            if not exception_message:
                # self.logger.debug(f"**step.to_template_format_dict(): {step.to_template_format_dict()}")
                with contextlib.suppress(IndexError, AttributeError):
                    exception_message = exception.args[0]
            return self.get_model_requery_history(
                error_template=template,
                **step.to_template_format_dict(),
                **getattr(exception, "extra_info", {}),
                exception_message=exception_message,
            )

        n_format_fails = 0
        while n_format_fails < self.max_requeries:
            try:
                return await self.forward(history)

            # Errors that are raised

            except KeyboardInterrupt:
                raise
            except EOFError:
                raise

            # Errors that cause requery

            except FormatError as e:
                n_format_fails += 1
                history = handle_error_with_retry(exception=e, template=self.tools.config.format_error_template, n_requeries=n_format_fails)
            except _BlockedActionError as e:
                n_format_fails += 1
                history = handle_error_with_retry(
                    exception=e, template=self.tools.config.filter.blocklist_error_template, n_requeries=n_format_fails
                )
            except ContentPolicyViolationError:
                self.logger.warning("Content policy violation, trying to resample")
                n_format_fails += 1
                # Try if simply resampling helps here
                pass
            except BashIncorrectSyntaxError as e:
                n_format_fails += 1
                history = handle_error_with_retry(
                    exception=e,
                    template=self.templates.shell_check_error_template,
                    n_requeries=n_format_fails,
                )
            except _RetryWithOutput as e:
                history = handle_error_with_retry(
                    exception=e,
                    template=self.templates.next_step_template,
                    n_requeries=n_format_fails,
                )
            except _RetryWithoutOutput:
                pass
                # Requery with the same template as the last step

            # Errors that cause exit

            except _ExitForfeit:
                self.logger.info("Exiting due to forfeit")
                return await handle_error_with_autosubmission(
                    "exit_forfeit",
                    "Exiting due to forfeit",
                )

            except _TotalExecutionTimeExceeded:
                self.logger.exception("Exiting due to total execution time exceeded")
                return await handle_error_with_autosubmission(
                    "exit_total_execution_time",
                    "Exit due to total execution time exceeded",
                )

            except CommandTimeoutError:
                self.logger.exception("Exiting due to multiple consecutive command timeouts")
                return await handle_error_with_autosubmission(
                    "exit_command_timeout",
                    "Exit due to multiple consecutive command timeouts",
                )

            except ContextWindowExceededError:
                return await handle_error_with_autosubmission(
                    "exit_context",
                    "Exit due to context window",
                )
            except TotalCostLimitExceededError:
                raise
            except CostLimitExceededError:
                return await handle_error_with_autosubmission(
                    "exit_cost",
                    "Exit due to cost limit",
                )
            except RetryError as e:
                self.logger.exception(f"Exiting due to retry error: {e}")
                return await handle_error_with_autosubmission(
                    "exit_api",
                    f"Exit due to retry error: {e}",
                )
            except SwerexException as e:
                self.logger.exception(f"Exiting due to environment error: {e}")
                return await handle_error_with_autosubmission(
                    "exit_environment_error",
                    f"Exit due to environment error: {e}",
                )
            except RuntimeError as e:
                self.logger.exception(f"Exiting due to runtime error: {e}")
                return await handle_error_with_autosubmission(
                    "exit_error",
                    f"Exit due to runtime error: {e}",
                )
            except Exception as e:
                self.logger.exception(f"Exiting due to unknown error: {e}")
                return await handle_error_with_autosubmission(
                    "exit_error",
                    f"Exit due to unknown error: {e}",
                )
        self.logger.exception(
            "Exit due to repeated format/blocklist/bash syntax errors",
            exc_info=True,
        )
        return await handle_error_with_autosubmission(
            "exit_format",
            "Exit due to repeated format/blocklist/bash syntax errors",
        )

    def add_step_to_trajectory(self, step: StepOutput) -> None:
        trajectory_step = TrajectoryStep(
            {
                "action": step.action,
                "observation": step.observation,
                "response": step.output,
                "thought": step.thought,
                "execution_time": step.execution_time,
                "state": step.state,
                "query": step.query,
                "extra_info": step.extra_info,
                "reasoning_content": step.reasoning_content,
                "output_tokens": step.output_tokens,
            },
        )
        self.trajectory.append(trajectory_step)

    async def step(self) -> StepOutput:
        """Run a step of the agent. This is a wrapper around `self.forward_with_handling`
        with additional bookkeeping:

        1. Update message history with performed action and observation
        2. Update trajectory with the final executed result
        3. Update the info dictionary

        Returns:
            step_output: step output (same as the output of `self.forward_with_handling`)
        """

        assert self._env is not None
        self._chook.on_step_start()

        n_step = len(self.trajectory) + 1
        self.logger.info("%s STEP %d %s", "=" * 25 + f"{self.uuid} ", n_step, "=" * 25)
        step_output = await self.forward_with_handling(self.input_ids)
        self.add_step_to_history(step_output)

        self.info["submission"] = step_output.submission
        self.info["exit_status"] = step_output.exit_status  # type: ignore
        # Use await for concurrent file reads
        if hasattr(self._env, "resume_sandbox"):
            await self._env.resume_sandbox()
        self.info.update(await self._get_edited_files_with_context(patch=step_output.submission or ""))
        if hasattr(self._env, "pause_sandbox"):
            await self._env.pause_sandbox()
        # self.info["model_stats"] = self.model.stats.model_dump()

        self.add_step_to_trajectory(step_output)

        self._chook.on_step_done(step=step_output, info=self.info)
        return step_output

    async def run(
        self,
        env: ContainerEnv,
        output_dir: Path = Path("."),
    ) -> AgentRunResult:
        """Run the agent on a problem instance. This method contains the
        main loop that repeatedly calls `self._step` until the problem is solved.

        Args:
            env: The environment to run the agent on.
            output_dir: Directory to save the trajectory to
        """

        await self.setup(env=env, problem_statement=self.problem_statement, output_dir=output_dir)

        # Run action/observation loop
        self._chook.on_run_start()
        step_output = StepOutput()
        while not step_output.done:
            step_output = await self.step()
            # self.save_trajectory()
        self._chook.on_run_done(trajectory=self.trajectory, info=self.info)

        self.logger.info("Trajectory saved to %s", self.traj_path)

        # Here we want to return the "global" information (e.g., submission should
        # be the best submission instead of the last one, etc.), so we get it from the traj file
        data = self.get_trajectory_data()
        # self.logger.debug(f"get_trajectory_data -> self.trajectory: {self.trajectory}")
        # set result to sample
        self.sample.prompts = self.init_input_ids
        self.sample.tokens = self.input_ids[len(self.init_input_ids) :]
        self.sample.loss_mask = self.loss_mask[len(self.init_input_ids) :]
        # Note: diff() will be called after run() returns, so don't override with potentially empty submission
        self.sample.m.rollout.patch = data["info"].get("submission", None)
        return AgentRunResult(info=data["info"], trajectory=data["trajectory"])


# ==============================================================================
# DefaultAgentWrapper - 包装 DefaultAgent (用于 validate 阶段)
# ==============================================================================


class DefaultAgentWrapper(AbstractAgent):
    """
    Wrapper for DefaultAgent that handles problem_statement parameter.

    This wrapper:
    1. Wraps the original DefaultAgent from swe-agent
    2. Stores problem_statement and passes it to run() method
    3. Provides the same interface as RLTokenAgent for consistency
    """

    def __init__(
        self,
        *,
        templates: TemplateConfig,
        tools: ToolHandler | ToolConfig,
        history_processors: list[HistoryProcessor],
        model: Any,
        max_requeries: int = 3,
        name: str = "main",
        problem_statement: ProblemStatement | ProblemStatementConfig,
        sample: SWESample | None = None,
    ):
        """Initialize the wrapper."""
        self.problem_statement = problem_statement
        self.sample = sample

        # Convert ToolConfig to SiiToolHandler for E2B compatibility
        if isinstance(tools, ToolConfig):
            tool_handler = SiiToolHandler(tools)
        else:
            tool_handler = tools

        # Create original DefaultAgent
        self._agent = DefaultAgent(
            templates=templates,
            tools=tool_handler,
            history_processors=history_processors,
            model=model,
            max_requeries=max_requeries,
            name=name,
        )

        # Expose key attributes
        self.name = self._agent.name
        self.tools = self._agent.tools
        self.history = self._agent.history
        self.trajectory = self._agent.trajectory
        self.info = self._agent.info

    def add_hook(self, hook):
        """Add hook to the underlying agent."""
        self._agent.add_hook(hook)

    async def run(
        self,
        env: ContainerEnv,
        output_dir: Path = Path("."),
    ) -> AgentRunResult:
        """Run the agent on a problem instance (async).

        Awaits upstream ``DefaultAgent.run`` directly.
        """
        swe_env = env._env if hasattr(env, "_env") else env
        result = await self._agent.run(
            env=swe_env,
            problem_statement=self.problem_statement,
            output_dir=output_dir,
        )

        # Store patch / exit_status in rollout for evaluation.
        # 优先用 self._agent.info（写入 traj 的同一个 dict，最第一手），
        # 落回 result.info 兜底。不传出 exit_status 会让 _should_eval 只能拿到
        # 默认 None，validate 阶段所有样本都会被当成 ABORTED 跳过 eval。
        if self.sample is not None:
            agent_info = getattr(self._agent, "info", {}) or {}
            info = agent_info if agent_info else (result.info or {})
            self.sample.m.rollout.patch = info.get("submission", None)
            self.sample.m.rollout.exit_status = info.get("exit_status", None)

        return result


# ==============================================================================
# LiteLLMAgentBuilder - 使用 DefaultAgent + LiteLLMModel (用于 validate 阶段)
# ==============================================================================


class LiteLLMAgentBuilder(AgentBuilder):
    """使用 LiteLLMModel + sglang API 的 AgentBuilder（用于 validate 阶段）

    特点：
    - 使用原始 DefaultAgent（无 token tracking）
    - 通过 LiteLLMModel 调用 sglang OpenAI-compatible API
    - 动态从 SglangEngine 获取 api_base (ip:port)
    - 配置可复用 agent 的配置（通过 agentflow() 合并）
    """

    def __init__(self, config: dict):
        config = config.copy()
        config.pop("name", None)
        # 分离 model 配置和 agent 配置
        # 保存 model 配置（字典形式），在 build() 时使用
        self.model_config_dict = config.get("model", {}).copy()
        # DefaultAgentConfig 需要一个有效的 ModelConfig
        # 我们先验证 model_config_dict，创建一个 GenericAPIModelConfig 实例
        # 注意：这个 config 中的 model 不会在 build() 中使用，
        # 我们会在 build() 时创建新的 LiteLLMModel 实例（带正确的 api_base）
        model_config = GenericAPIModelConfig.model_validate(self.model_config_dict)
        config["model"] = model_config
        # 解析 agent 配置
        self.agent_config = DefaultAgentConfig.model_validate(config)
        # 初始化 logger
        self.logger = get_logger("litellm_agent_builder", emoji="🔧")

    def build(self, sample: SWESample) -> Agent:
        """构建 DefaultAgent + LiteLLMModel

        从 sample.model.engine 动态获取 sglang 服务的 ip:port
        """
        # 从 SglangModel 获取 engine
        if not hasattr(sample.model, "engine"):
            raise ValueError("sample.model must have 'engine' attribute (SglangModel)")

        engine = sample.model.engine
        api_base = f"http://{engine.ip}:{engine.port}/v1"

        self.logger.info(f"[LiteLLMAgentBuilder] Using sglang API at: {api_base}")

        # 构建模型配置（使用已分离的 model_config_dict）
        model_config_dict = self.model_config_dict.copy()
        # 动态设置 api_base
        model_config_dict["api_base"] = api_base

        # 对于自定义模型（通过 OpenAI-compatible API），使用 custom_llm_provider 参数
        if "completion_kwargs" not in model_config_dict:
            model_config_dict["completion_kwargs"] = {}
        # 告诉 LiteLLM 使用 OpenAI provider，但使用自定义的 api_base
        model_config_dict["completion_kwargs"]["custom_llm_provider"] = "openai"
        # 设置超时时间（默认 1800s 可能不够，对于复杂的 SWE 任务）
        model_config_dict["completion_kwargs"]["timeout"] = 7200  # 60 minutes

        model_config = GenericAPIModelConfig.model_validate(model_config_dict)

        # 创建 LiteLLMModel（使用 sglang OpenAI-compatible API）
        model = LiteLLMModel(model_config, self.agent_config.tools)

        # 构建 problem_statement
        instance = sample.m.data.runtime_meta.instance.copy()
        if "problem_statement" in instance:
            instance.pop("problem_statement")
        if "repo" in instance:
            instance.pop("repo")

        if sample.m.data.issue_images:
            instance.pop("issue_images", None)
            problem_statement = SWEBenchMultimodalProblemStatement(
                text=sample.m.data.problem_statement,
                issue_images=sample.m.data.issue_images,
                id=instance["instance_id"],
                extra_fields=instance,
            )
        else:
            problem_statement = TextProblemStatement(
                text=sample.m.data.problem_statement,
                id=instance["instance_id"],
                extra_fields=instance,
            )

        # 创建 DefaultAgentWrapper，包装原始 DefaultAgent
        # DefaultAgent.run() 需要 problem_statement 参数，但 flow 不传递它
        # 所以我们使用 wrapper 存储 problem_statement 并在 run() 时传递
        return DefaultAgentWrapper(
            templates=self.agent_config.templates,
            tools=self.agent_config.tools,
            history_processors=self.agent_config.history_processors,
            model=model,
            max_requeries=self.agent_config.max_requeries,
            name=self.agent_config.name,
            problem_statement=problem_statement,
            sample=sample,  # Pass sample to save patch
        )
