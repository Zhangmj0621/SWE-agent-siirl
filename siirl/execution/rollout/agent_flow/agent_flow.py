import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from siirl.data_coordinator.sample import Sample
from siirl.engine.rollout.sglang_engine import SglangEngine

from ..agentflow import AgentFlow, ModelResponse, load_agentflow
from ..utils import (
    ContextWindowExceededError,
    EnvCreateError,
    RolloutGenerationAborted,
    SglangGenerationAborted,
)

LLMEngine = Any  # siirl.engine.rollout.sglang_engine.SglangEngine


# ==============================================================================
# Configuration Classes (compatible with SWE-slime format)
# ==============================================================================


class SGLangModelConfig(BaseModel):
    """Configuration for SGLangModel, compatible with SWE-slime's SGLangModelConfig.

    All fields are optional to ensure compatibility without requiring full configuration.
    """

    model_config = ConfigDict(extra="allow")

    # Basic model settings
    name: str = Field(default="sglang-model", description="Name of the model")
    temperature: float = 0.0
    top_p: float | None = 1.0

    # Limits (optional, won't error if not set or set to 0)
    per_instance_cost_limit: float = 0.0
    per_instance_call_limit: int = 0
    total_cost_limit: float = 0.0

    # Delay control (optional)
    delay: float = 0.0

    # Token limits (optional)
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None

    # API settings (optional)
    api_base: str | None = None
    api_key: SecretStr | None = None

    tool_parser: str | None = None

    message_separator: str = "\n"


@dataclass
class InstanceStats:
    """Track usage statistics for a single instance/sample."""

    instance_cost: float = 0.0
    tokens_sent: int = 0
    tokens_received: int = 0
    api_calls: int = 0


# ==============================================================================
# SglangModel - Shared engine with per-sample state management
# ==============================================================================


class SglangModel:
    """SglangEngine wrapper with per-sample state management.

    This class:
    - Shares the SglangEngine across all instances (for efficiency)
    - Maintains per-sample state (stats, limits, delays)
    - Compatible with SWE-slime's SGLangModel configuration format
    """

    def __init__(
        self,
        engine: SglangEngine,
        config: SGLangModelConfig = None,
        swe_cfg: dict = None,
        is_validate: bool = False,
    ) -> None:
        """Initialize SglangModel.

        Args:
            engine: Shared SglangEngine instance
            config: Per-sample configuration (limits, delays, etc.)
            swe_cfg: SWE-agent configuration (for tools setup)
            is_validate: Whether this is a validation run
        """
        self.engine = engine
        self.config = config or SGLangModelConfig()
        self.tools = None
        self.is_validate = is_validate

        # Per-sample state
        self.stats = InstanceStats()
        self.last_query_time = 0.0
        self._current_sample = None  # Reference to current sample for seed extraction

        # Setup tools from swe_cfg
        if swe_cfg and (agent_cfg := swe_cfg.get("agent")) and agent_cfg.get("tools"):
            from sweagent.tools.tools import ToolConfig

            tools_config = ToolConfig.model_validate(agent_cfg.get("tools", {}))
            self.tools = tools_config

        # Setup sampling params from config
        sampling_params = {
            "skip_special_tokens": True,
        }
        if self.config.temperature:
            sampling_params["temperature"] = self.config.temperature
        if self.config.max_output_tokens:
            sampling_params["max_new_tokens"] = self.config.max_output_tokens
        if self.config.top_p:
            sampling_params["top_p"] = self.config.top_p
        self.sampling_params = sampling_params

        # Calculate total context window for limit checking
        self.total_context_window = None
        if self.config.max_input_tokens and self.config.max_output_tokens:
            self.total_context_window = self.config.max_input_tokens + self.config.max_output_tokens

    def reset_stats(self):
        """Reset statistics (e.g., for new sample)."""
        self.stats = InstanceStats()
        self.last_query_time = 0.0

    def set_sample(self, sample):
        """Set the current sample for seed extraction.

        Args:
            sample: Sample object with extra_info that may contain 'request_seed'
        """
        self._current_sample = sample

    def _get_request_seed(self, **kwargs) -> int | None:
        """Get request seed from multiple sources with priority.

        Priority order:
        1. kwargs['request_seed'] - explicit override
        2. sample.extra_info['request_seed'] - from sample data
        3. None - no seed (use default behavior)

        Returns:
            int | None: The request seed to use, or None if not specified
        """
        # Priority 1: Explicit override via kwargs
        if "request_seed" in kwargs:
            seed = kwargs["request_seed"]
            if seed is not None:
                logger.debug(f"[SglangModel] Using request_seed from kwargs: {seed}")
                return int(seed)

        # Priority 2: From sample's extra_info
        if self._current_sample and hasattr(self._current_sample, "extra_info"):
            extra_info = self._current_sample.extra_info
            if isinstance(extra_info, dict) and "request_seed" in extra_info:
                seed = extra_info["request_seed"]
                if seed is not None:
                    logger.debug(f"[SglangModel] Using request_seed from sample.extra_info: {seed}")
                    return int(seed)

        return None

    def _check_limits(self, input_tokens: int):
        """Check if any limits would be exceeded.

        Raises:
            ContextWindowExceededError: If input tokens exceed total context window
            RuntimeError: If call limit or cost limit exceeded
        """
        # Check token limits from config
        # Only raise error if input reaches or exceeds total context window
        if self.total_context_window is not None and len(input_tokens) >= self.total_context_window:
            msg = f"Input tokens {len(input_tokens)} exceed total context window {self.total_context_window}"
            raise ContextWindowExceededError(msg)

        # Check call limit
        if 0 < self.config.per_instance_call_limit <= self.stats.api_calls:
            msg = f"API calls {self.stats.api_calls} exceeds limit {self.config.per_instance_call_limit}"
            logger.warning(f"[SglangModel] {msg}")
            raise RuntimeError(msg)

        # Check cost limit
        if 0 < self.config.per_instance_cost_limit <= self.stats.instance_cost:
            msg = f"Instance cost {self.stats.instance_cost} exceeds limit {self.config.per_instance_cost_limit}"
            logger.warning(f"[SglangModel] {msg}")
            raise RuntimeError(msg)

    async def _apply_delay(self):
        """Apply delay between queries if configured."""
        if self.config.delay > 0:
            elapsed = time.time() - self.last_query_time
            if elapsed < self.config.delay:
                await asyncio.sleep(self.config.delay - elapsed)

    def _update_stats(self, input_tokens: int, output_tokens: int):
        """Update statistics after a query."""
        self.stats.tokens_sent += input_tokens
        self.stats.tokens_received += output_tokens
        self.stats.api_calls += 1
        # Note: we don't track actual cost since it's 0 for local models

    def parse_tools(self, response: str, tools: list[dict[str, Any]], parser: str = "qwen25"):
        """
        This function mimics the function call parser API from
        https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py#L952
        But running locally
        """
        import uuid

        from litellm.types.utils import ChatCompletionMessageToolCall
        from litellm.types.utils import Function as LiteLLMFunction
        from sglang.srt.function_call.function_call_parser import FunctionCallParser
        from sglang.srt.managers.io_struct import Function as SGLangFunction
        from sglang.srt.managers.io_struct import Tool

        tools_list = [
            Tool(
                function=SGLangFunction(
                    name=tool["function"]["name"],
                    description=tool["function"]["description"],
                    parameters=tool["function"]["parameters"],
                ),
                type=tool["type"],
            )
            for tool in tools
        ]
        parser = FunctionCallParser(tools=tools_list, tool_call_parser=parser)

        # Strip leading whitespace to fix parser issues
        response = response.lstrip()

        normal_text, calls = parser.parse_non_stream(response)

        tool_calls: list[ChatCompletionMessageToolCall] = []

        for call in calls:
            # SGLang parser output may vary a bit; normalize defensively.
            # Handle Pydantic v2 (model_dump), Pydantic v1 (dict), and plain dict
            if hasattr(call, "model_dump"):
                d = call.model_dump()
            elif hasattr(call, "dict"):
                d = call.dict()
            elif isinstance(call, dict):
                d = call
            else:
                logger.warning(f"[SglangModel] Unexpected call type: {type(call)}, skipping")
                continue
            fn_block = d.get("function") if isinstance(d.get("function"), dict) else {}
            if not isinstance(fn_block, dict):
                fn_block = {}

            name = None
            if isinstance(fn_block.get("name"), str) and fn_block.get("name"):
                name = fn_block.get("name")
            elif isinstance(d.get("name"), str) and d.get("name"):
                name = d.get("name")
            if not isinstance(name, str) or not name:
                continue

            args = None
            if "arguments" in fn_block:
                args = fn_block.get("arguments")
            elif "parameters" in fn_block:
                args = fn_block.get("parameters")
            elif "arguments" in d:
                args = d.get("arguments")
            elif "parameters" in d:
                args = d.get("parameters")
            if args is None:
                args = "{}"
            if isinstance(args, dict):
                args = json.dumps(args, ensure_ascii=False)
            if not isinstance(args, str):
                args = str(args)

            tool_index = -1
            if isinstance(fn_block.get("tool_index"), int):
                tool_index = fn_block["tool_index"]
            elif isinstance(d.get("tool_index"), int):
                tool_index = d["tool_index"]
            elif isinstance(d.get("index"), int):
                tool_index = d["index"]

            call_id = d.get("id")
            if not isinstance(call_id, str) or not call_id:
                call_id = f"call_{uuid.uuid4().hex[:24]}"

            tool_calls.append(
                ChatCompletionMessageToolCall(
                    index=tool_index,
                    function=LiteLLMFunction(name=name, arguments=args),
                    id=call_id,
                    type="function",
                )
            )

        result = {"normal_text": normal_text, "calls": tool_calls}

        return result

    def parse_normal_text(self, text: str, parser: str = "qwen25") -> tuple[str | None, str | None]:
        """Parse model normal text into (content, reasoning_content).

        For parser == "qwen25":
        - reasoning_content: content inside the first <think> block (if present)
        - content: text after the closing </think> tag (if present)

        Either field may be missing.
        """
        assert parser in ["qwen25"], f"Unsupported parser: {parser}"

        if not isinstance(text, str) or not text:
            return None, None

        if parser == "qwen25":
            open_tag = "<think>"
            close_tag = "</think>"
            start = text.find(open_tag)
            if start == -1:
                # No think tag: treat the whole text as content.
                return text, None

            # Extract reasoning
            after_open = start + len(open_tag)
            end = text.find(close_tag, after_open)
            if end == -1:
                # Unterminated think: everything after <think> is reasoning; no content.
                reasoning = text[after_open:].strip()
                return None, (reasoning if reasoning else None)

            reasoning = text[after_open:end].strip()
            after_close = end + len(close_tag)
            # If </think> exists, content starts from the first non-newline char after it.
            while after_close < len(text) and text[after_close] in ("\n", "\r"):
                after_close += 1
            content = text[after_close:]

            return (content if content else None), (reasoning if reasoning else None)

        # Unreachable due to assert; keep for future extension.
        return text, None

    async def query(
        self,
        input_tokens: list[int],
        messages: list[dict] = None,
        is_validate: bool = None,
        timeout: int | None = None,
        request_seed: int | None = None,
        **kwargs,
    ) -> ModelResponse:
        """Query with state management (limits, delays, stats).

        This method:
        1. Checks limits (call limit, cost limit, token limit)
        2. Applies delay if configured
        3. Queries the shared engine
        4. Updates statistics

        Args:
            input_tokens: Token IDs for the input
            messages: Message history (for logging/debugging)
            is_validate: Whether this is validation mode
            timeout: Request timeout in seconds
            request_seed: Random seed for this request (overrides sample.extra_info)
            **kwargs: Additional parameters (may include request_seed override)

        Returns:
            ModelResponse with generated text and metadata
        """
        # Use instance is_validate as default if not explicitly provided
        if is_validate is None:
            is_validate = self.is_validate

        # Get request seed from multiple sources
        final_request_seed = self._get_request_seed(request_seed=request_seed, **kwargs)

        logger.debug(
            f"[SglangModel.query] Starting, tokens={len(input_tokens)}, "
            f"api_calls={self.stats.api_calls}, sampling_params={self.sampling_params}, "
            f"is_validate={is_validate}, request_seed={final_request_seed}"
        )

        # Check limits before querying
        self._check_limits(input_tokens)

        # Apply delay if configured
        await self._apply_delay()

        # Query the shared engine
        if len(input_tokens) > self.tokenizer.model_max_length > 0:
            msg = f"Input tokens {len(input_tokens)} exceed max tokens of model config {self.tokenizer.model_max_length}"
            raise ContextWindowExceededError(msg)
        if self.engine.max_model_len <= len(input_tokens):
            msg = f"Input tokens {len(input_tokens)} exceed max_model_len of engine config {self.engine.max_model_len}"
            raise ContextWindowExceededError(msg)

        logger.debug("[SglangModel.query] Calling engine.generate")
        task = self.engine.generate(
            input_ids=input_tokens,
            is_validate=is_validate,
            sampling_params=self.sampling_params,
            request_seed=final_request_seed,
        )
        if timeout is None:
            logger.debug("[SglangModel.query] Waiting for response...")
            text, responses, rollout_log_prob, routed_experts = await task
            logger.debug(f"[SglangModel.query] Got response, len={len(text)}")
        else:
            logger.debug(f"[SglangModel.query] Waiting with timeout={timeout}...")
            text, responses, rollout_log_prob, routed_experts = await asyncio.wait_for(task, timeout)

        # Parse tool calls if configured
        if self.tools:
            parser = self.config.tool_parser
            try:
                parse_result = self.parse_tools(text, self.tools.tools, parser=parser)
            except Exception as e:
                logger.info(f"[SglangModel] Failed to parse tools: {e}")
                logger.debug(f"[SglangModel] Response: {text}")
                raise e
            tool_calls = parse_result["calls"]
            normal_text = parse_result["normal_text"]

            content, reasoning_content = self.parse_normal_text(normal_text, parser=parser)
            model_response = ModelResponse(
                output=content,
                output_tokens=responses,
                log_probs=rollout_log_prob,
                experts=None,
                raw={},
            )
            if tool_calls is not None and len(tool_calls) != 0:
                # Keep downstream schema stable (agents expect list[dict])
                model_response.tool_calls = [call.to_dict() for call in tool_calls]
            if reasoning_content:
                # Per request: handle once, and keep both fields identical
                model_response.reasoning_content = reasoning_content
                model_response.thinking_blocks = reasoning_content

        # Update statistics
        output_tokens = len(model_response.output_tokens) if model_response.output_tokens else 0
        self._update_stats(len(input_tokens), output_tokens)
        self.last_query_time = time.time()

        return model_response

    def query_for_swe(
        self,
        history: "list[int] | list[dict]",
        action_prompt: str = "> ",
    ) -> dict:
        """Synchronous query method compatible with SWE-agent AbstractModel.

        This method provides a sync interface that SWE-agent expects, while
        internally calling the async query() method.

        Args:
            history: Chat history (list of dict) or token IDs (list of int)
            action_prompt: Action prompt string (default: ">")

        Returns:
            dict with keys:
                - message: str - model output text
                - output: str - same as message
                - output_tokens: list[int] - generated token IDs
                - log_probs: list[float] - log probabilities
                - tool_calls: list[dict] - optional tool calls
                - reasoning_content: str - optional reasoning
                - thinking_blocks: str - optional thinking blocks
        """
        import asyncio

        # Handle history as list[int] (tokens) or list[dict] (messages)
        if isinstance(history, list) and len(history) > 0:
            if isinstance(history[0], int):
                # history is token IDs
                input_tokens = history
                messages = None
            elif isinstance(history[0], dict):
                # history is messages, need to tokenize
                # For now, pass empty tokens and let the engine handle it
                input_tokens = []
                messages = history
            else:
                raise ValueError(f"Unsupported history type: {type(history[0])}")
        else:
            input_tokens = []
            messages = []

        # Run async query in a dedicated thread to avoid event loop conflicts
        import queue
        import threading

        def run_in_thread(result_queue):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                future = self.query(input_tokens=input_tokens, messages=messages)
                result = loop.run_until_complete(future)
                result_queue.put(result)
            except Exception as e:
                result_queue.put(e)
            finally:
                loop.close()

        result_queue = queue.Queue()
        thread = threading.Thread(target=run_in_thread, args=(result_queue,), name=f"SglangModel_query-{id(self)}", daemon=True)
        thread.start()
        thread.join(timeout=3600)

        if thread.is_alive():
            raise TimeoutError("Query timed out after 3600s")

        response = result_queue.get()
        if isinstance(response, Exception):
            raise response
        # Convert ModelResponse to dict format expected by SWE-agent
        result = {
            "message": response.output,
            "output": response.output,
            "output_tokens": response.output_tokens or [],
            "rollout_log_probs": response.log_probs or [],
        }

        # Add optional fields if present
        if response.tool_calls:
            result["tool_calls"] = response.tool_calls
        if response.reasoning_content:
            result["reasoning_content"] = response.reasoning_content
        if response.thinking_blocks:
            result["thinking_blocks"] = response.thinking_blocks

        return result

    @property
    def tokenizer(self):
        return self.engine.tokenizer


# ==============================================================================
# AgentFlowCallable - Entry point for sample processing
# ==============================================================================


class AgentFlowCallable:
    """Callable that creates a new model instance per sample."""

    __slots__ = ["flow", "engine", "model_config", "swe_cfg", "use_router"]

    def __init__(
        self,
        flow: AgentFlow,
        engine: SglangEngine,
        model_config: SGLangModelConfig = None,
        swe_cfg: dict = None,
    ):
        self.flow = flow
        self.engine = engine
        self.model_config = model_config or SGLangModelConfig()
        self.swe_cfg = swe_cfg
        self.use_router = False  # Set True during validate to use router load balancing

    async def __call__(self, sample: Sample, reward_fn=None, is_validate=False, request_seed: int | None = None):
        # is_generate is actually is_validate flag (passed from NaiveExecutor)
        # For SWE flows we use the upstream-compatible variant so swe-agent's
        # RLTokenAgent / DefaultAgent can consume ``model`` directly without an
        # adapter (token_manager, reset_rollout_state, upstream-shaped query dict).
        if self.swe_cfg and self.swe_cfg.get("agent"):
            from .swe_sglang_model import SweSglangModel

            model = SweSglangModel(self.engine, self.model_config, self.swe_cfg, is_validate=is_validate)
        else:
            model = SglangModel(self.engine, self.model_config, self.swe_cfg, is_validate=is_validate)
        # Set the current sample for seed extraction
        model.set_sample(sample)

        # If request_seed is provided, store it in sample.extra_info for SglangModel to use
        if request_seed is not None:
            if sample.extra_info is None:
                sample.extra_info = {}
            sample.extra_info["request_seed"] = request_seed
            logger.debug(f"[AgentFlowCallable] Set request_seed={request_seed} for sample {getattr(sample, 'uid', 'unknown')}")

        # Store weight_version (training step) in sample.extra_info for eval to use
        if sample.extra_info is None:
            sample.extra_info = {}
        sample.extra_info["weight_version"] = self.engine._weight_version
        logger.debug(f"[AgentFlowCallable] Set weight_version={self.engine._weight_version} for sample {getattr(sample, 'uid', 'unknown')}")

        # Preprocess the sample with this sample's model.
        # Partial-rollout inbound: propagate ``sample.partial_agent_data``
        # into the dict passed to preprocess so SWEAgentFlow can switch
        # into the resume branch (attach to pod, restore agent state).
        sample_data = dict(sample.extra_info) if isinstance(sample.extra_info, dict) else {}
        if getattr(sample, "partial_agent_data", None):
            sample_data["partial_agent_data"] = sample.partial_agent_data
        s = await self.flow.preprocess(sample_data, model=model, is_validate=is_validate)

        # Rebind the model's _current_sample to the SWESample so
        # SweSglangModel._pop_sample_partial can see
        # ``partial_response_ids`` / ``partial_rollout_log_prob`` /
        # ``partial_loss_mask`` (which live on SWESample, not the siirl
        # Pydantic Sample). Seed / rid already captured on the earlier
        # set_sample call above, so this rebind is safe.
        if hasattr(model, "_current_sample"):
            model._current_sample = s

        try:
            await self.flow.generate(s)
            await self.flow.reward(s)
        except RolloutGenerationAborted:
            # Already wrapped — just let NaiveExecutor.generate see it and
            # put_partial the sample.
            raise
        except SglangGenerationAborted:
            # SWEAgentFlow writes partial_agent_data onto ``s.m.rollout``;
            # move it onto the siirl Sample before handing off.
            partial = (
                getattr(getattr(s, "m", None), "rollout", None)
                and getattr(s.m.rollout, "partial_agent_data", None)
            )
            if partial:
                sample.partial_agent_data = partial
                raise RolloutGenerationAborted(sample)
            # No usable partial (env detach failed, or abort hit before we
            # got a snapshot). Fall through to the generic fail path so the
            # sample still makes it into the batch with reward=0.
            logger.warning(
                "[AgentFlowCallable] Abort without partial; falling through to fail"
            )
            s.reward = 0.0
        except EnvCreateError as e:
            logger.warning(f"Create Env Failed, set sample to default None {e}", exc_info=True)
            logger.warning(f"EnvCreateError type: {type(e).__name__}, message: {str(e)}")
            logger.warning("EnvCreateError traceback:", exc_info=True)
            s.reward = 0.0
            # Set exit_status to indicate environment creation failure
            # This helps distinguish from other types of aborts (e.g., exit_format)
            if hasattr(s, "m") and hasattr(s.m, "rollout"):
                s.m.rollout.exit_status = "env_create_failed"
        except Exception as e:
            logger.error(f"Failed to execute rollout and reward. Error type: {type(e).__name__}", exc_info=True)
            logger.error(f"Error message: {str(e)}")
            logger.error("Full traceback:", exc_info=True)
            import traceback

            logger.error(f"Detailed traceback:\n{traceback.format_exc()}")
            s.reward = 0.0
        else:
            # Normal completion: clear any stale partial so a future retry
            # of this sample doesn't accidentally resume from old state.
            sample.partial_agent_data = None

        # Check if rollout generated any tokens
        if len(s.tokens) == 0:
            if not s.reward:
                s.reward = 0.0
                s.status = s.Status.FAILED
                logger.warning("[AgentFlowCallable] No tokens generated for sample, skipping")
            # Return minimal valid data to avoid TransformerEngine crash
            # Use pad_token_id if available, otherwise use 0
            pad_token_id = getattr(s.model.tokenizer, "pad_token_id", 0)
            if pad_token_id is None:
                pad_token_id = 0
            sample.prompts = np.array([pad_token_id], dtype=np.int64)
            sample.responses = np.array([pad_token_id], dtype=np.int64)
            sample.rollout_log_prob = np.array([0.0], dtype=np.float32)
            sample.response_mask = np.array([0], dtype=np.int64)
            sample.rewards = cast(float, s.reward)
            return sample

        # TODO: 对齐 sample；暂时只赋值 naive_flow 里的那些
        # Ensure prompts is not empty
        if len(s.prompts) == 0:
            logger.warning("[AgentFlowCallable] No prompts in sample, using pad token")
            pad_token_id = getattr(s.model.tokenizer, "pad_token_id", 0)
            if pad_token_id is None:
                pad_token_id = 0
            sample.prompts = np.array([pad_token_id], dtype=np.int64)
        else:
            sample.prompts = np.array(s.prompts)

        sample.responses = np.array(s.tokens, dtype=np.int64)

        # Filter rollout_log_prob to only include response part, not prompt
        # Some inference engines return log_probs for full sequence (prompt + response)
        # We only want the response part to match with training framework expectations
        rollout_log_probs_full = np.array(s.rollout_log_probs, dtype=np.float32)
        response_length = len(s.tokens)

        if len(rollout_log_probs_full) > response_length:
            # rollout_log_probs contains prompt part, slice to response only
            rollout_log_probs_response = rollout_log_probs_full[-response_length:]
            logger.debug(
                f"[AgentFlowCallable] Sliced rollout_log_prob from {len(rollout_log_probs_full)} " f"to {response_length} (response only)"
            )
            sample.rollout_log_prob = rollout_log_probs_response
        else:
            # rollout_log_probs already only contains response part
            sample.rollout_log_prob = rollout_log_probs_full

        # TRUNCATED 样本：保留真实 reward 参与 GRPO group baseline，但将 response_mask
        # 全部置 0，使其不贡献到 policy gradient（对齐 Agentic_RL 的 remove_sample 语义）。
        loss_mask_arr = np.array(s.loss_mask, dtype=np.int64)
        if s.status == s.Status.TRUNCATED:
            logger.info(f"[AgentFlowCallable] Zeroing response_mask for TRUNCATED sample " f"(len={len(loss_mask_arr)}, reward={s.reward})")
            loss_mask_arr = np.zeros_like(loss_mask_arr)
        sample.response_mask = loss_mask_arr
        sample.rewards = cast(float, s.reward)

        # Record multiturn interaction count (API calls) in timing_info
        if sample.timing_info is None:
            sample.timing_info = {}
        sample.timing_info["multiturn_turns"] = model.stats.api_calls

        return sample

    def __repr__(self) -> str:
        return "AgentflowCallable"


# ==============================================================================
# build_agentflow - Factory function
# ==============================================================================


def build_agentflow(
    config: dict,
    engine: LLMEngine,
    model_config: SGLangModelConfig = None,
) -> Callable[..., Awaitable]:
    """Build agentflow from siirl executor.

    Args:
        config: Agent configuration dict (may contain 'model' field with SGLangModelConfig)
        engine: Shared SglangEngine instance
        model_config: Optional SGLangModelConfig for per-sample limits/delays.
                     If None, will be loaded from config['model']

    Returns:
        Callable that processes samples with independent model state

    Example:
        ```python
        build_agentflow(
            {
                "name": "swe",
                "runtime": {"name": "swefactory"},
                "agent": {"name": "minisweagent"},
                "environment": {"name": "k8s"},
                "model": {
                    "per_instance_call_limit": 100,
                    "delay": 0.1,
                },
            },
            engine,
        )
        ```
    """
    # Load model_config from config['agent']['model'] if not explicitly provided
    logger.info(f"[build_agentflow] Input config has 'model': {'model' in config}, keys: {list(config.keys())}")
    if "agent" in config and isinstance(config["agent"], dict) and "model" in config["agent"]:
        logger.info(f"[build_agentflow] config['agent']['model'] = {config['agent']['model']}")

    if model_config is None and "agent" in config and isinstance(config["agent"], dict) and "model" in config["agent"]:
        model_cfg_dict = config["agent"]["model"]
        # Pydantic will automatically ignore extra fields
        logger.info(f"[build_agentflow] Creating SGLangModelConfig with dict: {model_cfg_dict}")
        model_config = SGLangModelConfig(**model_cfg_dict)
        logger.info(f"[build_agentflow] Loaded model config from config['agent']['model']: {model_config}")

        # Log loaded fields (excluding api_key for security)
        logged_cfg = model_config.model_dump(exclude={"api_key"})
        logger.debug(f"[build_agentflow] Model config: {logged_cfg}")

    # Load flow without model (model will be created per sample)
    flow = load_agentflow(config)

    # Return callable that creates model per sample
    return AgentFlowCallable(flow, engine, model_config, config)
