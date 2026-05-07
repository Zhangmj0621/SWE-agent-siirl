"""Upstream-parity SGLangModel for siirl's SWE flow.

Structurally mirrors ``sweagent.agent.models.SGLangModel`` (token_manager,
incremental tokenisation, parse_response, upstream-shaped return dict,
pydantic ``InstanceStats``) so any method ``RLTokenAgent`` / ``DefaultAgent``
calls on the model is present. The only divergence is ``_single_query``:
instead of ``httpx.AsyncClient.post(f"{api_base}/generate", ...)``, we call
``self.engine.generate(...)`` — siirl's ``SglangEngine`` already runs an
SGLang HTTP server and is what the trainer pushes weights into.

Design choice: **do not subclass siirl's ``SglangModel``**. Every time we
did, an upstream field (``token_manager``, ``new_prompt_token_ids``,
``model_dump``) was silently missing and surfaced only at runtime. Owning
all methods here means that class of bug can't happen again.

siirl-side compatibility surface preserved:
    - ``self.engine`` — used by sii_sweagent for sglang ip:port lookup
    - ``self.tokenizer`` property — used by RLTokenAgentWrapper for state
    - ``self.set_sample(sample)`` — used by AgentFlowCallable
    - ``self.stats.api_calls`` — used by agent_flow.timing_info
    - ``self.tools`` (``ToolConfig``) / ``self.config`` (``SGLangModelConfig``)
"""

from __future__ import annotations

import asyncio
import base64
import copy
import time
import uuid
from typing import Any

import numpy as np
from loguru import logger

from sweagent.agent.models import (
    GLOBAL_STATS,
    GLOBAL_STATS_LOCK,
    InstanceStats,
)
from sweagent.agent.response_parsing import _detect_think_and_return_ori_think
from sweagent.agent.token_manager import TokenManager
from sweagent.types import History, HistoryItem

from siirl.execution.rollout.utils import ContextWindowExceededError

from .agent_flow import SGLangModelConfig


class SweSglangModel:
    """Upstream-parity SGLangModel routed through siirl's SglangEngine."""

    def __init__(
        self,
        engine,
        config: SGLangModelConfig | None = None,
        swe_cfg: dict | None = None,
        is_validate: bool = False,
    ) -> None:
        self.engine = engine
        self.config = config or SGLangModelConfig()
        self.swe_cfg = swe_cfg
        self.is_validate = is_validate

        # upstream-shape state
        self.stats = InstanceStats()
        self.token_manager = TokenManager()
        self._processed_message_count = 0

        # tool schema — parsed from swe_cfg the same way siirl's SglangModel did
        self.tools = None
        if swe_cfg and (agent_cfg := swe_cfg.get("agent")) and agent_cfg.get("tools"):
            from sweagent.tools.tools import ToolConfig

            self.tools = ToolConfig.model_validate(agent_cfg.get("tools", {}))

        # tool/reasoning parser names — upstream's SGLangModelConfig hardcodes
        # qwen25. siirl's config doesn't carry these; default to qwen25.
        self.tool_call_parser = getattr(self.config, "tool_call_parser", None) or getattr(
            self.config, "tool_parser", None
        ) or "qwen25"
        self.reasoning_parser = getattr(self.config, "reasoning_parser", None) or self.tool_call_parser

        self.model_max_input_tokens = self.config.max_input_tokens or 0
        self.model_max_output_tokens = self.config.max_output_tokens or 4096

        # sampling params defaults (mirrors siirl's SglangModel so engine
        # behaviour is identical) — engine's _get_sampling_params still
        # overlays val_kwargs when is_validate=True.
        sampling_params: dict[str, Any] = {"skip_special_tokens": True}
        if self.config.temperature:
            sampling_params["temperature"] = self.config.temperature
        if self.config.max_output_tokens:
            sampling_params["max_new_tokens"] = self.config.max_output_tokens
        if self.config.top_p:
            sampling_params["top_p"] = self.config.top_p
        self.sampling_params = sampling_params

        # siirl-specific: per-sample state
        self.last_query_time = 0.0
        self._current_sample = None
        # Stable request id for the SGLang server. One uuid per sample,
        # preserved across every engine.generate() within the same rollout
        # and across partial-rollout abort/resume. Assigned in set_sample.
        # Matches naive_flow's agent_data.rid convention.
        self._rid: str = ""

    # ------------------------------------------------------------------ #
    # siirl-side compatibility surface
    # ------------------------------------------------------------------ #

    @property
    def tokenizer(self):
        return self.engine.tokenizer

    def set_sample(self, sample) -> None:
        """Called by AgentFlowCallable to inject per-sample context (seed + rid).

        For partial-rollout resume, the sample carries the original rid in
        ``sample.partial_agent_data["rid"]``; reuse it so the SGLang server
        keeps correlating the continued turns as one logical request.
        """
        self._current_sample = sample
        partial = getattr(sample, "partial_agent_data", None) or {}
        self._rid = partial.get("rid") or uuid.uuid4().hex

    def reset_stats(self) -> None:
        self.stats = InstanceStats()
        self.last_query_time = 0.0

    # ------------------------------------------------------------------ #
    # upstream AbstractModel hook
    # ------------------------------------------------------------------ #

    def reset_rollout_state(self) -> None:
        """Called by RLTokenAgent.setup at the start of each rollout."""
        self.token_manager.reset()
        self._processed_message_count = 0
        self.reset_stats()

    def commit_step_tokens(
        self,
        *,
        new_prompt_token_ids: list[int] | None,
        new_prompt_logprobs: list[float] | None,
        output_tokens: list[int] | None,
        output_logprobs: list[float] | None,
        history_len_at_query: int | None,
    ) -> None:
        """Promote a successful _single_query's tokens into token_manager.

        Called exactly once per step from RLTokenAgent.add_step_to_history —
        i.e., after forward_with_handling returned a final StepOutput, so
        retried (failed) attempts never reach this path and token_manager
        stays consistent with self.history.
        """
        new_prompt_token_ids = list(new_prompt_token_ids or [])
        output_tokens = list(output_tokens or [])
        if new_prompt_token_ids:
            if not new_prompt_logprobs or len(new_prompt_logprobs) != len(new_prompt_token_ids):
                new_prompt_logprobs = [0.0] * len(new_prompt_token_ids)
            self.token_manager.add_prompt(new_prompt_token_ids, list(new_prompt_logprobs))
        if output_tokens:
            if not output_logprobs or len(output_logprobs) != len(output_tokens):
                output_logprobs = [0.0] * len(output_tokens)
            self.token_manager.add_response(output_tokens, list(output_logprobs))
        if history_len_at_query is not None:
            self._processed_message_count = int(history_len_at_query) + 1

    # ------------------------------------------------------------------ #
    # Limits / delays / seed — siirl-native
    # ------------------------------------------------------------------ #

    def _get_request_seed(self) -> int | None:
        if self._current_sample is None:
            return None
        extra = getattr(self._current_sample, "extra_info", None)
        if not isinstance(extra, dict):
            return None
        seed = extra.get("request_seed")
        return int(seed) if seed is not None else None

    def _check_limits(self, input_tokens_len: int) -> None:
        # call / cost / context limits — same shape as siirl SglangModel
        total_cw = None
        if self.config.max_input_tokens and self.config.max_output_tokens:
            total_cw = self.config.max_input_tokens + self.config.max_output_tokens
        if total_cw is not None and input_tokens_len >= total_cw:
            raise ContextWindowExceededError(
                f"Input tokens {input_tokens_len} exceed total context window {total_cw}"
            )
        if 0 < self.config.per_instance_call_limit <= self.stats.api_calls:
            raise RuntimeError(
                f"API calls {self.stats.api_calls} exceeds limit {self.config.per_instance_call_limit}"
            )
        if 0 < self.config.per_instance_cost_limit <= self.stats.instance_cost:
            raise RuntimeError(
                f"Instance cost {self.stats.instance_cost} exceeds limit {self.config.per_instance_cost_limit}"
            )

    async def _sleep(self) -> None:
        """Upstream-parity _sleep: respect config.delay against GLOBAL_STATS timestamp."""
        elapsed = time.time() - GLOBAL_STATS.last_query_timestamp
        if elapsed < self.config.delay:
            await asyncio.sleep(self.config.delay - elapsed)
        async with GLOBAL_STATS_LOCK:
            GLOBAL_STATS.last_query_timestamp = time.time()

    # ------------------------------------------------------------------ #
    # Incremental tokenisation — ported verbatim from upstream SGLangModel
    # (sweagent/agent/models.py:994-1102).
    # ------------------------------------------------------------------ #

    def _history_to_messages(self, history: History) -> list[dict[str, Any]]:
        history = copy.deepcopy(history)

        def get_role(history_item: HistoryItem) -> str:
            # siirl's SGLangModelConfig has no convert_system_to_user knob;
            # keep system as-is (matches upstream default False).
            return history_item["role"]

        messages: list[dict[str, Any]] = []
        for history_item in history:
            role = get_role(history_item)
            if role == "tool":
                tool_call_ids = history_item.get("tool_call_ids")
                message = {
                    "role": role,
                    "content": history_item["content"],
                    "tool_call_id": tool_call_ids[0] if tool_call_ids else None,
                }
            elif (tool_calls := history_item.get("tool_calls")) is not None:
                message = {"role": role, "content": history_item["content"], "tool_calls": tool_calls}
            else:
                message = {"role": role, "content": history_item["content"]}
            if "cache_control" in history_item:
                message["cache_control"] = history_item["cache_control"]
            if "reasoning_content" in history_item and history_item["reasoning_content"] is not None:
                message["reasoning_content"] = history_item["reasoning_content"]
            if "thinking_blocks" in history_item and history_item["thinking_blocks"] is not None:
                message["thinking_blocks"] = history_item["thinking_blocks"]
            messages.append(message)
        return messages

    def _sort_incremental_history(self, history: History) -> History:
        sorted_history: History = []
        tool_buffer: list[HistoryItem] = []

        def flush_tool_buffer() -> None:
            nonlocal tool_buffer
            if not tool_buffer:
                return
            tool_buffer = sorted(
                tool_buffer,
                key=lambda item: (item.get("tool_call_ids") or [""])[0],
            )
            sorted_history.extend(tool_buffer)
            tool_buffer = []

        for item in history:
            if item.get("role") == "tool":
                tool_buffer.append(item)
            else:
                flush_tool_buffer()
                sorted_history.append(item)
        flush_tool_buffer()
        return sorted_history

    def _tokenize_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tools: list[dict] | None = None,
    ) -> list[int]:
        return list(
            self.tokenizer.apply_chat_template(
                conversation=messages,
                tools=tools,
                add_generation_prompt=add_generation_prompt,
                tokenize=True,
                return_dict=False,
            )
        )

    def tokenize_prompt_messages(self, history: History) -> list[int] | None:
        messages = self._history_to_messages(history)
        tools_schema = self.tools.tools if self.tools is not None else None

        if len(self.token_manager) == 0:
            return self._tokenize_messages(messages, add_generation_prompt=True, tools=tools_schema)

        if len(history) > self._processed_message_count:
            new_history = self._sort_incremental_history(history[self._processed_message_count :])
            fake_history: History = [
                {
                    "role": "user",
                    "content": "ONLY FOR INCREMENTAL TOKENIZATION",
                    "message_type": "observation",
                }
            ]
            fake_messages = self._history_to_messages(fake_history)
            full_ids = self._tokenize_messages(
                fake_messages + self._history_to_messages(new_history),
                add_generation_prompt=True,
            )
            prefix_ids = self._tokenize_messages(
                fake_messages,
                add_generation_prompt=False,
            )
            # Prepend message_separator so the boundary between token_manager's
            # end (which finishes at <|im_end|> from the sampler) and the new
            # delta's start (<|im_start|>...) matches the chat-template format.
            # Sampler does not emit the trailing \n after <|im_end|>, so we put
            # it back here. Config default is "\n"; set to "" to disable.
            separator_ids: list[int] = []
            sep = getattr(self.config, "message_separator", "\n")
            if sep:
                separator_ids = list(
                    self.tokenizer.encode(sep, add_special_tokens=False)
                )
            return separator_ids + full_ids[len(prefix_ids) :]

        return None

    def _validate_incremental_tokens(self, history: History, new_prompt_token_ids: list[int] | None) -> None:
        # if not self.config.debug_check_incremental_tokens:
        #     return
        expected = self._tokenize_messages(
            self._history_to_messages(history), add_generation_prompt=True, tools=self.tools.tools
        )
        actual = self.token_manager.token_ids + (new_prompt_token_ids or [])
        if expected != actual:
            raise AssertionError(
                f"Incremental tokenization drift detected: expected {len(expected)} tokens, got {len(actual)}"
            )

    # ------------------------------------------------------------------ #
    # Response parsing
    #
    # Equivalent to upstream ``parse_tool_calls_with_sglang`` but bypasses
    # ``sglang.srt.parser.reasoning_parser.ReasoningParser`` — the sglang
    # version shipped with siirl's training image raises
    # ``ValueError: Unsupported model type: qwen25`` on it (upstream tests
    # ran against a different sglang build). For qwen25 the only reason
    # that parser was invoked was to fetch the ``<think>`` / ``</think>``
    # token strings, which are hardcoded anyway.
    # ------------------------------------------------------------------ #

    def parse_tools(self, response: str, tools: list[dict[str, Any]], parser: str = "qwen25"):
        """
        This function mimics the function call parser API from
        https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py#L952
        But running locally
        """
        import uuid

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

        tool_calls: list[dict[str, Any]] = []

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
                {
                    "index": tool_index,
                    "function": {"name": name, "arguments": args},
                    "id": call_id,
                    "type": "function",
                }
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

    def parse_response(self, text: str) -> dict[str, Any]:
        parsed = parse_tool_calls_with_sglang(
            text=text,
            tools=self.tools.tools,
            tool_call_parser=self.tool_call_parser,
            reasoning_parser=self.reasoning_parser,
        )
        output: dict[str, Any] = {"message": parsed["message"]}
        if parsed.get("tool_calls"):
            output["tool_calls"] = parsed["tool_calls"]
        if parsed.get("reasoning_content"):
            output["reasoning_content"] = parsed["reasoning_content"]
        return output

    # ------------------------------------------------------------------ #
    # Stats updater (upstream-shape, async to match upstream contract)
    # ------------------------------------------------------------------ #

    async def _update_stats(self, *, input_tokens: int, output_tokens: int, cost: float) -> None:
        async with GLOBAL_STATS_LOCK:
            GLOBAL_STATS.total_cost += cost
        self.stats.instance_cost += cost
        self.stats.tokens_sent += input_tokens
        self.stats.tokens_received += output_tokens
        self.stats.context = input_tokens + output_tokens
        self.stats.api_calls += 1
        if 0 < self.config.total_cost_limit < GLOBAL_STATS.total_cost:
            raise RuntimeError("Total cost limit exceeded")
        if 0 < self.config.per_instance_cost_limit < self.stats.instance_cost:
            raise RuntimeError("Instance cost limit exceeded")
        if 0 < self.config.per_instance_call_limit < self.stats.api_calls:
            raise RuntimeError("Per instance call limit exceeded")

    # ------------------------------------------------------------------ #
    # Core query path
    # ------------------------------------------------------------------ #

    async def _single_query(
        self,
        history: History,
        n: int | None = None,
        temperature: float | None = None,
    ) -> list[dict[str, Any]]:
        await self._sleep()
        new_prompt_token_ids = self.tokenize_prompt_messages(history)
        # self._validate_incremental_tokens(history, new_prompt_token_ids)
        input_ids = self.token_manager.token_ids + (new_prompt_token_ids or [])
        input_tokens = len(input_ids)

        if (
            self.model_max_input_tokens is not None
            and self.model_max_input_tokens > 0
            and input_tokens > self.model_max_input_tokens
        ):
            raise ContextWindowExceededError(
                f"Input tokens {input_tokens} exceed max tokens {self.model_max_input_tokens}"
            )
        self._check_limits(input_tokens)

        sampling_params = dict(self.sampling_params)
        if temperature is not None:
            sampling_params["temperature"] = float(temperature)

        request_seed = self._get_request_seed()
        return_routed_experts = bool(getattr(self.config, "return_routed_experts", False))

        # Engine replaces upstream's httpx POST — same semantics, in-process.
        # Engine returns: text, output_ids (list[int]), rollout_log_prob (list[float]),
        # routed_experts (np.ndarray[int32] or None).
        # rid is stable for this sample's lifetime (see set_sample) so abort /
        # resume correlates turns of the same logical request on the server.
        text, output_tokens, rollout_log_probs, routed_experts_raw = await self.engine.generate(
            input_ids=input_ids,
            is_validate=self.is_validate,
            sampling_params=sampling_params,
            request_seed=request_seed,
            return_routed_experts=return_routed_experts,
            rid=self._rid,
        )

        output_tokens = list(output_tokens or [])
        output_logprobs = list(rollout_log_probs or [])

        # zmj TODO: only return fake new_prompt_logprobs now since it is not used
        new_prompt_logprobs = [0.0] * len(new_prompt_token_ids) if new_prompt_token_ids else []

        # NOTE: Do NOT commit token_manager here — that used to happen inline and
        # caused retries (FormatError etc.) to see a token_manager contaminated
        # by the failed turn, with no <|im_start|>assistant\n tail. Instead,
        # RLTokenAgent.add_step_to_history calls commit_step_tokens() after
        # forward_with_handling returns successfully. Retries remain idempotent
        # because token_manager / _processed_message_count are untouched here.

        routed_experts = _routed_experts_to_b64(routed_experts_raw)

        parsed = self.parse_response(text)

        output: dict[str, Any] = {
            "message": parsed["message"],
            "output": parsed["message"],
            "new_prompt_token_ids": new_prompt_token_ids or [],
            "new_prompt_logprobs": new_prompt_logprobs,
            "history_len_at_query": len(history),
            "output_tokens": output_tokens,
            "rollout_log_probs": output_logprobs,
            "rollout_routed_experts": routed_experts,
        }
        if parsed.get("tool_calls"):
            output["tool_calls"] = parsed["tool_calls"]
        if parsed.get("reasoning_content"):
            output["reasoning_content"] = parsed["reasoning_content"]
            output["thinking_blocks"] = parsed["reasoning_content"]

        # Update statisticsxw
        self.last_query_time = time.time()

        await self._update_stats(input_tokens=input_tokens, output_tokens=len(output_tokens), cost=0.0)
        return [output]

    async def _query(
        self,
        history: History,
        n: int | None = None,
        temperature: float | None = None,
    ) -> list[dict[str, Any]]:
        # siirl engine handles replication upstream of us; only n==1 is valid.
        if n is not None and n != 1:
            raise AssertionError(
                f"SweSglangModel only supports n=1 (engine replicates one level up); got n={n}"
            )
        return await self._single_query(history, temperature=temperature)

    async def query(
        self,
        history: History | list[int],
        n: int = 1,
        temperature: float | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Match upstream SGLangModel.query signature."""
        if not history:
            message_history: History = []
        elif isinstance(history[0], dict):
            message_history = history  # type: ignore[assignment]
        else:
            raise TypeError("SweSglangModel.query expects list[dict] history for incremental tokenisation")

        try:
            result = await self._query(message_history, n=n, temperature=temperature)
        except Exception as exc:
            logger.debug(f"[SweSglangModel.query] error: {type(exc).__name__}: {exc}")
            raise

        if n is None or n == 1:
            return result[0]
        return result

    # ------------------------------------------------------------------ #
    # Partial rollout: model-side dump / restore.
    #
    # naive_flow stores a flat (prompts_ids / response_mask / rollout_log_prob)
    # view of the rollout tokens. We expose the same shape so the wrapper can
    # build ``sample.partial_agent_data`` using naive_flow's field names, and
    # rehydrate the TokenManager segments on resume.
    # ------------------------------------------------------------------ #

    def dump_state(self) -> dict[str, Any]:
        """Snapshot model-side state for ``sample.partial_agent_data``.

        ``prompts_ids / response_mask / rollout_log_prob`` match naive_flow's
        key names. ``swe_*`` keys are SWE-specific extras that the wrapper
        merges into the final partial dict.
        """
        tm = self.token_manager
        return {
            # naive_flow-aligned (flat) token view
            "prompts_ids": list(tm.token_ids),
            "response_ids": [
                tid for seg in tm._segments if seg.is_response for tid in seg.token_ids
            ],
            "response_mask": list(tm.loss_mask),
            "rollout_log_prob": list(tm.logprobs),
            # SWE-specific model state
            "swe_model_stats": self.stats.model_dump(),
            "swe_processed_message_count": int(self._processed_message_count),
            # rid — same convention as naive_flow agent_data.rid
            "rid": self._rid,
        }

    def restore_state(self, partial: dict[str, Any]) -> None:
        """Rehydrate TokenManager + stats + counters from ``partial_agent_data``.

        TokenManager gets reconstructed from the flat view by splitting
        ``prompts_ids`` along ``response_mask`` 0/1 runs — each contiguous
        run becomes a ``TokenSegment``.
        """
        self.token_manager.reset()
        self._rehydrate_token_manager(
            prompts_ids=partial["prompts_ids"],
            response_mask=partial["response_mask"],
            rollout_log_prob=partial["rollout_log_prob"],
        )
        self._processed_message_count = int(partial["swe_processed_message_count"])
        stats_dict = partial["swe_model_stats"]
        # InstanceStats is a pydantic model; construct from the dumped dict.
        self.stats = type(self.stats)(**stats_dict)
        # _rid already restored by set_sample; partial["rid"] is authoritative.
        self._rid = partial.get("rid") or self._rid

    def _rehydrate_token_manager(
        self,
        *,
        prompts_ids: list[int],
        response_mask: list[int],
        rollout_log_prob: list[float],
    ) -> None:
        """Split the flat token view back into TokenSegments by mask runs."""
        if not prompts_ids:
            return
        if not (len(prompts_ids) == len(response_mask) == len(rollout_log_prob)):
            raise ValueError(
                "[SweSglangModel._rehydrate_token_manager] length mismatch: "
                f"prompts_ids={len(prompts_ids)}, mask={len(response_mask)}, "
                f"logprob={len(rollout_log_prob)}"
            )
        cur_mask = response_mask[0]
        buf_ids: list[int] = []
        buf_lp: list[float] = []

        def _flush() -> None:
            if not buf_ids:
                return
            if cur_mask:
                self.token_manager.add_response(buf_ids, buf_lp)
            else:
                self.token_manager.add_prompt(buf_ids, buf_lp)

        for tid, mask, lp in zip(prompts_ids, response_mask, rollout_log_prob):
            if mask != cur_mask:
                _flush()
                buf_ids = []
                buf_lp = []
                cur_mask = mask
            buf_ids.append(int(tid))
            buf_lp.append(float(lp))
        _flush()

def _routed_experts_to_b64(routed_experts: Any) -> str:
    """Engine decodes routed_experts to ndarray; upstream carries it as base64 str."""
    if routed_experts is None:
        return ""
    if isinstance(routed_experts, str):
        return routed_experts
    if isinstance(routed_experts, np.ndarray):
        return base64.b64encode(routed_experts.tobytes()).decode("ascii")
    if isinstance(routed_experts, (bytes, bytearray)):
        return base64.b64encode(bytes(routed_experts)).decode("ascii")
    return ""

def parse_tool_calls_with_sglang(
    text: str,
    tools: list[dict],
    tool_call_parser: str,
    reasoning_parser: str,
) -> dict:
    from sglang.srt.entrypoints.openai.protocol import Function as SglFunction
    from sglang.srt.entrypoints.openai.protocol import Tool as SglTool
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    
    import json, traceback, uuid

    if not isinstance(text, str) or not text:
        return {"message": "", "tool_calls": None, "reasoning_content": None}

    think_start_token = "<think>"
    think_end_token = "</think>"

    reasoning_content, content_text = _detect_think_and_return_ori_think(
        text,
        think_start_token,
        think_end_token,
    )

    if reasoning_content:
        if think_start_token:
            reasoning_content = reasoning_content.replace(think_start_token, "", 1)
        if reasoning_content.endswith(think_end_token):
            reasoning_content = reasoning_content[: -len(think_end_token)]
        reasoning_content = reasoning_content.strip() or None
    else:
        reasoning_content = None

    if not tools or not content_text:
        return {
            "message": content_text,
            "tool_calls": None,
            "reasoning_content": reasoning_content,
        }

    sgl_tools = [
        SglTool(type=tool.get("type", "function"), function=SglFunction(**tool["function"]))
        for tool in tools
    ]
    parser = FunctionCallParser(sgl_tools, tool_call_parser)

    try:
        if parser.has_tool_call(content_text):
            message_text, call_info_list = parser.parse_non_stream(content_text)
            tool_calls = []
            for call in call_info_list:
                name = call.function.name if hasattr(call, "function") else call.name
                arguments = (
                    call.function.arguments
                    if hasattr(call, "function") and hasattr(call.function, "arguments")
                    else getattr(call, "arguments", getattr(call, "parameters", "{}"))
                )
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                elif not isinstance(arguments, str):
                    arguments = str(arguments)
                tool_calls.append(
                    {
                        "type": "function",
                        "id": getattr(call, "id", None) or f"call_{uuid.uuid4().hex[:24]}",
                        "function": {
                            "name": name,
                            "arguments": arguments,
                        },
                    }
                )
            return {
                "message": message_text,
                "tool_calls": tool_calls or None,
                "reasoning_content": reasoning_content,
            }
    except Exception as exc:
        logger.error("Tool call parsing error: %s", exc)
        traceback.print_exc()

    return {
        "message": content_text,
        "tool_calls": None,
        "reasoning_content": reasoning_content,
    }
    
