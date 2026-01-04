from typing import Optional, Callable, Awaitable, Any, cast
import asyncio
import numpy as np
from loguru import logger

from siirl.data_coordinator.sample import Sample

from ..agentflow import ModelResponse, load_agentflow, AgentFlow

LLMEngine = Any  # siirl.engine.rollout.sglang_engine.SglangEngine


class SglangModel:
    """Wrapper for SIIRL LLMEngine for agentflow.Model;
    its query response contains no raw data
    """

    __slots__ = [
        "engine",
    ]

    def __init__(self, engine: LLMEngine) -> None:
        self.engine = engine

    async def query(
        self,
        input_tokens: list[int],
        messages: list[dict],
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> ModelResponse:
        # see https://docs.sglang.io/basic_usage/sampling_params.html#core-parameters
        # also https://github.com/sgl-project/sglang/blob/main/sgl-model-gateway/src/protocols/sampling_params.rs
        sampling_params = {}
        if max_tokens is not None:
            sampling_params["max_new_tokens"]
        task = self.engine.generate(input_tokens, {"max_new_tokens": max_tokens})
        if timeout is None:
            text, responses, rollout_log_prob = await task
        else:
            text, responses, rollout_log_prob = await asyncio.wait_for(task, timeout)
        return ModelResponse(
            output=text,
            output_tokens=responses,
            log_probs=rollout_log_prob,
            experts=None,
            raw={},
        )

    @property
    def tokenizer(self):
        return self.engine.tokenizer


class AgentFlowCallable:
    __slots__ = ["flow"]

    def __init__(self, flow: AgentFlow):
        self.flow = flow

    async def __call__(
        self,
        sample: Sample,
        reward_fn=None,
        is_generate=False
    ):
        try:
            sample_data = sample.extra_info
            s = self.flow.preprocess(sample_data)
            await self.flow.generate(s)
            await self.flow.reward(s)

            # TODO: 对齐 sample；暂时只赋值 naive_flow 里的那些
            sample.responses = np.array(s.tokens, dtype=np.int64)
            sample.rollout_log_prob = np.array(s.rollout_log_probs, dtype=np.float32)
            sample.response_mask = np.array(s.loss_mask, dtype=np.int64)
            sample.rewards = cast(float, s.reward)
        except Exception as e:
            logger.error(f"SWE Instance fail with exception: {e}")
            sample.responses = np.array([])
            sample.rollout_log_prob = np.array([])
            sample.response_mask = np.array([])
            sample.rewards = 0.0
        return sample

    def __repr__(self) -> str:
        return f"AgentflowCallable"


def build_agentflow(config: dict, engine: LLMEngine) -> Callable[..., Awaitable]:
    """Build agentflow from siirl executor

    Try Call with

    ```python
    build_agentflow(
        {
            "name": "swe",
            "runtime": {"name": "swefactory"},
            "agent": {"name": "minisweagent"},
            "environment": {"name": "k8s"},
        },
        engine,
    )
    ```
    """
    model = SglangModel(engine)
    flow = load_agentflow(config, model)
    return AgentFlowCallable(flow)
