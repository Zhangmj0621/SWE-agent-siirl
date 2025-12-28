from typing import Optional, Callable, Awaitable, Any
import asyncio
import numpy as np

from siirl.data_coordinator.sample import Sample, Samples2Dict

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
        sample_data = Samples2Dict([sample]).to_dict(convert_tensors=True)
        s = self.flow.preprocess(sample_data)
        await self.flow.generate(s)
        await self.flow.reward(s)

        # TODO: 对齐 sample；暂时只赋值 naive_flow 里的那些
        sample.responses = np.array(s.tokens, dtype=np.int64)
        sample.rollout_log_prob = np.array(s.rollout_log_probs, dtype=np.float32)
        sample.response_mask = np.array(s.loss_mask, dtype=np.int64)
        sample.rewards = cast(float, s.reward)

        return sample


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
