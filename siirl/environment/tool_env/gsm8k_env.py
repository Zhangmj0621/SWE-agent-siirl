# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from uuid import uuid4

from siirl.environment import EnvResponse
from siirl.utils.reward_score import math

from .base_tool_env import ToolEnv
from .utils.schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class Gsm8kToolEnv(ToolEnv):
    """A demo tool for calculating the reward of gsm8k.

    - `get_openai_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "calc_gsm8k_reward",
                "description": "A tool for calculating the reward of gsm8k",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The answer to the question",
                        },
                    },
                    "required": ["answer"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: str | None = None, ground_truth: str | None = None, **kwargs) -> tuple[str, EnvResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        if ground_truth is None:
            ground_truth = kwargs.get("create_kwargs", {}).get("ground_truth", None)
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id, EnvResponse()

    async def reset(self, instance_id: str | None = None, ground_truth: str | None = None):
        return self.create(instance_id, ground_truth)

    async def step(self, action: dict) -> tuple[EnvResponse, float, dict]:
        instance_id = action.get("instance_id")
        answer = action.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)
        if answer.startswith("#### "):
            self._instance_dict[instance_id]["response"] = answer
        else:
            self._instance_dict[instance_id]["response"] = "#### " + answer
        reward = await self.calc_reward(instance_id)
        # penalty for non improved answer submission
        tool_reward = 0.0 if reward > self._instance_dict[instance_id]["reward"] else -0.05
        # update the reward
        self._instance_dict[instance_id]["reward"] = reward

        return EnvResponse(text=f"Current parsed {answer=} {reward=}", rewards=tool_reward)

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return math.compute_score(self._instance_dict[instance_id]["response"], self._instance_dict[instance_id]["ground_truth"])

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
