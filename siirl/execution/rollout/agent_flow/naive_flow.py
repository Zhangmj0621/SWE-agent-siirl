# Copyright 2025, Shanghai Innovation Institute.
# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import asyncio
import json
import os
import time
from typing import Dict, Any
import numpy as np
from loguru import logger

from siirl.data_coordinator.sample import Sample
from siirl.params import SiiRLArguments, MultiturnArguments
from siirl.utils.reward_score import default_compute_score
from siirl.execution.rollout.utils import AgentData, AgentState
from siirl.engine.rollout.sglang_engine import SglangEngine
from siirl.environment import EnvManager, EnvResponse
from siirl.environment import initialize_env
from siirl.environment.tool_env.utils.tool_parser import ToolParser, FunctionCall
from siirl.execution.rollout.utils import format_gpt_oss_tool_response_manually, add_generation_prompt_for_gpt_oss
class NaiveFlow():
    def __init__(self, config:SiiRLArguments, engine) -> None:
        self.config = config
        self.engine = engine
        self.system_prompt = engine.tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True,
        )
        self.multiturn_config = config.rollout.multiturn
        self.max_length = config.rollout.max_model_len
        self.max_response_length = config.data.max_response_length
        self.max_parallel_calls = self.multiturn_config.max_parallel_calls
        self.max_env_response_length = self.multiturn_config.max_env_response_length
        self.max_env_turns = self.multiturn_config.max_env_turns
        self.max_assistant_turns = self.multiturn_config.max_assistant_turns
        self.env_response_truncate_side = self.multiturn_config.env_response_truncate_side
        self.env = None
        self.init_env()
        
    def init_env(self):
        multiturn_config = self.config.rollout.multiturn
        env_type = multiturn_config.env_type
        if env_type:
            self.env = initialize_env(multiturn_config)
            if env_type == "tool_env":
                tool_format = multiturn_config.env_kwargs.get('tool_format', 'hermes')
                self.env.tool_parser = ToolParser.get_tool_parser(tool_format, self.engine.tokenizer)
                self.env.tool_parser_name = tool_format
            
    async def __call__(self, sample: Sample, reward_fn = None, is_validate = False):
        """
        Naive rollout flow implementation for single-turn text generation and reward calculation.
        Generates response from prompt using inference engine, creates response mask, and computes reward score.
        
        Args:
            sample: Sample object containing prompt and metadata for generation
            sampling_params: Dictionary of LLM sampling parameters (temperature, top_p, max_new_tokens, etc.)
            engine: Inference engine instance (e.g., SglangEngine) for text generation
            reward_fn: Optional custom reward function (if None, uses default_compute_score)
        
        Returns:
            Sample object with generated response, log probabilities, response mask, and reward score

        """
        # Track timing for performance analysis
        generation_duration = 0.0
        
        # Generate response and log probabilities from prompt using inference engine
        loop = asyncio.get_event_loop()
        agent_data = AgentData(raw_prompt = sample.raw_prompt.tolist())
        while agent_data.state != AgentState.TERMINATED:
            if agent_data.state == AgentState.PENDING:
                agent_data.state = await self._handle_pending_state(agent_data, loop)
            elif agent_data.state == AgentState.GENERATING:
                gen_start = time.time()
                agent_data.state = await self._handle_generating_state(agent_data, is_validate)
                generation_duration += time.time() - gen_start
            elif agent_data.state == AgentState.PROCESSING_ENV:
                agent_data.state = await self._handle_processing_envs_state(agent_data, loop)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED
        
        # Create response mask (all 1s since all generated tokens are valid)
        response_ids = agent_data.prompts_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompts_ids[: len(agent_data.prompts_ids) - len(agent_data.response_mask)]
        sample.responses = response_ids
        sample.prompts = prompt_ids
        sample.response_mask = agent_data.response_mask
        sample.rollout_log_prob = np.array(agent_data.rollout_log_prob, dtype=np.float32)
        
        # Track reward computation time
        reward_start = time.time()
        
        if agent_data.env_rewards:
            sample.rewards = sum(agent_data.env_rewards)
        else:
            # Extract metadata for reward calculation
            data_source = sample.data_source  # Source/type of the sample data
            ground_truth = sample.reward_model['ground_truth']  # Ground truth/reference for reward calculation
            
            # Calculate reward score (use custom function if provided, otherwise default)
            if reward_fn:
                # Use custom reward function with decoded response text
                rewards = reward_fn(
                    data_source = data_source, 
                    solution_str = self.engine.tokenizer.decode(sample.responses), 
                    ground_truth = ground_truth
                )
            else:
                # Use default reward scoring function
                rewards = default_compute_score(
                    data_source = data_source, 
                    solution_str = self.engine.tokenizer.decode(sample.responses), 
                    ground_truth = ground_truth
                )
            # Assign computed reward to sample
            sample.rewards = rewards
        
        reward_duration = time.time() - reward_start
        
        # Store timing info for aggregation in executor
        sample._generation_duration = generation_duration
        sample._reward_duration = reward_duration
        
        return sample

    async def _handle_processing_envs_state(self, agent_data:AgentData, loop):
        tasks = []
        env_call_name = []
        env_messages = []
        for env_call in agent_data.env_calls[: self.max_parallel_calls]:
            tasks.append(self._step(env_call, agent_data.env_kwargs))
            env_call_name.append(env_call.name)
        
        env_responses = await asyncio.gather(*tasks)
        for env_response in env_responses:
            # Create message from env response
            # Text-only content
            message = {"role": "tool", "content": env_response.text or ""}

            env_messages.append(message)

            if env_response.rewards is not None:
                agent_data.env_rewards.append(env_response.rewards)
        agent_data.messages.extend(env_messages)
      
        if self.env.tool_parser_name == "gpt-oss":
            logger.debug("manually format tool responses for gpt-oss")
            # Format tool responses manually
            tool_response_texts = []
            for i, env_msg in enumerate(env_messages):
                actual_env_name = env_call_name[i]
                formatted = format_gpt_oss_tool_response_manually(env_msg["content"], actual_env_name)
                tool_response_texts.append(formatted)
            
            tool_response_text = add_generation_prompt_for_gpt_oss("".join(tool_response_texts))
            response_ids = await loop.run_in_executor(
                None, lambda: self.engine.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            response_ids = await loop.run_in_executor(
                None,
                lambda:  self.engine.tokenizer.apply_chat_template(env_messages, add_generation_prompt=True, tokenize=True),
            )
            response_ids = response_ids[len(self.system_prompt) :]
        if len(agent_data.response_mask) + len(response_ids) >= self.max_response_length:
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask
        agent_data.prompts_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        agent_data.rollout_log_prob += [0.0] * len(response_ids)
        agent_data.env_turns += 1
        for env_response in env_responses:
            if env_response.complete:
                return AgentState.TERMINATED
        return AgentState.GENERATING
        
    
    
    
    async def _step(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any]
    ) -> tuple[FunctionCall, float, dict]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.env.env_name[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            action = {'instance_id': instance_id, **tool_args}
            env_response:EnvResponse = await tool.step(action)
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return (
                EnvResponse(
                    text=f"Error when executing tool: {e}", 
                )
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        env_response_text = env_response.text
        
        if env_response_text and len(env_response_text) > self.max_env_response_length:
            if self.env_response_truncate_side == "left":
                env_response_text = env_response_text[: self.max_env_response_length] + "...(truncated)"
            elif self.env_response_truncate_side == "right":
                env_response_text = "(truncated)..." + env_response_text[-self.max_env_response_length :]
            else:
                length = self.max_env_response_length // 2
                env_response_text = env_response_text[:length] + "...(truncated)..." + env_response_text[-length:]
        env_response.text = env_response_text
        return env_response

    
    
    async def _handle_generating_state(self, agent_data:AgentData, is_validate = False):
        _, response_ids, rollout_log_prob = await self.engine.generate(agent_data.prompts_ids, is_validate)
        agent_data.response_ids = response_ids
        agent_data.rollout_log_prob += rollout_log_prob
        agent_data.prompts_ids += response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        agent_data.assistant_turns += 1
        
        
        if (len(agent_data.response_mask) >=  self.max_response_length):
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_env_turns and agent_data.env_turns >= self.max_env_turns:
            return AgentState.TERMINATED
        
        if self.multiturn_config.env_type == "tool_env" and self.env:
            _, agent_data.env_calls = await self.env.tool_parser.extract_tool_calls(agent_data.response_ids)
            if agent_data.env_calls:
                return AgentState.PROCESSING_ENV
            
        else:
            assert NotImplementedError(f"{self.multiturn_config.env_type} has not support")
        
        return AgentState.TERMINATED
        
        
        

    async def _handle_pending_state(self, agent_data:AgentData, loop):
        prompts = await loop.run_in_executor(
                None,
                lambda: self.engine.tokenizer.apply_chat_template(
                    agent_data.messages,
                    tools=self.env.tool_schemas if self.env else None,
                    add_generation_prompt=True,
                    tokenize=True,
                ),
            )
        agent_data.prompts_ids = prompts
        return AgentState.GENERATING