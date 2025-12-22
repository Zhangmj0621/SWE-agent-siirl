# Copyright 2025, Shanghai Innovation Institute. All rights reserved.
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
import asyncio
from typing import Dict

from siirl.data_coordinator.sample import Sample

from siirl.utils.reward_score import default_compute_score
    

async def naive_flow(sample: Sample, sampling_params: Dict, engine, reward_fn = None):
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
    
    Todo:
        Add support for multi-turn conversation generation
    """
    # todo: add multiturn (support multi-turn conversation generation)
    # Generate response and log probabilities from prompt using inference engine
    _, sample.responses, sample.rollout_log_prob = await engine.generate(sample.prompts, sampling_params)
    
    # Create response mask (all 1s since all generated tokens are valid)
    sample.response_mask = [1] * len(sample.responses)
    
    # Extract metadata for reward calculation
    data_source = sample.data_source  # Source/type of the sample data
    ground_truth = sample.reward_model['ground_truth']  # Ground truth/reference for reward calculation
    
    # Calculate reward score (use custom function if provided, otherwise default)
    if reward_fn:
        # Use custom reward function with decoded response text
        rewards = reward_fn(
            data_source = data_source, 
            solution_str = engine.tokenizer.decode(sample.responses), 
            ground_truth = ground_truth
        )
    else:
        # Use default reward scoring function
        rewards = default_compute_score(
            data_source = data_source, 
            solution_str = engine.tokenizer.decode(sample.responses), 
            ground_truth = ground_truth
        )
    
    # Assign computed reward to sample
    sample.rewards = rewards
    
    return sample