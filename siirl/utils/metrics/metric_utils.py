# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""
Metrics computation functions for RL training.

Simplified version reused from siiRL-github/siirl/utils/metrics/metric_utils.py
Can be extended as needed.
"""

import os
import psutil
import torch
from typing import Any, Dict, Optional, Tuple
from tensordict import TensorDict

from .utils import StdStats


def _compute_response_info(batch: TensorDict) -> Dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.
    
    Args:
        batch: A TensorDict object containing batch data with responses and attention masks.
        
    Returns:
        A dictionary containing:
            - response_mask: Attention mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    response_length = batch["responses"].shape[-1]
    prompt_mask = batch["attention_mask"][:, :-response_length]

    if "response_mask" not in batch:
        response_mask = batch["attention_mask"][:, -response_length:]
    else:
        response_mask = batch["response_mask"]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metric(data: TensorDict) -> Dict[str, float]:
    """
    Computes various metrics from a batch of data for RL training.
    
    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data.
    
    Args:
        data: A TensorDict object containing batch data.
        
    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if available)
            - response/length/mean, max, min: Statistics about response lengths
            - prompt/length/mean, max, min: Statistics about prompt lengths
    """
    metrics = {}
    
    # Score metrics
    if "token_level_scores" in data:
        sequence_score = data["token_level_scores"].sum(-1)
        metrics["critic/score/mean"] = torch.mean(sequence_score).detach().item()
        metrics["critic/score/max"] = torch.max(sequence_score).detach().item()
        metrics["critic/score/min"] = torch.min(sequence_score).detach().item()
    
    # Reward metrics
    if "token_level_rewards" in data:
        sequence_reward = data["token_level_rewards"].sum(-1)
        metrics["critic/rewards/mean"] = torch.mean(sequence_reward).detach().item()
        metrics["critic/rewards/max"] = torch.max(sequence_reward).detach().item()
        metrics["critic/rewards/min"] = torch.min(sequence_reward).detach().item()
    
    # Advantage metrics
    if "advantages" in data:
        advantages = data["advantages"]
        if "response_mask" in data:
            response_mask = data["response_mask"].bool()
            valid_adv = torch.masked_select(advantages, response_mask)
        else:
            valid_adv = advantages.flatten()
        
        if valid_adv.numel() > 0:
            metrics["critic/advantages/mean"] = torch.mean(valid_adv).detach().item()
            metrics["critic/advantages/max"] = torch.max(valid_adv).detach().item()
            metrics["critic/advantages/min"] = torch.min(valid_adv).detach().item()
    
    # Returns metrics
    if "returns" in data:
        returns = data["returns"]
        if "response_mask" in data:
            response_mask = data["response_mask"].bool()
            valid_returns = torch.masked_select(returns, response_mask)
        else:
            valid_returns = returns.flatten()
        
        if valid_returns.numel() > 0:
            metrics["critic/returns/mean"] = torch.mean(valid_returns).detach().item()
            metrics["critic/returns/max"] = torch.max(valid_returns).detach().item()
            metrics["critic/returns/min"] = torch.min(valid_returns).detach().item()
    
    # Values metrics (if critic is used)
    if "values" in data:
        values = data["values"]
        if "response_mask" in data:
            response_mask = data["response_mask"].bool()
            valid_values = torch.masked_select(values, response_mask)
        else:
            valid_values = values.flatten()
        
        if valid_values.numel() > 0:
            metrics["critic/values/mean"] = torch.mean(valid_values).detach().item()
            metrics["critic/values/max"] = torch.max(valid_values).detach().item()
            metrics["critic/values/min"] = torch.min(valid_values).detach().item()
            
            # Compute explained variance if returns are also available
            if "returns" in data:
                valid_returns = torch.masked_select(data["returns"], response_mask) if "response_mask" in data else data["returns"].flatten()
                if valid_returns.numel() > 0:
                    return_diff_var = torch.var(valid_returns - valid_values)
                    return_var = torch.var(valid_returns)
                    metrics["critic/vf_explained_var"] = (1.0 - return_diff_var / (return_var + 1e-5)).detach().item()
    
    # Response and prompt length metrics
    if "responses" in data and "attention_mask" in data:
        try:
            response_info = _compute_response_info(data)
            prompt_length = response_info["prompt_length"]
            response_length = response_info["response_length"]
            max_response_length = data["responses"].shape[-1]
            max_prompt_length = prompt_length.shape[-1] if len(prompt_length.shape) > 0 else 0
            
            metrics["response/length/mean"] = torch.mean(response_length).detach().item()
            metrics["response/length/max"] = torch.max(response_length).detach().item()
            metrics["response/length/min"] = torch.min(response_length).detach().item()
            metrics["response/clip_ratio/mean"] = torch.mean(torch.eq(response_length, max_response_length).float()).detach().item()
            
            metrics["prompt/length/mean"] = torch.mean(prompt_length).detach().item()
            metrics["prompt/length/max"] = torch.max(prompt_length).detach().item()
            metrics["prompt/length/min"] = torch.min(prompt_length).detach().item()
        except Exception:
            pass  # Skip length metrics if computation fails
    
    # System info
    metrics["perf/process_cpu_mem_used_gb"] = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    
    return metrics


def compute_timing_metrics(batch: TensorDict, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    """
    Computes timing metrics for different processing stages in training.
    
    Args:
        batch: A TensorDict object containing batch data.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
        
    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds (where applicable)
    """
    metrics = {}
    
    # Raw timing metrics
    for name, value in timing_raw.items():
        metrics[f"timing_s/{name}"] = value
    
    # Per-token timing metrics (if we have response info)
    if "responses" in batch and "attention_mask" in batch:
        try:
            response_info = _compute_response_info(batch)
            num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
            num_response_tokens = torch.sum(response_info["response_length"]).item()
            num_overall_tokens = num_prompt_tokens + num_response_tokens

            num_tokens_of_section = {
                "gen": num_response_tokens,
                **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
            }

            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys()):
                if num_tokens_of_section[name] > 0:
                    metrics[f"timing_per_token_ms/{name}"] = timing_raw[name] * 1000 / num_tokens_of_section[name]
        except Exception:
            pass  # Skip per-token metrics if computation fails
    
    return metrics


def compute_throughput_metrics(batch: TensorDict, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    """
    Computes throughput metrics for training.
    
    Args:
        batch: A TensorDict object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.
        
    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU
    """
    # Get total tokens - support both list and tensor formats
    if "global_token_num" in batch:
        global_token_num = batch["global_token_num"]
        if hasattr(global_token_num, 'data'):
            # NonTensorData wrapper
            total_num_tokens = sum(global_token_num.data) if isinstance(global_token_num.data, list) else global_token_num.data
        elif isinstance(global_token_num, list):
            total_num_tokens = sum(global_token_num)
        else:
            total_num_tokens = global_token_num
    else:
        # Fallback: estimate from attention mask
        if "attention_mask" in batch:
            total_num_tokens = torch.sum(batch["attention_mask"]).item()
        else:
            total_num_tokens = 0
    
    time = timing_raw.get("step", 1.0)
    
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus) if time > 0 and n_gpus > 0 else 0,
    }


def compute_log_prob_diff_metrics(
    data: TensorDict,
) -> Tuple[Dict[str, float], Optional[StdStats]]:
    """
    Computes metrics for the difference between rollout log probs and recomputed log probs.
    
    This metric is important in RL to monitor the discrepancy between:
    - rollout_log_probs: log probs from the inference engine during rollout
    - old_log_probs: log probs recomputed by the training framework
    
    Args:
        data: A TensorDict containing:
            - rollout_log_probs: log probs from rollout inference engine
            - old_log_probs: log probs recomputed by training framework
            - response_mask: mask for valid response tokens
            
    Returns:
        Tuple of:
            - Dictionary with max and mean metrics
            - StdStats object for distributed std calculation (None if no valid data)
    """
    metrics = {}
    std_stats = None
    
    if "rollout_log_probs" not in data or "old_log_probs" not in data:
        return metrics, std_stats
    
    # Convert log probs to probs for comparison (same as siiRL-github)
    rollout_probs = torch.exp(data["rollout_log_probs"])
    actor_probs = torch.exp(data["old_log_probs"])
    
    # Compute absolute difference
    probs_diff = torch.abs(rollout_probs.cpu() - actor_probs.cpu())
    
    # Apply mask if available
    if "response_mask" in data:
        mask = data["response_mask"].bool().cpu()
        valid_diff = torch.masked_select(probs_diff, mask)
    else:
        valid_diff = probs_diff.flatten()
    
    if valid_diff.numel() > 0:
        metrics["training/rollout_probs_diff_max"] = torch.max(valid_diff).item()
        metrics["training/rollout_probs_diff_mean"] = torch.mean(valid_diff).item()
        
        # Create StdStats for distributed std calculation
        std_stats = StdStats.from_tensor(valid_diff)
    
    return metrics, std_stats

