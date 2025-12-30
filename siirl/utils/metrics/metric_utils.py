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
"""

import os
import psutil
import torch
import ray
import numpy as np
import pandas as pd

from collections import defaultdict
from typing import Any, Dict, Optional, Tuple, List
from tensordict import TensorDict
from scipy.stats import mode

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
    - rollout_log_prob: log probs from the inference engine during rollout
    - old_log_probs: log probs recomputed by the training framework
    
    Args:
        data: A TensorDict containing:
            - rollout_log_prob: log probs from rollout inference engine
            - old_log_probs: log probs recomputed by training framework
            - response_mask: mask for valid response tokens
            
    Returns:
        Tuple of:
            - Dictionary with max and mean metrics
            - StdStats object for distributed std calculation (None if no valid data)
    """
    metrics = {}
    std_stats = None
    
    if "rollout_log_prob" not in data or "old_log_probs" not in data:
        return metrics, std_stats
    
    # Convert log probs to probs for comparison
    rollout_probs = torch.exp(data["rollout_log_prob"])
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
        metrics["actor/rollout_probs_diff_max"] = torch.max(valid_diff).item()
        metrics["actor/rollout_probs_diff_mean"] = torch.mean(valid_diff).item()
        
        # Create StdStats for distributed std calculation
        std_stats = StdStats.from_tensor(valid_diff)
    
    return metrics, std_stats


def extract_rollout_timing_metrics(data: TensorDict) -> Dict[str, Any]:
    """
    Extracts rollout timing metrics from batch data.
    
    Each sample in the batch carries timing_info recorded during rollout:
    - rollout_start_at: Unix timestamp when rollout started
    - rollout_end_at: Unix timestamp when rollout ended
    - rollout_duration: Total rollout time (seconds)
    - generation_duration: LLM generation time (seconds)
    - reward_duration: Reward computation time (seconds)
    
    Args:
        data: A TensorDict containing batch data. The timing_info is expected
              to be stored in data["timing_info"] as a list of dicts.
              
    Returns:
        A dictionary containing:
            - perf/delta_time/rollout_per_sample: Average rollout duration per sample
            - perf/delta_time/generation_per_sample: Average generation duration per sample
            - perf/delta_time/reward_per_sample: Average reward duration per sample
            - _earliest_rollout_start_at: Earliest rollout start timestamp (for e2e calculation)
    """
    metrics = {}
    
    # Try to extract timing_info from data
    timing_info_list = None
    
    if "timing_info" in data:
        timing_info_raw = data["timing_info"]
        # Handle NonTensorData wrapper
        if hasattr(timing_info_raw, 'data'):
            timing_info_list = timing_info_raw.data
        elif isinstance(timing_info_raw, list):
            timing_info_list = timing_info_raw
    
    if not timing_info_list or len(timing_info_list) == 0:
        return metrics
    
    # Extract timing values from each sample
    rollout_durations = []
    generation_durations = []
    reward_durations = []
    rollout_start_times = []
    
    for timing_info in timing_info_list:
        if isinstance(timing_info, dict):
            if "rollout_duration" in timing_info:
                rollout_durations.append(timing_info["rollout_duration"])
            if "generation_duration" in timing_info:
                generation_durations.append(timing_info["generation_duration"])
            if "reward_duration" in timing_info:
                reward_durations.append(timing_info["reward_duration"])
            if "rollout_start_at" in timing_info:
                rollout_start_times.append(timing_info["rollout_start_at"])
    
    # Compute average metrics
    if rollout_durations:
        metrics["perf/delta_time/rollout_per_sample"] = sum(rollout_durations) / len(rollout_durations)
    if generation_durations:
        metrics["perf/delta_time/generation_per_sample"] = sum(generation_durations) / len(generation_durations)
    if reward_durations:
        metrics["perf/delta_time/reward_per_sample"] = sum(reward_durations) / len(reward_durations)
    
    # Store earliest rollout start for e2e latency calculation (internal use)
    if rollout_start_times:
        metrics["_earliest_rollout_start_at"] = min(rollout_start_times)
    
    return metrics

# validate metrics, inherited from siirl
def _calculate_bootstrap_metrics(group: pd.DataFrame, variable_name: str, subset_size: int, n_bootstrap: int = 1000) -> Dict[str, Any]:
    """Performs fully vectorized bootstrap sampling to estimate statistics.

    This is the core computational engine. It avoids all Python loops by using
    NumPy's vectorized indexing and Scipy's vectorized mode calculation to
    efficiently compute metrics for thousands of bootstrap samples at once.

    Args:
        group: DataFrame containing the data for a single prompt, including the
               target variable column and potentially a 'pred' column.
        variable_name: The name of the column to perform bootstrap sampling on.
        subset_size: The size of each bootstrap sample (referred to as 'N').
        n_bootstrap: The number of bootstrap iterations to perform.

    Returns:
        A dictionary containing the calculated mean and StdStats objects for
        best-of-N, worst-of-N, and majority-vote-of-N metrics.
    """
    metrics = {}
    variable_values = group[variable_name].to_numpy()

    # --- Step 1: Generate all random indices for all bootstrap samples at once.
    # This creates a 2D array of shape (n_bootstrap, subset_size), where each
    # row is a set of indices for one bootstrap sample.
    bootstrap_indices = np.random.choice(len(variable_values), size=(n_bootstrap, subset_size), replace=True)

    # --- Step 2: Gather all bootstrap data samples using advanced indexing.
    # This efficiently creates a 2D array of the actual data values for all samples.
    bootstrap_data = variable_values[bootstrap_indices]

    # --- Step 3: Vectorized calculation for best-of-N and worst-of-N.
    # np.max/min along axis=1 finds the best/worst value within each sample.
    # The result is a 1D array of shape (n_bootstrap,).
    max_values_per_sample = np.max(bootstrap_data, axis=1)
    min_values_per_sample = np.min(bootstrap_data, axis=1)

    # Calculate mean and create StdStats for distributed std calculation
    metrics[f"best@{subset_size}/mean"] = np.mean(max_values_per_sample)
    metrics[f"best@{subset_size}/std"] = StdStats.from_values(max_values_per_sample.tolist())
    metrics[f"worst@{subset_size}/mean"] = np.mean(min_values_per_sample)
    metrics[f"worst@{subset_size}/std"] = StdStats.from_values(min_values_per_sample.tolist())

    # --- Step 4: Vectorized calculation for majority vote ('maj').
    if "pred" in group.columns:
        prediction_values = group["pred"].to_numpy()
        bootstrap_predictions = prediction_values[bootstrap_indices]

        # Find the mode (most frequent prediction) for each bootstrap sample.
        # `scipy.stats.mode` is vectorized and can operate along an axis.
        modes_per_sample = mode(bootstrap_predictions, axis=1, keepdims=True)[0]

        # To get the value associated with the majority vote, we find the *first*
        # occurrence of the mode in each sample, replicating the original logic.
        # `argmax` on the boolean mask provides the index of the first `True`.
        mask = bootstrap_predictions == modes_per_sample
        first_match_indices = np.argmax(mask, axis=1)

        # Use the derived indices to gather the final majority vote values.
        # This requires indexing the i-th row of `bootstrap_data` with the i-th index.
        majority_values = bootstrap_data[np.arange(n_bootstrap), first_match_indices]

        metrics[f"maj@{subset_size}/mean"] = np.mean(majority_values)
        metrics[f"maj@{subset_size}/std"] = StdStats.from_values(majority_values.tolist())

    return metrics


@ray.remote
def _process_prompt_group_task(group: pd.DataFrame, numeric_variables: List[str], seed: int) -> pd.DataFrame:
    """A Ray remote task to process metrics for a single prompt group.

    This function serves as the parallel unit of work. It takes a DataFrame
    for one prompt, calculates all standard and bootstrapped metrics, and
    returns a tidy DataFrame of the results.

    Args:
        group: DataFrame containing all data for a single prompt.
        numeric_variables: A list of column names to calculate metrics for.
        seed: The random seed to ensure reproducible results for this task.

    Returns:
        A tidy DataFrame with columns ['data_source', 'prompt', 'var_name',
        'metric_name', 'value'], containing all calculated metrics for the group.
        For std metrics, 'value' will be a StdStats object.
    """
    # Seed the random number generator for this specific worker.
    np.random.seed(seed)

    # Extract identifying information from the group.
    data_source = group["data_source"].iloc[0]
    prompt = group["prompt"].iloc[0]
    num_responses = len(group)

    # Store results in a list of dictionaries for efficient DataFrame creation.
    results = []
    for var_name in numeric_variables:
        base_info = {"data_source": data_source, "prompt": prompt, "var_name": var_name}

        # --- Calculate standard (non-bootstrapped) metrics ---
        results.append({**base_info, "metric_name": f"mean@{num_responses}", "value": group[var_name].mean()})
        
        if num_responses > 1:
            # Use StdStats for std calculation instead of direct std computation
            values = group[var_name].tolist()
            std_stats = StdStats.from_values(values)
            results.append({**base_info, "metric_name": f"std@{num_responses}", "value": std_stats})

            # For pooled_std, we also use StdStats - it contains all the information needed
            # to calculate pooled std in a distributed manner:
            # pooled_variance = Σ(sum_sq - sum²/count) / Σ(count - 1)
            # The StdStats object contains sum, sum_sq, count which allows us to compute
            # both the numerator (sum_sq - sum²/count) and denominator (count - 1)
            results.append({**base_info, "metric_name": "pooled_std", "value": std_stats})

            # --- Calculate bootstrapped metrics for various sample sizes ---
            bootstrap_sizes = sorted(list(set([2**i for i in range(1, 10) if 2**i < num_responses] + [num_responses])))

            for size in bootstrap_sizes:
                bootstrap_results = _calculate_bootstrap_metrics(group, var_name, subset_size=size)
                for metric_name, value in bootstrap_results.items():
                    results.append({**base_info, "metric_name": metric_name, "value": value})

    return pd.DataFrame(results)


def aggregate_validation_metrics(data_sources: List[str], sample_inputs: List[str], infos_dict: Dict[str, List[Any]], seed: int = 42) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_inputs: List of input prompts corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value_or_stdstats
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": StdStats object for distributed std calculation
        - "pooled_std": StdStats object for distributed pooled std calculation
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": StdStats object for distributed std calculation of best values
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": StdStats object for distributed std calculation of worst values
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": StdStats object for distributed std calculation of majority voting (if "pred" exists)

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_inputs = ["prompt1", "prompt1", "prompt2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = aggregate_validation_metrics(data_sources, sample_inputs, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # --- 1. Data Consolidation ---
    # Combine all input lists into a single, unified DataFrame.
    df = pd.DataFrame({"data_source": data_sources, "prompt": sample_inputs, **infos_dict})
    numeric_vars = [col for col, dtype in df.dtypes.items() if pd.api.types.is_numeric_dtype(dtype)]

    # --- 2. Task Preparation ---
    # Split the DataFrame into a list of smaller DataFrames, one for each prompt group.
    prompt_groups = [group for _, group in df.groupby(["data_source", "prompt"])]

    # --- 3. Parallel Dispatch ---
    # Launch all processing tasks concurrently. `ray.remote` returns immediately
    # with a future (ObjectRef) for each task.
    futures = [_process_prompt_group_task.remote(group, numeric_vars, int(seed)) for group in prompt_groups]

    # --- 4. Result Collection ---
    # `ray.get` blocks until all tasks are complete and retrieves their results.
    processed_df_list = ray.get(futures)
    
    if not processed_df_list:
        return {}
    processed_df = pd.concat(processed_df_list)    

    # --- 5. Final Aggregation ---
    # Handle different types of metrics separately
    output_dict = defaultdict(lambda: defaultdict(dict))
    
    # Group by data_source, var_name, and metric_name
    for (data_source, var_name, metric_name), group in processed_df.groupby(["data_source", "var_name", "metric_name"]):
        values = group["value"].tolist()
        
        # Check if this is a std metric (contains StdStats objects)
        if values and isinstance(values[0], StdStats):
            # Aggregate StdStats objects using the distributed calculation
            total_sum = sum(stats.sum for stats in values)
            total_sum_sq = sum(stats.sum_sq for stats in values)
            total_count = sum(stats.count for stats in values)
            
            # Create aggregated StdStats object
            aggregated_stats = StdStats(sum=total_sum, sum_sq=total_sum_sq, count=total_count)
            output_dict[data_source][var_name][metric_name] = aggregated_stats
        else:
            # For non-std metrics, take the mean across prompts
            output_dict[data_source][var_name][metric_name] = sum(values) / len(values)

    return output_dict
