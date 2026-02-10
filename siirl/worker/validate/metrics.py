# Copyright 2026, Shanghai Innovation Institute. All rights reserved.
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

"""Validation metric aggregation helpers."""

from collections import defaultdict

import numpy as np
from loguru import logger

from siirl.data_coordinator import Sample
from siirl.utils.metrics import aggregate_validation_metrics


def aggregate_and_log_validation_metrics(
    all_payloads: list[Sample],
) -> dict[str, float]:
    """
    Aggregates all validation results and logs performance (rank 0 only).

    This method:
    1. Calls _aggregate_validation_results to compute final metrics
    2. Logs a detailed performance breakdown of the validation process
    3. Reports total validation time

    Args:
        all_payloads: All validation payloads gathered from all ranks

    Returns:
        Dict[str, float]: Final aggregated validation metrics
    """
    if not all_payloads:
        logger.warning("Validation finished with no results gathered on Rank 0 to aggregate.")
        return {}
    final_metrics = aggregate_validation_results(all_payloads)
    return final_metrics


def aggregate_validation_results(all_payloads: list[Sample]) -> dict[str, float]:
    """
    Computes the final metric dictionary from all gathered validation payloads.

    This method processes validation results to compute:
    - Mean/majority/best metrics for different data sources
    - Pass@N accuracy metrics
    - Per-data-source test scores

    Args:
        all_payloads: All validation payloads from all ranks

    Returns:
        Dict[str, float]: Final validation metrics organized by data source and metric type
    """
    data_sources = [p.data_source for p in all_payloads]
    sample_inputs = [p.prompt_texts for p in all_payloads]

    infos_dict = defaultdict(list)
    for p in all_payloads:
        infos_dict["reward"].append(p.rewards)

    data_src2var2metric2val = aggregate_validation_metrics(
        data_sources=data_sources,
        sample_inputs=sample_inputs,
        infos_dict=infos_dict,
    )

    metric_dict = {}
    for data_source, var2metric2val in data_src2var2metric2val.items():
        core_var = "acc" if "acc" in var2metric2val else "reward"
        for var_name, metric2val in var2metric2val.items():
            if not metric2val:
                continue

            # Robustly parse '@N' to prevent crashes from malformed metric names.
            n_max_values = []
            for name in metric2val:
                if "@" in name and "/mean" in name:
                    try:
                        n_val = int(name.split("@")[-1].split("/")[0])
                        n_max_values.append(n_val)
                    except (ValueError, IndexError):
                        continue  # Ignore malformed metric names

            n_max = max(n_max_values) if n_max_values else 1

            for metric_name, metric_val in metric2val.items():
                is_core_metric = (
                    (var_name == core_var)
                    and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                    and (f"@{n_max}" in metric_name)
                )

                metric_sec = "val-core" if is_core_metric else "val-aux"
                pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                metric_dict[pfx] = metric_val

    # Re-calculate test_score per data source
    data_source_rewards = defaultdict(list)
    for p in all_payloads:
        data_source_rewards[p.data_source].append(p.rewards)

    for source, rewards in data_source_rewards.items():
        if rewards:
            metric_dict[f"val/test_score/{source}"] = np.mean(rewards)

    return metric_dict
