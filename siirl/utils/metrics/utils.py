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
Metric data structures and aggregation functions.

Reused from siiRL-github/siirl/execution/metric_worker/utils.py
"""

import torch
from statistics import mean
from typing import List, Any
from dataclasses import dataclass


@dataclass
class Metric:
    """Single metric entry with name, value, and expected world_size."""
    name: str
    value: Any  # Can be float or List[float]
    world_size: int


def MetricFunc(name: str):
    """
    Automatically select aggregation function based on metric name.
    
    Rules:
    - "min" in name → MinMetric
    - "max" in name → MaxMetric
    - "sum" or "total" in name → SumMetric
    - else → MeanMetric
    
    Args:
        name: Metric name string
        
    Returns:
        Aggregation function
    """
    if "min" in name:
        return MinMetric
    elif "max" in name:
        return MaxMetric
    elif "sum" in name or "total" in name:
        return SumMetric
    else:
        return MeanMetric


def _flatten_values(metrics: List[Metric]) -> List[float]:
    """Flatten metrics list to value list (supports value being a list)."""
    return [v
        for metric in metrics
        for v in (metric.value if isinstance(metric.value, list) else [metric.value])]


def SumMetric(metrics: List[Metric]) -> float:
    """Aggregate metrics by sum."""
    values = _flatten_values(metrics)
    return sum(values)


def MeanMetric(metrics: List[Metric]) -> float:
    """Aggregate metrics by mean."""
    values = _flatten_values(metrics)
    return mean(values)


def MaxMetric(metrics: List[Metric]) -> float:
    """Aggregate metrics by max."""
    values = _flatten_values(metrics)
    return max(values)


def MinMetric(metrics: List[Metric]) -> float:
    """Aggregate metrics by min."""
    values = _flatten_values(metrics)
    return min(values)

