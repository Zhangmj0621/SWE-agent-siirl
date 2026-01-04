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

Supports distributed aggregation including proper std calculation using
the parallel variance formula: Var(X) = E[X²] - E[X]²
"""

import math
import torch
from statistics import mean
from typing import List, Any, Union, Dict
from dataclasses import dataclass


@dataclass
class Metric:
    """Single metric entry with name, value, and expected world_size."""
    name: str
    value: Any  # Can be float, List[float], or StdStats
    world_size: int


@dataclass
class StdStats:
    """
    Statistics needed for distributed std calculation.
    
    For correct distributed std computation, each worker submits:
    - sum: sum of values
    - sum_sq: sum of squared values
    - count: number of values
    
    The global std is then computed as:
        std = sqrt(global_sum_sq/N - (global_sum/N)²)
    
    This is mathematically equivalent to computing std on all data combined.
    
    Proof of correctness:
        Let X = {x_1, ..., x_N} be the global dataset split across K workers.
        Worker k has subset X_k with n_k elements.
        
        Global mean: μ = (Σ_k sum_k) / N where N = Σ_k n_k
        Global variance: σ² = E[X²] - E[X]² = (Σ_k sum_sq_k) / N - μ²
        
        This formula is exact because:
        - E[X²] = (1/N) * Σ_{i=1}^{N} x_i² = (1/N) * Σ_k sum_sq_k
        - E[X]² = μ² = ((1/N) * Σ_k sum_k)²
        
        No approximations are made; this is algebraically identical to
        computing variance on the full dataset.
    """
    sum: float
    sum_sq: float
    count: int
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "StdStats":
        """Create StdStats from a tensor."""
        return cls(
            sum=tensor.sum().item(),
            sum_sq=(tensor ** 2).sum().item(),
            count=tensor.numel()
        )
    
    @classmethod
    def from_values(cls, values: List[float]) -> "StdStats":
        """Create StdStats from a list of values."""
        return cls(
            sum=sum(values),
            sum_sq=sum(v ** 2 for v in values),
            count=len(values)
        )
    
    def to_dict(self) -> Dict[str, Union[float, int]]:
        """Convert to dict for serialization."""
        return {"sum": self.sum, "sum_sq": self.sum_sq, "count": self.count}
    
    @classmethod
    def from_dict(cls, d: Dict[str, Union[float, int]]) -> "StdStats":
        """Create from dict."""
        return cls(sum=d["sum"], sum_sq=d["sum_sq"], count=d["count"])


def MetricFunc(name: str):
    """
    Automatically select aggregation function based on metric name.
    
    Rules:
    - "pooled_std" → PooledStdMetric (distributed pooled std calculation)
    - "std" in name → StdMetric (distributed std calculation)
    - "min" in name → MinMetric
    - "max" in name → MaxMetric
    - "sum" or "total" in name → SumMetric
    - else → MeanMetric
    
    Args:
        name: Metric name string
        
    Returns:
        Aggregation function
    """
    if name == "pooled_std":
        return PooledStdMetric
    elif "std" in name:
        return StdMetric
    elif "min" in name:
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


def PooledStdMetric(metrics: List[Metric]) -> float:
    """
    Aggregate pooled std from distributed StdStats using the correct pooled variance formula.
    
    Pooled standard deviation formula:
    pooled_std = sqrt(Σ(ni - 1) * si² / Σ(ni - 1))
    
    Where:
    - ni is the sample size of group i
    - si² is the sample variance of group i (ddof=1)
    - (ni - 1) * si² = sum of squared deviations = sum_sq - sum²/count
    
    This is mathematically exact for computing pooled std from distributed data.
    
    Args:
        metrics: List of Metric objects where value is StdStats
        
    Returns:
        Pooled standard deviation
    """
    total_sum_sq_dev = 0.0  # Σ(ni - 1) * si²
    total_df = 0            # Σ(ni - 1)
    
    for metric in metrics:
        if isinstance(metric.value, StdStats):
            stats = metric.value
            if stats.count > 1:  # Need at least 2 samples to compute variance
                # Calculate sum of squared deviations: (ni - 1) * si²
                sum_sq_dev = stats.sum_sq - (stats.sum ** 2) / stats.count
                df = stats.count - 1
                
                total_sum_sq_dev += sum_sq_dev
                total_df += df
        elif isinstance(metric.value, dict):
            # Support dict format for serialization
            stats_dict = metric.value
            count = stats_dict.get("count", 0)
            if count > 1:
                sum_val = stats_dict.get("sum", 0)
                sum_sq = stats_dict.get("sum_sq", 0)
                
                sum_sq_dev = sum_sq - (sum_val ** 2) / count
                df = count - 1
                
                total_sum_sq_dev += sum_sq_dev
                total_df += df
    
    if total_df == 0:
        return 0.0
    
    # Pooled variance = Σ(sum_sq_dev) / Σ(df)
    pooled_variance = total_sum_sq_dev / total_df
    
    # Protect against numerical errors that could make variance slightly negative
    pooled_variance = max(0.0, pooled_variance)
    
    return math.sqrt(pooled_variance)

def StdMetric(metrics: List[Metric]) -> float:
    """
    Aggregate std from distributed StdStats using parallel variance formula.
    
    Formula: Var(X) = E[X²] - E[X]² = (Σx²/N) - (Σx/N)²
    
    This is mathematically exact for computing global std from distributed data.
    
    Args:
        metrics: List of Metric objects where value is StdStats
        
    Returns:
        Global standard deviation
    """
    total_sum = 0.0
    total_sum_sq = 0.0
    total_count = 0
    
    for metric in metrics:
        if isinstance(metric.value, StdStats):
            total_sum += metric.value.sum
            total_sum_sq += metric.value.sum_sq
            total_count += metric.value.count
        elif isinstance(metric.value, dict):
            # Support dict format for serialization
            total_sum += metric.value.get("sum", 0)
            total_sum_sq += metric.value.get("sum_sq", 0)
            total_count += metric.value.get("count", 0)
    
    if total_count == 0:
        return 0.0
    
    mean_val = total_sum / total_count
    # Var(X) = E[X²] - E[X]²
    variance = (total_sum_sq / total_count) - (mean_val ** 2)
    
    # Protect against numerical errors that could make variance slightly negative
    variance = max(0.0, variance)
    
    return math.sqrt(variance)

