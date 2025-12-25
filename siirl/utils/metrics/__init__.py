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
siiRL-agentic Metrics Module

Provides distributed metrics collection and aggregation functionality.
Reused from siiRL-github with minimal modifications.

Usage:
    from siirl.utils.metrics import MetricWorker, MetricClient
    
    # Create MetricWorker (Ray Actor)
    metric_worker = MetricWorker.remote()
    ray.get(metric_worker.start.remote())
    
    # Create MetricClient in each Trainer
    client = MetricClient(metric_worker)
    client.submit_metric({"loss": 0.5}, world_size=4)
    client.wait_submit()
    
    # Get aggregated results
    final_metrics = client.wait_final_res()
"""

from .utils import Metric, MetricFunc, MeanMetric, SumMetric, MaxMetric, MinMetric
from .metric_worker import MetricWorker, MetricClient
from .metric_utils import compute_data_metric, compute_timing_metrics, compute_throughput_metrics

__all__ = [
    # Data structures and aggregation functions
    "Metric",
    "MetricFunc",
    "MeanMetric",
    "SumMetric",
    "MaxMetric",
    "MinMetric",
    # Ray Actor and Client
    "MetricWorker",
    "MetricClient",
    # Metric computation functions
    "compute_data_metric",
    "compute_timing_metrics",
    "compute_throughput_metrics",
]

