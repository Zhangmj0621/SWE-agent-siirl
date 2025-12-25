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
MetricWorker and MetricClient for distributed metrics collection.

Reused from siiRL-github/siirl/execution/metric_worker/metric_worker.py
"""

import ray
import asyncio
from ray.actor import ActorHandle
from typing import Optional, Any, Dict, List
from loguru import logger
from tensordict import TensorDict

from siirl.utils.metrics.utils import Metric, MetricFunc


class MetricClient:
    """
    Client class for interacting with the MetricWorker actor.
    
    Provides methods to submit metrics, wait for submissions to complete,
    and retrieve final aggregated metrics from the worker.
    
    Usage:
        client = MetricClient(metric_worker_handle)
        client.submit_metric({"loss": 0.5}, world_size=4)
        client.wait_submit()
        final_metrics = client.wait_final_res()
    """
    
    def __init__(self, metric_worker: ActorHandle):
        """
        Initialize MetricClient with a reference to the MetricWorker actor.
        
        Args:
            metric_worker: Ray actor handle for the MetricWorker instance
        """
        self.metric_worker = metric_worker
        self.fut: List = []  # List to track pending metric submission futures
        
    def stop(self):
        """Stop the metric worker and terminate its processing loop."""
        ray.get(self.metric_worker.stop.remote())
    
    def submit_metric(self, metrics: Dict[str, Any], world_size: int):
        """
        Submit a dictionary of metrics to the worker for aggregation.
        
        Args:
            metrics: Dictionary containing metric names and values
            world_size: Total number of processes in the distributed system
        """
        self.fut.append(self.metric_worker.submit_metric.remote(metrics, world_size))
    
    def wait_submit(self):
        """Wait for all pending metric submissions to complete."""
        if self.fut:
            ray.get(self.fut)
            self.fut = []
    
    def wait_final_res(self) -> Dict[str, float]:
        """
        Retrieve the final aggregated metrics from the worker.
        
        Returns:
            Dictionary of aggregated metrics
        """
        return ray.get(self.metric_worker.wait_final_res.remote())
    
    # === Convenience methods for common metric computations ===
    
    def compute_local_data_metric(self, data: TensorDict, world_size: int):
        """
        Compute and submit batch-related metrics.
        
        Args:
            data: TensorDict containing the data to process
            world_size: Total number of processes
        """
        from siirl.utils.metrics.metric_utils import compute_data_metric
        metrics = compute_data_metric(data)
        self.submit_metric(metrics, world_size)

    def compute_local_throughput_metrics(self, data: TensorDict, timing_raw: Dict[str, float], n_gpu: int, world_size: int):
        """
        Compute and submit throughput metrics.
        
        Args:
            data: TensorDict containing relevant data (e.g., token counts)
            timing_raw: Dictionary containing raw timing data
            n_gpu: Number of GPUs used
            world_size: Total number of processes
        """
        from siirl.utils.metrics.metric_utils import compute_throughput_metrics
        metrics = compute_throughput_metrics(data, timing_raw, n_gpu)
        self.submit_metric(metrics, world_size)
        
    def compute_local_timing_metrics(self, data: TensorDict, timing_raw: Dict[str, float], world_size: int):
        """
        Compute and submit timing metrics.
        
        Args:
            data: TensorDict containing relevant data
            timing_raw: Dictionary containing raw timing data
            world_size: Total number of processes
        """
        from siirl.utils.metrics.metric_utils import compute_timing_metrics
        metrics = compute_timing_metrics(data, timing_raw)
        self.submit_metric(metrics, world_size)


@ray.remote(num_cpus=0.5)
class MetricWorker:
    """
    Ray actor responsible for aggregating metrics from distributed processes.
    
    Runs an asynchronous loop to process incoming metrics, aggregate them
    across all processes, and provide final results when requested.
    
    Usage:
        metric_worker = MetricWorker.remote()
        ray.get(metric_worker.start.remote())
        
        # Submit metrics from different processes
        ray.get(metric_worker.submit_metric.remote({"loss": 0.5}, world_size=4))
        
        # Get aggregated results
        results = ray.get(metric_worker.wait_final_res.remote())
    """
    
    def __init__(self) -> None:
        from siirl.utils.logger.logging_utils import set_basic_config
        set_basic_config()
        
        self.metric_queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        self.process_task: Optional[asyncio.Task] = None
        self.step = 0
        self.final_metrics: Dict[str, float] = {}
        self.working_metrics: Dict[str, List[Metric]] = {}
    
    async def start(self):
        """
        Start the metrics processing loop.
        
        Initializes and starts the asynchronous loop that processes metrics
        from the queue.
        """
        if self.is_running:
            return
        
        self.is_running = True
        self.process_task = asyncio.create_task(self._process_metrics_loop())
        logger.info("MetricWorker started")

    async def submit_metric(self, metric: Dict[str, Any], world_size: int):
        """
        Submit a metric dictionary to the worker's processing queue.
        
        Args:
            metric: Dictionary of metric names and values
            world_size: Total number of processes in the distributed system
        """
        await self.metric_queue.put((metric, world_size))
    
    async def stop(self):
        """Stop the metrics processing loop and clean up resources."""
        self.is_running = False
        if self.process_task:
            self.process_task.cancel()
            try:
                await self.process_task
            except asyncio.CancelledError:
                pass
    
    async def compute_metric(self, metric_name: str, metrics: List[Metric]):
        """
        Compute the final aggregated value for a metric.
        
        Uses the appropriate metric function to aggregate values from all processes.
        
        Args:
            metric_name: Name of the metric to compute
            metrics: List of Metric objects containing values from each process
        """
        metric_func = MetricFunc(metric_name)
        result = metric_func(metrics)
        self.working_metrics.pop(metric_name)
        
        # Rename timing metrics for consistency in output
        if metric_name.startswith("timing_s/"):
            metric_name = metric_name.replace("timing_s/", "perf/delta_time/")
            
        self.final_metrics[metric_name] = result
           
    async def parse_metric(self, metric_data: tuple):
        """
        Process incoming metric data and aggregate when all processes have submitted.
        
        Collects metric values from each process and triggers computation when
        all values (one per process) have been received.
        
        Args:
            metric_data: Tuple containing (metric_dict, world_size)
        """
        metric_dict, world_size = metric_data
        
        for key, value in metric_dict.items():
            metric = Metric(name=key, value=value, world_size=world_size)
            
            if key in self.working_metrics:
                metrics = self.working_metrics[key]
                metrics.append(metric)
                # Check if we have received values from all processes
                if len(metrics) >= world_size:
                    await self.compute_metric(key, metrics)
            else:
                self.working_metrics[key] = [metric]
    
    async def _process_metrics_loop(self):
        """
        Main loop for processing metrics from the queue.
        
        Continuously retrieves and processes metric data while the worker is running.
        """
        while self.is_running:
            try:
                metric_data = await asyncio.wait_for(self.metric_queue.get(), timeout=0.1)
                await self.parse_metric(metric_data)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def wait_final_res(self) -> Dict[str, float]:
        """
        Wait for all metrics to be processed and return the final results.
        
        Ensures all remaining metrics in the queue are processed, computes any
        remaining aggregated values, and returns the final metrics.
        
        Returns:
            Dictionary of final aggregated metrics
        """
        await self.stop()
        
        # Process any remaining metrics in the queue
        while not self.metric_queue.empty():
            try:
                metric_data = self.metric_queue.get_nowait()
                await self.parse_metric(metric_data)
            except asyncio.QueueEmpty:
                break
        
        # Compute any metrics still in working set
        items = list(self.working_metrics.items())
        for key, value in items:
            await self.compute_metric(key, value)
        
        # Restart the worker for potential future use
        await self.start()
        
        # Capture and reset metrics before returning
        final_metrics = self.final_metrics
        self.final_metrics = {}
        self.working_metrics = {}
        return final_metrics

