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
import time
import ray
import re
import os
import asyncio
import multiprocessing

from loguru import logger
from ray.util import list_named_actors


from sglang_router.launch_router import RouterArgs, launch_router

from siirl.utils.enums import DistributedEnv
from siirl.utils.net_utils.net import get_net_interface_ip, get_free_port
from siirl.params.training_args import SiiRLArguments
from siirl.worker.ray_utils import get_random_string, RayClassWithInitArgs
from siirl.worker.rollout.rollout_worker import RolloutWorker
from siirl.engine.rollout.sglang_engine import wait_until_ok


class RolloutManager:
    """
    Manages the lifecycle of rollout workers and SGLang router in a distributed training environment.
    Coordinates worker initialization, engine setup, router deployment, and rollout process execution.
    """
    def __init__(self, config:SiiRLArguments, pgs, data_coordinator_handle, rollout_gpu = 8):
        """
        Initialize RolloutManager with core configuration and resources.
        
        Args:
            config: SiiRLArguments containing all training/rollout configuration parameters
            pgs: List of Ray placement groups for distributed resource allocation
            data_coordinator_handle: Ray handle to data coordinator for data management
            rollout_gpu: Total number of GPUs allocated for rollout process (default: 8)
        """
        # Unique prefix for actor naming to avoid collision in distributed environment
        self.name_prefix: str = get_random_string(length=6)
        self.config = config
        self.pgs = pgs  # Ray placement groups for resource management
        self.rollout_gpu = rollout_gpu  # Total GPUs for rollout
        self.device_name = config.trainer.device  # Target device (e.g., "cuda")
        # Calculate number of GPUs per engine (TP size constraint)
        self.num_gpu_per_engine = min(config.rollout.tensor_model_parallel_size, config.trainer.n_gpus_per_node)
        # Total number of engine instances based on GPU allocation
        self.num_engine = rollout_gpu // self.num_gpu_per_engine
        self.data_coordinator = data_coordinator_handle  # Handle to data coordinator actor
        
        self.router_address = None  # Address of SGLang router (ip:port)
        self.worker_handle = []  # List of Ray actor handles for RolloutWorkers
        self.worker_urls = []  # List of worker URLs for router configuration
        # Initialize Ray-wrapped RolloutWorker class with configuration
        self.rollout_ray_class = RayClassWithInitArgs(
            ray.remote(RolloutWorker),
            config,
        ) 
        # Initialize worker nodes, engines, router and start rollout process
        self.init_worker()
        self.init_engine()
        self.start_router()
        self.start_rollout()
        
    def init_worker(self):
        """
        Initialize RolloutWorker actors across all placement groups and GPU ranks.
        Creates a worker instance for each GPU rank in the distributed setup.
        """
        rank = -1  # Global rank counter for distributed training
        # Iterate over placement groups and local GPU ranks to create workers
        for pg_index, placement_group in enumerate(self.pgs):
            for local_rank in range(self.config.trainer.n_gpus_per_node):
                rank += 1
                worker = self._create_worker(
                    rank=rank,
                    local_rank = local_rank,
                    pg_index=pg_index,
                    placement_group=placement_group,
                    num_gpus=0.2,  # GPU resource allocation per worker
                    device_name=self.device_name,
                )
                self.worker_handle.append(worker)
        
        
    def _create_worker(self, rank, local_rank, pg_index, placement_group, num_gpus, device_name):
        """
        Create a single RolloutWorker Ray actor with distributed environment configuration.
        
        Args:
            rank: Global distributed training rank
            local_rank: Local GPU rank on the node
            pg_index: Index of the placement group
            placement_group: Ray placement group for resource allocation
            num_gpus: Number of GPUs to allocate for this worker
            device_name: Target device name (e.g., "cuda")
        
        Returns:
            Ray actor handle to the created RolloutWorker instance
        """
        # --- 1. Set distributed environment variables ---
        env_vars = {
            DistributedEnv.WORLD_SIZE.value: str(self.rollout_gpu),  # Total number of GPUs in rollout cluster
            DistributedEnv.RANK.value: str(rank),  # Global rank
            DistributedEnv.LOCAL_RANK.value: str(local_rank),  # Local rank on node
            DistributedEnv.WG_PREFIX.value: self.name_prefix,  # Unique prefix for worker group
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",  # Preserve CUDA device visibility
        }
        # Pass through GLOO network interface configuration if present
        if os.getenv('GLOO_SOCKET_IFNAME'):
            env_vars['GLOO_SOCKET_IFNAME'] = os.getenv('GLOO_SOCKET_IFNAME')

        # --- 2. Generate a unique and descriptive actor name ---
        # Extract base class name from Ray actor representation
        base_class_repr = type(self.rollout_ray_class.cls).__name__  # e.g., "ActorClass(DAGWorker)"
        match = re.search(r"ActorClass\(([^)]+)\)", base_class_repr)
        actor_class_name = match.group(1) if match else base_class_repr
        # Create unique actor name with prefix, class name and ranks
        actor_name = f"{self.name_prefix}_{actor_class_name}_{pg_index}:{rank}"
        
        # --- 3. Set actor-specific configuration options ---
        self.rollout_ray_class.update_options(
            {
                "runtime_env": {"env_vars": env_vars},  # Set distributed env vars
                "name": actor_name,  # Assign unique actor name
            }
        )

        # --- 4. Create the Ray actor instance ---
        logger.debug(f"Creating actor '{actor_name}' with rank {rank}.")
        worker = self.rollout_ray_class(
            placement_group=placement_group, 
            placement_group_bundle_idx=local_rank, 
            num_gpus=num_gpus, 
            device_name=device_name
        )
        return worker
        
    def init_engine(self):
        """
        Initialize SGLang engine instances on RolloutWorkers with distributed configuration.
        Configures tensor parallelism (TP), data parallelism (DP) and distributed communication addresses.
        """
        # Default rollout worker starts at GPU0 of each node
        tp_size = self.config.rollout.tensor_model_parallel_size  # Tensor parallelism size
        dp_size = self.rollout_gpu // self.config.rollout.tensor_model_parallel_size  # Data parallelism size
        pp_size = 1  # Pipeline parallelism (not used in current setup)
        
        # Prepare distributed initialization addresses for cross-node TP communication
        dist_init_addr = []
        # Need cross-node TP init address when TP size exceeds GPUs per node
        if tp_size > self.config.trainer.n_gpus_per_node:
            for dp_rank in range(dp_size):
                # Get IP:port from TP0 worker of each DP group
                dist_init_addr.append(ray.get(self.worker_handle[dp_rank * tp_size].get_ip_port.remote()))
        else:
            # No cross-node TP needed, use None for all DP ranks
            dist_init_addr = [None] * dp_size        
        
        # Asynchronously initialize engines on workers
        future = []
        for rank, worker in enumerate(self.worker_handle):
            # Only initialize engine on TP0 workers (SGLang requirement)
            if rank % self.config.rollout.tensor_model_parallel_size == 0 or rank % self.config.trainer.n_gpus_per_node == 0:
                # Get network configuration from worker
                ip = ray.get(worker.get_ip.remote())
                port = ray.get(worker.get_free_port.remote())
                nccl_port = ray.get(worker.get_free_port.remote())
                future.append(
                    worker.init_engine.remote(rank, dist_init_addr[rank // tp_size], ip, port, nccl_port)
                )
                # Record worker URL for router configuration
                self.worker_urls.append(f"http://{ip}:{port}")
        # Wait for all engine initialization to complete
        ray.get(future)  

    def get_rollout_worker_on_tp0(self):
        result = []
        for rank, worker in enumerate(self.worker_handle):
            # Only initialize engine on TP0 workers (SGLang requirement)
            if rank % self.config.rollout.tensor_model_parallel_size == 0 or rank % self.config.trainer.n_gpus_per_node == 0: 
                result.append(worker)
        return result

    def start_rollout(self):
        """
        Start the rollout process on all TP0 RolloutWorker instances.
        Triggers async rollout execution in background threads on workers.
        """
        future = []
        for rank, worker in enumerate(self.worker_handle):
            # Only start rollout on TP0 workers (SGLang engine leaders)
            if rank % self.config.rollout.tensor_model_parallel_size == 0:
                future.append(
                    worker.start_rollout.remote(self.router_address, self.data_coordinator, self.num_engine)
                )
        # Wait for all rollout processes to start
        ray.get(future)
        
    def start_router(self, request_timeout: int = 3600):
        """
        Start SGLang router process and configure it with worker URLs.
        Performs health check to ensure router is operational before proceeding.
        
        Args:
            request_timeout: Router request timeout in seconds (default: 100)
        """
        # Get router IP and port from config (auto-generate if not specified)
        router_ip, router_port = self.config.rollout.router_ip, self.config.rollout.router_port
        if not router_ip:
            router_ip = get_net_interface_ip()  # Auto-detect local IP
        if not router_port:
            router_port = get_free_port(router_ip)  # Find free port
        router_address = f"{router_ip}:{router_port}"
        
        # Configure router arguments
        router_args = RouterArgs(
            host=router_ip,
            port=router_port,
            worker_urls=self.worker_urls,  # List of rollout worker URLs
            balance_abs_threshold=0,  # Load balancing threshold
            log_level="warn",  # Router log level
            request_timeout_secs=3600,  # Request timeout
        )
        # Start router in separate process
        router_process = multiprocessing.Process(target=launch_router, args=(router_args,))
        router_process.daemon = True  # Set as daemon to exit with main process
        router_process.start()
        
        # Wait for router to become healthy (3 second initial delay)
        time.sleep(3)
        wait_until_ok(f"http://{router_address}/health", process=router_process)
        self.router_address = router_address
        logger.info(f"Launch Sglang Router at {self.router_address}")
        
        # Update all TP0 workers with router address
        future = []
        for rank, worker in enumerate(self.worker_handle):
            if rank % self.config.rollout.tensor_model_parallel_size == 0:
                future.append(
                    worker.set_router.remote(self.router_address)
                )
        ray.get(future)