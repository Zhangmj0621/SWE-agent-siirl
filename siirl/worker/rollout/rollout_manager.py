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
import multiprocessing
import os
import re
import time
import traceback
from collections import deque

import ray

from siirl.params.training_args import SiiRLArguments
from siirl.utils.enums import DistributedEnv
from siirl.utils.net_utils.net import get_free_port, get_net_interface_ip
from siirl.worker.ray_utils import GPUResources, RayClassWithInitArgs, get_random_string


@ray.remote
class RolloutManager:
    """
    Manages the lifecycle of rollout workers and SGLang router in a distributed training environment.

    Key Design:
    - Each RolloutWorker actor corresponds to one SGLang process
    - Single-node TP: 1 actor manages tp_size GPUs
    - Cross-node TP: 1 actor per node, each manages gpus_per_node GPUs

    Example (6 GPUs, tp_size=2):
        - Creates 3 actors, each managing 2 GPUs
        - 3 SGLang processes (TP groups)

    Example (16 GPUs across 2 nodes, tp_size=8):
        - Creates 2 actors (1 per node), each managing 8 GPUs
        - 2 SGLang processes form 1 TP group (cross-node)
    """

    def __init__(
        self,
        config: SiiRLArguments,
        gpu_resources: GPUResources,
        data_coordinator_handle,
        coordinator=None,
        metric_worker=None,
    ):
        """
        Initialize RolloutManager with configuration and GPU resources.

        Args:
            config: SiiRLArguments containing all training/rollout configuration.
            gpu_resources: GPUResources from allocate_resources() containing
                           placement group and allocated GPU bundle indices.
            data_coordinator_handle: Ray handle to DataCoordinator for data management.
            coordinator: Ray handle to TaskCoordinator for lifecycle management.
            metric_worker: Ray handle to Metric_worker
        """
        # Lazy imports to avoid serialization issues with file handles
        from siirl.utils.logger.logging_utils import set_basic_config
        from siirl.worker.rollout.rollout_worker import RolloutWorker

        # Configure logging for this Ray actor process
        # (worker_process_setup_hook only works for task workers, not actors)
        set_basic_config()

        self.name_prefix: str = get_random_string(length=6)
        self.config = config
        self.coordinator = coordinator  # TaskCoordinator for lifecycle management

        # Store GPUResources for centralized access
        self.gpu_resources = gpu_resources
        self.pg = gpu_resources.pg
        self.rollout_gpu = gpu_resources.num_gpus

        self.device_name = config.trainer.device
        self.tp_size = config.rollout.tensor_model_parallel_size
        self.n_gpus_per_node = config.trainer.n_gpus_per_node

        # === Key metrics for rollout/engine management ===
        # GPUs managed by each rollout (capped at node boundary)
        self.gpus_per_rollout = min(self.tp_size, self.n_gpus_per_node)
        # Total number of RolloutWorker rollout to create
        self.num_workers = self.rollout_gpu // self.gpus_per_rollout
        # Number of TP groups (logical inference engines)
        self.num_tp_groups = self.rollout_gpu // self.tp_size
        # Number of rollout per TP group (>1 for cross-node TP)
        self.rollout_per_tp_group = self.tp_size // self.gpus_per_rollout
        self.dp_size = self.num_tp_groups
        # ray_handle
        self.data_coordinator = data_coordinator_handle
        self.metric_worker = metric_worker

        self.router_address = None
        self.worker_handle = []
        self.worker_urls = []

        # Cache for dist_init_addr (used in cross-node TP)
        self._dist_init_addrs = {}

        # Initialize Ray-wrapped RolloutWorker class with configuration
        self.rollout_ray_class = RayClassWithInitArgs(ray.remote(RolloutWorker), config, self.dp_size, metric_worker)
        # used for dataloader
        self.total_training_steps, self.num_train_batches = ray.get(self.data_coordinator.epoch_info.remote())
        self.start_epoch = 0
        self.batches_to_skip = 0
        self.event = asyncio.Event()
        self.global_steps = 0  # will be reset by actor checkpoint, but maybe not correct in fully async mode

        # Initialize workers, engines, router and start rollout
        self.message_queue = deque()

    def init(self):
        self.init_worker()
        self.init_engine()
        self.start_router()
        self.start_rollout()

    def init_worker(self):
        """
        Initialize RolloutWorker actors.

        Creates one actor per SGLang process:
        - Single-node TP: num_rollout_workers = rollout_gpu / tp_size
        - Cross-node TP: num_rollout_workers = rollout_gpu / min(tp_size, gpus_per_node)

        Each actor is placed on the first GPU bundle it manages.
        """
        from loguru import logger

        res = self.gpu_resources

        logger.info("[RolloutManager.init_worker] Configuration:")
        logger.info(f"  rollout_gpu={self.rollout_gpu}, tp_size={self.tp_size}, n_gpus_per_node={self.n_gpus_per_node}")
        logger.info(f"  gpus_per_rollout={self.gpus_per_rollout}, num_actors={self.num_workers}")
        logger.info(f"  num_tp_groups={self.num_tp_groups}, rollout_per_tp_group={self.rollout_per_tp_group}")
        logger.info(f"  GPU indices: {res.indices}, local_ranks: {res.local_ranks}")

        for worker_idx in range(self.num_workers):
            # Calculate the first GPU index this actor manages
            first_gpu_idx = worker_idx * self.gpus_per_rollout
            bundle_idx = res.indices[first_gpu_idx]
            local_rank = res.local_ranks[first_gpu_idx]

            worker = self._create_worker(
                rank=worker_idx,
                local_rank=local_rank,
                bundle_idx=bundle_idx,
                num_gpus=0.2,  # Fractional GPU for Ray scheduling
                device_name=self.device_name,
            )
            self.worker_handle.append(worker)

            logger.debug(
                f"Actor {worker_idx}: bundle_idx={bundle_idx}, "
                f"local_rank={local_rank}, manages GPUs [{first_gpu_idx}:{first_gpu_idx + self.gpus_per_rollout}]"
            )

    def _create_worker(self, rank, local_rank, bundle_idx, num_gpus, device_name):
        """
        Create a single RolloutWorker Ray actor.

        Args:
            rank: Actor index (0 to num_rollout_workers-1)
            local_rank: Local GPU rank on the node (for env vars)
            bundle_idx: Bundle index in the placement group
            num_gpus: Fractional GPU allocation for Ray scheduling
            device_name: Target device name (e.g., "cuda")

        Returns:
            Ray actor handle to the created RolloutWorker instance
        """
        # Set distributed environment variables
        env_vars = {
            DistributedEnv.WORLD_SIZE.value: str(self.num_workers),
            DistributedEnv.RANK.value: str(rank),
            DistributedEnv.LOCAL_RANK.value: str(local_rank),
            DistributedEnv.WG_PREFIX.value: self.name_prefix,
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        }
        if os.getenv("GLOO_SOCKET_IFNAME"):
            env_vars["GLOO_SOCKET_IFNAME"] = os.getenv("GLOO_SOCKET_IFNAME")

        # Generate unique actor name
        base_class_repr = type(self.rollout_ray_class.cls).__name__
        match = re.search(r"ActorClass\(([^)]+)\)", base_class_repr)
        actor_class_name = match.group(1) if match else base_class_repr
        actor_name = f"{self.name_prefix}_{actor_class_name}_actor{rank}_bundle{bundle_idx}"

        self.rollout_ray_class.update_options(
            {
                "runtime_env": {"env_vars": env_vars},
                "name": actor_name,
            }
        )

        from loguru import logger

        logger.debug(f"Creating actor '{actor_name}'")

        worker = self.rollout_ray_class(
            placement_group=self.pg,
            placement_group_bundle_idx=bundle_idx,
            num_gpus=num_gpus,
            device_name=device_name,
        )
        return worker

    def _build_engine_configs(self) -> list:
        """
        Build configuration for each SGLang process.

        Each actor corresponds to one SGLang process. For cross-node TP,
        multiple actors (one per node) form a single TP group with shared dist_init_addr.

        Returns:
            List of dicts, one per actor:
            {
                "worker_idx": int,
                "tp_group_idx": int,
                "base_gpu_id": int,
                "node_rank": int,
                "nnodes": int,
                "dist_init_addr": str | None,
                "is_tp0": bool,
            }
        """
        from loguru import logger

        configs = []
        res = self.gpu_resources

        for worker_idx in range(self.num_workers):
            # Determine which TP group this actor belongs to
            tp_group_idx = worker_idx // self.rollout_per_tp_group
            # Determine node_rank within the TP group
            node_rank = worker_idx % self.rollout_per_tp_group
            nnodes = self.rollout_per_tp_group

            # Get base_gpu_id from GPUResources
            first_gpu_idx = worker_idx * self.gpus_per_rollout
            base_gpu_id = res.local_ranks[first_gpu_idx]

            # Handle dist_init_addr for cross-node TP
            if nnodes > 1:
                if node_rank == 0:
                    # First actor in TP group: generate and cache dist_init_addr
                    dist_init_addr = ray.get(self.worker_handle[worker_idx].get_ip_port.remote())
                    self._dist_init_addrs[tp_group_idx] = dist_init_addr
                    logger.info(
                        f"TP Group {tp_group_idx}: Cross-node TP with {nnodes} nodes, "
                        f"dist_init_addr={dist_init_addr}"
                    )
                else:
                    # Other actors in TP group: use cached dist_init_addr
                    dist_init_addr = self._dist_init_addrs[tp_group_idx]
            else:
                dist_init_addr = None

            configs.append(
                {
                    "worker_idx": worker_idx,
                    "tp_group_idx": tp_group_idx,
                    "base_gpu_id": base_gpu_id,
                    "node_rank": node_rank,
                    "nnodes": nnodes,
                    "dist_init_addr": dist_init_addr,
                    "is_tp0": (node_rank == 0),  # Only node_rank=0 registers with router
                }
            )

            logger.debug(
                f"Actor {worker_idx}: tp_group={tp_group_idx}, node_rank={node_rank}, "
                f"nnodes={nnodes}, base_gpu_id={base_gpu_id}, is_tp0={node_rank == 0}"
            )

        return configs

    def init_engine(self):
        """
        Initialize SGLang engine on each RolloutWorker actor.

        Each actor starts one SGLang process. For cross-node TP,
        actors in the same TP group share dist_init_addr and coordinate via NCCL.
        """
        from loguru import logger

        logger.info("[RolloutManager.init_engine] Starting SGLang engine initialization")

        engine_configs = self._build_engine_configs()

        logger.info(f"  Built {len(engine_configs)} engine configs:")
        for cfg in engine_configs:
            logger.info(
                f"    Actor {cfg['worker_idx']}: tp_group={cfg['tp_group_idx']}, "
                f"node_rank={cfg['node_rank']}/{cfg['nnodes']}, "
                f"base_gpu_id={cfg['base_gpu_id']}, is_tp0={cfg['is_tp0']}, "
                f"dist_init_addr={cfg['dist_init_addr']}"
            )

        futures = []
        for cfg in engine_configs:
            worker = self.worker_handle[cfg["worker_idx"]]

            # Get network configuration from worker
            ip = ray.get(worker.get_ip.remote())
            port = ray.get(worker.get_free_port.remote())
            nccl_port = ray.get(worker.get_free_port.remote())

            future = worker.init_engine.remote(
                rank=cfg["worker_idx"],
                dist_init_addr=cfg["dist_init_addr"],
                ip=ip,
                port=port,
                nccl_port=nccl_port,
                base_gpu_id=cfg["base_gpu_id"],
                node_rank=cfg["node_rank"],
                nnodes=cfg["nnodes"],
            )
            futures.append(future)

            # Only TP0 (node_rank=0) registers with router
            if cfg["is_tp0"]:
                self.worker_urls.append(f"http://{ip}:{port}")

        ray.get(futures)
        logger.info(
            f"Initialized {self.num_workers} SGLang processes "
            f"({self.num_tp_groups} TP groups, {len(self.worker_urls)} router endpoints)"
        )

    def set_step(self, step: int):
        self.global_steps = step

    def get_rollout_worker_on_tp0(self):
        """
        Get RolloutWorker handles for TP0 actors only.

        These are the actors with node_rank=0 in their TP group,
        responsible for serving inference requests.

        Returns:
            List of Ray actor handles for TP0 RolloutWorkers.
        """
        result = []
        for worker_idx in range(self.num_workers):
            # TP0 actors have worker_idx divisible by rollout_per_tp_group
            if worker_idx % self.rollout_per_tp_group == 0:
                result.append(self.worker_handle[worker_idx])
        return result

    def start_rollout(self):
        """
        Start the rollout process on TP0 RolloutWorker actors.
        Only TP0 actors run the executor; other actors only participate in TP communication.
        """
        futures = []
        for worker_idx in range(self.num_workers):
            if worker_idx % self.rollout_per_tp_group == 0:
                futures.append(
                    self.worker_handle[worker_idx].start_rollout.remote(
                        self.router_address, self.data_coordinator, self.dp_size
                    )
                )
        ray.get(futures)

    def start_router(self, request_timeout: int = 3600):
        """
        Start SGLang router process and configure it with worker URLs.
        """
        from sglang_router.launch_router import RouterArgs, launch_router

        from siirl.engine.rollout.sglang_engine import wait_until_ok

        router_ip = self.config.rollout.router_ip or get_net_interface_ip()
        router_port = self.config.rollout.router_port or get_free_port(router_ip)
        router_address = f"{router_ip}:{router_port}"

        router_args = RouterArgs(
            host=router_ip,
            port=router_port,
            worker_urls=self.worker_urls,
            balance_abs_threshold=0,
            log_level="warn",
            request_timeout_secs=3600,
        )

        router_process = multiprocessing.Process(target=launch_router, args=(router_args,))
        router_process.daemon = True
        router_process.start()

        time.sleep(3)
        wait_until_ok(f"http://{router_address}/health", process=router_process)
        self.router_address = router_address

        from loguru import logger

        logger.info(f"Launch SGLang Router at {self.router_address}")

        # Update TP0 workers with router address
        futures = []
        for worker_idx in range(self.num_workers):
            if worker_idx % self.rollout_per_tp_group == 0:
                futures.append(self.worker_handle[worker_idx].set_router.remote(self.router_address))
        ray.get(futures)

    def get_router_address(self):
        """Get the router address for external access."""
        return self.router_address

    async def run_dataloader(self):
        from loguru import logger

        total_epochs = self.config.trainer.total_epochs
        val_num_batch, val_batch_size = ray.get(self.data_coordinator.val_info.remote())
        dp_val_batch = (val_batch_size + self.dp_size - 1) // self.dp_size
        val_before_train = self.config.trainer.val_before_train
        for epoch in range(self.start_epoch, total_epochs):
            for batch_idx in range(self.num_train_batches):
                is_last_step = self.global_steps >= self.total_training_steps
                if epoch == self.start_epoch and batch_idx < (self.global_steps % self.num_train_batches):
                    continue
                await self.event.wait()
                self.event.clear()
                if val_before_train:
                    await self.validate(val_num_batch, dp_val_batch)
                    val_before_train = False
                self.global_steps += 1
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    await self.validate(val_num_batch, dp_val_batch)
                logger.info(f"Start Rollout Step {self.global_steps}")
                await self.data_coordinator.run_dataloader.remote(epoch)

    def next_rollout(self):
        self.event.set()
        return self.router_address

    async def validate(self, val_num_batch, val_batch_size):
        """
        Trigger validate rollout.
        """
        from loguru import logger

        for _ in range(val_num_batch):
            await self.data_coordinator.run_dataloader.remote(is_validate=True)
        logger.info("Starting validate rollout...")
        rollout_workers = self.get_rollout_worker_on_tp0()
        futures = [
            rollout_worker.validate.remote(val_batch_size * val_num_batch, self.global_steps)
            for rollout_worker in rollout_workers
        ]
        await asyncio.gather(*futures)
        val_metrics = await self.metric_worker.wait_final_res.remote()
        logger.info(f"Step-{self.global_steps} Validate Metrics: {val_metrics}")
        self.message_queue.append((val_metrics, self.global_steps))
        return

    async def get_metrics(self):
        """
        return metrics for train actor, it will be write to tracker
        """
        result = list(self.message_queue)
        self.message_queue.clear()
        return result

    def should_stop(self) -> bool:
        """
        Check if rollout should stop based on coordinator status.

        Returns:
            True if should stop, False otherwise
        """
        if self.coordinator:
            try:
                return ray.get(self.coordinator.should_stop.remote())
            except Exception as e:
                from loguru import logger

                logger.warning(f"[RolloutManager] Failed to check coordinator: {e}")
                logger.warning(f"[RolloutManager] Traceback:\n{traceback.format_exc()}")
                return False
        return False

    def report_failure(self, reason: str):
        """
        Report a failure to the coordinator.

        Args:
            reason: Description of the failure
        """
        from loguru import logger

        logger.error(f"[RolloutManager] Failure: {reason}")

        if self.coordinator:
            try:
                ray.get(self.coordinator.report_failure.remote("rollout_manager", reason))
            except Exception as e:
                logger.warning(f"[RolloutManager] Failed to report to coordinator: {e}")
                logger.warning(f"[RolloutManager] Traceback:\n{traceback.format_exc()}")

    def cleanup(self):
        """
        Clean up rollout resources: stop workers and router.

        Should be called when shutting down gracefully.
        """
        from loguru import logger

        logger.info("[RolloutManager] Starting cleanup...")

        # Stop all workers
        for i, worker in enumerate(self.worker_handle):
            try:
                ray.kill(worker)
                logger.debug(f"[RolloutManager] Killed worker {i}")
            except Exception as e:
                logger.warning(f"[RolloutManager] Failed to kill worker {i}: {e}")
                logger.warning(f"[RolloutManager] Traceback:\n{traceback.format_exc()}")

        self.worker_handle = []
        self.worker_urls = []

        logger.info("[RolloutManager] Cleanup completed")
