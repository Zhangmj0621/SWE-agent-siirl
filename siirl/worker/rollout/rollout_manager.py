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
import contextlib
import multiprocessing
import os
import re
import time
import traceback
from collections import defaultdict, deque

import ray

from siirl.params.training_args import SiiRLArguments
from siirl.utils.enums import DistributedEnv
from siirl.utils.net_utils.net import (
    SGLANG_DIST_INIT_START_PORT,
    SGLANG_HTTP_START_PORT,
    SGLANG_ROUTER_START_PORT,
    get_free_port,
    get_net_interface_ip,
)
from siirl.worker.ray_utils import GPUResources, RayClassWithInitArgs, get_random_string

VALIDATE_REUSE_SYNC_TIMEOUT_S = 120
VALIDATE_REUSE_SYNC_LOG_INTERVAL_S = 5.0
VALIDATE_PROGRESS_POLL_INTERVAL_S = 2.0
VALIDATE_REUSE_GRACEFUL_SHUTDOWN_TIMEOUT_S = 15
VALIDATE_REUSE_RECREATE_COOLDOWN_S = 3.0
VALIDATE_REUSE_PORT_STRIDE = 128
VALIDATE_REUSE_PORT_CYCLE = 10
VALIDATE_REUSE_PORT_RETRY_SLOTS = 3


def compute_validate_reuse_topology(train_gpus: int, tp_size: int, n_gpus_per_node: int) -> dict | None:
    if train_gpus <= 0 or tp_size <= 0 or n_gpus_per_node <= 0:
        return None

    gpus_per_rollout = min(tp_size, n_gpus_per_node)
    if tp_size % gpus_per_rollout != 0:
        return None

    num_tp_groups = train_gpus // tp_size
    if num_tp_groups == 0:
        return None

    usable_gpus = num_tp_groups * tp_size
    num_workers = usable_gpus // gpus_per_rollout
    rollout_per_tp_group = tp_size // gpus_per_rollout

    return {
        "usable_gpus": usable_gpus,
        "num_workers": num_workers,
        "num_tp_groups": num_tp_groups,
        "gpus_per_rollout": gpus_per_rollout,
        "rollout_per_tp_group": rollout_per_tp_group,
    }


def split_validate_samples(samples: list, num_workers: int) -> list[list]:
    if num_workers <= 0:
        return []
    shards = [[] for _ in range(num_workers)]
    for idx, sample in enumerate(samples):
        shards[idx % num_workers].append(sample)
    return shards


def split_validate_reuse_sync_workers(worker_infos: list[dict], trainer_node_ip: str, trainer_local_rank: int) -> tuple[list, list]:
    distributed_workers = []
    tensor_workers = []
    for info in worker_infos:
        worker = info["worker"]
        local_ranks = info.get("local_ranks") or []
        if info.get("node_ip") == trainer_node_ip and trainer_local_rank in local_ranks:
            tensor_workers.append(worker)
        else:
            distributed_workers.append(worker)
    return distributed_workers, tensor_workers


def rollout_to_train_step(rollout_step: int) -> int:
    return max(rollout_step - 1, 0)


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
        train_gpu_resources: GPUResources | None = None,
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
        self.train_gpu_resources = train_gpu_resources
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
        self.router_process = None
        self.worker_handle = []
        self.worker_urls = []
        self._validate_reuse_workers = []
        self._validate_reuse_tp0_workers = []
        self._validate_reuse_tp0_worker_infos = []
        self._validate_reuse_sync_required = False
        self._validate_reuse_synced_ranks: set[int] = set()
        self._validate_reuse_begin_ranks: set[int] = set()
        self._validate_reuse_sync_plan_logged_ranks: set[int] = set()
        self._validate_reuse_topology = None
        self._validate_reuse_phase = "IDLE"
        self._validate_reuse_abort_reason = ""
        self._validate_reuse_session_counter = 0
        self._validate_reuse_active_session_id: int | None = None
        self._validate_reuse_last_destroy_ts = 0.0
        self._validate_reuse_port_window_idx = -1
        self._validate_reuse_enabled = bool(
            getattr(config.trainer, "validate_reuse_train_gpus", False) and self.train_gpu_resources is not None
        )
        self._trainer_world_size = self.train_gpu_resources.num_gpus if self.train_gpu_resources is not None else 0

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
        self._val_time_acc = 0.0
        self._validate_progress_active = False
        self._validate_progress_total = 0
        self._validate_progress_done = 0
        self._validate_progress_workers_active = 0
        self._validate_progress_workers_total = 0
        self._validate_progress_step = 0
        self._validate_progress_last_update_ts = 0.0

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

    def _create_worker(
        self,
        rank,
        local_rank,
        bundle_idx,
        num_gpus,
        device_name,
        num_cpus: float | None = None,
        world_size: int | None = None,
        worker_prefix: str | None = None,
        rollout_ray_class: RayClassWithInitArgs | None = None,
    ):
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
            DistributedEnv.WORLD_SIZE.value: str(world_size if world_size is not None else self.num_workers),
            DistributedEnv.RANK.value: str(rank),
            DistributedEnv.LOCAL_RANK.value: str(local_rank),
            DistributedEnv.WG_PREFIX.value: worker_prefix or self.name_prefix,
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        }
        if os.getenv("GLOO_SOCKET_IFNAME"):
            env_vars["GLOO_SOCKET_IFNAME"] = os.getenv("GLOO_SOCKET_IFNAME")

        # Generate unique actor name
        target_rollout_ray_class = rollout_ray_class or self.rollout_ray_class
        base_class_repr = type(target_rollout_ray_class.cls).__name__
        match = re.search(r"ActorClass\(([^)]+)\)", base_class_repr)
        actor_class_name = match.group(1) if match else base_class_repr
        actor_name = f"{worker_prefix or self.name_prefix}_{actor_class_name}_actor{rank}_bundle{bundle_idx}"

        actor_options = {
            "runtime_env": {"env_vars": env_vars},
            "name": actor_name,
        }
        if num_cpus is not None:
            actor_options["num_cpus"] = num_cpus
        target_rollout_ray_class.update_options(actor_options)

        from loguru import logger

        logger.debug(f"Creating actor '{actor_name}'")

        worker = target_rollout_ray_class(
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

            # Handle dist_init_addr for cross-node TP (use port range 20000+)
            if nnodes > 1:
                if node_rank == 0:
                    # First actor in TP group: generate and cache dist_init_addr
                    dist_init_start_port = SGLANG_DIST_INIT_START_PORT + tp_group_idx  # Unique start port per TP group
                    dist_init_addr = ray.get(self.worker_handle[worker_idx].get_ip_port.remote(start_port=dist_init_start_port))
                    self._dist_init_addrs[tp_group_idx] = dist_init_addr
                    logger.info(f"TP Group {tp_group_idx}: Cross-node TP with {nnodes} nodes, " f"dist_init_addr={dist_init_addr}")
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

        Port allocation is done inside each worker with socket holding to prevent
        port races. worker_urls are collected after init_engine completes.
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

        # Phase 1: Allocate ports by node (avoids race conditions)
        # Group workers by node IP and cache IPs to avoid redundant remote calls
        worker_ips = {}  # worker_idx -> ip
        node_workers = defaultdict(list)
        for cfg in engine_configs:
            worker = self.worker_handle[cfg["worker_idx"]]
            ip = ray.get(worker.get_ip.remote())
            worker_ips[cfg["worker_idx"]] = ip
            node_workers[ip].append((worker, cfg))

        # Allocate ports per node using first worker on each node
        worker_ports = {}  # worker_idx -> port
        for _, workers_on_node in node_workers.items():
            first_worker = workers_on_node[0][0]
            num_ports = len(workers_on_node)
            ports = ray.get(first_worker.allocate_ports.remote(start_port=SGLANG_HTTP_START_PORT, count=num_ports))
            for i, (_, cfg) in enumerate(workers_on_node):
                worker_ports[cfg["worker_idx"]] = ports[i]

        # Initialize engines with allocated ports
        init_futures = []
        worker_info = []
        for cfg in engine_configs:
            worker = self.worker_handle[cfg["worker_idx"]]
            ip = worker_ips[cfg["worker_idx"]]
            port = worker_ports[cfg["worker_idx"]]

            future = worker.init_engine.remote(
                rank=cfg["worker_idx"],
                dist_init_addr=cfg["dist_init_addr"],
                ip=ip,
                port=port,
                base_gpu_id=cfg["base_gpu_id"],
                node_rank=cfg["node_rank"],
                nnodes=cfg["nnodes"],
            )
            init_futures.append(future)
            worker_info.append({"worker": worker, "cfg": cfg, "ip": ip, "port": port})

        ray.get(init_futures)

        # Phase 2: Launch servers (may retry with new ports on conflict)
        launch_futures = []
        for info in worker_info:
            launch_futures.append(info["worker"].launch_server.remote())
        ray.get(launch_futures)

        # Phase 3: Collect ACTUAL worker URLs (after possible port retries)
        self.worker_urls = []
        for info in worker_info:
            if info["cfg"]["is_tp0"]:
                actual_port = ray.get(info["worker"].get_port.remote())
                self.worker_urls.append(f"http://{info['ip']}:{actual_port}")

        logger.info(
            f"Initialized {self.num_workers} SGLang processes "
            f"({self.num_tp_groups} TP groups, {len(self.worker_urls)} router endpoints)"
        )

    def set_step(self, step: int):
        from loguru import logger

        step = max(0, int(step))
        self.global_steps = step

        if self.num_train_batches > 0:
            self.start_epoch = min(step // self.num_train_batches, self.config.trainer.total_epochs)
            self.batches_to_skip = step % self.num_train_batches
        else:
            self.start_epoch = 0
            self.batches_to_skip = 0

        if self.total_training_steps > 0 and self.global_steps >= self.total_training_steps:
            self.start_epoch = self.config.trainer.total_epochs
            self.batches_to_skip = 0

        logger.info(
            "[RolloutManager] Resume cursor updated "
            f"global_steps={self.global_steps} start_epoch={self.start_epoch} batches_to_skip={self.batches_to_skip}"
        )

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

    def get_validate_reuse_sync_workers(self, trainer_rank: int):
        if not self._validate_reuse_sync_required:
            return []
        return self._validate_reuse_tp0_workers

    def _reset_validate_reuse_sync_state(self):
        self._validate_reuse_sync_required = False
        self._validate_reuse_phase = "IDLE"
        self._validate_reuse_abort_reason = ""
        self._validate_reuse_active_session_id = None
        self._validate_reuse_begin_ranks.clear()
        self._validate_reuse_synced_ranks.clear()
        self._validate_reuse_sync_plan_logged_ranks.clear()

    def get_validate_reuse_sync_plan(self, trainer_rank: int):
        if not self._validate_reuse_sync_required:
            return {"distributed_workers": [], "tensor_workers": [], "phase": "IDLE"}
        if self.train_gpu_resources is None or trainer_rank < 0 or trainer_rank >= self.train_gpu_resources.num_gpus:
            return {
                "distributed_workers": self._validate_reuse_tp0_workers,
                "tensor_workers": [],
                "phase": self._validate_reuse_phase,
            }

        from loguru import logger

        trainer_node_ip = self.train_gpu_resources.node_ips[trainer_rank]
        trainer_local_rank = self.train_gpu_resources.local_ranks[trainer_rank]
        distributed_workers, tensor_workers = split_validate_reuse_sync_workers(
            self._validate_reuse_tp0_worker_infos,
            trainer_node_ip,
            trainer_local_rank,
        )
        if trainer_rank not in self._validate_reuse_sync_plan_logged_ranks:
            self._validate_reuse_sync_plan_logged_ranks.add(trainer_rank)
            logger.info(
                "[RolloutManager] Validate reuse sync plan "
                f"trainer_rank={trainer_rank} trainer_node={trainer_node_ip} trainer_local_rank={trainer_local_rank} "
                f"distributed_workers={len(distributed_workers)} tensor_workers={len(tensor_workers)} "
                f"phase={self._validate_reuse_phase}"
            )
        return {
            "distributed_workers": distributed_workers,
            "tensor_workers": tensor_workers,
            "phase": self._validate_reuse_phase,
        }

    def start_validate_reuse_sync_session(self) -> int:
        if not self._validate_reuse_sync_required:
            return -1
        if self._validate_reuse_active_session_id is None:
            self._validate_reuse_session_counter += 1
            self._validate_reuse_active_session_id = self._validate_reuse_session_counter
            self._validate_reuse_phase = "COLLECTING"
            self._validate_reuse_abort_reason = ""
            self._validate_reuse_begin_ranks.clear()
        return self._validate_reuse_active_session_id

    def mark_validate_reuse_begin(self, trainer_rank: int, session_id: int):
        if not self._validate_reuse_sync_required:
            return {"accepted": False, "reason": "sync_not_required", "phase": self._validate_reuse_phase}
        if session_id != self._validate_reuse_active_session_id:
            return {"accepted": False, "reason": "stale_session", "phase": self._validate_reuse_phase}
        if self._validate_reuse_phase == "ABORTED":
            return {"accepted": False, "reason": self._validate_reuse_abort_reason, "phase": self._validate_reuse_phase}

        self._validate_reuse_begin_ranks.add(trainer_rank)
        if len(self._validate_reuse_begin_ranks) >= self._trainer_world_size > 0:
            self._validate_reuse_phase = "RUNNING"
        return {
            "accepted": True,
            "phase": self._validate_reuse_phase,
            "begun": len(self._validate_reuse_begin_ranks),
            "world_size": self._trainer_world_size,
        }

    def get_validate_reuse_sync_gate(self, session_id: int):
        if not self._validate_reuse_sync_required:
            return {"proceed": False, "phase": "IDLE", "reason": "sync_not_required"}
        if session_id != self._validate_reuse_active_session_id:
            return {"proceed": False, "phase": self._validate_reuse_phase, "reason": "stale_session"}
        if self._validate_reuse_phase == "ABORTED":
            return {
                "proceed": False,
                "phase": self._validate_reuse_phase,
                "reason": self._validate_reuse_abort_reason or "aborted",
            }
        if len(self._validate_reuse_begin_ranks) >= self._trainer_world_size > 0:
            self._validate_reuse_phase = "RUNNING"
            return {"proceed": True, "phase": self._validate_reuse_phase}
        return {"proceed": None, "phase": self._validate_reuse_phase}

    def abort_validate_reuse_sync_session(self, session_id: int, reason: str):
        if not self._validate_reuse_sync_required:
            return {"aborted": False, "phase": "IDLE"}
        if session_id != self._validate_reuse_active_session_id:
            return {"aborted": False, "phase": self._validate_reuse_phase, "reason": "stale_session"}
        self._validate_reuse_phase = "ABORTED"
        self._validate_reuse_abort_reason = reason
        return {"aborted": True, "phase": self._validate_reuse_phase, "reason": reason}

    def mark_validate_reuse_synced(self, trainer_rank: int, session_id: int | None = None):
        if not self._validate_reuse_sync_required:
            return {"accepted": False, "reason": "sync_not_required"}
        if session_id is not None and session_id != self._validate_reuse_active_session_id:
            return {"accepted": False, "reason": "stale_session"}
        if self._validate_reuse_phase == "ABORTED":
            return {"accepted": False, "reason": self._validate_reuse_abort_reason}

        from loguru import logger

        self._validate_reuse_synced_ranks.add(trainer_rank)
        synced = len(self._validate_reuse_synced_ranks)
        missing = sorted(set(range(self._trainer_world_size)) - self._validate_reuse_synced_ranks)
        logger.info(
            "[RolloutManager] Validate reuse synced "
            f"trainer_rank={trainer_rank} synced={synced}/{self._trainer_world_size} missing={missing}"
        )
        if synced >= self._trainer_world_size > 0:
            self._validate_reuse_phase = "DONE"
            self._validate_reuse_sync_required = False
        return {"accepted": True, "phase": self._validate_reuse_phase, "synced": synced}

    async def _wait_validate_reuse_synced(self, timeout_s: int = VALIDATE_REUSE_SYNC_TIMEOUT_S) -> bool:
        if not self._validate_reuse_sync_required:
            return True
        if self._trainer_world_size <= 0:
            return False

        from loguru import logger

        start = time.time()
        next_log_at = start + VALIDATE_REUSE_SYNC_LOG_INTERVAL_S
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._validate_reuse_phase == "ABORTED":
                logger.warning("[RolloutManager] Validate reuse sync aborted " f"reason={self._validate_reuse_abort_reason or 'unknown'}")
                return False
            if len(self._validate_reuse_synced_ranks) >= self._trainer_world_size:
                logger.info(
                    "[RolloutManager] Validate reuse sync completed "
                    f"in {time.time() - start:.2f}s synced={len(self._validate_reuse_synced_ranks)}/{self._trainer_world_size}"
                )
                return True
            now = time.time()
            if now >= next_log_at:
                missing = sorted(set(range(self._trainer_world_size)) - self._validate_reuse_synced_ranks)
                logger.info(
                    "[RolloutManager] Waiting validate reuse sync "
                    f"elapsed={now - start:.2f}s synced={len(self._validate_reuse_synced_ranks)}/{self._trainer_world_size} "
                    f"missing={missing}"
                )
                next_log_at = now + VALIDATE_REUSE_SYNC_LOG_INTERVAL_S
            await asyncio.sleep(0.05)
        return False

    def _destroy_validate_reuse_pool(self):
        if not self._validate_reuse_workers:
            self._validate_reuse_tp0_workers = []
            self._validate_reuse_tp0_worker_infos = []
            self._validate_reuse_topology = None
            self._validate_reuse_sync_plan_logged_ranks.clear()
            return

        from loguru import logger

        shutdown_timeout_s = max(1, int(VALIDATE_REUSE_GRACEFUL_SHUTDOWN_TIMEOUT_S))
        shutdown_futures = [w.shutdown_engine.remote() for w in self._validate_reuse_workers]
        try:
            ray.get(shutdown_futures, timeout=shutdown_timeout_s)
        except Exception as e:
            logger.warning(f"[RolloutManager] Validate reuse engine shutdown failed: {e}")

        for i, worker in enumerate(self._validate_reuse_workers):
            with contextlib.suppress(Exception):
                ray.kill(worker)
                logger.debug(f"[RolloutManager] Killed validate reuse worker {i}")

        self._validate_reuse_workers = []
        self._validate_reuse_tp0_workers = []
        self._validate_reuse_tp0_worker_infos = []
        self._validate_reuse_topology = None
        self._validate_reuse_sync_plan_logged_ranks.clear()
        self._validate_reuse_last_destroy_ts = time.time()

    def _next_validate_reuse_port_bases(self) -> tuple[int, int]:
        stride = max(1, int(VALIDATE_REUSE_PORT_STRIDE))
        cycle = max(1, int(VALIDATE_REUSE_PORT_CYCLE))
        self._validate_reuse_port_window_idx = (self._validate_reuse_port_window_idx + 1) % cycle
        offset = self._validate_reuse_port_window_idx * stride
        http_port_base = SGLANG_HTTP_START_PORT + 2000 + offset
        dist_port_base = SGLANG_DIST_INIT_START_PORT + 1000 + offset
        return http_port_base, dist_port_base

    def _init_validate_reuse_pool(self) -> list:
        if not self._validate_reuse_enabled or self.train_gpu_resources is None:
            return []

        from loguru import logger

        from siirl.worker.rollout.rollout_worker import RolloutWorker

        cooldown_s = max(0.0, float(VALIDATE_REUSE_RECREATE_COOLDOWN_S))
        if self._validate_reuse_last_destroy_ts > 0 and cooldown_s > 0:
            elapsed = time.time() - self._validate_reuse_last_destroy_ts
            if elapsed < cooldown_s:
                sleep_s = cooldown_s - elapsed
                logger.info(f"[RolloutManager] Waiting {sleep_s:.2f}s before recreating validate reuse workers")
                time.sleep(sleep_s)

        http_port_base, dist_port_base = self._next_validate_reuse_port_bases()
        retry_slots = max(1, int(VALIDATE_REUSE_PORT_RETRY_SLOTS))
        logger.info(
            "[RolloutManager] Validate reuse port window "
            f"window_idx={self._validate_reuse_port_window_idx} http_port_base={http_port_base} "
            f"dist_port_base={dist_port_base} retry_slots={retry_slots}"
        )

        train_pairs = list(zip(self.train_gpu_resources.indices, self.train_gpu_resources.local_ranks, strict=False))

        topology = compute_validate_reuse_topology(
            len(train_pairs),
            self.tp_size,
            self.n_gpus_per_node,
        )
        if topology is None:
            logger.info("[RolloutManager] Validate GPU reuse disabled: no full TP group available on training GPUs")
            return []

        self._validate_reuse_topology = topology
        use_gpus = topology["usable_gpus"]
        num_workers = topology["num_workers"]
        num_tp_groups = topology["num_tp_groups"]
        gpus_per_rollout = topology["gpus_per_rollout"]
        rollout_per_tp_group = topology["rollout_per_tp_group"]

        selected_pairs = train_pairs[:use_gpus]
        indices = [idx for idx, _ in selected_pairs]
        local_ranks = [lr for _, lr in selected_pairs]

        reuse_prefix = f"{self.name_prefix}_valreuse"
        reuse_ray_class = RayClassWithInitArgs(ray.remote(RolloutWorker), self.config, num_tp_groups, self.metric_worker)

        workers = []
        for worker_idx in range(num_workers):
            first_gpu_idx = worker_idx * gpus_per_rollout
            bundle_idx = indices[first_gpu_idx]
            local_rank = local_ranks[first_gpu_idx]
            worker = self._create_worker(
                rank=worker_idx,
                local_rank=local_rank,
                bundle_idx=bundle_idx,
                num_gpus=0,
                device_name=self.device_name,
                num_cpus=0,
                world_size=num_workers,
                worker_prefix=reuse_prefix,
                rollout_ray_class=reuse_ray_class,
            )
            workers.append(worker)

        dist_init_addrs = {}
        configs = []
        for worker_idx in range(num_workers):
            tp_group_idx = worker_idx // rollout_per_tp_group
            node_rank = worker_idx % rollout_per_tp_group
            first_gpu_idx = worker_idx * gpus_per_rollout
            base_gpu_id = local_ranks[first_gpu_idx]
            worker_local_ranks = local_ranks[first_gpu_idx : first_gpu_idx + gpus_per_rollout]
            if rollout_per_tp_group > 1:
                if node_rank == 0:
                    dist_init_start_port = dist_port_base + tp_group_idx
                    dist_init_addr = ray.get(workers[worker_idx].get_ip_port.remote(start_port=dist_init_start_port))
                    dist_init_addrs[tp_group_idx] = dist_init_addr
                else:
                    dist_init_addr = dist_init_addrs[tp_group_idx]
            else:
                dist_init_addr = None
            configs.append(
                {
                    "worker_idx": worker_idx,
                    "base_gpu_id": base_gpu_id,
                    "node_rank": node_rank,
                    "nnodes": rollout_per_tp_group,
                    "dist_init_addr": dist_init_addr,
                    "is_tp0": node_rank == 0,
                    "worker_local_ranks": worker_local_ranks,
                }
            )

        worker_ips = {}
        node_workers = defaultdict(list)
        for cfg in configs:
            worker = workers[cfg["worker_idx"]]
            ip = ray.get(worker.get_ip.remote())
            worker_ips[cfg["worker_idx"]] = ip
            node_workers[ip].append((worker, cfg))

        worker_reserved_ports = {}
        for _, workers_on_node in node_workers.items():
            first_worker = workers_on_node[0][0]
            worker_count = len(workers_on_node)
            ports = ray.get(first_worker.allocate_ports.remote(start_port=http_port_base, count=worker_count * retry_slots))
            for i, (_, cfg) in enumerate(workers_on_node):
                worker_reserved_ports[cfg["worker_idx"]] = [ports[i + j * worker_count] for j in range(retry_slots)]

        init_futures = []
        for cfg in configs:
            worker = workers[cfg["worker_idx"]]
            reserved_ports = worker_reserved_ports[cfg["worker_idx"]]
            init_futures.append(
                worker.init_engine.remote(
                    rank=cfg["worker_idx"],
                    dist_init_addr=cfg["dist_init_addr"],
                    ip=worker_ips[cfg["worker_idx"]],
                    port=reserved_ports[0],
                    base_gpu_id=cfg["base_gpu_id"],
                    node_rank=cfg["node_rank"],
                    nnodes=cfg["nnodes"],
                )
            )
        ray.get(init_futures)
        launch_futures = []
        for cfg in configs:
            worker = workers[cfg["worker_idx"]]
            reserved_ports = worker_reserved_ports[cfg["worker_idx"]]
            launch_futures.append(worker.launch_server.remote(max_retries=len(reserved_ports), reserved_ports=reserved_ports))
        ray.get(launch_futures)

        tp0_workers = []
        tp0_infos = []
        for cfg in configs:
            if cfg["is_tp0"]:
                worker = workers[cfg["worker_idx"]]
                tp0_workers.append(worker)
                tp0_infos.append(
                    {
                        "worker": worker,
                        "node_ip": worker_ips[cfg["worker_idx"]],
                        "local_ranks": cfg["worker_local_ranks"],
                    }
                )
        ray.get([w.init_validate_executor.remote(self.data_coordinator, num_tp_groups) for w in tp0_workers])

        logger.info(f"[RolloutManager] Validate GPU reuse enabled: {use_gpus} train GPUs, " f"{len(tp0_workers)} extra TP0 workers")
        self._validate_reuse_workers = workers
        self._validate_reuse_tp0_workers = tp0_workers
        self._validate_reuse_tp0_worker_infos = tp0_infos
        return tp0_workers

    def start_rollout(self):
        """
        Start the rollout process on TP0 RolloutWorker actors.
        Only TP0 actors run the executor; other actors only participate in TP communication.
        """
        futures = []
        for worker_idx in range(self.num_workers):
            if worker_idx % self.rollout_per_tp_group == 0:
                futures.append(
                    self.worker_handle[worker_idx].start_rollout.remote(self.router_address, self.data_coordinator, self.dp_size)
                )
        ray.get(futures)

    def start_router(self, request_timeout: int = 3600):
        """
        Start SGLang router process and configure it with worker URLs.
        """
        from sglang_router.launch_router import RouterArgs, launch_router

        from siirl.engine.rollout.sglang_engine import wait_until_ok

        router_ip = self.config.rollout.router_ip or get_net_interface_ip()
        # Use sequential port allocation starting from 25000 for router
        router_port = self.config.rollout.router_port or get_free_port(router_ip, start_port=SGLANG_ROUTER_START_PORT)
        router_address = f"{router_ip}:{router_port}"

        router_args = RouterArgs(
            host=router_ip,
            port=router_port,
            worker_urls=self.worker_urls,
            balance_abs_threshold=0,
            log_level="warn",
            request_timeout_secs=3600,
        )

        self.router_process = multiprocessing.Process(target=launch_router, args=(router_args,))
        self.router_process.daemon = True
        self.router_process.start()

        time.sleep(3)
        wait_until_ok(f"http://{router_address}/health", process=self.router_process)
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
                if self.total_training_steps > 0 and self.global_steps >= self.total_training_steps:
                    logger.info(
                        "[RolloutManager] Reached total training steps, stop dataloader loop "
                        f"global_steps={self.global_steps} total_training_steps={self.total_training_steps}"
                    )
                    self.report_completed()
                    return
                if epoch == self.start_epoch and batch_idx < self.batches_to_skip:
                    continue
                await self.event.wait()
                self.event.clear()
                if val_before_train:
                    await self.validate(val_num_batch, dp_val_batch)
                    val_before_train = False
                next_step = self.global_steps + 1
                is_last_step = self.total_training_steps > 0 and next_step >= self.total_training_steps
                has_batch = await self.data_coordinator.run_dataloader.remote(epoch)
                if not has_batch:
                    reason = (
                        "[RolloutManager] Dataloader exhausted before expected training completion "
                        f"epoch={epoch} batch_idx={batch_idx} global_steps={self.global_steps} "
                        f"total_training_steps={self.total_training_steps}"
                    )
                    logger.warning(reason)
                    if self.total_training_steps > 0 and self.global_steps >= self.total_training_steps:
                        self.report_completed()
                    else:
                        self.report_failure(reason)
                    return
                self.global_steps = next_step
                if self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    await self.validate(val_num_batch, dp_val_batch)
                logger.info(f"Start Rollout Step {rollout_to_train_step(self.global_steps)} " f"(rollout_index={self.global_steps})")

    def next_rollout(self):
        self.event.set()
        return self.router_address

    def _set_validate_progress_state(self, *, active: bool, total: int, done: int, workers_active: int, workers_total: int):
        self._validate_progress_active = active
        self._validate_progress_total = max(0, int(total))
        self._validate_progress_done = max(0, int(done))
        self._validate_progress_workers_active = max(0, int(workers_active))
        self._validate_progress_workers_total = max(0, int(workers_total))
        self._validate_progress_step = int(self.global_steps)
        self._validate_progress_last_update_ts = time.time()

    def get_validate_progress_snapshot(self):
        return {
            "active": self._validate_progress_active,
            "total": self._validate_progress_total,
            "done": self._validate_progress_done,
            "workers_active": self._validate_progress_workers_active,
            "workers_total": self._validate_progress_workers_total,
            "step": self._validate_progress_step,
            "last_update_ts": self._validate_progress_last_update_ts,
        }

    async def _monitor_validate_progress(self, assigned_workers: list[tuple], total_samples: int):
        total_samples = max(0, int(total_samples))
        worker_total = len(assigned_workers)
        if total_samples <= 0 or worker_total <= 0:
            self._set_validate_progress_state(
                active=False,
                total=total_samples,
                done=total_samples,
                workers_active=0,
                workers_total=worker_total,
            )
            return

        self._set_validate_progress_state(
            active=True,
            total=total_samples,
            done=0,
            workers_active=worker_total,
            workers_total=worker_total,
        )

        try:
            while True:
                progress_refs = [worker.get_validate_progress.remote() for worker, _ in assigned_workers]
                progresses = await asyncio.gather(*progress_refs)

                done_samples = 0
                workers_active = 0
                for progress, (_, expected_total) in zip(progresses, assigned_workers, strict=False):
                    done = int(progress.get("done", 0))
                    done_samples += min(max(done, 0), expected_total)
                    if progress.get("active", False):
                        workers_active += 1

                done_samples = min(done_samples, total_samples)
                self._set_validate_progress_state(
                    active=done_samples < total_samples,
                    total=total_samples,
                    done=done_samples,
                    workers_active=workers_active,
                    workers_total=worker_total,
                )

                if done_samples >= total_samples:
                    return
                await asyncio.sleep(VALIDATE_PROGRESS_POLL_INTERVAL_S)
        finally:
            self._set_validate_progress_state(
                active=False,
                total=total_samples,
                done=total_samples,
                workers_active=0,
                workers_total=worker_total,
            )

    async def validate(self, val_num_batch, val_batch_size):
        """
        Trigger validate rollout.
        """
        from loguru import logger

        from siirl.worker.rollout.validator import aggregate_and_log_validation_metrics

        val_start = time.time()
        self._set_validate_progress_state(active=False, total=0, done=0, workers_active=0, workers_total=0)
        try:
            for _ in range(val_num_batch):
                has_val_batch = await self.data_coordinator.run_dataloader.remote(is_validate=True)
                if not has_val_batch:
                    logger.warning("[RolloutManager] Validation dataloader exhausted before filling expected batches")
                    break

            if self._validate_reuse_enabled:
                try:
                    self._init_validate_reuse_pool()
                except Exception as e:
                    logger.warning(f"[RolloutManager] Validate GPU reuse init failed, fallback to rollout-only: {e}")
                    self._destroy_validate_reuse_pool()

            if self._validate_reuse_tp0_workers:
                self._validate_reuse_sync_required = True
                self._validate_reuse_phase = "IDLE"
                self._validate_reuse_abort_reason = ""
                self._validate_reuse_active_session_id = None
                self._validate_reuse_begin_ranks.clear()
                self._validate_reuse_synced_ranks.clear()
                self._validate_reuse_sync_plan_logged_ranks.clear()
                logger.info(
                    "[RolloutManager] Start validate reuse sync wait "
                    f"trainer_world_size={self._trainer_world_size} reuse_tp0_workers={len(self._validate_reuse_tp0_workers)}"
                )
                synced = await self._wait_validate_reuse_synced()
                if not synced:
                    missing = sorted(set(range(self._trainer_world_size)) - self._validate_reuse_synced_ranks)
                    logger.warning(
                        "[RolloutManager] Validate GPU reuse sync timeout, fallback to rollout-only "
                        f"synced={len(self._validate_reuse_synced_ranks)}/{self._trainer_world_size} missing={missing}"
                    )
                    self._reset_validate_reuse_sync_state()
                    self._destroy_validate_reuse_pool()

            logger.info("Starting validate rollout...")
            validate_workers = self.get_rollout_worker_on_tp0() + self._validate_reuse_tp0_workers
            all_val_samples = []
            drain_batch_size = max(val_batch_size * val_num_batch, len(validate_workers))
            while True:
                val_batch = await self.data_coordinator.get_dataloader.remote(batch_size=drain_batch_size, is_validate=True)
                if not val_batch:
                    break
                all_val_samples.extend(val_batch)

            shards = split_validate_samples(all_val_samples, len(validate_workers))
            logger.info(
                f"Validate dispatch: workers={len(validate_workers)}, "
                f"samples={len(all_val_samples)}, shard_sizes={[len(shard) for shard in shards]}"
            )
            assigned_workers = [(worker, len(shards[idx])) for idx, worker in enumerate(validate_workers) if shards[idx]]
            futures = [
                worker.validate_assigned.remote(shards[idx], self.global_steps)
                for idx, worker in enumerate(validate_workers)
                if shards[idx]
            ]
            progress_task = None
            if assigned_workers:
                progress_task = asyncio.create_task(self._monitor_validate_progress(assigned_workers, len(all_val_samples)))
            try:
                results = await asyncio.gather(*futures) if futures else []
            finally:
                if progress_task is not None:
                    if not progress_task.done():
                        progress_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await progress_task
            all_samples = []
            for samples, _ in results:
                all_samples.extend(samples)
            raw_val_metrics = aggregate_and_log_validation_metrics(all_samples)
            val_metrics = raw_val_metrics
            if self.metric_worker is not None and raw_val_metrics:
                await self.metric_worker.submit_metric.remote(raw_val_metrics, 1)
                val_metrics = await self.metric_worker.wait_final_res.remote()
            if val_metrics:
                from siirl.utils.metrics import restore_weighted_metrics

                val_metrics = restore_weighted_metrics(val_metrics)
            logger.info(f"Step-{self.global_steps} Validate Metrics: {val_metrics}")
            self.message_queue.append((val_metrics, self.global_steps))
        finally:
            self._reset_validate_reuse_sync_state()
            self._destroy_validate_reuse_pool()
            self._val_time_acc += time.time() - val_start
        return

    async def get_metrics(self):
        """
        return metrics for train actor, it will be write to tracker
        """
        result = list(self.message_queue)
        self.message_queue.clear()
        return result

    def pop_validation_time(self) -> float:
        """
        Return and reset accumulated validation time.
        Used by trainer to exclude validation from train throughput metrics.
        """
        val_time = self._val_time_acc
        self._val_time_acc = 0.0
        return val_time

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

    def report_completed(self):
        from loguru import logger

        logger.info("[RolloutManager] Reporting task completion")
        if self.coordinator:
            try:
                ray.get(self.coordinator.report_completed.remote("rollout_manager"))
            except Exception as e:
                logger.warning(f"[RolloutManager] Failed to report completion: {e}")
                logger.warning(f"[RolloutManager] Traceback:\n{traceback.format_exc()}")

    def cleanup(self):
        """
        Clean up rollout resources: stop workers and router.

        Should be called when shutting down gracefully.
        """
        from loguru import logger

        logger.info("[RolloutManager] Starting cleanup...")

        self._destroy_validate_reuse_pool()

        # Gracefully shutdown engine processes first
        shutdown_futures = [w.shutdown_engine.remote() for w in self.worker_handle]
        try:
            ray.get(shutdown_futures, timeout=30)
        except Exception as e:
            logger.warning(f"[RolloutManager] Engine shutdown timed out or failed: {e}")

        # Then kill Ray actors
        for i, worker in enumerate(self.worker_handle):
            try:
                ray.kill(worker)
                logger.debug(f"[RolloutManager] Killed worker {i}")
            except Exception as e:
                logger.warning(f"[RolloutManager] Failed to kill worker {i}: {e}")
                logger.warning(f"[RolloutManager] Traceback:\n{traceback.format_exc()}")

        # Terminate router process
        if self.router_process and self.router_process.is_alive():
            self.router_process.terminate()
            self.router_process.join(timeout=5)
            if self.router_process.is_alive():
                self.router_process.kill()
                self.router_process.join(timeout=3)
            logger.info("[RolloutManager] Router process terminated")

        self.worker_handle = []
        self.worker_urls = []
        self.router_process = None

        logger.info("[RolloutManager] Cleanup completed")
