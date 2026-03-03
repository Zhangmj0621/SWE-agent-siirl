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

import os
import socket

import ray
from loguru import logger
from ray.actor import ActorHandle

from siirl.params.training_args import SiiRLArguments
from siirl.utils.enums import DistributedEnv
from siirl.worker.actor.trainer import Trainer
from siirl.worker.ray_utils import GPUResources


class TrainerGroup:
    """
    Manages a group of Trainers for distributed RL training.
    Each Trainer manages one actor, one ref, and optionally one critic (for PPO).
    Supports multiple algorithms (PPO, GRPO) and training backends (megatron, fsdp, etc.)

    Model requirements by algorithm:
    - PPO: actor + critic + ref (per Trainer)
    - GRPO: actor + ref (per Trainer)
    """

    def __init__(
        self,
        config: SiiRLArguments,
        gpu_resources: GPUResources,
        data_coordinator,
        rollout_manager=None,
        coordinator=None,
        metric_worker: ActorHandle | None = None,
    ) -> None:
        """
        Initialize TrainerGroup with configuration and resource handles.

        Args:
            config: Training configuration
            gpu_resources: GPUResources containing placement group and GPU indices for training
            data_coordinator: Ray handle to DataCoordinator
            rollout_manager: Ray handle to RolloutManager for weight synchronization
            coordinator: Ray handle to TaskCoordinator for lifecycle management
            metric_worker: Ray handle to MetricWorker for distributed metrics collection
        """
        self.config = config
        self.data_coordinator = data_coordinator
        self.rollout_manager = rollout_manager
        self.coordinator = coordinator
        self.metric_worker = metric_worker

        # GPU resources from allocate_resources()
        self.pg = gpu_resources.pg  # Ray placement group
        self.gpu_indices = gpu_resources.indices  # Allocated GPU bundle indices
        self.local_ranks = gpu_resources.local_ranks  # Local GPU IDs for each bundle
        self.node_ips = gpu_resources.node_ips  # Node IPs for each bundle
        self.num_gpus = gpu_resources.num_gpus  # Total GPUs for training
        self.is_shared = gpu_resources.is_shared  # Whether in colocated mode

        self.trainers: list[Trainer] = []

        self.use_critic = self.config.actor_ref.algorithm.adv_estimator == "ppo"

        self.master_addr, self.master_ports = self._resolve_master_endpoint()
        self._validate_distributed_setup()

    def set_rollout_manager(self, rollout_manager):
        """
        Set the rollout manager for weight synchronization.

        Args:
            rollout_manager: Ray handle to RolloutManager
        """
        self.rollout_manager = rollout_manager

    # ---- Distributed endpoint resolution & validation ----

    @staticmethod
    def _resolve_ip(addr: str) -> str | None:
        """Resolve an address (IP or hostname) to an IP string; returns None on failure."""
        try:
            socket.inet_aton(addr)
            return addr
        except OSError:
            try:
                return socket.gethostbyname(addr)
            except socket.gaierror:
                return None

    def _allocate_master_port_local(self, master_addr: str) -> str:
        """Find an available rendezvous port on the current process host."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = str(s.getsockname()[1])
        logger.info(f"Auto-allocated MASTER_PORT={port} for master_addr={master_addr} (single-node mode)")
        return port

    def _is_multi_node(self) -> bool:
        """Best-effort multi-node detection for rendezvous port policy."""
        cfg_nnodes = getattr(getattr(self.config, "trainer", None), "nnodes", None)
        if cfg_nnodes is not None:
            try:
                return int(cfg_nnodes) > 1
            except (TypeError, ValueError):
                pass
        return len(set(self.node_ips)) > 1

    def _resolve_master_endpoint(self) -> tuple[str, str]:
        """
        Derive the Torch rendezvous endpoint from the actual trainer topology.

        MASTER_ADDR is always set to self.node_ips[0] (the node hosting rank 0),
        regardless of what the environment variable says. This prevents the common
        multi-node failure where MASTER_ADDR points to a Ray head that hosts no
        trainer, causing all ranks to connect-timeout.

        MASTER_PORT priority: TRAIN_MASTER_PORT > MASTER_PORT > auto-allocate.
        """
        if not self.node_ips:
            raise RuntimeError("No actor node IPs available; cannot determine MASTER_ADDR")

        master_addr = self.node_ips[0]

        env_master_addr = os.getenv("MASTER_ADDR")
        if env_master_addr and env_master_addr != master_addr:
            resolved = self._resolve_ip(env_master_addr)
            if resolved != master_addr:
                logger.warning(
                    f"Overriding MASTER_ADDR: environment has '{env_master_addr}' "
                    f"but rank0 trainer is on '{master_addr}'. "
                    f"Using '{master_addr}' for Torch rendezvous."
                )

        master_port = os.getenv("TRAIN_MASTER_PORT") or os.getenv("MASTER_PORT")
        if master_port is None:
            if self._is_multi_node():
                raise RuntimeError(
                    "Missing TRAIN_MASTER_PORT/MASTER_PORT in multi-node setup. "
                    "Please set TRAIN_MASTER_PORT explicitly to a reachable fixed port."
                )
            master_port = self._allocate_master_port_local(master_addr)

        return master_addr, master_port

    def _validate_distributed_setup(self):
        """Fail-fast checks on topology consistency and endpoint validity."""
        n_indices = len(self.gpu_indices)
        n_local = len(self.local_ranks)
        n_ips = len(self.node_ips)
        if not (n_indices == n_local == n_ips == self.num_gpus):
            raise RuntimeError(
                f"Trainer distributed layout mismatch: "
                f"gpu_indices={n_indices}, local_ranks={n_local}, "
                f"node_ips={n_ips}, num_gpus={self.num_gpus}"
            )

        if len(set(self.gpu_indices)) != n_indices:
            raise RuntimeError(f"Duplicate GPU bundle indices: {self.gpu_indices}")

        try:
            port = int(self.master_ports)
        except (ValueError, TypeError) as err:
            raise RuntimeError(f"MASTER_PORT is not a valid integer: {self.master_ports}") from err
        if not (1024 <= port <= 65535):
            raise RuntimeError(f"MASTER_PORT out of range [1024, 65535]: {port}")

        if self._resolve_ip(self.master_addr) is None:
            raise RuntimeError(f"Invalid MASTER_ADDR '{self.master_addr}': cannot resolve to IP")

    def _build_trainer_env(self, rank: int, local_rank: int) -> dict[str, str]:
        """Build the runtime environment variables for a single trainer process."""
        env_vars = {
            DistributedEnv.WORLD_SIZE.value: str(self.num_gpus),
            DistributedEnv.RANK.value: str(rank),
            DistributedEnv.LOCAL_RANK.value: str(local_rank),
            DistributedEnv.MASTER_ADDR.value: self.master_addr,
            DistributedEnv.MASTER_PORT.value: self.master_ports,
            "DIST_INIT_METHOD": "env://",
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
        }

        gloo_ifname = os.getenv("GLOO_SOCKET_IFNAME")
        if gloo_ifname:
            env_vars["GLOO_SOCKET_IFNAME"] = gloo_ifname

        return env_vars

    # ---- Actor lifecycle ----

    def init_actors(self):
        """
        Initialize trainers by creating Trainer instances and wrapping them as Ray Actors.
        Each Trainer manages its own actor, ref, and optionally critic models.
        Uses GPU bundle indices and local_ranks from allocated GPUResources for precise GPU assignment.
        """
        logger.info(f"[TrainerGroup.init_actors] Creating {self.num_gpus} trainers")
        logger.info(f"  gpu_indices={self.gpu_indices}, local_ranks={self.local_ranks}")
        logger.info(f"  node_ips={self.node_ips}, is_shared={self.is_shared}")
        if not (len(self.gpu_indices) == len(self.local_ranks) == len(self.node_ips) == self.num_gpus):
            raise ValueError(
                "TrainerGroup resource shape mismatch: "
                f"indices={len(self.gpu_indices)}, local_ranks={len(self.local_ranks)}, "
                f"node_ips={len(self.node_ips)}, num_gpus={self.num_gpus}"
            )

        for rank, (bundle_idx, local_rank) in enumerate(zip(self.gpu_indices, self.local_ranks, strict=False)):
            env_vars = self._build_trainer_env(rank, local_rank)
            logger.info(
                f"  Creating Trainer rank={rank}: bundle_idx={bundle_idx}, local_rank={local_rank}, "
                f"node_ip={self.node_ips[rank]}, env={{WORLD_SIZE={self.num_gpus}, RANK={rank}, "
                f"LOCAL_RANK={local_rank}, MASTER_ADDR={self.master_addr}, MASTER_PORT={self.master_ports}}}"
            )

            TrainerActor = ray.remote(Trainer)

            trainer_options = {
                "runtime_env": {"env_vars": env_vars},
                "name": f"trainer_rank{rank}_bundle{bundle_idx}",
                "num_gpus": 1,
            }

            # Create trainer with placement group scheduling
            trainer_handle = TrainerActor.options(
                **trainer_options,
                placement_group=self.pg,
                placement_group_bundle_index=bundle_idx,
            ).remote(
                config=self.config,
                rank=rank,
                local_rank=local_rank,
                world_size=self.num_gpus,
                use_critic=self.use_critic,
                data_coordinator=self.data_coordinator,
                coordinator=self.coordinator,
                rollout_manager=self.rollout_manager,
                metric_worker=self.metric_worker,
            )

            self.trainers.append(trainer_handle)

        # Initialize models on all trainers
        futures = [trainer.init_models.remote() for trainer in self.trainers]
        ray.get(futures)

        # Set rollout workers and setup param sync if rollout_manager is available
        if self.rollout_manager is not None:
            futures = [trainer.setup_param_sync.remote() for trainer in self.trainers]
            ray.get(futures)

        logger.success(f"Successfully initialized {len(self.trainers)} trainers with their models")

    def load_checkpoint(self):
        """Load checkpoint for all trainers."""
        if not self.trainers:
            logger.warning("No trainers available for checkpoint loading")
            return

        logger.info("Loading checkpoints for all trainers")
        futures = [trainer.load_checkpoint.remote() for trainer in self.trainers]
        ray.get(futures)
        logger.success("Checkpoint loaded for all trainers")

    def train(self):
        """
        Execute training loop.

        MetricTracker is created inside each Trainer (only rank=0 creates one).
        """
        batch_size = self.config.data.train_batch_size * self.config.rollout.n
        futures = [trainer.train.remote(batch_size) for trainer in self.trainers]
        ray.get(futures)

    def put_weight(self):
        """
        Extract trained model parameters from actor and update to RolloutManager.

        In colocated mode, wraps sync with offload/resume to avoid OOM when
        both SGLang and trainer share the same GPU.
        """
        logger.info("Extracting model weights from actor workers")

        if not self.trainers:
            logger.warning("No trainers available for weight extraction")
            return

        if self.rollout_manager is None:
            logger.warning("RolloutManager not set, cannot update weights")
            return

        is_colocate = getattr(self.config.trainer, "colocate", False)
        timeout = max(1, int(getattr(self.config.trainer, "colocate_timeout_s", 60)))
        offloaded = False
        try:
            if is_colocate:
                ray.get(self.rollout_manager.offload_for_train.remote(timeout_s=timeout), timeout=timeout)
                offloaded = True
            futures = [trainer.update_rollout_weight.remote() for trainer in self.trainers]
            ray.get(futures)
        finally:
            if offloaded:
                ray.get(self.rollout_manager.resume_after_sync.remote(timeout_s=timeout), timeout=timeout)

        logger.info("Weight update completed")
