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

import ray
import os
from loguru import logger
from typing import List, Optional

from siirl.params.training_args import SiiRLArguments
from siirl.utils.enums import DistributedEnv
from siirl.engine.actor.utils import get_master_info
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
    ) -> None:
        """
        Initialize TrainerGroup with configuration and resource handles.

        Args:
            config: Training configuration
            gpu_resources: GPUResources containing placement group and GPU indices for training
            data_coordinator: Ray handle to DataCoordinator
            rollout_manager: Ray handle to RolloutManager for weight synchronization
        """
        self.config = config
        self.data_coordinator = data_coordinator
        self.rollout_manager = rollout_manager

        # GPU resources from allocate_resources()
        self.pg = gpu_resources.pg  # Ray placement group
        self.gpu_indices = gpu_resources.indices  # Allocated GPU bundle indices
        self.local_ranks = gpu_resources.local_ranks  # Local GPU IDs for each bundle
        self.node_ips = gpu_resources.node_ips  # Node IPs for each bundle
        self.num_gpus = gpu_resources.num_gpus  # Total GPUs for training
        self.is_shared = gpu_resources.is_shared  # Whether in colocated mode

        self.trainers: List[Trainer] = []

        self.use_critic = self.config.actor_ref.algorithm.adv_estimator == "ppo"

        # TODO: add robust port access
        self.master_addr, self.master_ports = get_master_info()

    def set_rollout_manager(self, rollout_manager):
        """
        Set the rollout manager for weight synchronization.
        
        Args:
            rollout_manager: Ray handle to RolloutManager
        """
        self.rollout_manager = rollout_manager

    def init_actors(self):
        """
        Initialize trainers by creating Trainer instances and wrapping them as Ray Actors.
        Each Trainer manages its own actor, ref, and optionally critic models.
        Uses GPU bundle indices and local_ranks from allocated GPUResources for precise GPU assignment.
        """
        logger.info(f"[TrainerGroup.init_actors] Creating {self.num_gpus} trainers")
        logger.info(f"  gpu_indices={self.gpu_indices}, local_ranks={self.local_ranks}")
        logger.info(f"  node_ips={self.node_ips}, is_shared={self.is_shared}")
        
        # Iterate over allocated GPU bundle indices and their local ranks
        for rank, (bundle_idx, local_rank) in enumerate(zip(self.gpu_indices, self.local_ranks)):
            env_vars = {
                DistributedEnv.WORLD_SIZE.value: str(self.num_gpus),
                DistributedEnv.RANK.value: str(rank),
                DistributedEnv.LOCAL_RANK.value: str(local_rank),
                DistributedEnv.MASTER_ADDR.value: self.master_addr,
                DistributedEnv.MASTER_PORT.value: self.master_ports,
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            }
            logger.debug(f"  Creating Trainer rank={rank}: bundle_idx={bundle_idx}, local_rank={local_rank}, "
                        f"env={{WORLD_SIZE={self.num_gpus}, RANK={rank}, LOCAL_RANK={local_rank}, "
                        f"MASTER_ADDR={self.master_addr}, MASTER_PORT={self.master_ports}}}")

            if os.getenv('GLOO_SOCKET_IFNAME'):
                env_vars['GLOO_SOCKET_IFNAME'] = os.getenv('GLOO_SOCKET_IFNAME')

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
            )

            self.trainers.append(trainer_handle)

        # Initialize models on all trainers
        futures = [trainer.init_models.remote() for trainer in self.trainers]
        ray.get(futures)

        # Set rollout workers and setup param sync if rollout_manager is available
        if self.rollout_manager is not None:
            rollout_workers = ray.get(self.rollout_manager.get_rollout_worker_on_tp0.remote())
            futures = [trainer.set_rollout_workers.remote(rollout_workers) for trainer in self.trainers]
            ray.get(futures)

            futures = [trainer.setup_param_sync.remote() for trainer in self.trainers]
            ray.get(futures)

        logger.success(f"Successfully initialized {len(self.trainers)} trainers with their models")

    def train(self, num_epochs: int = 1):
        """
        Execute training loop.

        Args:
            num_epochs: Number of training epochs
        """
        batch_size = self.config.actor_ref.actor.ppo_mini_batch_size
        futures = [trainer.train.remote(batch_size) for trainer in self.trainers]
        ray.get(futures)
        logger.info(f"Training completed for {num_epochs} epochs")

    def put_weight(self):
        """
        Extract trained model parameters from actor and update to RolloutManager.
        Supports model weight synchronization for rollout/inference.
        """
        logger.info("Extracting model weights from actor workers")

        if not self.trainers:
            logger.warning("No trainers available for weight extraction")
            return

        if self.rollout_manager is None:
            logger.warning("RolloutManager not set, cannot update weights")
            return

        futures = [trainer.update_rollout_weight.remote() for trainer in self.trainers]
        ray.get(futures)
        logger.info("Weight update completed")
