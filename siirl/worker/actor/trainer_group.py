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


class TrainerGroup:
    """
    Manages a group of Trainers for distributed RL training.
    Supports PPO and GRPO algorithms with train/inference separation architecture.
    """

    def __init__(
        self,
        config: SiiRLArguments,
        data_coordinator,
        num_gpus: int,
        placement_groups: Optional[List] = None,
    ) -> None:
        self.config = config
        self.data_coordinator = data_coordinator
        self.num_gpus = num_gpus
        self.placement_groups = placement_groups
        self.trainers: List[Trainer] = []
        self.use_critic = self.config.actor_ref.algorithm.adv_estimator == "ppo"
        self.master_addr, self.master_ports = get_master_info()

    def init_actors(self):
        """
        Initialize trainers by creating Trainer instances and wrapping them as Ray Actors.
        Each Trainer manages its own actor, ref, and optionally critic models.
        """
        n_gpus_per_node = self.config.trainer.n_gpus_per_node

        for rank in range(self.num_gpus):
            node_idx = rank // n_gpus_per_node
            local_rank = rank % n_gpus_per_node
            pg = self.placement_groups[node_idx] if self.placement_groups else None
            bundle_index = local_rank

            env_vars = {
                DistributedEnv.WORLD_SIZE.value: str(self.num_gpus),
                DistributedEnv.RANK.value: str(rank),
                DistributedEnv.LOCAL_RANK.value: str(local_rank),
                DistributedEnv.MASTER_ADDR.value: self.master_addr,
                DistributedEnv.MASTER_PORT.value: self.master_ports,
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            }

            if os.getenv('GLOO_SOCKET_IFNAME'):
                env_vars['GLOO_SOCKET_IFNAME'] = os.getenv('GLOO_SOCKET_IFNAME')

            TrainerActor = ray.remote(Trainer)
            trainer_options = {
                "runtime_env": {"env_vars": env_vars},
                "name": f"trainer_{rank}",
                "num_gpus": 1,
            }

            trainer_handle = TrainerActor.options(
                **trainer_options,
                placement_group=pg,
                placement_group_bundle_index=bundle_index,
            ).remote(
                config=self.config,
                rank=rank,
                local_rank=local_rank,
                world_size=self.num_gpus,
                use_critic=self.use_critic,
                data_coordinator=self.data_coordinator,
            )

            self.trainers.append(trainer_handle)

        ray.get([trainer.init_models.remote() for trainer in self.trainers])

        futures = [trainer.set_rollout_workers.remote(rollout_manager.get_rollout_worker_on_tp0()) for trainer in self.trainers]
        ray.get(futures)

        futures = [trainer.setup_param_sync.remote() for trainer in self.trainers]
        ray.get(futures)

        logger.success(f"Initialized {len(self.trainers)} trainers")

    def train(self):
        batch_size = self.config.actor_ref.actor.ppo_mini_batch_size
        ray.get([trainer.train.remote(batch_size) for trainer in self.trainers])

    def put_weight(self):
        if not self.trainers:
            logger.warning("No trainers available for weight extraction")
            return
        ray.get([trainer.update_rollout_weight.remote() for trainer in self.trainers])
        logger.warning("Weight update functionality not implemented")
        
        

