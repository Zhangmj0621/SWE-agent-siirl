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
import time
from tensordict import stack
from megatron.core import parallel_state as mpu

from siirl.engine.actor.megatron_actor import ActorWorker, ReferenceWorker, CriticWorker
from siirl.engine.param_sync.update_weight import ParamSyncDistribute
from siirl.algorithm.advantage import compute_advantage
from siirl.utils.distributed_utils import init_gloo_group


class Trainer:
    """
    Single training unit managing actor, reference, and optionally critic models.
    Each Trainer handles data fetching and training execution for one GPU.
    """

    def __init__(
        self,
        config,
        rank: int,
        local_rank: int,
        world_size: int,
        use_critic: bool = False,
        data_coordinator=None,
    ):
        self.config = config
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.use_critic = use_critic
        self.data_coordinator = data_coordinator

        self.rollout_manager = None

        # Initialize models (will be created in init_models method)
        self.actor_worker = None
        self.ref_worker = None
        self.critic_worker = None
        self.dp_rank = None
        self.dp_world_size = None

    def init_models(self):
        self.actor_worker = ActorWorker(config=self.config.actor_ref)
        self.actor_worker.init_model()

        self.ref_worker = ReferenceWorker(config=self.config.actor_ref)
        self.ref_worker.init_model()

        if self.use_critic:
            self.critic_worker = CriticWorker(config=self.config.critic)
            self.critic_worker.init_model()

        self.dp_rank = mpu.get_data_parallel_rank()
        self.dp_world_size = mpu.get_data_parallel_world_size()


    def set_rollout_workers(self, rollout_workers):
        self.rollout_workers = rollout_workers

    def setup_param_sync(self):
        assert self.actor_worker is not None,"must init models first"
        assert self.rollout_workers is not None, "must set rollout workers"
        self.param_sync = ParamSyncDistribute(config=self.config.actor_ref, model=self.actor_worker.actor_module, bridge=self.actor_worker.bridge)
        init_gloo_group()
        
    # @timer
    def update_rollout_weight(self):
        assert self.param_sync is not None, "must setup param sync first"
        if isinstance(self.param_sync,ParamSyncDistribute):
            # TODO support elastic rollout connection
            rollout_workers = self.rollout_manager.get_rollout_worker_on_tp0.remote()
            if any(self.param_sync.has_connected_to_actor(x) for x in rollout_workers):   
                self.param_sync.setup_param_sync_group(rollout_workers)
        self.param_sync.update_weights()

    def has_critic(self):
        return self.critic_worker is not None

    def get_batch(self, batch_size: int):
        if self.data_coordinator is None:
            raise RuntimeError("DataCoordinator not available")

        batch_ref = ray.get(
            self.data_coordinator.get_batch.remote(
                batch_size=batch_size,
                dp_rank=self.dp_rank,
                balance_partitions=self.dp_world_size,
            )
        )

        if not batch_ref:
            return None

        batch_data_list = ray.get(batch_ref)

        return stack(batch_data_list, dim=0)

    def train_step(self, batch_data):
        data_with_logprobs = self.actor_worker.compute_log_prob(batch_data)
        data_with_ref = self.ref_worker.compute_ref_log_prob(data_with_logprobs)

        if self.use_critic:
            data_with_values = self.critic_worker.compute_values(data_with_ref)
        else:
            data_with_values = data_with_ref

        algo_config = self.config.actor_ref.algorithm
        adv_estimator = algo_config.adv_estimator
        gamma = algo_config.gamma
        lam = algo_config.lam

        data_for_update = compute_advantage(
            data=data_with_values,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
        )

        actor_result = self.actor_worker.update_actor(data_for_update)

        if self.use_critic:
            critic_result = self.critic_worker.update_critic(data_for_update)
            metrics = {"actor": actor_result, "critic": critic_result}
        else:
            metrics = {"actor": actor_result}

        return metrics

    def train(self, batch_size: int):
        """Continuous training loop that processes batches as they become available."""
        while True:
            batch_data = self.get_batch(batch_size)
            if batch_data is not None:
                self.train_step(batch_data)
            time.sleep(0.1)