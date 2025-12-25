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
import os
import torch.distributed as dist
import traceback
from loguru import logger
from megatron.core import parallel_state as mpu

from siirl.engine.actor.megatron_actor import ActorWorker, ReferenceWorker, CriticWorker
from siirl.engine.param_sync.update_weight import ParamSyncDistributed
from siirl.algorithm.advantage import compute_advantage
from siirl.utils.distributed_utils import init_gloo_group
from siirl.data_coordinator.sample import Samples2Dict


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
        coordinator=None,
        rollout_manager = None,
    ):
        self.config = config
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.use_critic = use_critic
        self.data_coordinator = data_coordinator
        self.coordinator = coordinator  # TaskCoordinator for lifecycle management

        self.rollout_manager = rollout_manager

        # Initialize models (will be created in init_models method)
        self.actor_worker = None
        self.ref_worker = None
        self.critic_worker = None
        self.dp_rank = None
        self.dp_world_size = None
        
        # Training state
        self.global_step = 0

        # Log trainer initialization info
        node_ip = ray.util.get_node_ip_address()
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
        ray_gpu_ids = ray.get_gpu_ids()
        logger.info(f"[Trainer.__init__] rank={rank}, local_rank={local_rank}, world_size={world_size}")
        logger.info(f"  node_ip={node_ip}, CUDA_VISIBLE_DEVICES={cuda_visible}, ray_gpu_ids={ray_gpu_ids}")

    def init_models(self):
        logger.info(f"[Trainer.init_models] rank={self.rank} starting model initialization...")
        
        self.actor_worker = ActorWorker(config=self.config.actor_ref)
        self.actor_worker.init_model()
        logger.info(f"[Trainer.init_models] rank={self.rank} ActorWorker initialized")

        self.ref_worker = ReferenceWorker(config=self.config.actor_ref)
        self.ref_worker.init_model()
        logger.info(f"[Trainer.init_models] rank={self.rank} ReferenceWorker initialized")

        if self.use_critic:
            self.critic_worker = CriticWorker(config=self.config.critic)
            self.critic_worker.init_model()
            logger.info(f"[Trainer.init_models] rank={self.rank} CriticWorker initialized")

        self.dp_rank = mpu.get_data_parallel_rank()
        self.dp_world_size = mpu.get_data_parallel_world_size()
        
        logger.success(f"[Trainer.init_models] rank={self.rank} completed: dp_rank={self.dp_rank}, dp_world_size={self.dp_world_size}")


    def set_rollout_manager(self, rollout_manager):
        self.rollout_manager = rollout_manager

    def setup_param_sync(self):
        assert self.actor_worker is not None,"must init models first"
        assert self.rollout_manager is not None, "must set rollout_manager"
        self.param_sync = ParamSyncDistributed(config=self.config, model=self.actor_worker.actor_module, bridge=self.actor_worker.bridge)
        init_gloo_group()
        
    # @timer
    def update_rollout_weight(self):
        assert self.param_sync is not None, "must setup param sync first"
        if isinstance(self.param_sync,ParamSyncDistributed):
            # TODO support elastic rollout connection
            rollout_workers = ray.get(self.rollout_manager.get_rollout_worker_on_tp0.remote())
            if any(not self.param_sync.has_connected_to_actor(x) for x in rollout_workers):   
                self.param_sync.setup_param_sync_group(rollout_workers)
        self.param_sync.update_weights()

    def has_critic(self):
        return self.critic_worker is not None

    def get_batch(self, batch_size: int):
        if self.data_coordinator is None:
            raise RuntimeError("DataCoordinator not available")

        batch_size = batch_size // self.dp_world_size 

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

        ray.get(self.data_coordinator.reset_cache.remote()) if self.rank == 0 else None
        dist.barrier()

        return Samples2Dict(batch_data_list)

    def train_step(self, batch_data):
        step_start_time = time.time()
        logger.info(f"[Trainer.train_step] rank={self.rank} dp_rank={self.dp_rank} step={self.global_step} starting")

        logger.info(f"[Trainer.train_step] step={self.global_step} computing actor log probs")
        data_with_logprobs = self.actor_worker.compute_log_prob(batch_data)

        logger.info(f"[Trainer.train_step] step={self.global_step} computing reference log probs")
        data_with_ref = self.ref_worker.compute_ref_log_prob(data_with_logprobs)

        if self.use_critic:
            logger.info(f"[Trainer.train_step] step={self.global_step} computing critic values")
            data_with_values = self.critic_worker.compute_values(data_with_ref)
        else:
            data_with_values = data_with_ref

        algo_config = self.config.actor_ref.algorithm
        adv_estimator = algo_config.adv_estimator
        gamma = algo_config.gamma
        lam = algo_config.lam

        logger.info(f"[Trainer.train_step] step={self.global_step} computing advantages with {adv_estimator}")
        data_for_update = compute_advantage(
            data=data_with_values,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
        )

        logger.info(f"[Trainer.train_step] step={self.global_step} updating actor")
        actor_result = self.actor_worker.update_actor(data_for_update)

        if self.use_critic:
            logger.info(f"[Trainer.train_step] step={self.global_step} updating critic")
            critic_result = self.critic_worker.update_critic(data_for_update)
            metrics = {"actor": actor_result, "critic": critic_result}
        else:
            metrics = {"actor": actor_result}

        step_duration = time.time() - step_start_time
        logger.success(f"[Trainer.train_step] rank={self.rank} dp_rank={self.dp_rank} step={self.global_step} completed in {step_duration:.2f}s")

        return metrics

    def _check_should_stop(self) -> bool:
        """Check if training should stop based on coordinator signal."""
        if not self.coordinator:
            return False
        try:
            return ray.get(self.coordinator.should_stop.remote())
        except Exception as e:
            logger.warning(f"[Trainer rank={self.rank}] Failed to check coordinator: {e}")
            logger.warning(f"[Trainer rank={self.rank}] Traceback:\n{traceback.format_exc()}")
            return False

    def _report_failure(self, error_msg: str):
        """Report failure to coordinator."""
        if not self.coordinator:
            return
        try:
            ray.get(self.coordinator.report_failure.remote(
                source=f"trainer_{self.rank}",
                reason=error_msg
            ))
        except Exception as e:
            logger.warning(f"[Trainer rank={self.rank}] Failed to report failure: {e}")
            logger.warning(f"[Trainer rank={self.rank}] Traceback:\n{traceback.format_exc()}")

    def train(self, batch_size: int):
        """
        Continuous training loop that processes batches as they become available.
        
        The loop checks for stop signals from TaskCoordinator and handles:
        - Graceful shutdown (coordinator signal)
        - Error propagation (reports failures to coordinator)
        """
        logger.info(f"[Trainer rank={self.rank}] Starting training loop, batch_size={batch_size}")
        
        try:
            while True:
                # Check stop signal
                if self._check_should_stop():
                    logger.info(f"[Trainer rank={self.rank}] Stop signal received, exiting...")
                    break

                batch_data = self.get_batch(batch_size)
                if batch_data is None:
                    time.sleep(0.1)
                    continue

                self.train_step(batch_data)
                self.update_rollout_weight()

                self.global_step += 1

                time.sleep(0.01)

        except Exception as e:
            error_msg = f"Training failed at step {self.global_step}: {e}"
            logger.error(f"[Trainer rank={self.rank}] {error_msg}")
            logger.error(f"[Trainer rank={self.rank}] Full traceback:\n{traceback.format_exc()}")
            self._report_failure(error_msg)
            raise
        
        logger.info(f"[Trainer rank={self.rank}] Training loop ended at step {self.global_step}")
