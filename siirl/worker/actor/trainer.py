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
from typing import Optional
from ray.actor import ActorHandle
from megatron.core import parallel_state as mpu

from siirl.engine.actor.megatron_actor import ActorWorker, ReferenceWorker, CriticWorker
from siirl.engine.param_sync.update_weight import ParamSyncDistributed
from siirl.algorithm.advantage import compute_advantage
from siirl.utils.distributed_utils import init_gloo_group
from siirl.data_coordinator.sample import Samples2Dict
from siirl.worker.actor.checkpoint_manager import CheckpointManager

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
        rollout_manager=None,
        metric_worker: Optional[ActorHandle] = None,
    ):
        # NOTE: Logging is auto-configured via worker_process_setup_hook in ray.init()
        
        self.config = config
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.use_critic = use_critic
        self.data_coordinator = data_coordinator
        self.coordinator = coordinator  # TaskCoordinator for lifecycle management

        self.rollout_manager = rollout_manager

        # Initialize MetricClient for distributed metrics collection
        self.metric_client = None
        if metric_worker is not None:
            from siirl.utils.metrics import MetricClient
            self.metric_client = MetricClient(metric_worker)
            logger.info(f"[Trainer rank={rank}] MetricClient initialized")

        # MetricTracker will be created in init_models()
        # Only rank=0 (global rank) creates a tracker (same as siiRL-github)
        self.tracker = None

        # Initialize models (will be created in init_models method)
        self.actor_worker = None
        self.ref_worker = None
        self.critic_worker = None
        self.dp_rank = None
        self.dp_world_size = None
        self.checkpoint_manager = None

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

        self.checkpoint_manager = CheckpointManager(
            config=self.config,
            rank=self.rank,
            world_size=self.world_size,
            actor_worker=self.actor_worker,
            ref_worker=self.ref_worker,
            critic_worker=self.critic_worker,
            data_coordinator=self.data_coordinator,
            dp_rank=self.dp_rank,
            dp_world_size=self.dp_world_size,
        )


        # Initialize MetricTracker only on rank=0 (global rank, same as siiRL-github)
        if self.rank == 0:
            self._init_tracker()

        logger.success(f"[Trainer.init_models] rank={self.rank} completed: dp_rank={self.dp_rank}, dp_world_size={self.dp_world_size}")

    def _init_tracker(self):
        """
        Initialize MetricTracker for logging (only called on rank=0).

        Configures backends based on config settings, similar to siiRL-github's
        DAGWorker._initialize_worker() pattern.
        """
        from siirl.utils.logger import MetricTracker
        
        logger.info(f"[Trainer rank={self.rank}] Rank 0: Initializing MetricTracker...")

        # Configure backends based on config settings
        backends = ["console"]  # Always include console
        backend_configs = {}

        # Check for wandb config
        if hasattr(self.config, 'trainer') and hasattr(self.config.trainer, 'logger'):
            logger_backends = self.config.trainer.logger
            if isinstance(logger_backends, list):
                backends = logger_backends
            elif isinstance(logger_backends, str):
                backends = [logger_backends]

        # Check for wandb proxy
        if hasattr(self.config, 'trainer') and hasattr(self.config.trainer, 'wandb_proxy'):
            if self.config.trainer.wandb_proxy:
                backend_configs["wandb"] = {"proxy": self.config.trainer.wandb_proxy}

        # Get project and experiment names
        project_name = getattr(self.config.trainer, 'project_name', 'siirl_agentic')
        experiment_name = getattr(self.config.trainer, 'experiment_name', f'exp_{int(time.time())}')

        try:
            self.tracker = MetricTracker(
                project_name=project_name,
                experiment_name=experiment_name,
                backends=backends,
                config=self.config.to_dict() if hasattr(self.config, 'to_dict') else {},
                backend_configs=backend_configs,
            )
            logger.success(f"[Trainer rank={self.rank}] MetricTracker initialized: backends={backends}")
        except Exception as e:
            logger.error(f"[Trainer rank={self.rank}] Failed to initialize MetricTracker: {e}")
            self.tracker = None

    def load_checkpoint(self):
        """Load checkpoint and return global step."""
        if self.checkpoint_manager is None:
            logger.warning(f"[Trainer rank={self.rank}] Checkpoint manager not initialized")
            return 0

        global_step = self.checkpoint_manager.load_checkpoint()
        self.global_step = global_step
        logger.info(f"[Trainer rank={self.rank}] Loaded checkpoint, resuming from step {global_step}")
        return global_step


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

        ray.get(self.data_coordinator.clear_cache.remote()) if self.rank == 0 else None
        dist.barrier()

        return Samples2Dict(batch_data_list)

    def train_step(self, batch_data):
        step_start_time = time.time()
        timing_raw = {}
        logger.info(f"[Trainer.train_step] rank={self.rank} dp_rank={self.dp_rank} step={self.global_step} starting")

        logger.info(f"[Trainer.train_step] step={self.global_step} computing actor log probs")
        data_with_logprobs = self.actor_worker.compute_log_prob(batch_data)

        logger.info(f"[Trainer.train_step] step={self.global_step} computing reference log probs")
        ref_start = time.time()
        data_with_ref = self.ref_worker.compute_ref_log_prob(data_with_logprobs)
        timing_raw["ref"] = time.time() - ref_start

        if self.use_critic:
            logger.info(f"[Trainer.train_step] step={self.global_step} computing critic values")
            values_start = time.time()
            data_with_values = self.critic_worker.compute_values(data_with_ref)
            timing_raw["values"] = time.time() - values_start
        else:
            data_with_values = data_with_ref

        algo_config = self.config.actor_ref.algorithm
        adv_estimator = algo_config.adv_estimator
        gamma = algo_config.gamma
        lam = algo_config.lam

        logger.info(f"[Trainer.train_step] step={self.global_step} computing advantages with {adv_estimator}")
        adv_start = time.time()
        data_for_update = compute_advantage(
            data=data_with_values,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
        )
        timing_raw["adv"] = time.time() - adv_start

        logger.info(f"[Trainer.train_step] step={self.global_step} updating actor")
        update_actor_start = time.time()
        actor_result = self.actor_worker.update_actor(data_for_update)
        timing_raw["update_actor"] = time.time() - update_actor_start

        if self.use_critic:
            logger.info(f"[Trainer.train_step] step={self.global_step} updating critic")
            update_critic_start = time.time()
            critic_result = self.critic_worker.update_critic(data_for_update)
            timing_raw["update_critic"] = time.time() - update_critic_start
            metrics = {"actor": actor_result, "critic": critic_result}
        else:
            metrics = {"actor": actor_result}

        step_duration = time.time() - step_start_time
        timing_raw["step"] = step_duration

        # Submit metrics to MetricWorker for aggregation
        if self.metric_client is not None:
            try:
                from siirl.utils.metrics import compute_data_metric, compute_throughput_metrics
                
                # Compute and submit data metrics
                data_metrics = compute_data_metric(data_for_update)
                self.metric_client.submit_metric(data_metrics, self.dp_world_size)

                # Compute and submit throughput metrics
                n_gpus = self.world_size
                throughput_metrics = compute_throughput_metrics(data_for_update, timing_raw, n_gpus)
                self.metric_client.submit_metric(throughput_metrics, self.dp_world_size)

                # Submit timing metrics
                timing_metrics = {f"timing_s/{k}": v for k, v in timing_raw.items()}
                self.metric_client.submit_metric(timing_metrics, self.dp_world_size)

                # Submit actor/critic update metrics
                flat_metrics = {}
                for prefix, result_dict in metrics.items():
                    if isinstance(result_dict, dict):
                        for k, v in result_dict.items():
                            flat_metrics[f"{prefix}/{k}"] = v
                if flat_metrics:
                    self.metric_client.submit_metric(flat_metrics, self.dp_world_size)

            except Exception as e:
                logger.warning(f"[Trainer rank={self.rank}] Failed to submit metrics: {e}")

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
        - Metrics aggregation and logging (rank=0 handles logging via self.tracker)

        Args:
            batch_size: Training batch size
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

                # Wait for all metric submissions and aggregate (rank=0 does the logging)
                if self.metric_client is not None:
                    try:
                        self.metric_client.wait_submit()

                        # Only rank=0 (global rank) aggregates and logs to tracker
                        if self.rank == 0 and self.tracker is not None:
                            aggregated_metrics = self.metric_client.wait_final_res()
                            aggregated_metrics["training/global_step"] = self.global_step
                            self.tracker.log(aggregated_metrics, step=self.global_step)

                    except Exception as e:
                        logger.warning(f"[Trainer rank={self.rank}] Metric aggregation failed: {e}")

                self.update_rollout_weight()
                ray.get(self.rollout_manager.next_rollout.remote())
                self.global_step += 1

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    logger.info(f"[Trainer rank={self.rank}] Saving checkpoint at step {self.global_step}")
                    self.checkpoint_manager.save_checkpoint(self.global_step)

                time.sleep(0.01)

        except Exception as e:
            error_msg = f"Training failed at step {self.global_step}: {e}"
            logger.error(f"[Trainer rank={self.rank}] {error_msg}")
            logger.error(f"[Trainer rank={self.rank}] Full traceback:\n{traceback.format_exc()}")
            self._report_failure(error_msg)
            raise
        
        finally:
            # Ensure all pending metrics are submitted before exiting
            if self.metric_client is not None:
                try:
                    self.metric_client.wait_submit()
                except Exception:
                    pass

            # Close MetricTracker (only rank=0 has one)
            if self.tracker is not None:
                try:
                    self.tracker.finish()
                    logger.info(f"[Trainer rank={self.rank}] MetricTracker closed")
                except Exception as e:
                    logger.warning(f"[Trainer rank={self.rank}] Error closing MetricTracker: {e}")

        logger.info(f"[Trainer rank={self.rank}] Training loop ended at step {self.global_step}")
