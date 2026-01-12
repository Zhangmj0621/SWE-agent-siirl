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
import time
import datetime
import traceback

import ray
import torch
import torch.distributed as dist
from loguru import logger
from megatron.core import parallel_state as mpu
from ray.actor import ActorHandle

from siirl.algorithm.advantage import compute_advantage
from siirl.data_coordinator.sample import Samples2Dict
from siirl.engine.actor.megatron_actor import ActorWorker, CriticWorker, ReferenceWorker
from siirl.engine.param_sync.update_weight import ParamSyncDistributed
from siirl.utils.distributed_utils import init_gloo_group
from siirl.utils.timer import Timer, TimerCollection
from siirl.worker.actor.checkpoint_manager import CheckpointManager
from siirl.utils.backend.device import get_nccl_backend, get_torch_device
from siirl.params import SiiRLArguments, TrainingArguments
from siirl.engine.actor.utils import set_random_seed

def global_initialize_model_parallel(config: TrainingArguments):
    """Initialize Megatron model parallel groups"""
    megatron_config = config

    rank = int(os.environ["LOCAL_RANK"])
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend=get_nccl_backend(),
            timeout=datetime.timedelta(seconds=600),
            init_method=os.environ.get("DIST_INIT_METHOD", None),
        )
        get_torch_device().set_device(rank)

        if megatron_config.sequence_parallel:
            os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=megatron_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=megatron_config.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=megatron_config.virtual_pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank=None,
            use_sharp=False,
            context_parallel_size=megatron_config.context_parallel_size,
            expert_model_parallel_size=megatron_config.expert_model_parallel_size,
            expert_tensor_parallel_size=megatron_config.expert_tensor_parallel_size,
            nccl_communicator_config_path=None,
        )
        set_random_seed(seed=megatron_config.seed)


class Trainer:
    """
    Single training unit managing actor, reference, and optionally critic models.
    Each Trainer handles data fetching and training execution for one GPU.
    """

    def __init__(
        self,
        config: SiiRLArguments,
        rank: int,
        local_rank: int,
        world_size: int,
        use_critic: bool = False,
        data_coordinator=None,
        coordinator=None,
        rollout_manager=None,
        metric_worker: ActorHandle | None = None,
    ):
        # Configure logging for this Ray actor process
        # (worker_process_setup_hook only works for task workers, not actors)
        from siirl.utils.logger.logging_utils import set_basic_config

        set_basic_config()

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

        # MetricTracker will be created in init_models() (only on rank=0)
        self.tracker = None

        # Initialize models (will be created in init_models method)
        self.actor_worker = None
        self.ref_worker = None
        self.critic_worker = None
        self.dp_rank = None
        self.dp_world_size = None
        self.tp_rank = None
        self.pp_rank = None
        self.should_submit_metrics = False  # Will be set in init_models()

        self.checkpoint_manager = None

        # Training state
        self.global_step = 0

        # Local batch cache for handling async data fetch race conditions
        # When some dp_ranks get data while others don't, the ones with data
        # cache it locally and wait for the next round
        self._local_batch_cache = None

        # Log trainer initialization info
        node_ip = ray.util.get_node_ip_address()
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
        ray_gpu_ids = ray.get_gpu_ids()
        logger.info(f"[Trainer.__init__] rank={rank}, local_rank={local_rank}, world_size={world_size}")
        logger.info(f"  node_ip={node_ip}, CUDA_VISIBLE_DEVICES={cuda_visible}, ray_gpu_ids={ray_gpu_ids}")

    def init_models(self):
        logger.info(f"[Trainer.global_initialize_model_parallel] rank={self.rank} starting model parallel initialization...")
        global_initialize_model_parallel(self.config.trainer)

        logger.info(f"[Trainer.init_models] rank={self.rank} starting model initialization...")

        self.actor_worker = ActorWorker(config=self.config)
        self.actor_worker.init_model()
        logger.info(f"[Trainer.init_models] rank={self.rank} ActorWorker initialized")

        self.ref_worker = ReferenceWorker(config=self.config)
        self.ref_worker.init_model()
        logger.info(f"[Trainer.init_models] rank={self.rank} ReferenceWorker initialized")

        if self.use_critic:
            self.critic_worker = CriticWorker(config=self.config)
            self.critic_worker.init_model()
            logger.info(f"[Trainer.init_models] rank={self.rank} CriticWorker initialized")

        # Use with_context_parallel=True for dp_rank/dp_world_size:
        # - Ensures CP group ranks have the same dp_rank (they process the same batch's different sequence parts)
        # - Matches the DP group used in _sync_batch_availability
        self.dp_rank = mpu.get_data_parallel_rank(with_context_parallel=True)
        self.dp_world_size = mpu.get_data_parallel_world_size(with_context_parallel=True)
        self.tp_rank = mpu.get_tensor_model_parallel_rank()
        self.pp_rank = mpu.get_pipeline_model_parallel_rank()
        self.cp_rank = mpu.get_context_parallel_rank()

        # Only TP rank 0, PP rank 0, and CP rank 0 should submit metrics to avoid duplicates
        self.should_submit_metrics = self.tp_rank == 0 and self.pp_rank == 0 and self.cp_rank == 0

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

        # Initialize MetricTracker only on global rank=0
        if self.rank == 0:
            self._init_tracker()

        logger.success(
            f"[Trainer.init_models] rank={self.rank} completed: dp_rank={self.dp_rank}, "
            f"dp_world_size={self.dp_world_size}, tp_rank={self.tp_rank}, "
            f"pp_rank={self.pp_rank}, cp_rank={self.cp_rank}"
        )

    def _init_tracker(self):
        """
        Initialize MetricTracker for logging (only called on rank=0).

        Configures backends based on config settings.
        """
        from siirl.utils.logger import MetricTracker

        logger.info(f"[Trainer rank={self.rank}] Rank 0: Initializing MetricTracker...")

        # Configure backends based on config settings
        backends = ["console"]  # Always include console
        backend_configs = {}

        # Check for wandb config
        if hasattr(self.config, "trainer") and hasattr(self.config.trainer, "logger"):
            logger_backends = self.config.trainer.logger
            if isinstance(logger_backends, list):
                backends = logger_backends
            elif isinstance(logger_backends, str):
                backends = [logger_backends]

        # Check for wandb proxy
        if hasattr(self.config, "trainer") and hasattr(self.config.trainer, "wandb_proxy"):
            if self.config.trainer.wandb_proxy:
                backend_configs["wandb"] = {"proxy": self.config.trainer.wandb_proxy}

        # Get project and experiment names
        project_name = getattr(self.config.trainer, "project_name", "siirl_agentic")
        experiment_name = getattr(self.config.trainer, "experiment_name", f"exp_{int(time.time())}")

        try:
            self.tracker = MetricTracker(
                project_name=project_name,
                experiment_name=experiment_name,
                backends=backends,
                config=self.config.to_dict() if hasattr(self.config, "to_dict") else {},
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
        ray.get(self.rollout_manager.set_step.remote(global_step))
        self.global_step = global_step
        logger.info(f"[Trainer rank={self.rank}] Loaded checkpoint, resuming from step {global_step}")
        return global_step

    def set_rollout_manager(self, rollout_manager):
        self.rollout_manager = rollout_manager

    def setup_param_sync(self):
        assert self.actor_worker is not None, "must init models first"
        assert self.rollout_manager is not None, "must set rollout_manager"
        self.param_sync = ParamSyncDistributed(
            config=self.config,
            model=self.actor_worker.actor_module,
            bridge=self.actor_worker.bridge,
        )
        init_gloo_group()

    # @timer
    def update_rollout_weight(self):
        assert self.param_sync is not None, "must setup param sync first"
        if isinstance(self.param_sync, ParamSyncDistributed):
            # TODO support elastic rollout connection
            rollout_workers = ray.get(self.rollout_manager.get_rollout_worker_on_tp0.remote())
            if any(not self.param_sync.has_connected_to_actor(x) for x in rollout_workers):
                self.param_sync.setup_param_sync_group(rollout_workers)
        self.param_sync.update_weights()

    def has_critic(self):
        return self.critic_worker is not None

    def get_current_weight_version(self) -> int:
        """Get current weight version from param_sync."""
        if hasattr(self, "param_sync") and self.param_sync is not None:
            return self.param_sync.weight_version
        return 0

    def _compute_min_version(self) -> int:
        """
        Compute minimum acceptable weight version based on off-policy config.

        off_policy_step controls version staleness tolerance:
        - 0: strict on-policy, only accept current version
        - 1: accept data up to 1 version behind
        - 2: accept data up to 2 versions behind
        """
        current_version = self.get_current_weight_version()
        off_policy_step = self.config.trainer.off_policy_step
        return max(0, current_version - off_policy_step)

    def _sync_batch_availability(self, batch_ref) -> bool:
        """
        Synchronize batch data availability across ALL ranks.

        Two-phase synchronization:
        1. DP group sync: ensure all ranks within same DP group agree on data availability
        2. Global sync: ensure all DP groups agree

        Args:
            batch_ref: The batch reference fetched from DataCoordinator (can be empty list)

        Returns:
            True if ALL ranks (across all DP groups) have data, False otherwise.

        Side effect:
            If sync fails but this rank has data, caches it in self._local_batch_cache
            for use in the next get_batch call.
        """
        # Phase 1: DP group synchronization
        # Use all_reduce with MIN on DP group: if any DP rank has 0 (no data), result is 0
        # We use DP group because different DP ranks fetch different data partitions
        has_data = torch.tensor([1 if batch_ref else 0], dtype=torch.int32, device="cuda")
        dp_group = mpu.get_data_parallel_group(with_context_parallel=True)
        dist.all_reduce(has_data, op=dist.ReduceOp.MIN, group=dp_group)
        dp_has_data = has_data.item() == 1

        # Phase 2: Global synchronization
        # Ensures all ranks across ALL DP groups agree on whether to proceed.
        global_has_data = torch.tensor([1 if dp_has_data else 0], dtype=torch.int32, device="cuda")
        dist.all_reduce(global_has_data, op=dist.ReduceOp.MIN)  # Uses default (global) process group
        all_have_data = global_has_data.item() == 1

        if not all_have_data and batch_ref:
            # Global sync failed, but this rank has data - cache it for next round
            self._local_batch_cache = batch_ref
            logger.debug(f"[Trainer rank={self.rank}] Caching batch data locally due to global sync failure")

        return all_have_data

    def get_batch(self, batch_size: int):
        """
        Get a batch of data for training with proper synchronization across ALL ranks.

        This method handles the race condition where different DP ranks may receive
        data at different times due to async data production. It uses:
        1. Local cache to store data that was fetched but couldn't be used (due to sync failure)
        2. Two-phase sync via _sync_batch_availability:
           - Phase 1: DP group all_reduce(MIN) for data partition consistency
           - Phase 2: Global all_reduce(MIN) to ensure all DP groups agree

        The two-phase sync guarantees that either ALL ranks return data and execute
        the subsequent barrier, or ALL ranks return None and skip the barrier.

        Args:
            batch_size: Total batch size (will be divided by dp_world_size)

        Returns:
            TensorDict with batch data, or None if data not available for all ranks
        """
        if self.data_coordinator is None:
            raise RuntimeError("DataCoordinator not available")

        batch_size = batch_size // self.dp_world_size

        # Priority 1: Use locally cached data from previous failed sync
        if self._local_batch_cache is not None:
            batch_ref = self._local_batch_cache
            self._local_batch_cache = None
            logger.debug(f"[Trainer rank={self.rank}] Using locally cached batch data")
        else:
            # Priority 2: Fetch from DataCoordinator
            min_version = self._compute_min_version()
            batch_ref = ray.get(
                self.data_coordinator.get_batch.remote(
                    batch_size=batch_size,
                    dp_rank=self.dp_rank,
                    balance_partitions=self.dp_world_size,
                    min_version=min_version,
                )
            )

        # Synchronize: ensure all DP ranks have data before proceeding
        if not self._sync_batch_availability(batch_ref):
            return None

        # All DP ranks have data, proceed with training
        batch_data_list = ray.get(batch_ref)

        # Clear DataCoordinator cache and sync before entering train_step
        if self.rank == 0:
            ray.get(self.data_coordinator.clear_cache.remote())
        dist.barrier()

        return Samples2Dict(batch_data_list)

    def train_step(self, batch_data):
        timers = TimerCollection()

        with timers["step"]:
            data_with_logprobs = self.actor_worker.compute_log_prob(batch_data)

            # Compute entropy from log probs
            entropy_loss = None
            if "entropys" in data_with_logprobs and "response_mask" in data_with_logprobs:
                from siirl.algorithm.loss import agg_loss

                entropys = data_with_logprobs["entropys"]
                response_mask = data_with_logprobs["response_mask"]
                loss_agg_mode = self.config.actor_ref.actor.loss_agg_mode
                entropy_loss = agg_loss(entropys, response_mask.to(entropys.device), loss_agg_mode)

            with timers["ref"]:
                data_with_ref = self.ref_worker.compute_ref_log_prob(data_with_logprobs)

            if self.use_critic:
                with timers["values"]:
                    data_with_values = self.critic_worker.compute_values(data_with_ref)
            else:
                data_with_values = data_with_ref

            algo_config = self.config.actor_ref.algorithm
            adv_estimator = algo_config.adv_estimator
            gamma = algo_config.gamma
            lam = algo_config.lam

            with timers["adv"]:
                data_for_update = compute_advantage(
                    data=data_with_values,
                    adv_estimator=adv_estimator,
                    gamma=gamma,
                    lam=lam,
                )

            with timers["update_actor"]:
                actor_result = self.actor_worker.update_actor(data_for_update)

            # Extract metrics from TensorDict (stored in data["metrics"] by update_actor)
            actor_metrics = actor_result.get("metrics", {})
            if hasattr(actor_metrics, "data"):  # NonTensorData wrapper
                actor_metrics = actor_metrics.data

            # Add entropy loss to actor metrics (computed earlier from compute_log_prob)
            if entropy_loss is not None:
                actor_metrics["actor/entropy_loss"] = entropy_loss.item()

            if self.use_critic:
                with timers["update_critic"]:
                    critic_result = self.critic_worker.update_critic(data_for_update)

                # Extract critic metrics
                critic_metrics = critic_result.get("metrics", {})
                if hasattr(critic_metrics, "data"):
                    critic_metrics = critic_metrics.data

                metrics = {"actor": actor_metrics, "critic": critic_metrics}
            else:
                metrics = {"actor": actor_metrics}

        timing_raw = timers.to_dict()

        # Submit metrics to MetricWorker for aggregation
        # Only TP rank 0 and PP rank 0 should submit to avoid duplicates
        if self.metric_client is not None and self.should_submit_metrics:
            try:
                from siirl.utils.metrics import (
                    compute_data_metric,
                    compute_log_prob_diff_metrics,
                    compute_throughput_metrics,
                    extract_rollout_timing_metrics,
                )

                # Compute and submit data metrics
                data_metrics = compute_data_metric(data_for_update)
                self.metric_client.submit_metric(data_metrics, self.dp_world_size)

                # Compute and submit throughput metrics
                # Note: throughput will be recalculated in train() with correct step_interval
                n_gpus = self.world_size
                throughput_metrics = compute_throughput_metrics(data_for_update, timing_raw, n_gpus)
                self.metric_client.submit_metric(throughput_metrics, self.dp_world_size)

                # Submit timing metrics (train_step internal timings)
                timing_metrics = {f"timing_s/{k}": v for k, v in timing_raw.items()}
                self.metric_client.submit_metric(timing_metrics, self.dp_world_size)

                # Extract and submit rollout timing metrics from batch data
                rollout_timing = extract_rollout_timing_metrics(data_for_update)
                if rollout_timing:
                    # Separate internal key from metrics to submit
                    earliest_start = rollout_timing.pop("_earliest_rollout_start_at", None)
                    if rollout_timing:
                        self.metric_client.submit_metric(rollout_timing, self.dp_world_size)

                # Submit actor/critic update metrics
                # Note: metrics from megatron_actor.py already have proper prefixes (e.g. "actor/pg_loss", "perf/mfu/actor")
                # so we just merge them directly without adding another prefix
                flat_metrics = {}
                for _, result_dict in metrics.items():
                    if isinstance(result_dict, dict):
                        flat_metrics.update(result_dict)
                if flat_metrics:
                    self.metric_client.submit_metric(flat_metrics, self.dp_world_size)

                # Compute and submit log prob diff metrics (rollout vs training)
                # max/mean can be aggregated normally, std needs special handling via StdStats
                diff_metrics, std_stats = compute_log_prob_diff_metrics(data_for_update)
                if diff_metrics:
                    self.metric_client.submit_metric(diff_metrics, self.dp_world_size)

                # Submit std stats separately for proper distributed std calculation
                if std_stats is not None:
                    self.metric_client.submit_metric({"actor/rollout_probs_diff_std": std_stats}, self.dp_world_size)

            except Exception as e:
                logger.warning(f"[Trainer rank={self.rank}] Failed to submit metrics: {e}")

        logger.success(
            f"[Trainer.train_step] rank={self.rank} dp_rank={self.dp_rank} step={self.global_step} completed in {timers['step'].formatted}"
        )

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
            ray.get(self.coordinator.report_failure.remote(source=f"trainer_{self.rank}", reason=error_msg))
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

        # Track step interval for accurate throughput calculation
        last_step_end_time = None

        try:
            while True:
                # Check stop signal
                if self._check_should_stop():
                    logger.info(f"[Trainer rank={self.rank}] Stop signal received, exiting...")
                    break

                # Update rollout weights and record timing
                with Timer("weight_sync") as weight_sync_timer:
                    self.update_rollout_weight()

                # run dataloader for train
                if self.rank == 0:
                    ray.get(self.rollout_manager.next_rollout.remote())

                # Record get_batch timing
                with Timer("get_batch") as get_batch_timer:
                    while (batch_data := self.get_batch(batch_size)) is None:
                        time.sleep(0.1)

                # compare
                self.train_step(batch_data)

                # Calculate step_interval (time between consecutive step completions)
                current_step_end_time = time.time()
                step_interval = None
                if last_step_end_time is not None:
                    step_interval = current_step_end_time - last_step_end_time
                last_step_end_time = current_step_end_time

                # Wait for all metric submissions and aggregate (rank=0 does the logging)
                # Only TP rank 0 and PP rank 0 submit metrics, so only they need to wait
                if self.metric_client is not None and self.should_submit_metrics:
                    try:
                        self.metric_client.wait_submit()
                    except Exception as e:
                        logger.warning(f"[Trainer rank={self.rank}] Metric submission wait failed: {e}")

                # Barrier sync to ensure all ranks are synchronized
                dist.barrier()

                # Only rank=0 (global rank) aggregates and logs to tracker
                if self.rank == 0 and self.tracker is not None and self.metric_client is not None:
                    try:
                        aggregated_metrics = self.metric_client.wait_final_res()
                        aggregated_metrics["training/global_step"] = self.global_step
                        aggregated_metrics["perf/delta_time/weight_sync"] = weight_sync_timer.elapsed
                        aggregated_metrics["perf/delta_time/get_batch"] = get_batch_timer.elapsed

                        # Add step_interval for accurate throughput measurement
                        if step_interval is not None:
                            aggregated_metrics["perf/delta_time/step_interval"] = step_interval
                            # Recalculate throughput using step_interval (system throughput)
                            total_tokens = aggregated_metrics.get("perf/total_num_tokens", 0)
                            if step_interval > 0 and total_tokens > 0:
                                aggregated_metrics["perf/throughput"] = total_tokens / (step_interval * self.world_size)
                        self.tracker.log(aggregated_metrics, step=self.global_step)

                        # get rollout validate metrics
                        val_metrics = ray.get(self.rollout_manager.get_metrics.remote())
                        for metrics, global_step in val_metrics:
                            self.tracker.log(metrics, global_step)

                    except Exception as e:
                        logger.warning(f"[Trainer rank={self.rank}] Metric aggregation failed: {e}")

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
            # Only ranks that submitted metrics need to wait
            if self.metric_client is not None and self.should_submit_metrics:
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
