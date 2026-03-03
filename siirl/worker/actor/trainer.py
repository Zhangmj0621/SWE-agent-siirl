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

import contextlib
import datetime
import os
import time
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
from siirl.engine.actor.utils import set_random_seed
from siirl.engine.param_sync.update_weight import ParamSyncColocated, ParamSyncDistributed
from siirl.params import SiiRLArguments, TrainingArguments
from siirl.utils.backend.device import get_nccl_backend, get_torch_device
from siirl.utils.distributed_utils import get_gloo_group, init_gloo_group
from siirl.utils.logger.memory_profiler import MemoryProfiler
from siirl.utils.megatron.megatron_utils import offload_megatron_model_to_cpu
from siirl.utils.timer import Timer, TimerCollection
from siirl.worker.actor.checkpoint_manager import CheckpointManager
from siirl.worker.validate.reuse.constants import SYNC_RETRY_SLEEP_S
from siirl.worker.validate.reuse.trainer_sync import ValidateGateDecision, ValidateReuseTrainerSync

TRAIN_NO_BATCH_BACKOFF_S = 0.1


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

        # Set Megatron global args for recompute (activation checkpointing)
        # This is required because Megatron's forward_backward_func reads from global args
        try:
            from argparse import Namespace

            from megatron.training.global_vars import get_args, set_args

            # Try to get existing args, or create new one
            try:
                megatron_args = get_args()
                if megatron_args is None:
                    megatron_args = Namespace()
            except Exception:
                megatron_args = Namespace()

            # Set recompute parameters for memory optimization
            megatron_args.recompute_granularity = "full"
            megatron_args.recompute_method = "uniform"
            megatron_args.recompute_num_layers = 1
            megatron_args.distribute_saved_activations = False

            set_args(megatron_args)
            logger.debug("[Memory] Enabled Megatron activation recompute: granularity=full, method=uniform")
        except ImportError:
            pass  # Megatron global_vars not available, skip recompute config

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
        self.param_sync = None
        self._validate_reuse_sync: ValidateReuseTrainerSync | None = None
        self._colocate_scope_weights_offloaded = False

        # Training state
        self.global_step = 0
        # Subtract prior checkpoint save overhead from next-step perf accounting.
        self._pending_ckpt_excluded_time = 0.0
        # EMA of step_interval from non-validation steps.  Used as a floor
        # when excluding validation time to avoid over-exclusion that removes
        # the normal generation-pipeline lag.
        self._step_interval_ema: float = 0.0
        self._step_interval_ema_alpha: float = 0.3

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

        # Use with_context_parallel=False for dp_rank/dp_world_size:
        # - Ensures CP group ranks have the same dp_rank (they process the same batch's different sequence parts)
        # - Matches the DP group used in _sync_batch_availability
        self.dp_rank = mpu.get_data_parallel_rank(with_context_parallel=False)
        self.dp_world_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
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
        if hasattr(self.config, "trainer") and hasattr(self.config.trainer, "wandb_proxy") and self.config.trainer.wandb_proxy:
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
        self._maybe_init_validate_reuse_sync()

    def setup_param_sync(self):
        assert self.actor_worker is not None, "must init models first"
        assert self.rollout_manager is not None, "must set rollout_manager"
        cls = ParamSyncColocated if self.config.trainer.colocate else ParamSyncDistributed
        self.param_sync = cls(
            config=self.config,
            model=self.actor_worker.actor_module,
            bridge=self.actor_worker.bridge,
        )
        init_gloo_group()
        logger.info(f"[Trainer rank={self.rank}] param_sync={cls.__name__} (colocate={self.config.trainer.colocate})")

        # For colocated mode, fetch topology snapshot and pass to setup
        if self.config.trainer.colocate and isinstance(self.param_sync, ParamSyncColocated):
            rollout_workers = ray.get(self.rollout_manager.get_rollout_worker_on_tp0.remote())
            rollout_topology = ray.get(self.rollout_manager.get_colocate_topology_snapshot.remote())
            self.param_sync.setup_param_sync_group(rollout_workers, rollout_topology=rollout_topology)
        else:
            rollout_workers = ray.get(self.rollout_manager.get_rollout_worker_on_tp0.remote())
            self.param_sync.setup_param_sync_group(rollout_workers)

        self._maybe_init_validate_reuse_sync()

    def _broadcast_rank0_int(self, local_value: int) -> int:
        """Broadcast a rank-0 int32 scalar to all ranks via Gloo."""
        value = torch.tensor([int(local_value)], dtype=torch.int32)
        dist.broadcast(value, src=0, group=get_gloo_group())
        return int(value.item())

    def _broadcast_rank0_error(self, local_error: Exception | None) -> Exception | None:
        """Broadcast rank 0 error flag to all ranks via Gloo so every rank fails consistently.

        Args:
            local_error: The exception caught on rank 0 (None on non-rank-0 or success).

        Returns:
            The original exception on rank 0, a RuntimeError placeholder on other ranks
            if rank 0 failed, or None if no error.
        """
        has_error = self._broadcast_rank0_int(1 if local_error is not None else 0)
        if has_error == 0:
            return None
        if local_error is not None:
            return local_error
        return RuntimeError(f"[Trainer rank={self.rank}] rank 0 reported failure (see rank 0 logs)")

    def _broadcast_rank0_bool(self, local_value: bool) -> bool:
        """Broadcast a rank-0 bool decision to keep all ranks on one control path."""
        return self._broadcast_rank0_int(1 if local_value else 0) == 1

    @property
    def _is_colocate(self) -> bool:
        return self.config.trainer.colocate and isinstance(self.param_sync, ParamSyncColocated)

    @property
    def _rpc_timeout_s(self) -> int:
        return max(1, int(getattr(self.config.trainer, "param_sync_rpc_timeout_s", 120)))

    @property
    def _colocate_timeout_s(self) -> int:
        return max(1, int(getattr(self.config.trainer, "colocate_timeout_s", 60)))

    def _cuda_debug_snapshot(self) -> dict[str, float | int | str]:
        if not torch.cuda.is_available():
            return {"cuda_available": 0}
        try:
            device = torch.cuda.current_device()
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            return {
                "cuda_available": 1,
                "device": int(device),
                "allocated_gb": round(torch.cuda.memory_allocated(device) / (1024**3), 3),
                "reserved_gb": round(torch.cuda.memory_reserved(device) / (1024**3), 3),
                "free_gb": round(free_bytes / (1024**3), 3),
                "total_gb": round(total_bytes / (1024**3), 3),
            }
        except Exception as e:
            return {"cuda_available": 1, "snapshot_error": repr(e)}

    def _next_weight_version_hint(self, bump_weight_version: bool = True) -> int:
        current = getattr(self.param_sync, "weight_version", None)
        if current is None:
            return -1
        try:
            current_int = int(current)
        except Exception:
            return -1
        return current_int + (1 if bump_weight_version else 0)

    def _build_colocate_trace_id(self, phase: str, bump_weight_version: bool = True) -> str:
        next_weight_version = self._next_weight_version_hint(bump_weight_version=bump_weight_version)
        return f"{phase}-step{self.global_step}-rank{self.rank}" f"-nextwv{next_weight_version}-ts{int(time.time() * 1000)}"

    def _log_colocate_trace(self, stage: str, trace_id: str, **fields) -> None:
        return

    def _wait_validate_gate(self):
        """Block until validate-reuse gate allows proceeding."""
        while True:
            if self._validate_reuse_sync is not None:
                self._validate_reuse_sync.try_sync()
            gate_decision = self._validate_reuse_sync.wait_idle() if self._validate_reuse_sync is not None else ValidateGateDecision.PROCEED
            if gate_decision is ValidateGateDecision.RETRY_SYNC:
                time.sleep(SYNC_RETRY_SLEEP_S)
                continue
            break

    def _colocate_offload(self, trace_id: str):
        """Rank-0 calls offload_for_train, broadcasts errors to all ranks."""
        error = None
        weights_offloaded = False
        self._log_colocate_trace("offload_start", trace_id=trace_id)
        if self.rank == 0:
            try:
                weights_offloaded = bool(
                    ray.get(
                        self.rollout_manager.offload_for_train.remote(
                            timeout_s=self._colocate_timeout_s,
                            trace_id=trace_id,
                        )
                    )
                )
            except Exception as e:
                logger.error(f"[Trainer rank=0] offload_for_train failed: {e}")
                error = e
        error = self._broadcast_rank0_error(error)
        if error is not None:
            self._log_colocate_trace("offload_failed", trace_id=trace_id, error=repr(error))
            raise error
        self._colocate_scope_weights_offloaded = self._broadcast_rank0_bool(weights_offloaded)
        self._log_colocate_trace("offload_done", trace_id=trace_id)

    def _colocate_resume(self, trace_id: str):
        """Rank-0 calls resume_after_sync, broadcasts errors to all ranks."""
        error = None
        self._log_colocate_trace("resume_start", trace_id=trace_id)
        if self.rank == 0:
            try:
                ray.get(self.rollout_manager.resume_after_sync.remote(timeout_s=self._colocate_timeout_s, trace_id=trace_id))
            except Exception as e:
                logger.error(f"[Trainer rank=0] resume_after_sync failed: {e}")
                error = e
        error = self._broadcast_rank0_error(error)
        if error is not None:
            self._log_colocate_trace("resume_failed", trace_id=trace_id, error=repr(error))
            raise error
        self._log_colocate_trace("resume_done", trace_id=trace_id)

    @contextlib.contextmanager
    def _colocate_offload_scope(self, label: str, trace_id: str):
        """Ensure colocated rollout memory is resumed after offload."""
        self._log_colocate_trace("offload_scope_enter", trace_id=trace_id, label=label)
        self._colocate_offload(trace_id=trace_id)
        primary_error = None
        try:
            yield
        except Exception as e:
            primary_error = e
            try:
                self._log_colocate_trace("offload_scope_error", trace_id=trace_id, label=label, error=repr(e))
            except Exception:
                logger.opt(exception=True).debug(f"[Trainer rank={self.rank}] trace logging failed in offload_scope_error label={label}")
            raise
        finally:
            try:
                self._colocate_resume(trace_id=trace_id)
            except Exception as resume_error:
                if primary_error is None:
                    self._colocate_scope_weights_offloaded = False
                    raise
                logger.error(f"[Trainer rank={self.rank}] resume_after_sync failed during {label} cleanup: {resume_error}")
                try:
                    self._log_colocate_trace("offload_scope_resume_error", trace_id=trace_id, label=label, error=repr(resume_error))
                except Exception:
                    logger.opt(exception=True).debug(
                        f"[Trainer rank={self.rank}] trace logging failed in offload_scope_resume_error label={label}"
                    )
            self._colocate_scope_weights_offloaded = False
            try:
                self._log_colocate_trace("offload_scope_exit", trace_id=trace_id, label=label)
            except Exception:
                logger.opt(exception=True).debug(f"[Trainer rank={self.rank}] trace logging failed in offload_scope_exit label={label}")

    def update_rollout_weight(self, trace_id: str | None = None):
        """Sync trainer weights to rollout workers.

        In colocated mode the train loop manages offload/resume around this call.
        """
        trace_id = trace_id or self._build_colocate_trace_id("sync", bump_weight_version=True)
        rollout_workers = ray.get(self.rollout_manager.get_rollout_worker_on_tp0.remote())
        self._sync_rollout_workers(rollout_workers, trace_id=trace_id)

    def _sync_rollout_workers(self, rollout_workers, tensor_workers=None, bump_weight_version=True, trace_id: str | None = None):
        assert self.param_sync is not None, "must setup param sync first"
        trace_id = trace_id or self._build_colocate_trace_id("sync", bump_weight_version=bump_weight_version)
        tensor_workers = tensor_workers or []
        if not rollout_workers and not tensor_workers:
            return
        sync_start = time.monotonic()
        current_weight_version = int(getattr(self.param_sync, "weight_version", -1))
        target_weight_version = self._next_weight_version_hint(bump_weight_version=bump_weight_version)
        self._log_colocate_trace(
            "sync_start",
            trace_id=trace_id,
            current_weight_version=current_weight_version,
            target_weight_version=target_weight_version,
            rollout_worker_count=len(rollout_workers),
            tensor_worker_count=len(tensor_workers),
            bump_weight_version=int(bool(bump_weight_version)),
            backend=getattr(self.param_sync, "_sync_backend", "distributed"),
        )

        # Load actor model to GPU before weight sync (needed when param_offload=True)
        if self.actor_worker._is_offload_param:
            from siirl.utils.megatron.megatron_utils import load_megatron_model_to_gpu

            load_megatron_model_to_gpu(self.actor_worker.actor_module, load_grad=False)
            self._log_colocate_trace("sync_actor_loaded_to_gpu", trace_id=trace_id)

        # If rollout WEIGHTS were released, onload them before IPC update.
        # update_weights_from_tensor requires destination weights to be resident.
        is_colocate = self.config.trainer.colocate and isinstance(self.param_sync, ParamSyncColocated)
        needs_onload_for_sync = is_colocate and bool(getattr(self, "_colocate_scope_weights_offloaded", False))
        if needs_onload_for_sync:
            rpc_timeout_s = max(1, int(getattr(self.config.trainer, "param_sync_rpc_timeout_s", 120)))
            onload_error = None
            if self.rank == 0:
                try:
                    self._log_colocate_trace("sync_onload_weights_start", trace_id=trace_id, timeout_s=rpc_timeout_s)
                    ray.get(self.rollout_manager.onload_weights_for_sync.remote(timeout_s=rpc_timeout_s, trace_id=trace_id))
                except Exception as e:
                    logger.error(f"[Trainer rank=0] onload_weights_for_sync failed: {e}")
                    onload_error = e
            onload_error = self._broadcast_rank0_error(onload_error)
            if onload_error is not None:
                self._log_colocate_trace("sync_onload_weights_failed", trace_id=trace_id, error=repr(onload_error))
                raise onload_error
            self._log_colocate_trace("sync_onload_weights_done", trace_id=trace_id, timeout_s=rpc_timeout_s)

        try:
            if isinstance(self.param_sync, ParamSyncDistributed):
                if rollout_workers and any(not self.param_sync.has_connected_to_actor(x) for x in rollout_workers):
                    self._log_colocate_trace("sync_setup_group_start", trace_id=trace_id)
                    if isinstance(self.param_sync, ParamSyncColocated):
                        rollout_topology = ray.get(self.rollout_manager.get_colocate_topology_snapshot.remote())
                        self.param_sync.setup_param_sync_group(rollout_workers, rollout_topology=rollout_topology)
                    else:
                        self.param_sync.setup_param_sync_group(rollout_workers)
                    self._log_colocate_trace("sync_setup_group_done", trace_id=trace_id)
                self.param_sync.update_weights_mixed(
                    rollout_workers,
                    tensor_workers,
                    bump_weight_version=bump_weight_version,
                    trace_id=trace_id,
                )
            else:
                self.param_sync.update_weights()
            current_weight_version = int(getattr(self.param_sync, "weight_version", -1))
            self._log_colocate_trace(
                "sync_done",
                trace_id=trace_id,
                elapsed_ms=round((time.monotonic() - sync_start) * 1000, 2),
                current_weight_version=current_weight_version,
            )
        finally:
            # Ensure model is offloaded even on sync failure to avoid GPU memory leak
            if self.actor_worker._is_offload_param:
                offload_megatron_model_to_cpu(self.actor_worker.actor_module)
                get_torch_device().empty_cache()
                self._log_colocate_trace("sync_actor_offloaded_to_cpu", trace_id=trace_id)

    def _get_regular_rollout_workers(self) -> list:
        if self.rollout_manager is None:
            return []
        rpc_timeout_s = max(1, int(getattr(self.config.trainer, "param_sync_rpc_timeout_s", 120)))
        return ray.get(self.rollout_manager.get_rollout_worker_on_tp0.remote(), timeout=rpc_timeout_s)

    def _ensure_regular_rollout_workers(self, regular_workers) -> None:
        if isinstance(self.param_sync, ParamSyncDistributed):
            if regular_workers and any(not self.param_sync.has_connected_to_actor(x) for x in regular_workers):
                if isinstance(self.param_sync, ParamSyncColocated):
                    rollout_topology = ray.get(self.rollout_manager.get_colocate_topology_snapshot.remote())
                    self.param_sync.setup_param_sync_group(regular_workers, rollout_topology=rollout_topology)
                else:
                    self.param_sync.setup_param_sync_group(regular_workers)
            return
        self._sync_rollout_workers(regular_workers, bump_weight_version=False)

    def _maybe_init_validate_reuse_sync(self) -> None:
        if self.rollout_manager is None or self.param_sync is None:
            self._validate_reuse_sync = None
            return
        self._validate_reuse_sync = ValidateReuseTrainerSync(
            rollout_manager=self.rollout_manager,
            rank=self.rank,
            config=self.config,
            sync_workers_fn=self._sync_rollout_workers,
            get_regular_workers_fn=self._get_regular_rollout_workers,
            ensure_regular_workers_fn=self._ensure_regular_rollout_workers,
            get_current_weight_version_fn=self.get_current_weight_version,
        )

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

        # Optional memory profiling (enabled via SIIRL_MEMORY_PROFILE=1)
        memory_profiler = MemoryProfiler.create_if_enabled(self.global_step, self.rank)

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

            # Memory optimization: offload actor before ref inference to avoid peak memory
            # when both actor and ref models are on GPU simultaneously
            if self.actor_worker._is_offload_param:
                offload_megatron_model_to_cpu(self.actor_worker.actor_module)
                get_torch_device().empty_cache()

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

                # Throughput is recalculated in train() with corrected step_interval.
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
                    _earliest_start = rollout_timing.pop("_earliest_rollout_start_at", None)  # noqa: F841
                    if rollout_timing:
                        self.metric_client.submit_metric(rollout_timing, self.dp_world_size)

                # Metrics from megatron_actor.py already include prefixes.
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

        # Export memory snapshot if profiling enabled
        if memory_profiler:
            memory_profiler.export_snapshot()

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

    def _report_completed(self):
        """Report training completion to coordinator."""
        if not self.coordinator:
            return
        try:
            ray.get(self.coordinator.report_completed.remote(source=f"trainer_{self.rank}"))
        except Exception as e:
            logger.warning(f"[Trainer rank={self.rank}] Failed to report completion: {e}")
            logger.warning(f"[Trainer rank={self.rank}] Traceback:\n{traceback.format_exc()}")

    def _pop_validation_time_once(self, step: int, step_start: float, step_end: float) -> float:
        """Read and reset validation-time overlap for one step window (rank 0 only)."""
        if self.rank != 0 or self.rollout_manager is None:
            return 0.0
        try:
            val_time = float(
                ray.get(
                    self.rollout_manager.pop_validation_time_overlap.remote(
                        float(step_start),
                        float(step_end),
                        int(step),
                    )
                )
            )
            if val_time > 0:
                logger.info(f"[Trainer rank=0] Validation time pop step={int(step)} val_time={val_time:.2f}s")
            return val_time
        except Exception as e:
            logger.warning(f"[Trainer rank={self.rank}] Failed to fetch validation time: {e}")
            return 0.0

    def _compute_step_timing(self, train_e2e: float, validation_excluded: float) -> dict[str, float]:
        """Compute step timing with validation/checkpoint exclusions.

        When validation is excluded, the naive subtraction ``train_e2e - val``
        removes the normal generation-pipeline lag that exists on every step
        (the time ``get_batch`` waits for the inference server to finish the
        previous batch).  This makes validation steps report a *shorter*
        step_interval — and therefore *higher* throughput — than non-validation
        steps, creating a periodic throughput spike.

        Fix: clamp the validation-adjusted step_interval to be no less than
        the recent EMA of non-validation step intervals.  On non-validation
        steps, update the EMA so it tracks the true steady-state step time.
        """
        step_interval_raw = max(train_e2e - max(validation_excluded, 0.0), 0.0)
        checkpoint_excluded = min(self._pending_ckpt_excluded_time, step_interval_raw)
        step_interval = max(step_interval_raw - checkpoint_excluded, 0.0)

        is_validation_step = validation_excluded > 0.0
        if is_validation_step and self._step_interval_ema > 0:
            # Clamp: validation step should not report a shorter interval than
            # the recent non-validation baseline.
            step_interval = max(step_interval, self._step_interval_ema)
            step_interval_raw = max(step_interval_raw, self._step_interval_ema)
        elif not is_validation_step and step_interval > 0:
            # Update EMA from non-validation steps only.
            alpha = self._step_interval_ema_alpha
            if self._step_interval_ema <= 0:
                self._step_interval_ema = step_interval  # seed
            else:
                self._step_interval_ema = alpha * step_interval + (1 - alpha) * self._step_interval_ema

        return {
            "validation_excluded": validation_excluded,
            "step_interval_raw": step_interval_raw,
            "checkpoint_excluded": checkpoint_excluded,
            "step_interval": step_interval,
        }

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
        total_training_steps = int(getattr(self.config.actor_ref.actor.optim, "total_training_steps", 0) or 0)

        try:
            # Colocated bootstrap: push trainer weights to rollout before the first
            # generation so that a resumed checkpoint doesn't produce stale data.
            if self._is_colocate:
                logger.info(f"[Trainer rank={self.rank}] Colocated bootstrap: syncing weights before first generation")
                bootstrap_trace_id = self._build_colocate_trace_id("bootstrap", bump_weight_version=True)
                with self._colocate_offload_scope("bootstrap", trace_id=bootstrap_trace_id):
                    self.update_rollout_weight(trace_id=bootstrap_trace_id)
            while True:
                # Check stop signal
                if self._check_should_stop():
                    logger.info(f"[Trainer rank={self.rank}] Stop signal received, exiting...")
                    break
                if total_training_steps > 0 and self.global_step >= total_training_steps:
                    logger.info(
                        f"[Trainer rank={self.rank}] Reached total training steps "
                        f"({self.global_step}/{total_training_steps}), exiting..."
                    )
                    if self.rank == 0:
                        self._report_completed()
                    break

                train_e2e_start_time = time.time()

                if self._is_colocate:
                    # Colocated: generate -> offload -> train -> sync -> resume
                    # Rollout uses weights synced at the end of the previous step.
                    if self.rank == 0:
                        ray.get(self.rollout_manager.next_rollout.remote())

                    if self._validate_reuse_sync is not None:
                        self._validate_reuse_sync.try_sync()

                    with Timer("get_batch") as get_batch_timer:
                        while (batch_data := self.get_batch(batch_size)) is None:
                            did_sync = self._validate_reuse_sync.try_sync() if self._validate_reuse_sync is not None else False
                            gate_decision = (
                                self._validate_reuse_sync.wait_idle()
                                if self._validate_reuse_sync is not None
                                else ValidateGateDecision.PROCEED
                            )
                            if gate_decision is ValidateGateDecision.RETRY_SYNC:
                                time.sleep(SYNC_RETRY_SLEEP_S)
                                continue
                            if not did_sync:
                                time.sleep(TRAIN_NO_BATCH_BACKOFF_S)

                    self._wait_validate_gate()
                    step_trace_id = self._build_colocate_trace_id("step", bump_weight_version=True)
                    self._log_colocate_trace("step_pipeline_start", trace_id=step_trace_id)
                    with self._colocate_offload_scope(f"step={self.global_step}", trace_id=step_trace_id):
                        self.train_step(batch_data)
                        self._log_colocate_trace("step_train_done", trace_id=step_trace_id)

                        with Timer("weight_sync") as weight_sync_timer:
                            self.update_rollout_weight(trace_id=step_trace_id)
                    self._log_colocate_trace("step_pipeline_done", trace_id=step_trace_id)
                else:
                    # Separated: sync -> generate -> get_batch -> train (original order)
                    with Timer("weight_sync") as weight_sync_timer:
                        self.update_rollout_weight()

                    if self.rank == 0:
                        ray.get(self.rollout_manager.next_rollout.remote())

                    if self._validate_reuse_sync is not None:
                        self._validate_reuse_sync.try_sync()

                    with Timer("get_batch") as get_batch_timer:
                        while (batch_data := self.get_batch(batch_size)) is None:
                            did_sync = self._validate_reuse_sync.try_sync() if self._validate_reuse_sync is not None else False
                            gate_decision = (
                                self._validate_reuse_sync.wait_idle()
                                if self._validate_reuse_sync is not None
                                else ValidateGateDecision.PROCEED
                            )
                            if gate_decision is ValidateGateDecision.RETRY_SYNC:
                                time.sleep(SYNC_RETRY_SLEEP_S)
                                continue
                            if not did_sync:
                                time.sleep(TRAIN_NO_BATCH_BACKOFF_S)

                    self._wait_validate_gate()
                    self.train_step(batch_data)

                # Wait for all metric submissions and aggregate (rank=0 does the logging)
                # Only TP rank 0 and PP rank 0 submit metrics, so only they need to wait
                if self.metric_client is not None and self.should_submit_metrics:
                    try:
                        self.metric_client.wait_submit()
                    except Exception as e:
                        logger.warning(f"[Trainer rank={self.rank}] Metric submission wait failed: {e}")

                # Barrier sync to ensure all ranks are synchronized
                dist.barrier()

                train_e2e_end_time = time.time()
                train_e2e = train_e2e_end_time - train_e2e_start_time
                val_time = self._pop_validation_time_once(self.global_step, train_e2e_start_time, train_e2e_end_time)
                step_timing = self._compute_step_timing(train_e2e, val_time)

                # Only rank=0 (global rank) aggregates and logs to tracker
                if self.rank == 0 and self.tracker is not None and self.metric_client is not None:
                    try:
                        from siirl.utils.metrics import restore_weighted_metrics

                        aggregated_metrics = self.metric_client.wait_final_res()
                        aggregated_metrics = restore_weighted_metrics(aggregated_metrics)

                        aggregated_metrics["training/global_step"] = self.global_step
                        aggregated_metrics["perf/delta_time/weight_sync"] = weight_sync_timer.elapsed
                        aggregated_metrics["perf/delta_time/get_batch"] = get_batch_timer.elapsed

                        # Recompute throughput after aggregation to avoid per-rank bias.
                        total_tokens = aggregated_metrics.get("perf/total_num_tokens", 0)
                        # Exclude validation time and prior checkpoint save time.
                        aggregated_metrics["perf/delta_time/validation_excluded"] = step_timing["validation_excluded"]
                        aggregated_metrics["perf/delta_time/step_interval_raw"] = step_timing["step_interval_raw"]
                        aggregated_metrics["perf/delta_time/checkpoint_save_excluded"] = step_timing["checkpoint_excluded"]
                        aggregated_metrics["perf/delta_time/step_interval"] = step_timing["step_interval"]
                        aggregated_metrics["perf/time_per_step"] = step_timing["step_interval"]
                        aggregated_metrics["perf/time_per_step_max"] = step_timing["step_interval"]

                        if self.config.trainer.colocate:
                            total_gpus = self.world_size
                            if total_gpus <= 0:
                                total_gpus = max(
                                    self.config.trainer.actor_gpus,
                                    self.config.trainer.nnodes * self.config.trainer.n_gpus_per_node,
                                )
                        else:
                            total_gpus = self.config.trainer.actor_gpus + self.config.trainer.rollout_gpus
                        total_gpus = max(total_gpus, 1)
                        aggregated_metrics["perf/total_gpus"] = total_gpus
                        aggregated_metrics["perf/gpu_mode"] = "colocated" if self.config.trainer.colocate else "separated"
                        if step_timing["step_interval"] > 0 and total_tokens > 0:
                            aggregated_metrics["perf/throughput"] = total_tokens / (step_timing["step_interval"] * total_gpus)
                        self.tracker.log(aggregated_metrics, step=self.global_step)

                        # get rollout validate metrics
                        val_metrics = ray.get(self.rollout_manager.get_metrics.remote())
                        for metrics, global_step in val_metrics:
                            self.tracker.log(metrics, global_step)

                    except Exception as e:
                        logger.warning(f"[Trainer rank={self.rank}] Metric aggregation failed: {e}")

                self.global_step += 1

                checkpoint_save_time = 0.0
                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    logger.info(f"[Trainer rank={self.rank}] Saving checkpoint at step {self.global_step}")
                    with Timer("save_checkpoint") as save_checkpoint_timer:
                        self.checkpoint_manager.save_checkpoint(self.global_step)
                    checkpoint_save_time = save_checkpoint_timer.elapsed

                # Carry save time into next-step exclusion.
                self._pending_ckpt_excluded_time = checkpoint_save_time

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
                with contextlib.suppress(Exception):
                    self.metric_client.wait_submit()

            # Close MetricTracker (only rank=0 has one)
            if self.tracker is not None:
                try:
                    self.tracker.finish()
                    logger.info(f"[Trainer rank={self.rank}] MetricTracker closed")
                except Exception as e:
                    logger.warning(f"[Trainer rank={self.rank}] Error closing MetricTracker: {e}")

        logger.info(f"[Trainer rank={self.rank}] Training loop ended at step {self.global_step}")
