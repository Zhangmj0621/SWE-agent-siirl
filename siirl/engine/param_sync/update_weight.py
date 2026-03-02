import gc
import socket
import time
import traceback
from abc import abstractmethod
from collections import defaultdict, deque
from collections.abc import Sequence
from datetime import timedelta

import ray
import torch
import torch.distributed as dist
from loguru import logger
from mbridge.core.bridge import Bridge
from megatron.core import mpu
from ray.actor import ActorHandle
from tqdm import tqdm

from siirl.params.training_args import SiiRLArguments
from siirl.utils.distributed_utils import get_gloo_group, init_process_group

from . import mbridge_patch  # noqa: F401
from .route_model import LaneRoute, RoutePlan, compute_route_plan, is_lane_leader, is_lane_participant, validate_route_plan


def _import_flattened_tensor_bucket():
    """Import FlattenedTensorBucket with dual-path compatibility."""
    try:
        from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
    except ImportError:
        from sglang.srt.model_executor.model_runner import FlattenedTensorBucket
    return FlattenedTensorBucket


def _ensure_monkey_patched():
    """Apply monkey_patch_torch_reductions once so tensor serialization uses CUDA IPC handles."""
    if getattr(_ensure_monkey_patched, "_done", False):
        return
    try:
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
    except ImportError:
        from sglang.srt.utils import monkey_patch_torch_reductions
    monkey_patch_torch_reductions()
    _ensure_monkey_patched._done = True


def _serialize_bucket_ipc(
    named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> tuple[list[bytes], list[dict[str, object]]]:
    """Serialize a bucket of named tensors via FlattenedTensorBucket + IPC handles.

    Returns serialized payloads and long-lived flattened bucket objects.
    """
    from sglang.srt.utils.common import MultiprocessingSerializer

    _ensure_monkey_patched()
    FlattenedTensorBucket = _import_flattened_tensor_bucket()

    # Group tensors by dtype (unless FlattenedTensorBucket handles mixed dtypes).
    if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
        groups: dict[str, list[tuple[str, torch.Tensor]]] = {"mixed": list(named_tensors)}
    else:
        groups = defaultdict(list)
        for name, tensor in named_tensors:
            groups[tensor.dtype].append((name, tensor))

    serialized: list[bytes] = []
    long_lived_buckets: list[dict[str, object]] = []
    for _dtype_key, tensors in groups.items():
        bucket = FlattenedTensorBucket(named_tensors=tensors)
        flattened_tensor = bucket.get_flattened_tensor()
        metadata = bucket.get_metadata()
        flattened_data = {
            "flattened_tensor": flattened_tensor,
            "metadata": metadata,
        }
        long_lived_buckets.append(flattened_data)
        # output_str=False -> bytes with CUDA IPC handle (no base64 overhead)
        serialized_part = MultiprocessingSerializer.serialize(flattened_data, output_str=False)
        serialized.append(serialized_part)
    return serialized, long_lived_buckets


def _shared_cache_len() -> int:
    try:
        from torch.multiprocessing import reductions

        cache = getattr(reductions, "shared_cache", None)
        if cache is None:
            return -1
        return len(cache)
    except Exception:
        return -1


def _cuda_memory_snapshot() -> dict[str, float | int | str]:
    snapshot: dict[str, float | int | str] = {}
    if not torch.cuda.is_available():
        snapshot["cuda_available"] = 0
        return snapshot

    try:
        device = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        stats = torch.cuda.memory_stats(device)
        snapshot.update(
            {
                "cuda_available": 1,
                "device": int(device),
                "allocated_gb": round(torch.cuda.memory_allocated(device) / (1024**3), 3),
                "reserved_gb": round(torch.cuda.memory_reserved(device) / (1024**3), 3),
                "free_gb": round(free_bytes / (1024**3), 3),
                "total_gb": round(total_bytes / (1024**3), 3),
                "active_gb": round(stats.get("active_bytes.all.current", 0) / (1024**3), 3),
                "inactive_split_gb": round(stats.get("inactive_split_bytes.all.current", 0) / (1024**3), 3),
                "shared_cache_len": _shared_cache_len(),
            }
        )
    except Exception as e:
        snapshot["snapshot_error"] = repr(e)
    return snapshot


def _tensor_bytes(named_tensors: Sequence[tuple[str, torch.Tensor]]) -> int:
    return sum(tensor.numel() * tensor.element_size() for _, tensor in named_tensors)


class ParamSyncInterface:
    def __init__(self, config: SiiRLArguments, model: Sequence[torch.nn.Module], bridge: Bridge):
        self.config = config
        self.model = model
        self.bridge = bridge
        self.weight_version = 0
        self._model_update_groups = None

    @abstractmethod
    def setup_param_sync_group(self, rollout_workers: Sequence[ActorHandle]):
        pass

    @abstractmethod
    def update_weights(self) -> None:
        pass


class ParamSyncDistributed(ParamSyncInterface):
    def __init__(self, config: SiiRLArguments, model: Sequence[torch.nn.Module], bridge: Bridge):
        super().__init__(config, model, bridge)
        self.rollout_worker_connected = set()
        self.rollout_workers = []
        self.tensor_rollout_workers = []
        self._connected_rollout_workers: list[ActorHandle] = []
        self._connected_rollout_worker_ids: set[str] = set()
        self.param_sync_unhealthy = False
        self._current_sync_bucket_count = 0
        self._sync_total_ms_samples = deque(maxlen=256)
        self._sync_success_count = 0
        self._sync_failure_count = 0

    def _rpc_timeout_s(self) -> int:
        return max(1, int(getattr(self.config.trainer, "param_sync_rpc_timeout_s", 120)))

    def _sync_metric_group_name(self) -> str:
        return getattr(self, "_group_name", self.__class__.__name__)

    def _record_sync_bucket(self) -> None:
        self._current_sync_bucket_count += 1

    def _log_colocate_mem_debug(self, stage: str, **fields) -> None:
        return

    @staticmethod
    def _percentile(values: Sequence[float], quantile: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        sorted_values = sorted(values)
        idx = int(round((len(sorted_values) - 1) * quantile))
        idx = max(0, min(idx, len(sorted_values) - 1))
        return sorted_values[idx]

    def _log_sync_metrics(self, elapsed_ms: float, success: bool) -> None:
        self._sync_total_ms_samples.append(elapsed_ms)
        if success:
            self._sync_success_count += 1
        else:
            self._sync_failure_count += 1
        total = self._sync_success_count + self._sync_failure_count
        failure_rate = self._sync_failure_count / max(1, total)
        p50 = self._percentile(self._sync_total_ms_samples, 0.50)
        p95 = self._percentile(self._sync_total_ms_samples, 0.95)
        logger.info(
            f"[{self._sync_metric_group_name()}] "
            f"sync_total_ms={elapsed_ms:.2f} "
            f"sync_total_ms_p50={p50:.2f} "
            f"sync_total_ms_p95={p95:.2f} "
            f"sync_failure_rate={failure_rate:.4f} "
            f"sync_bucket_count={self._current_sync_bucket_count}"
        )

    def _normalize_rollout_workers(self, rollout_workers: Sequence[ActorHandle]) -> list[ActorHandle]:
        deduped_workers: list[ActorHandle] = []
        seen_actor_ids: set[str] = set()
        for worker in rollout_workers:
            actor_id_hex = worker._actor_id.hex()
            if actor_id_hex in seen_actor_ids:
                continue
            seen_actor_ids.add(actor_id_hex)
            deduped_workers.append(worker)
        return deduped_workers

    def has_connected_to_actor(self, actor: ActorHandle):
        return actor._actor_id.hex() in self.rollout_worker_connected

    def update_rollout_worker_connected(self, new_actors: Sequence[ActorHandle]):
        if not isinstance(new_actors, Sequence):
            new_actors = [new_actors]
        for actor in new_actors:
            actor_id_hex = actor._actor_id.hex()
            self.rollout_worker_connected.add(actor_id_hex)

    def setup_param_sync_group(self, rollout_workers: Sequence[ActorHandle]):
        # from Train DP 0 to all worker
        # each pp rank has its own group
        normalized_workers = self._normalize_rollout_workers(rollout_workers)
        self.rollout_workers = normalized_workers
        self._is_pp_src_rank = mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        if self._is_pp_src_rank:
            self._group_name = f"param_sync-pp_{pp_rank}"

        if self._is_pp_src_rank:
            if self.param_sync_unhealthy:
                raise RuntimeError(f"[{self._group_name}] Param sync is unhealthy and requires process restart")

            new_worker_ids = {worker._actor_id.hex() for worker in normalized_workers}
            if self._model_update_groups is not None and new_worker_ids == self._connected_rollout_worker_ids:
                return

            old_workers = list(self._connected_rollout_workers)
            old_worker_ids = set(self._connected_rollout_worker_ids)
            old_group = self._model_update_groups
            timeout_s = self._rpc_timeout_s()

            try:
                if old_group is not None:
                    disconnect_rollout_workers_from_distributed(self._group_name, old_group, old_workers, timeout_s=timeout_s)
                    self._model_update_groups = None

                self._model_update_groups = connect_rollout_workers_from_distributed(
                    self.config,
                    self._group_name,
                    normalized_workers,
                    timeout_s=timeout_s,
                )
                self._connected_rollout_workers = list(normalized_workers)
                self._connected_rollout_worker_ids = new_worker_ids
                self.rollout_worker_connected.clear()
                self.update_rollout_worker_connected(normalized_workers)
            except Exception:
                logger.exception(f"[{self._group_name}] Failed to setup param sync group")
                rollback_ok = False

                if old_workers:
                    try:
                        self._model_update_groups = connect_rollout_workers_from_distributed(
                            self.config,
                            self._group_name,
                            old_workers,
                            timeout_s=timeout_s,
                        )
                        self._connected_rollout_workers = old_workers
                        self._connected_rollout_worker_ids = old_worker_ids
                        self.rollout_worker_connected.clear()
                        self.update_rollout_worker_connected(old_workers)
                        rollback_ok = True
                        logger.warning(f"[{self._group_name}] Rolled back to previous param sync workers")
                    except Exception:
                        logger.exception(f"[{self._group_name}] Rollback failed after setup_param_sync_group failure")

                if not rollback_ok:
                    self._model_update_groups = None
                    self._connected_rollout_workers = []
                    self._connected_rollout_worker_ids.clear()
                    self.rollout_worker_connected.clear()
                    self.param_sync_unhealthy = True
                    logger.error(f"[{self._group_name}] Marked param sync as unhealthy after unrecoverable setup failure")
                raise
            logger.info(f"self._model_update_groups=={self._model_update_groups.size()}")

    def _update_weights_use_mbridge(self) -> None:
        """
        Pause → flush → all params → continue. Progress on PP source.
        """

        buffer_size = 0
        converted_named_tensors = []
        exported_params = 0
        exported_bytes = 0
        pbar = (
            tqdm(
                desc=f"[{self._group_name}] Update weights",
                total=0,
                dynamic_ncols=True,
                leave=False,
            )
            if self._is_pp_src_rank
            else None
        )
        try:
            generator = self.bridge._export_weights_in_current_pipeline_stage(self.model)
            for name, param in generator:
                exported_params += 1
                exported_bytes += param.numel() * param.element_size()
                buffer_size = self._update_param_sync_bucket(name, param, converted_named_tensors, buffer_size, pbar)

            if converted_named_tensors:
                self._update_bucket_weights(converted_named_tensors, pbar=pbar)
            self._log_colocate_mem_debug(
                stage="export_done",
                exported_params=exported_params,
                exported_mb=round(exported_bytes / (1024**2), 2),
            )
        finally:
            if pbar is not None:
                pbar.close()

    def _update_weights_naive(self) -> None:
        raise NotImplementedError("_update_weights_naive is not implemented, please set use_mbridge=True")

    @torch.no_grad()
    def update_weights(self) -> None:
        self.update_weights_mixed(self.rollout_workers, [])

    def _all_target_workers(self) -> list[ActorHandle]:
        return [*self.rollout_workers, *self.tensor_rollout_workers]

    @torch.no_grad()
    def update_weights_mixed(
        self,
        rollout_workers: Sequence[ActorHandle],
        tensor_rollout_workers: Sequence[ActorHandle] | None = None,
        bump_weight_version: bool = True,
        trace_id: str | None = None,
    ) -> None:
        if self.param_sync_unhealthy:
            raise RuntimeError("Param sync group is unhealthy; refusing to sync rollout weights")

        self.rollout_workers = list(rollout_workers)
        self.tensor_rollout_workers = list(tensor_rollout_workers or [])
        all_workers = self._all_target_workers()
        if not all_workers:
            return

        sync_started_at = time.monotonic()
        sync_success = False
        self._current_sync_bucket_count = 0
        timeout_s = self._rpc_timeout_s()
        try:
            if bump_weight_version:
                self.weight_version += 1
            if dist.get_rank() == 0:
                if self.tensor_rollout_workers:
                    logger.info(
                        f"[ParamSyncDistributed] Mixed weight sync: distributed_workers={len(self.rollout_workers)} "
                        f"tensor_workers={len(self.tensor_rollout_workers)} "
                        f"weight_version={self.weight_version} bump={bump_weight_version}"
                    )
                ray.get([worker.pause_generation.remote() for worker in all_workers], timeout=timeout_s)
                ray.get([worker.flush_cache.remote() for worker in all_workers], timeout=timeout_s)
            dist.barrier(group=get_gloo_group())

            if self.bridge is not None:
                self._update_weights_use_mbridge()
            else:
                self._update_weights_naive()

            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                self._check_weight_version()
                ray.get([worker.continue_generation.remote() for worker in all_workers], timeout=timeout_s)
            dist.barrier(group=get_gloo_group())
            sync_success = True
        finally:
            if dist.get_rank() == 0:
                elapsed_ms = (time.monotonic() - sync_started_at) * 1000
                self._log_sync_metrics(elapsed_ms, success=sync_success)

    def _check_weight_version(self):
        workers = self._all_target_workers()
        version_list = ray.get([worker.weight_version.remote() for worker in workers], timeout=self._rpc_timeout_s())
        for idx, v in enumerate(version_list):
            if v != self.weight_version:
                raise ValueError(
                    f"Weight version mismatch!, {idx}th rollout weight version: {v}, trainer weight version: {self.weight_version}"
                )
        return True

    def _update_param_sync_bucket(
        self,
        name: str,
        param: torch.nn.Parameter,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
        pbar: tqdm | None = None,
    ):
        if not self._is_pp_src_rank:
            return buffer_size
        buffer_limit = self.config.trainer.param_sync_buffer_size
        param_size = param.numel() * param.element_size()
        if buffer_size + param_size > buffer_limit:
            self._update_bucket_weights(converted_named_tensors, pbar=pbar)
            buffer_size = 0
        converted_named_tensors.append((name, param))
        buffer_size += param_size
        return buffer_size

    def _update_bucket_weights(
        self,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        pbar: tqdm | None = None,
    ) -> None:
        refs = []
        if self._is_pp_src_rank and self.rollout_workers:
            refs.extend(
                update_weights_from_distributed(
                    self._group_name,
                    self._model_update_groups,
                    self.weight_version,
                    self.rollout_workers,
                    converted_named_tensors,
                )
            )
        if self._is_pp_src_rank and self.tensor_rollout_workers:
            refs.extend(
                update_weights_from_tensor(
                    self.weight_version,
                    self.tensor_rollout_workers,
                    converted_named_tensors,
                    self.config.rollout.tensor_model_parallel_size,
                )
            )

        if refs:
            ray.get(refs, timeout=self._rpc_timeout_s())
            self._record_sync_bucket()
        converted_named_tensors.clear()
        if pbar is not None:
            pbar.update(1)


class ParamSyncColocated(ParamSyncDistributed):
    """Colocated weight sync using topology-aware lane routing.

    Flattened backend:
    - Each lane maps to one TP0 rollout worker.
    - Each source rank only syncs its colocated lane payload.
    - Lane leader performs one RPC per bucket part.

    Tensor backend keeps legacy rank0 sender behavior.
    """

    _SUPPORTED_BACKENDS = {"tensor", "flattened_bucket"}
    _MAX_SYNC_RETRIES = 2
    _MIN_RECLAIMABLE_BYTES = 1 << 30

    def __init__(self, config: SiiRLArguments, model: Sequence[torch.nn.Module], bridge: Bridge):
        super().__init__(config, model, bridge)
        self._sync_backend = self._resolve_sync_backend()
        self._max_sync_retries = self._MAX_SYNC_RETRIES
        self._tensor_path_logged = False
        self._trace_id = ""
        # Route plan for lane-based dispatch.
        self._route_plan: RoutePlan | None = None
        self._route_source_ranks: set[int] = set()
        self._route_epoch = 0
        # IPC gather group for my lane (created once during setup).
        self._ipc_gather_group = None
        self._ipc_gather_src = -1
        self._ipc_group_ready = False
        # Cached TP size from rollout config.
        self._tp_size = max(1, int(getattr(config.rollout, "tensor_model_parallel_size", 1)))
        self._refresh_sync_context()

    def _resolve_sync_backend(self) -> str:
        backend = getattr(self.config.rollout, "colocate_param_sync_backend", "flattened_bucket")

        backend = str(backend).strip().lower()
        alias = {
            "legacy_tensor": "tensor",
            "tensor_rpc": "tensor",
            "flattened": "flattened_bucket",
            "bucket": "flattened_bucket",
        }
        backend = alias.get(backend, backend)
        if backend not in self._SUPPORTED_BACKENDS:
            logger.warning(
                f"[ParamSyncColocated] Unknown rollout.colocate_param_sync_backend={backend!r}, " "falling back to 'flattened_bucket'."
            )
            backend = "flattened_bucket"
        return backend

    def _using_flattened_bucket(self) -> bool:
        return self._sync_backend == "flattened_bucket"

    def _should_enter_bucket_sync(self) -> bool:
        """Determine whether this rank should build/sync flattened buckets.

        Rules:
        - Tensor backend: keep legacy sender behavior (_is_pp_src_rank only).
        - Flattened backend with route plan: rank must belong to route source ranks.
        - Flattened backend without route plan: legacy sender behavior.
        """
        if not self._using_flattened_bucket():
            return self._is_pp_src_rank
        if self._route_plan is not None:
            return dist.get_rank() in self._route_source_ranks
        return self._is_pp_src_rank

    def _refresh_sync_context(self):
        self._is_pp_src_rank = mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        self._group_name = f"param_sync_colocated-pp_{pp_rank}"
        self._tp_size = max(1, int(getattr(self.config.rollout, "tensor_model_parallel_size", 1)))

    def _log_colocate_mem_debug(self, stage: str, **fields) -> None:
        if not self._should_enter_bucket_sync() and not self._is_pp_src_rank:
            return
        mem = _cuda_memory_snapshot()
        merged = {
            "group": self._group_name,
            "stage": stage,
            "trace_id": self._trace_id or "na",
            "backend": self._sync_backend,
            "weight_version": self.weight_version,
            "rank": dist.get_rank(),
            **mem,
            **fields,
        }
        payload = " ".join(f"{k}={v}" for k, v in merged.items())
        logger.info(payload)

    def _clear_shared_cache(self) -> None:
        """Evict CUDA IPC handles so their pinned GPU storage can be reclaimed."""
        if not self._should_enter_bucket_sync() and not self._is_pp_src_rank:
            return
        try:
            from torch.multiprocessing import reductions

            cache = getattr(reductions, "shared_cache", None)
            if cache is not None:
                if len(cache) > 0:
                    cache.clear()
            else:
                logger.warning(
                    "group={} stage=shared_cache_clear_skipped reason=shared_cache_unavailable",
                    self._group_name,
                )
        except Exception as e:
            logger.warning(
                "group={} stage=shared_cache_clear_failed error={}",
                self._group_name,
                repr(e),
            )

        if torch.cuda.is_available():
            try:
                torch.cuda.ipc_collect()
            except Exception as e:
                logger.warning(
                    "group={} stage=ipc_collect_failed error={}",
                    self._group_name,
                    repr(e),
                )

    def _compact_cuda_cache(self, stage: str, *, force: bool = False) -> None:
        if (not self._should_enter_bucket_sync() and not self._is_pp_src_rank) or not torch.cuda.is_available():
            return
        gc.collect()
        device = torch.cuda.current_device()
        reserved = torch.cuda.memory_reserved(device)
        allocated = torch.cuda.memory_allocated(device)
        reclaimable = max(0, reserved - allocated)
        if force or reclaimable >= self._MIN_RECLAIMABLE_BYTES:
            torch.cuda.empty_cache()

    def _sync_ipc_bucket_with_retry(
        self,
        serialized_named_tensors: list[bytes],
        timeout_s: int,
        *,
        bucket_idx: int,
        part_idx: int,
        part_count: int,
    ) -> None:
        retry_limit = self._max_sync_retries
        payload_bytes = (
            len(serialized_named_tensors[0])
            if serialized_named_tensors and isinstance(serialized_named_tensors[0], (bytes, bytearray, str))
            else -1
        )
        for attempt in range(1, retry_limit + 2):
            refs = [
                worker.param_sync_from_tensor.remote(
                    serialized_named_tensors=serialized_named_tensors,
                    flush_cache=False,
                    weight_version=str(self.weight_version),
                    load_format="flattened_bucket",
                    trace_id=self._trace_id,
                    bucket_idx=bucket_idx,
                    part_idx=part_idx,
                    part_count=part_count,
                )
                for worker in self.rollout_workers
            ]
            try:
                ray.get(refs, timeout=timeout_s)
                return
            except Exception:
                if attempt > retry_limit:
                    worker_alive = worker_dead = 0
                    exitcodes: list[int | None] = []
                    debug_error = ""
                    try:
                        states = ray.get([worker.get_debug_state.remote() for worker in self.rollout_workers], timeout=5)
                        for state in states:
                            if state.get("engine_alive"):
                                worker_alive += 1
                            else:
                                worker_dead += 1
                            exitcodes.append(state.get("engine_exitcode"))
                    except Exception as e:
                        debug_error = repr(e)
                    logger.error(
                        f"[{self._group_name}] IPC bucket sync failed after retries. retry_limit={retry_limit}\n"
                        f"{traceback.format_exc()}"
                    )
                    self._log_colocate_mem_debug(
                        stage="bucket_ipc_part_sync_failed",
                        bucket_idx=bucket_idx,
                        part_idx=part_idx,
                        part_count=part_count,
                        attempt=attempt,
                        payload_bytes=payload_bytes,
                        worker_alive=worker_alive,
                        worker_dead=worker_dead,
                        worker_exitcodes=exitcodes,
                        worker_debug_error=debug_error,
                    )
                    raise
                logger.warning(f"[{self._group_name}] IPC bucket sync retry {attempt}/{retry_limit} after failure")

    def _sync_bucket_with_tensor_path(
        self,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        timeout_s: int,
        pbar: tqdm | None = None,
    ) -> None:
        if dist.get_rank() == 0 and not self._tensor_path_logged:
            logger.warning(
                f"[{self._group_name}] Using tensor sync backend for colocated mode "
                "(explicit override; default backend is flattened_bucket)."
            )
            self._tensor_path_logged = True

        bucket_idx = self._current_sync_bucket_count + 1
        refs = update_weights_from_tensor(
            self.weight_version,
            self.rollout_workers,
            converted_named_tensors,
            self.config.rollout.tensor_model_parallel_size,
            trace_id=self._trace_id,
            bucket_idx=bucket_idx,
        )
        if refs:
            ray.get(refs, timeout=timeout_s)
            self._record_sync_bucket()
        converted_named_tensors.clear()
        if pbar is not None:
            pbar.update(1)
        self._compact_cuda_cache(stage="bucket_tensor")

    # ==========================================================================
    # Route Plan Building (for lane-based dispatch)
    # ==========================================================================

    def _build_colocate_route_plan(self, rollout_topology: dict) -> RoutePlan:
        """Build and validate the colocated route plan."""
        plan = compute_route_plan(
            rollout_topology=rollout_topology,
            rollout_workers=self.rollout_workers,
            config_tp_size=self._tp_size,
            route_epoch=self._route_epoch + 1,
        )
        validate_route_plan(plan, self._group_name)
        self._route_epoch = plan.route_epoch
        self._route_source_ranks = {rank for lane in plan.lanes for rank in lane.source_ranks}
        return plan

    def _ensure_ipc_gather_groups(self, plan: RoutePlan) -> None:
        """Create one gloo gather group per lane (TP>1 only)."""
        if self._ipc_group_ready:
            return

        if plan.tp_size == 1:
            self._ipc_group_ready = True
            return

        my_rank = dist.get_rank()
        for lane in plan.lanes:
            group = dist.new_group(ranks=lane.source_ranks, backend="gloo")
            if my_rank in lane.source_ranks:
                if self._ipc_gather_group is not None:
                    raise RuntimeError(f"[{self._group_name}] rank={my_rank} belongs to multiple lanes; invalid route plan")
                self._ipc_gather_group = group
                self._ipc_gather_src = lane.leader_rank

        if my_rank in self._route_source_ranks and self._ipc_gather_group is None:
            raise RuntimeError(f"[{self._group_name}] rank={my_rank} is route participant but has no IPC gather group")

        self._ipc_group_ready = True

    def _gather_lane_payload(self, lane: LaneRoute, local_payload: bytes) -> list[bytes] | None:
        """Gather payloads from all source ranks to leader rank via gloo."""
        my_rank = dist.get_rank()

        if len(lane.source_ranks) == 1:
            if my_rank == lane.leader_rank:
                return [local_payload]
            return None

        if self._ipc_gather_group is None:
            raise RuntimeError(f"[{self._group_name}] IPC gather group not initialized for TP>1 lane")

        if my_rank == lane.leader_rank:
            gathered: list[bytes | None] = [None] * len(lane.source_ranks)
        else:
            gathered = None

        dist.gather_object(
            local_payload,
            object_gather_list=gathered,
            dst=lane.leader_rank,
            group=self._ipc_gather_group,
        )

        if my_rank != lane.leader_rank:
            return None

        if gathered is None or any(item is None for item in gathered):
            raise RuntimeError(f"lane gather returned incomplete data for leader rank={lane.leader_rank}")
        return [item for item in gathered if item is not None]

    def _sync_flattened_bucket_by_lanes(
        self,
        plan: RoutePlan,
        bucket_idx: int,
        part_idx: int,
        part_count: int,
        serialized_part: bytes,
        timeout_s: int,
    ) -> list:
        """Sync flattened bucket using lane-based routing.

        For each lane:
        1. Determine if this rank is a participant (in source_ranks)
        2. Gather serialized payload to leader (TP>1) or use directly (TP=1)
        3. Leader sends one RPC to target worker with flattened_bucket format

        Args:
            plan: Route plan with lane definitions
            bucket_idx: Current bucket index
            part_idx: Current part index within bucket
            part_count: Total parts in this bucket
            serialized_part: One pre-serialized IPC payload part
            timeout_s: RPC timeout
        """
        refs = []
        my_rank = dist.get_rank()

        for lane in plan.lanes:
            # Check if this rank participates in this lane
            if not is_lane_participant(my_rank, lane):
                continue

            # For flattened_bucket, use pre-serialized IPC parts
            # Each part is already in the correct flattened format
            local_payload = serialized_part

            # Gather to leader (no-op for TP=1)
            gathered = self._gather_lane_payload(lane, local_payload)

            # Only leader sends RPC
            if not is_lane_leader(my_rank, lane):
                continue

            if gathered is None:
                raise RuntimeError(f"lane gather returned None for leader rank={lane.leader_rank}")

            sync_key = f"{self.weight_version}:{plan.route_epoch}:{bucket_idx}:{part_idx}:{lane.lane.lane_idx}"

            refs.append(
                lane.target_worker.param_sync_from_tensor.remote(
                    serialized_named_tensors=gathered,
                    load_format="flattened_bucket",
                    flush_cache=False,
                    weight_version=str(self.weight_version),
                    trace_id=self._trace_id,
                    bucket_idx=bucket_idx,
                    part_idx=part_idx,
                    part_count=part_count,
                    sync_key=sync_key,
                    lane_idx=lane.lane.lane_idx,
                    route_epoch=plan.route_epoch,
                )
            )

        return refs

    def setup_param_sync_group(
        self,
        rollout_workers: Sequence[ActorHandle],
        rollout_topology: dict | None = None,
    ):
        """Setup colocated param sync group with optional topology-based routing.

        Args:
            rollout_workers: Ray actor handles for TP0 rollout workers
            rollout_topology: Optional topology snapshot for lane-based routing.
                             If provided and using flattened_bucket backend,
                             builds route plan for lane-leader dispatch.
        """
        normalized_workers = self._normalize_rollout_workers(rollout_workers)
        self.rollout_workers = normalized_workers
        self._refresh_sync_context()
        self.rollout_worker_connected.clear()
        self.update_rollout_worker_connected(normalized_workers)
        self._connected_rollout_workers = list(normalized_workers)
        self._connected_rollout_worker_ids = {w._actor_id.hex() for w in normalized_workers}

        # Reset lane gather state whenever worker topology is refreshed.
        self._ipc_gather_group = None
        self._ipc_gather_src = -1
        self._ipc_group_ready = False

        # Build route plan for flattened_bucket backend.
        if self._using_flattened_bucket():
            if rollout_topology is None:
                raise RuntimeError(f"[{self._group_name}] flattened_bucket requires rollout_topology in setup_param_sync_group")
            self._route_plan = self._build_colocate_route_plan(rollout_topology)
            self._ensure_ipc_gather_groups(self._route_plan)
            logger.info(
                f"[{self._group_name}] Colocated param sync: {len(normalized_workers)} workers, "
                f"lanes={len(self._route_plan.lanes)} tp_size={self._route_plan.tp_size} "
                f"topology_hash={self._route_plan.topology_hash} route_epoch={self._route_plan.route_epoch}"
            )
        else:
            self._route_plan = None
            self._route_source_ranks = set()
            logger.info(
                f"[{self._group_name}] Colocated param sync group: {len(normalized_workers)} workers " f"({self._sync_backend} backend)"
            )

    def _update_param_sync_bucket(
        self,
        name: str,
        param: torch.nn.Parameter,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
        pbar: tqdm | None = None,
    ):
        """Accumulate one parameter into current bucket for this rank.

        In flattened colocated mode, every route-participant rank builds its local
        bucket. In tensor mode, only legacy source rank builds buckets.
        """
        if not self._should_enter_bucket_sync():
            return buffer_size

        buffer_limit = self.config.trainer.param_sync_buffer_size
        param_size = param.numel() * param.element_size()
        if buffer_size + param_size > buffer_limit:
            self._update_bucket_weights(converted_named_tensors, pbar=pbar)
            buffer_size = 0
        converted_named_tensors.append((name, param))
        buffer_size += param_size
        return buffer_size

    @torch.no_grad()
    def update_weights_mixed(
        self,
        rollout_workers: Sequence[ActorHandle],
        tensor_rollout_workers: Sequence[ActorHandle] | None = None,
        bump_weight_version: bool = True,
        trace_id: str | None = None,
    ) -> None:
        """Colocated weight sync: skip NCCL, skip pause/flush/continue.

        GPU memory lifecycle (release/resume) is managed by the trainer via
        rollout_manager.offload_for_train() / resume_after_sync() BEFORE and
        AFTER this method is called.  This method only does:
          1. version bump
          2. barrier
          3. mbridge export + IPC bucket sync
          4. barrier + version check
          5. barrier
        """
        all_workers = self._normalize_rollout_workers([*rollout_workers, *(tensor_rollout_workers or [])])
        self._refresh_sync_context()

        if self.param_sync_unhealthy:
            raise RuntimeError("Param sync group is unhealthy; refusing to sync rollout weights")

        self.rollout_workers = list(all_workers)
        self.tensor_rollout_workers = []
        if not all_workers:
            return

        sync_started_at = time.monotonic()
        sync_success = False
        self._current_sync_bucket_count = 0
        try:
            target_weight_version = self.weight_version + (1 if bump_weight_version else 0)
            self._trace_id = trace_id or (f"auto-step{target_weight_version}-rank{dist.get_rank()}-ts{int(time.time() * 1000)}")
            if bump_weight_version:
                self.weight_version += 1
            self._log_colocate_mem_debug(
                stage="step_start",
                weight_version=self.weight_version,
                worker_count=len(all_workers),
            )

            # No pause_generation / flush_cache here — already released by trainer
            dist.barrier(group=get_gloo_group())

            if self.bridge is not None:
                self._update_weights_use_mbridge()
            else:
                raise NotImplementedError("Colocated mode requires use_mbridge=True")

            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                self._check_weight_version()
            # No continue_generation here — will be resumed by trainer
            dist.barrier(group=get_gloo_group())
            sync_success = True
        finally:
            self._clear_shared_cache()
            if dist.get_rank() == 0:
                elapsed_ms = (time.monotonic() - sync_started_at) * 1000
                self._log_sync_metrics(elapsed_ms, success=sync_success)
            self._log_colocate_mem_debug(
                stage="step_end",
                weight_version=self.weight_version,
                sync_success=int(sync_success),
                bucket_count=self._current_sync_bucket_count,
            )
            self._compact_cuda_cache(stage="step_end", force=True)
            self._trace_id = ""

    def _update_bucket_weights(
        self,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        pbar: tqdm | None = None,
    ) -> None:
        """Update weights for one bucket using configured backend.

        Flattened backend:
        - only route-participant ranks build/send bucket payloads;
        - each lane leader sends one RPC per bucket-part.

        Tensor backend:
        - keeps legacy rank0 sender behavior.
        """
        # Skip empty buckets
        if not converted_named_tensors:
            if pbar is not None:
                pbar.update(1)
            return

        # Check if this rank should participate in this backend/route.
        if not self._should_enter_bucket_sync() or not self.rollout_workers:
            converted_named_tensors.clear()
            if pbar is not None:
                pbar.update(1)
            return

        timeout_s = self._rpc_timeout_s()
        bucket_idx = self._current_sync_bucket_count + 1

        if not self._using_flattened_bucket():
            # Tensor path: only rank0 sends RPC
            if self._is_pp_src_rank:
                self._sync_bucket_with_tensor_path(converted_named_tensors, timeout_s=timeout_s, pbar=pbar)
            else:
                converted_named_tensors.clear()
                if pbar is not None:
                    pbar.update(1)
            return

        # Flattened bucket path with lane-based routing
        try:
            serialized_parts, long_lived_buckets = _serialize_bucket_ipc(converted_named_tensors)

            try:
                # Use lane-based routing if route plan is available
                if self._route_plan is not None:
                    # Lane-based dispatch: each lane leader sends one RPC
                    for part_idx, part in enumerate(serialized_parts, start=1):
                        refs = self._sync_flattened_bucket_by_lanes(
                            plan=self._route_plan,
                            bucket_idx=bucket_idx,
                            part_idx=part_idx,
                            part_count=len(serialized_parts),
                            serialized_part=part,
                            timeout_s=timeout_s,
                        )
                        if refs:
                            ray.get(refs, timeout=timeout_s)
                        self._record_sync_bucket()
                else:
                    raise RuntimeError(
                        f"[{self._group_name}] flattened_bucket requires route plan; "
                        "call setup_param_sync_group(..., rollout_topology=...) before syncing"
                    )
            finally:
                long_lived_buckets.clear()

        except Exception as exc:
            # Fail-fast: no automatic fallback to tensor
            self._log_colocate_mem_debug(
                stage="bucket_flattened_failed",
                bucket_idx=bucket_idx,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            if getattr(self.config.trainer, "colocate_flattened_fail_fast", True):
                raise RuntimeError(f"[{self._group_name}] flattened_bucket sync failed (fail-fast): {exc}") from exc
            raise

        converted_named_tensors.clear()
        if pbar is not None:
            pbar.update(1)
        self._compact_cuda_cache(stage="bucket_flattened")


def connect_rollout_workers_from_distributed(
    args: SiiRLArguments,
    group_name: str,
    rollout_workers: Sequence[ActorHandle],
    timeout_s: int | None = None,
) -> dist.ProcessGroup:
    """
    Create NCCL group: training rank 0 + all worker GPUs. Blocks until joined.
    """
    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    rollout_worker_num = len(rollout_workers)
    if rollout_worker_num <= 0:
        raise ValueError(f"{group_name}: rollout_workers is empty")
    rollout_gpu_per_worker = max(1, args.rollout.tensor_model_parallel_size)
    world_size = len(rollout_workers) * rollout_gpu_per_worker + 1
    logger.info(f"Group {group_name} is connecting to {rollout_workers}")
    refs = [
        worker.init_param_sync_group.remote(
            master_address,
            master_port,
            i * rollout_gpu_per_worker + 1,
            world_size,
            group_name,
            backend="nccl",
        )
        for i, worker in enumerate(rollout_workers)
    ]
    model_update_groups = init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_address}:{master_port}",
        world_size=world_size,
        rank=0,
        group_name=group_name,
        timeout=timedelta(seconds=timeout_s) if timeout_s else None,
    )
    ray.get(refs, timeout=timeout_s)
    return model_update_groups


def disconnect_rollout_workers_from_distributed(
    group_name,
    model_update_groups,
    rollout_workers,
    timeout_s: int | None = None,
):
    """
    Destroy NCCL on training and workers.
    """
    refs = [worker.destroy_weights_update_group.remote(group_name) for worker in rollout_workers]
    dist.destroy_process_group(model_update_groups)
    if refs:
        ray.get(refs, timeout=timeout_s)


def update_weights_from_distributed(
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int,
    rollout_workers: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> list[ray.ObjectRef]:
    """
    Send metadata (Ray), broadcast tensors (NCCL rank 0 → workers).
    """
    refs = [
        worker.param_sync_from_distributed.remote(
            names=[name for name, _ in converted_named_tensors],
            dtypes=[param.dtype for _, param in converted_named_tensors],
            shapes=[param.shape for _, param in converted_named_tensors],
            group_name=group_name,
            weight_version=str(weight_version),
        )
        for worker in rollout_workers
    ]

    handles = []
    for _, param in converted_named_tensors:
        handles.append(dist.broadcast(param.data, 0, group=group, async_op=True))
    for handle in handles:
        handle.wait()

    return refs


def update_weights_from_tensor(
    weight_version: int,
    rollout_workers: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
    tensor_model_parallel_size: int,
    trace_id: str | None = None,
    bucket_idx: int | None = None,
) -> list[ray.ObjectRef]:
    if not rollout_workers:
        return []

    trace_id = trace_id or "na"
    serialized_bucket = serialize_named_tensors(converted_named_tensors)
    serialized_named_tensors = [serialized_bucket for _ in range(max(1, tensor_model_parallel_size))]
    refs = [
        worker.param_sync_from_tensor.remote(
            serialized_named_tensors=serialized_named_tensors,
            flush_cache=False,
            weight_version=str(weight_version),
            trace_id=trace_id,
            bucket_idx=bucket_idx,
            part_idx=1,
            part_count=1,
        )
        for worker in rollout_workers
    ]
    return refs


def serialize_named_tensors(named_tensors: Sequence[tuple[str, torch.Tensor]]) -> str:
    # Patch torch reductions before CUDA tensor IPC serialization.
    monkey_patch_torch_reductions = None
    try:
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions as _patch_fn

        monkey_patch_torch_reductions = _patch_fn
    except ImportError:
        try:
            from sglang.srt.patch_torch import monkey_patch_torch_reductions as _patch_fn

            monkey_patch_torch_reductions = _patch_fn
        except ImportError:
            monkey_patch_torch_reductions = None

    if monkey_patch_torch_reductions is not None:
        monkey_patch_torch_reductions()

    from sglang.srt.utils.common import MultiprocessingSerializer

    return MultiprocessingSerializer.serialize(list(named_tensors), output_str=True)
