import socket
from abc import abstractmethod
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

    def _rpc_timeout_s(self) -> int:
        return max(1, int(getattr(self.config.trainer, "param_sync_rpc_timeout_s", 120)))

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
        pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_pp_src_rank else None

        generator = self.bridge._export_weights_in_current_pipeline_stage(self.model)

        for name, param in generator:
            buffer_size = self._update_param_sync_bucket(name, param, converted_named_tensors, buffer_size, pbar)

        if converted_named_tensors:
            self._update_bucket_weights(converted_named_tensors, pbar=pbar)

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
    ) -> None:
        if self.param_sync_unhealthy:
            raise RuntimeError("Param sync group is unhealthy; refusing to sync rollout weights")

        self.rollout_workers = list(rollout_workers)
        self.tensor_rollout_workers = list(tensor_rollout_workers or [])
        all_workers = self._all_target_workers()
        if not all_workers:
            return

        timeout_s = self._rpc_timeout_s()
        self.weight_version += 1
        if dist.get_rank() == 0:
            if self.tensor_rollout_workers:
                logger.info(
                    f"[ParamSyncDistributed] Mixed weight sync: distributed_workers={len(self.rollout_workers)} "
                    f"tensor_workers={len(self.tensor_rollout_workers)} weight_version={self.weight_version}"
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
        param_size = param.numel() * param.element_size()
        if buffer_size + param_size > self.config.trainer.param_sync_buffer_size:
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
        converted_named_tensors.clear()
        if pbar is not None:
            pbar.update(1)


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
) -> list[ray.ObjectRef]:
    if not rollout_workers:
        return []

    serialized_bucket = serialize_named_tensors(converted_named_tensors)
    serialized_named_tensors = [serialized_bucket for _ in range(max(1, tensor_model_parallel_size))]
    return [
        worker.param_sync_from_tensor.remote(
            serialized_named_tensors=serialized_named_tensors,
            flush_cache=False,
            weight_version=str(weight_version),
        )
        for worker in rollout_workers
    ]


def serialize_named_tensors(named_tensors: Sequence[tuple[str, torch.Tensor]]) -> str:
    from sglang.srt.utils.common import MultiprocessingSerializer

    return MultiprocessingSerializer.serialize(list(named_tensors), output_str=True)
