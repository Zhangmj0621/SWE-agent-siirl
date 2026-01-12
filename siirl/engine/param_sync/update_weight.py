import socket
from abc import abstractmethod
from collections.abc import Sequence

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


class ParamSyncInterface:
    def __init__(self, config: SiiRLArguments, model: Sequence[torch.nn.Module], bridge: Bridge):
        self.config = config
        self.model = model
        self.bridge = bridge
        if self.bridge is not None:
            pass
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
        self.rollout_workers = rollout_workers
        self._is_pp_src_rank = mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        if self._is_pp_src_rank:
            self._group_name = f"param_sync-pp_{pp_rank}"

        if self._is_pp_src_rank:
            if self._model_update_groups is not None:
                disconnect_rollout_workers_from_distributed(self._group_name, self._model_update_groups, rollout_workers)
            self._model_update_groups = connect_rollout_workers_from_distributed(self.config, self._group_name, rollout_workers)
            self.rollout_worker_connected.clear()
            self.update_rollout_worker_connected(rollout_workers)
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
            self._update_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)

    def _update_weights_naive(self) -> None:
        raise NotImplementedError("_update_weights_naive is not implemented, please set use_mbridge=True")

    @torch.no_grad()
    def update_weights(self) -> None:
        self.weight_version += 1
        if dist.get_rank() == 0:
            ray.get([worker.pause_generation.remote() for worker in self.rollout_workers])
            ray.get([worker.flush_cache.remote() for worker in self.rollout_workers])
        dist.barrier(group=get_gloo_group())

        if self.bridge is not None:
            self._update_weights_use_mbridge()
        else:
            self._update_weights_naive()

        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            self._check_weight_version()
            ray.get([worker.continue_generation.remote() for worker in self.rollout_workers])
        dist.barrier(group=get_gloo_group())

    def _check_weight_version(self):
        version_list = ray.get([worker.weight_version.remote() for worker in self.rollout_workers])
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
            return
        param_size = param.numel() * param.element_size()
        if buffer_size + param_size > self.config.trainer.param_sync_buffer_size:
            self._update_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)
            buffer_size = 0
        converted_named_tensors.append((name, param))
        buffer_size += param_size
        return buffer_size

    def _update_bucket_weights_from_distributed(
        self,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        pbar: tqdm | None = None,
    ) -> None:

        refs = update_weights_from_distributed(
            self._group_name,
            self._model_update_groups,
            self.weight_version,
            self.rollout_workers,
            converted_named_tensors,
        )

        ray.get(refs)
        converted_named_tensors.clear()
        pbar.update(1)


def connect_rollout_workers_from_distributed(
    args: SiiRLArguments, group_name: str, rollout_workers: Sequence[ActorHandle]
) -> dist.ProcessGroup:
    """
    Create NCCL group: training rank 0 + all worker GPUs. Blocks until joined.
    """
    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    rollout_worker_num = len(rollout_workers)
    rollout_gpu_per_worker = args.trainer.rollout_gpus // rollout_worker_num
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
    )
    ray.get(refs)
    return model_update_groups


def disconnect_rollout_workers_from_distributed(group_name, model_update_groups, rollout_workers):
    """
    Destroy NCCL on training and workers.
    """
    refs = [worker.destroy_weights_update_group.remote(group_name) for worker in rollout_workers]
    dist.destroy_process_group(model_update_groups)
    ray.get(refs)


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
