import socket
import mbridge
from mbridge.core.bridge import Bridge
from mbridge.core.util import unwrap_model
from abc import abstractmethod
import time
import ray
import torch
import torch.distributed as dist
from tqdm import tqdm
from siirl.worker.rollout.rollout_worker import RolloutWorker
from typing import List, Sequence
from siirl.utils.distributed_utils import init_process_group
from ray.actor import ActorHandle
from siirl.params.training_args import SiiRLArguments
from megatron.core import mpu

class ParamSyncInterface:
    def __init__(self,config:SiiRLArguments, model: Sequence[torch.nn.Module],bridge:Bridge,rollout_workers: Sequence[ActorHandle]):
        self.config = config
        self.model = model
        self.bridge = bridge
        self.rollout_workers = rollout_workers
        self.current_version = 0
        self.weights_info = None
        self.sync_group_initialized = False
        self.sync_group_name = "actor_rollout"
        self.wait_last_update = None
        self.wait_last_resume = None

    def setup_param_sync_group(self):
        pass

    @abstractmethod
    def update_weights(self) -> None:
        pass


class ParamSyncDistribute(ParamSyncInterface):
    def __init__(self,config:SiiRLArguments,model: Sequence[torch.nn.Module],bridge:Bridge, rollout_workers: Sequence[ActorHandle]):
        super().__init__(config,model,bridge,rollout_workers)

    def setup_param_sync_group(self):
        # from Train DP 0 to all engine
        # each pp rank has its own group
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        )
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        if self._is_pp_src_rank:
            self._group_name = f"param_sync-pp_{pp_rank}"

        if self._is_pp_src_rank:
            if self._model_update_groups is not None:
                disconnect_rollout_engines_from_distributed(
                    self._group_name, self._model_update_groups, self.rollout_workers
                )
            self._model_update_groups = connect_rollout_engines_from_distributed(
                self.config, self._group_name, self.rollout_workers
            )

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        Pause → flush → non-expert (TP) → expert (EP) → continue. Progress on PP source.
        """
        self.weight_version += 1
        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_workers])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_workers])
        # dist.barrier(group=get_gloo_group())

        buffer_size = 0
        converted_named_tensors = []
        pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_pp_src_rank else None

        generator = self.bridge.export_weights(self.model)

        for name, param in generator:
            buffer_size = self._update_param_sync_bucket(param,converted_named_tensors,buffer_size,pbar)

        if converted_named_tensors:
            self._update_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)

        if dist.get_rank() == 0:
            ray.get([engine.continue_generation.remote() for engine in self.rollout_workers])

    def _update_param_sync_bucket(
        self,
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
        buffer_size += param_size
        return buffer_size

    def _update_bucket_weights_from_distributed(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """
        Lock → broadcast → clear → unlock → pbar++. Lock prevents NCCL deadlock.
        """
        # lock the rollout engines to prevent dead lock on broadcast.
        # while not ray.get(self.rollout_engine_lock.acquire.remote()):
        #     time.sleep(0.1)

        refs = update_weights_from_distributed(
            self._group_name,
            self._model_update_groups,
            self.weight_version,
            self.rollout_workers,
            converted_named_tensors,
        )

        ray.get(refs)
        converted_named_tensors.clear()
        # ray.get(self.rollout_engine_lock.release.remote())
        pbar.update(1)


def connect_rollout_engines_from_distributed(
    args: SiiRLArguments, group_name: str, rollout_workers: Sequence[ActorHandle]
) -> dist.ProcessGroup:
    """
    Create NCCL group: training rank 0 + all engine GPUs. Blocks until joined.
    """
    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    world_size = len(rollout_workers) * args.trainer.n_gpus_per_node + 1

    refs = [
        worker.init_param_sync_group.remote(
            master_address,
            master_port,
            i * args.trainer.n_gpus_per_node + 1,
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

def disconnect_rollout_engines_from_distributed(group_name, model_update_groups, rollout_engines):
    """
    Destroy NCCL on training and engines.
    """
    refs = [engine.destroy_weights_update_group.remote(group_name) for engine in rollout_engines]
    dist.destroy_process_group(model_update_groups)
    ray.get(refs)


def update_weights_from_distributed(
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int,
    rollout_workers: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> list[ObjectRef]:
    """
    Send metadata (Ray), broadcast tensors (NCCL rank 0 → engines).
    """
    refs = [
        engine.param_sync_from_distributed.remote(
            names=[name for name, _ in converted_named_tensors],
            dtypes=[param.dtype for _, param in converted_named_tensors],
            shapes=[param.shape for _, param in converted_named_tensors],
            group_name=group_name,
            weight_version=str(weight_version),
        )
        for engine in rollout_workers
    ]

    handles = []
    for _, param in converted_named_tensors:
        handles.append(dist.broadcast(param.data, 0, group=group, async_op=True))
    for handle in handles:
        handle.wait()

    return refs