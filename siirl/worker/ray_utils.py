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

from dataclasses import dataclass
from typing import Any

import ray
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy

from siirl.params.training_args import SiiRLArguments


def get_random_string(length: int) -> str:
    import random
    import string

    letters_digits = string.ascii_letters + string.digits
    return "".join(random.choice(letters_digits) for _ in range(length))


class RayClassWithInitArgs:
    """A wrapper class for Ray actors with initialization arguments.

    This class extends ClassWithInitArgs to provide additional functionality for
    configuring and creating Ray actors with specific resource requirements and
    scheduling strategies.
    """

    def __init__(self, cls, *args, **kwargs) -> None:
        # self._options = kwargs.pop('options', dict())
        self.cls = cls
        self.args = args
        self.kwargs = kwargs
        self.fused_worker_used = False
        self._options = {}
        self._additional_resource = {}

    def set_additional_resource(self, additional_resource):
        """Set additional resource requirements for the actor.

        Args:
            additional_resource: Dictionary specifying additional resource requirements
        """
        self._additional_resource = additional_resource

    def update_options(self, options: dict):
        """Update the Ray actor creation options.

        Args:
            options: Dictionary of options to update
        """
        self._options.update(options)

    def __call__(
        self,
        placement_group,
        placement_group_bundle_idx,
        num_gpus=1,
        sharing_with=None,
        rank=0,
        device_name="cuda",
    ) -> Any:
        """Create and return a Ray actor with the configured options.

        Args:
            placement_group: Ray placement group for scheduling
            placement_group_bundle_idx: Index of the bundle in the placement group
            use_gpu: Whether to use GPU resources
            num_gpus: Number of GPUs to allocate
            sharing_with: Actor to share resources with

        Returns:
            A Ray actor handle with the configured options
        """
        # Do not mutate self.kwargs, as this object is shared across all ranks.
        local_kwargs = self.kwargs.copy()
        local_kwargs.pop("device_name", "cuda")

        if sharing_with is not None:
            target_node_id = ray.get(sharing_with.get_node_id.remote())
            cuda_visible_devices = ray.get(sharing_with.get_cuda_visible_devices.remote())
            options = {"scheduling_strategy": NodeAffinitySchedulingStrategy(node_id=target_node_id, soft=False)}
            return self.cls.options(**options).remote(*self.args, cuda_visible_devices=cuda_visible_devices, **local_kwargs)

        options = {
            "scheduling_strategy": PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=placement_group_bundle_idx,
            )
        }
        options.update(self._options)

        if device_name == "cuda":
            options["num_gpus"] = num_gpus
        if device_name == "npu":
            options["resources"] = {"NPU": num_gpus}

        if len(self._additional_resource) > 1:
            for k, v in self._additional_resource.items():
                options[k] = v

        # print("cls:", self.cls)
        # print("args: ", self.args)
        # print("kwargs: ", self.kwargs)
        return self.cls.options(**options).remote(*self.args, **local_kwargs)


# =============================================================================
# GPU Resource Allocation Module
# =============================================================================


@dataclass
class GPUResources:
    """
    GPU resource allocation result - a simple and intuitive data structure.

    Attributes:
        pg: Ray placement group for resource scheduling.
        indices: List of allocated GPU bundle indices within the placement group.
        local_ranks: List of local GPU ranks (CUDA device IDs) for each bundle.
        node_ips: List of node IP addresses for each bundle.
        num_gpus: Total number of GPUs allocated.
        is_shared: Whether in colocated mode (training and rollout share GPUs).

    Example:
        resources = allocate_resources(config)
        actor_res = resources["actor"]    # GPUResources(2 GPUs, exclusive)
        rollout_res = resources["rollout"] # GPUResources(6 GPUs, exclusive)

        # Access bundle index and local rank together:
        for rank, (bundle_idx, local_rank) in enumerate(zip(res.indices, res.local_ranks)):
            ...
    """

    pg: PlacementGroup
    indices: list[int]
    local_ranks: list[int]  # Local GPU IDs (CUDA device) for each bundle
    node_ips: list[str]  # Node IP addresses for each bundle
    num_gpus: int
    is_shared: bool = False

    def __repr__(self):
        mode = "shared" if self.is_shared else "exclusive"
        return f"GPUResources({self.num_gpus} GPUs, {mode})"


def allocate_resources(config: SiiRLArguments) -> dict[str, GPUResources]:
    """
    Allocate GPU resources - unified entry point.

    Automatically selects separated or colocated mode based on configuration.

    Args:
        config: SiiRLArguments configuration object

    Returns:
        Separated mode: {"actor": GPUResources, "rollout": GPUResources}
        Colocated mode: {"actor": GPUResources, "rollout": GPUResources}

    Example:
        # Separated mode (default)
        resources = allocate_resources(config)
        actor_res = resources["actor"]     # 2 GPUs for training
        rollout_res = resources["rollout"] # 6 GPUs for inference

        # Colocated mode
        config.trainer.colocate = True
        resources = allocate_resources(config)
        shared_actor_res = resources["actor"]    # 8 GPUs, shared between training and rollout
        shared_rollout_res = resources["rollout"]  # same object as shared_actor_res
    """
    cfg = config.trainer

    if cfg.colocate:
        return _allocate_colocated(config)
    else:
        return _allocate_separated(config)


def _allocate_separated(config: SiiRLArguments) -> dict[str, GPUResources]:
    """
    Separated mode: Training and inference use different GPUs.

    Args:
        config: SiiRLArguments configuration object.

    Returns:
        {"actor": GPUResources, "rollout": GPUResources}
    """
    from loguru import logger

    cfg = config.trainer

    actor_gpus = cfg.actor_gpus
    rollout_gpus = cfg.rollout_gpus
    total_gpus = actor_gpus + rollout_gpus

    logger.info(f"Allocating resources (separated mode): " f"{actor_gpus} GPUs for training, {rollout_gpus} GPUs for rollout")

    # Determine device type
    device = "GPU" if cfg.device == "cuda" else "NPU"

    # Create a single placement group containing all GPUs
    bundles = [{"CPU": 1, device: 1} for _ in range(total_gpus)]
    pg_name = f"siirl_resources_{get_random_string(6)}"
    pg = placement_group(bundles, strategy="PACK", name=pg_name)
    ray.get(pg.ready())

    # Sort bundle indices by node IP and GPU ID for consistency
    sorted_indices, local_ranks, node_ips = _sort_by_node(pg, total_gpus)

    # Allocate indices and local_ranks to different roles
    actor_indices = sorted_indices[:actor_gpus]
    actor_local_ranks = local_ranks[:actor_gpus]
    actor_node_ips = node_ips[:actor_gpus]
    rollout_indices = sorted_indices[actor_gpus:]
    rollout_local_ranks = local_ranks[actor_gpus:]
    rollout_node_ips = node_ips[actor_gpus:]

    logger.info("[GPU Allocation - Separated Mode]")
    logger.info(f"  Total bundles: {total_gpus}, Actor GPUs: {actor_gpus}, Rollout GPUs: {rollout_gpus}")
    logger.info("  Actor allocation:")
    for i, (idx, lr, ip) in enumerate(zip(actor_indices, actor_local_ranks, actor_node_ips, strict=False)):
        logger.info(f"    rank {i}: bundle_idx={idx}, local_rank={lr}, node={ip}")
    logger.info("  Rollout allocation:")
    for i, (idx, lr, ip) in enumerate(zip(rollout_indices, rollout_local_ranks, rollout_node_ips, strict=False)):
        logger.info(f"    idx {i}: bundle_idx={idx}, local_rank={lr}, node={ip}")

    return {
        "actor": GPUResources(
            pg=pg,
            indices=actor_indices,
            local_ranks=actor_local_ranks,
            node_ips=actor_node_ips,
            num_gpus=actor_gpus,
            is_shared=False,
        ),
        "rollout": GPUResources(
            pg=pg,
            indices=rollout_indices,
            local_ranks=rollout_local_ranks,
            node_ips=rollout_node_ips,
            num_gpus=rollout_gpus,
            is_shared=False,
        ),
    }


def _allocate_colocated(config: SiiRLArguments) -> dict[str, GPUResources]:
    """
    Colocated mode: Training and inference share the same GPUs.

    Args:
        config: SiiRLArguments configuration object.

    Returns:
        {"actor": GPUResources, "rollout": GPUResources}
    """
    from loguru import logger

    cfg = config.trainer

    # In colocated mode, use actor_gpus or default to all available GPUs
    total_gpus = cfg.actor_gpus if cfg.actor_gpus > 0 else (cfg.nnodes * cfg.n_gpus_per_node)
    validate_colocated_topology(config, total_gpus=total_gpus)
    if cfg.rollout_gpus != total_gpus:
        logger.info(f"Colocated mode: overriding trainer.rollout_gpus from {cfg.rollout_gpus} to {total_gpus}")
        cfg.rollout_gpus = total_gpus

    logger.info(f"Allocating resources (colocated mode): " f"{total_gpus} GPUs shared between training and rollout")

    # Determine device type
    device = "GPU" if cfg.device == "cuda" else "NPU"

    # Colocated mode allocates more CPU per bundle for concurrent operations
    bundles = [{"CPU": 2, device: 1} for _ in range(total_gpus)]
    pg_name = f"siirl_shared_{get_random_string(6)}"
    pg = placement_group(bundles, strategy="PACK", name=pg_name)
    ray.get(pg.ready())

    # Sort bundle indices by node IP and GPU ID for consistency
    sorted_indices, local_ranks, node_ips = _sort_by_node(pg, total_gpus)

    logger.info("[GPU Allocation - Colocated Mode]")
    logger.info(f"  Total shared GPUs: {total_gpus}")
    for i, (idx, lr, ip) in enumerate(zip(sorted_indices, local_ranks, node_ips, strict=False)):
        logger.info(f"    idx {i}: bundle_idx={idx}, local_rank={lr}, node={ip}")

    shared = GPUResources(
        pg=pg,
        indices=sorted_indices,
        local_ranks=local_ranks,
        node_ips=node_ips,
        num_gpus=total_gpus,
        is_shared=True,
    )
    return {"actor": shared, "rollout": shared}


def validate_colocated_topology(config: SiiRLArguments, total_gpus: int) -> None:
    cfg = config.trainer
    tp = cfg.tensor_model_parallel_size
    pp = cfg.pipeline_model_parallel_size
    cp = cfg.context_parallel_size
    ep = cfg.expert_model_parallel_size
    etp = cfg.expert_tensor_parallel_size
    rollout_tp = config.rollout.tensor_model_parallel_size
    n_gpus_per_node = cfg.n_gpus_per_node

    if total_gpus <= 0:
        raise ValueError(f"colocate: total_gpus must be > 0, got {total_gpus}")
    if tp <= 0 or pp <= 0 or cp <= 0:
        raise ValueError(f"colocate: tp/pp/cp must be > 0, got tp={tp}, pp={pp}, cp={cp}")
    if ep <= 0 or etp <= 0:
        raise ValueError(f"colocate: ep/etp must be > 0, got ep={ep}, etp={etp}")
    if rollout_tp <= 0:
        raise ValueError(f"colocate: rollout_tp must be > 0, got {rollout_tp}")
    if n_gpus_per_node <= 0:
        raise ValueError(f"colocate: n_gpus_per_node must be > 0, got {n_gpus_per_node}")

    megatron_unit = tp * pp * cp
    if total_gpus % megatron_unit != 0:
        raise ValueError(f"colocate: total_gpus={total_gpus} not divisible by tp*pp*cp={tp}*{pp}*{cp}={megatron_unit}")
    if tp % etp != 0:
        raise ValueError(f"colocate: tp={tp} not divisible by etp={etp}")

    dp_size = total_gpus // megatron_unit
    if ep > 1 and dp_size % ep != 0:
        raise ValueError(f"colocate: dp_size={dp_size} not divisible by ep={ep}")

    if total_gpus % rollout_tp != 0:
        raise ValueError(f"colocate: total_gpus={total_gpus} not divisible by rollout_tp={rollout_tp}")
    if rollout_tp > n_gpus_per_node and rollout_tp % n_gpus_per_node != 0:
        raise ValueError(f"colocate: cross-node TP requires rollout_tp({rollout_tp}) % n_gpus_per_node({n_gpus_per_node}) == 0")

    return


def _sort_by_node(pg: PlacementGroup, num_bundles: int) -> tuple[list[int], list[int], list[str]]:
    """
    Sort bundle indices by node IP and GPU ID to ensure consistency across runs.

    This ensures that rank assignment is deterministic when resuming from checkpoints,
    even if the Ray cluster is restarted.

    Args:
        pg: Ray placement group.
        num_bundles: Number of bundles in the placement group.

    Returns:
        Tuple of (sorted_indices, local_ranks, node_ips):
        - sorted_indices: List of bundle indices sorted by (node_ip, gpu_id)
        - local_ranks: List of local GPU IDs (CUDA device) for each bundle
        - node_ips: List of node IP addresses for each bundle
    """
    from loguru import logger

    # Must request GPU to get correct GPU IDs from ray.get_gpu_ids()
    @ray.remote(num_gpus=1)
    class _InfoActor:
        def info(self):
            return ray.util.get_node_ip_address(), ray.get_gpu_ids()

    # Create temporary actors to get node info for each bundle
    actors = [
        _InfoActor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=i,
            )
        ).remote()
        for i in range(num_bundles)
    ]

    # Get info and destroy temporary actors
    infos = ray.get([a.info.remote() for a in actors])
    for a in actors:
        ray.kill(a)

    # Build index list with (bundle_idx, ip, gpu_id)
    indexed = []
    for i in range(num_bundles):
        ip = infos[i][0]
        gpu_ids = infos[i][1]
        if not gpu_ids:
            raise RuntimeError(f"Bundle {i} has no GPU assigned. Check placement group configuration.")
        gpu_id = int(gpu_ids[0])
        indexed.append((i, ip, gpu_id))
        logger.debug(f"Bundle {i}: node={ip}, gpu_id={gpu_id}")

    # Sort by (IP as tuple of ints, GPU ID)
    def sort_key(x):
        idx, ip, gpu_id = x
        ip_parts = list(map(int, ip.split(".")))
        return (ip_parts, gpu_id)

    indexed.sort(key=sort_key)

    sorted_indices = [x[0] for x in indexed]
    local_ranks = [x[2] for x in indexed]  # GPU IDs as local ranks
    node_ips = [x[1] for x in indexed]  # Node IPs for each bundle

    logger.info(f"Sorted GPU allocation: indices={sorted_indices}, local_ranks={local_ranks}")

    return sorted_indices, local_ranks, node_ips
