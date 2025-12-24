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

import threading
import ray

from typing import Any, Dict, List, Optional, Tuple
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

    def update_options(self, options: Dict):
        """Update the Ray actor creation options.

        Args:
            options: Dictionary of options to update
        """
        self._options.update(options)

    def __call__(self, placement_group, placement_group_bundle_idx, num_gpus=1, sharing_with=None, rank=0, device_name="cuda") -> Any:
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

        options = {"scheduling_strategy": PlacementGroupSchedulingStrategy(placement_group=placement_group, placement_group_bundle_index=placement_group_bundle_idx)}
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

from dataclasses import dataclass


@dataclass
class GPUResources:
    """
    GPU resource allocation result - a simple and intuitive data structure.
    
    Attributes:
        pg: Ray placement group
        indices: List of allocated GPU bundle indices
        num_gpus: Number of GPUs
        is_shared: Whether in colocated mode
    
    Example:
        resources = allocate_resources(config)
        actor_res = resources["actor"]  # GPUResources(2 GPUs, exclusive)
        rollout_res = resources["rollout"]  # GPUResources(6 GPUs, exclusive)
    """
    pg: PlacementGroup
    indices: List[int]
    num_gpus: int
    is_shared: bool = False
    
    def __repr__(self):
        mode = "shared" if self.is_shared else "exclusive"
        return f"GPUResources({self.num_gpus} GPUs, {mode})"
    
    # === Reserved interfaces for colocated mode ===
    
    def request_exclusive(self, role: str) -> '_NoOpContext':
        """
        Request exclusive access (time-slicing for colocated mode).
        
        Args:
            role: Role name ("actor" or "rollout")
            
        Returns:
            Context manager for use with 'with' statement
        """
        if not self.is_shared:
            return _NoOpContext()
        # TODO: Implement colocated lock
        return _NoOpContext()
    
    def offload(self, role: str) -> None:
        """
        Offload model to CPU (memory management for colocated mode).
        
        Args:
            role: Role name
        """
        if not self.is_shared:
            return
        # TODO: Implement offload
        pass
    
    def onload(self, role: str) -> None:
        """
        Load model to GPU (memory management for colocated mode).
        
        Args:
            role: Role name
        """
        if not self.is_shared:
            return
        # TODO: Implement onload
        pass


class _NoOpContext:
    """No-op context manager (used in separated mode)."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


def allocate_resources(config: SiiRLArguments) -> Dict[str, GPUResources]:
    """
    Allocate GPU resources - unified entry point.
    
    Automatically selects separated or colocated mode based on configuration.
    
    Args:
        config: SiiRLArguments configuration object
        
    Returns:
        Separated mode: {"actor": GPUResources, "rollout": GPUResources}
        Colocated mode: {"shared": GPUResources}
        
    Example:
        # Separated mode (default)
        resources = allocate_resources(config)
        actor_res = resources["actor"]     # 2 GPUs for training
        rollout_res = resources["rollout"] # 6 GPUs for inference
        
        # Colocated mode
        config.trainer.colocate = True
        resources = allocate_resources(config)
        shared_res = resources["shared"]   # 8 GPUs, shared between training and rollout
    """
    cfg = config.trainer
    
    if cfg.colocate:
        return _allocate_colocated(config)
    else:
        return _allocate_separated(config)


def _allocate_separated(config: SiiRLArguments) -> Dict[str, GPUResources]:
    """
    Separated mode: Training and inference use different GPUs.
    
    Args:
        config: SiiRLArguments configuration object
        
    Returns:
        {"actor": GPUResources, "rollout": GPUResources}
    """
    from loguru import logger
    
    cfg = config.trainer
    
    actor_gpus = cfg.actor_gpus
    rollout_gpus = cfg.rollout_gpus
    total_gpus = actor_gpus + rollout_gpus
    
    logger.info(f"Allocating resources (separated mode): {actor_gpus} GPUs for training, {rollout_gpus} GPUs for rollout")
    
    # Determine device type
    device = "GPU" if cfg.device == "cuda" else "NPU"
    
    # Create placement group
    bundles = [{"CPU": 1, device: 1} for _ in range(total_gpus)]
    pg_name = f"siirl_resources_{get_random_string(6)}"
    pg = placement_group(bundles, strategy="PACK", name=pg_name)
    ray.get(pg.ready())
    
    # Sort by node and GPU ID
    sorted_indices = _sort_by_node(pg, total_gpus)
    
    # Allocate to different roles
    actor_indices = sorted_indices[:actor_gpus]
    rollout_indices = sorted_indices[actor_gpus:]
    
    logger.info(f"  Actor GPUs: bundle indices {actor_indices}")
    logger.info(f"  Rollout GPUs: bundle indices {rollout_indices}")
    
    return {
        "actor": GPUResources(pg=pg, indices=actor_indices, num_gpus=actor_gpus, is_shared=False),
        "rollout": GPUResources(pg=pg, indices=rollout_indices, num_gpus=rollout_gpus, is_shared=False),
    }


def _allocate_colocated(config: SiiRLArguments) -> Dict[str, GPUResources]:
    """
    Colocated mode: Training and inference share the same GPUs.
    
    Args:
        config: SiiRLArguments configuration object
        
    Returns:
        {"shared": GPUResources}
    """
    from loguru import logger
    
    cfg = config.trainer
    
    # In colocated mode, use actor_gpus or default to all GPUs
    total_gpus = cfg.actor_gpus if cfg.actor_gpus > 0 else (cfg.nnodes * cfg.n_gpus_per_node)
    
    logger.info(f"Allocating resources (colocated mode): {total_gpus} GPUs shared between training and rollout")
    
    # Determine device type
    device = "GPU" if cfg.device == "cuda" else "NPU"
    
    # Create placement group (colocated mode allocates more CPU)
    bundles = [{"CPU": 2, device: 1} for _ in range(total_gpus)]
    pg_name = f"siirl_shared_{get_random_string(6)}"
    pg = placement_group(bundles, strategy="PACK", name=pg_name)
    ray.get(pg.ready())
    
    # Sort by node and GPU ID
    sorted_indices = _sort_by_node(pg, total_gpus)
    
    logger.info(f"  Shared GPUs: bundle indices {sorted_indices}")
    
    return {
        "shared": GPUResources(pg=pg, indices=sorted_indices, num_gpus=total_gpus, is_shared=True),
    }


def _sort_by_node(pg: PlacementGroup, num_bundles: int) -> List[int]:
    """
    Sort bundle indices by node IP and GPU ID to ensure consistency across runs.
    
    Args:
        pg: Ray placement group
        num_bundles: Number of bundles
        
    Returns:
        Sorted list of bundle indices
    """
    @ray.remote(num_cpus=0.01)
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
    
    # Sort by (IP, GPU_ID)
    indexed = []
    for i in range(num_bundles):
        ip = infos[i][0]
        gpu_ids = infos[i][1]
        gpu_id = gpu_ids[0] if gpu_ids else 0
        indexed.append((i, ip, gpu_id))
    
    def sort_key(x):
        idx, ip, gpu_id = x
        ip_parts = list(map(int, ip.split(".")))
        return (ip_parts, gpu_id)
    
    indexed.sort(key=sort_key)
    
    return [x[0] for x in indexed]
