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
from loguru import logger
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy

from siirl.params.training_args import SiiRLArguments
from siirl.utils.backend.device import get_device_name


def sort_placement_group_by_node_ip(pgs: List[PlacementGroup]) -> List[PlacementGroup]:
    """
    Sort the placement groups by node ip, all bundles in a single placement group should be on the same node.

    FSDPCheckpointManager saves sharded model states and optimizer states in local storage, which requires RANK
    to be consistent across nodes when resume from checkpoint.

    With this function, if there's only one resource pool and there's no node change, RANK should be consistent
    across nodes in multiple ray jobs, even if the whole ray cluster is restarted.
    """
    node_ip = {node["NodeID"]: node["NodeManagerAddress"] for node in ray.nodes()}
    pg_ip = {}
    for pg in pgs:
        specs = ray._private.state.state.placement_group_table(pg.id)
        # all bunles should be on the same node
        node_id = specs["bundles_to_node_id"][0]
        pg_ip[pg.id] = node_ip[node_id]
    return sorted(pgs, key=lambda pg: pg_ip[pg.id])

def create_placement_groups(config: SiiRLArguments):
    """Create a placement group with the specified number of xPUs."""
    resource_pool_spec = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
    pg_name_prefix = f"siirl_group_{'_'.join([str(count) for count in resource_pool_spec])}:"
    # print(f"pg_name_prefix = {pg_name_prefix}")
    device_name = config.trainer.device
    if device_name == "npu":
        device_name = "NPU"
    elif device_name == "cuda":
        device_name = "GPU"
    bundle = {"CPU": 1, device_name: 1} 
    pg_scheme = [[bundle.copy() for _ in range(process_count)] for process_count in resource_pool_spec]
    pgs = [placement_group(bundles=bundles, strategy="STRICT_PACK", name=pg_name_prefix + str(idx)) for idx, bundles in enumerate(pg_scheme)]
    
    ray.get([pg.ready() for pg in pgs])
    return sort_placement_group_by_node_ip(pgs)

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