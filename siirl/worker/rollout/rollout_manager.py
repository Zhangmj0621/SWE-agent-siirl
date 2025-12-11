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

from sglang_router.launch_router import RouterArgs, launch_router

from siirl.utils.backend.net import get_net_interface_ip, get_free_port
from siirl.params.training_args import SiiRLArguments
from siirl.worker.resouce_pool import RayClassWithInitArgs
from siirl.worker.rollout.rollout_worker import RolloutWorker

def start_router(config: SiiRLArguments, worker_urls: list, request_timeout: int = 100):
    router_ip, router_port = config.rollout.router_ip, config.rollout.router_port
    if not config.rollout.router_ip:
        router_ip = get_net_interface_ip()
    if not config.rollout.router_port:
        router_port = get_free_port
    router_address = f"{router_ip}:{router_port}"
    router_args = RouterArgs(
        host=router_ip,
        port=router_port,
        worker_urls=worker_urls,
        balance_abs_threshold=0,
        log_level="warn",
        request_timeout_secs=request_timeout,
    )

    
class RolloutManager:
    def __init__(self, config:SiiRLArguments, pgs, local_gpu = 8):
        self.config = config
        self.pgs = pgs
        self.worker_handle = []
        self.local_gpu = local_gpu
        self.device_name = config.trainer.device
        # init rollout worker
        self.rollout_ray_class = RayClassWithInitArgs(
            ray.remote(RolloutWorker),
            config
        )
            
     
    def init_worker(self):
        rank = -1
        for pg_index, placement_group in enumerate(self.pgs):
            for _ in range(self.rollout_gpu):
                rank += 1
                worker = self._create_worker(
                    rank=rank,
                    pg_index=pg_index,
                    placement_group=placement_group,
                    num_gpus_per_worker=0.2,
                    device_name=self.device_name,
                )
                self.worker_handle.append(worker)

            if rank == 0:
                # Rank 0 worker is special: it establishes the master
                # address and port for the entire worker group.
                self._master_addr, self._master_port = self._get_register_center_and_master_info()
    
    
    def _create_worker(self, rank, pg_index, placement_group, num_gpus, device_name):
        
        
        
        pass
    def init_engine(self):
        pass
        
        tp_size = config.rollout.tensor_model_parallel_size
        dp_size = world_size / config.rollout.tensor_model_parallel_size
        pp_size = 1
          
            
        
    
        
        
        
        