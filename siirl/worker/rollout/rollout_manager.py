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
import time
import ray
import re
import os
import asyncio
import multiprocessing

from loguru import logger
from typing import Any, Dict, List, Optional, Tuple
from ray.util import list_named_actors


from sglang_router.launch_router import RouterArgs, launch_router

from siirl.utils.enums import DistributedEnv
from siirl.utils.backend.net import get_net_interface_ip, get_free_port
from siirl.params.training_args import SiiRLArguments
from siirl.worker.ray_utils import get_random_string, RayClassWithInitArgs
from siirl.worker.rollout.rollout_worker import RolloutWorker
from siirl.engine.rollout.sglang_engine import wait_until_ok





    
class RolloutManager:
    def __init__(self, config:SiiRLArguments, pgs, rollout_gpu = 8):
        self.name_prefix: str = get_random_string(length=6)
        self.config = config
        self.pgs = pgs
        self.worker_handle = []
        self.worker_urls = []
        self.rollout_gpu = rollout_gpu
        self.device_name = config.trainer.device
        
        self.num_gpu_per_engine = min(config.rollout.tensor_model_parallel_size,config.trainer.n_gpus_per_node)
        self.num_engine = rollout_gpu // self.num_gpu_per_engine
        # init rollout worker
        self.rollout_ray_class = RayClassWithInitArgs(
            ray.remote(RolloutWorker),
            config,
        ) 
        self.init_worker()
            
    def init_worker(self):
        rank = -1
        for pg_index, placement_group in enumerate(self.pgs):
            for local_rank in range(self.config.trainer.n_gpus_per_node):
                rank += 1
                worker = self._create_worker(
                    rank=rank,
                    local_rank = local_rank,
                    pg_index=pg_index,
                    placement_group=placement_group,
                    num_gpus=0.2,
                    device_name=self.device_name,
                )
                self.worker_handle.append(worker)
        self.init_engine()

    def _create_worker(self, rank, local_rank, pg_index, placement_group, num_gpus, device_name):
        
        # --- 1. set env
        env_vars = {
            DistributedEnv.WORLD_SIZE.value: str(self.rollout_gpu),
            DistributedEnv.RANK.value: str(rank),
            DistributedEnv.LOCAL_RANK.value: str(local_rank),
            DistributedEnv.WG_PREFIX.value: self.name_prefix,
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        }
        if os.getenv('GLOO_SOCKET_IFNAME'):
            env_vars['GLOO_SOCKET_IFNAME'] = os.getenv('GLOO_SOCKET_IFNAME')

        # --- 2. Generate a unique and descriptive actor name ---
        base_class_repr = type(self.rollout_ray_class.cls).__name__  # e.g., "ActorClass(DAGWorker)"
        match = re.search(r"ActorClass\(([^)]+)\)", base_class_repr)
        actor_class_name = match.group(1) if match else base_class_repr
        actor_name = f"{self.name_prefix}_{actor_class_name}_{pg_index}:{rank}"
        # --- 3. Set actor-specific options ---
        self.rollout_ray_class.update_options(
            {
                "runtime_env": {"env_vars": env_vars},
                "name": actor_name,
            }
        )

        # --- 4. Create the actor ---
        logger.debug(f"Creating actor '{actor_name}' with rank {rank}.")
        worker = self.rollout_ray_class(placement_group=placement_group, placement_group_bundle_idx=local_rank, num_gpus=0.2, device_name=device_name)
        return worker
        
    def init_engine(self):
        # default rollout worker will start at GPU0 of NODE
        tp_size = self.config.rollout.tensor_model_parallel_size
        dp_size = self.rollout_gpu // self.config.rollout.tensor_model_parallel_size
        pp_size = 1
        # if tp_size > gpus_per_node, sglang need  dist_init_addr of tp_0
        dist_init_addr = []
        if tp_size > self.config.trainer.n_gpus_per_node:
            for dp_rank in range(dp_size):
                dist_init_addr.append(ray.get(self.worker_handle[dp_rank * tp_size].get_ip_port.remote()))
        
        else:
            dist_init_addr = [None] * dp_size        
        future = []
        for rank, worker in enumerate(self.worker_handle):
            # sglang only each tp0 of each node should create worker
            ip = ray.get(worker.get_ip.remote())
            port = ray.get(worker.get_free_port.remote())
            nccl_port = ray.get(worker.get_free_port.remote())
            if rank % self.config.rollout.tensor_model_parallel_size == 0 or rank % self.config.trainer.n_gpus_per_node == 0:
                future.append(
                    worker.init_engine.remote(rank, dist_init_addr[rank // tp_size], ip, port, nccl_port)
                )
            self.worker_urls.append(f"http://{ip}:{port}")
        ray.get(future)  
        router_address = self.start_router()
        logger.info(f"Launch Sglang Router at {router_address}")
    
    def start_router(self, request_timeout: int = 100):
        router_ip, router_port = self.config.rollout.router_ip, self.config.rollout.router_port
        if not router_ip:
            router_ip = get_net_interface_ip()
        if not router_port:
            router_port = get_free_port(router_ip)
        router_address = f"{router_ip}:{router_port}"
        router_args = RouterArgs(
            host=router_ip,
            port=router_port,
            worker_urls=self.worker_urls,
            balance_abs_threshold=0,
            log_level="warn",
            request_timeout_secs=request_timeout,
        )
        router_process = multiprocessing.Process(target=launch_router, args=(router_args,))
        router_process.daemon = True
        router_process.start()
        time.sleep(3)
        wait_until_ok(f"http://{router_address}/health", process=router_process)
        return router_address