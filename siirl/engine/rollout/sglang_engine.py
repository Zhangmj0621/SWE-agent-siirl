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


import asyncio
import io
import os

from PIL import Image
import requests
import sglang as sgl

from sglang.srt.conversation import chat_templates
from sglang.test.test_utils import is_in_ci
from sglang.utils import async_stream_and_merge, stream_and_merge
from sglang.srt.entrypoints.http_server import launch_server


from siirl.params.training_args import SiiRLArguments
class SglangEngine:
    def __init__(self, rank: int, config: SiiRLArguments):
        self.rank = rank
        self.config = config
    
    
    
    def get_sglang_params(self, rank: int):
        
        
        config = self.config.rollout
        model_path=self.config.actor_rollout_ref.model.path,
        dtype=config.dtype,
        mem_fraction_static=config.gpu_memory_utilization,
        enable_memory_saver=True,
        base_gpu_id=0,
        gpu_id_step=1,
        tp_size=self._tp_size,
        node_rank=node_rank,
        load_format=load_format,
        dist_init_addr=dist_init_addr,
        nnodes=nnodes,
        trust_remote_code=trust_remote_code,
        # NOTE(linjunrong): add rank to prevent SGLang generate same port inside PortArgs.init_new
        # when random.seed is being set during training
        port=30000 + rank,
        # NOTE(Chenyang): if you want to debug the SGLang engine output
        # please set the following parameters
        # Otherwise, it will make the engine run too slow
        # log_level="INFO",
        # log_requests=True,
        # log_requests_level=2,
        # max_running_requests=1,
        mm_attention_backend="fa3",
        attention_backend="fa3",
        # In async mode, we want token in token out.
        skip_tokenizer_init=self.config.mode == "async",
        
        
        nnodes = -(config.tensor_model_parallel_size // len(self.visible_devices_set))
        if nnodes > 1:
            ip = get_ip()
            port = get_open_port() if port is None else port
            [ip, port] = broadcast_pyobj(
                [ip, port],
                rank=self._rank,
                dist_group=self._device_mesh_cpu.get_group("tp"),
                src=self._device_mesh_cpu["tp"].mesh[0].item(),
                force_cpu_device=False,
            )
            dist_init_addr = f"[{ip}]:{port}" if is_ipv6(ip) else f"{ip}:{port}"
        else:
            dist_init_addr = None
        
        
        args = {
            "model_path": self.config.actor_rollout_ref.model.path,
            "dtype": config.dtype,
            "mem_fraction_static": config.gpu_memory_utilization,
            "enable_memory_saver": True,
            "base_gpu_id": 0,
            "gpu_id_step": 1,
            "tp_size": config.tensor_model_parallel_size,
            "node_rank": -1,
            "load_format": config.load_format,
            "dist_init_addr": -1,
            "nnodes": -1,
            "trust_remote_code": config.trust_remote_code,
            "max_running_requests": c,
            # NOTE(linjunrong): add rank to prevent SGLang generate same port inside PortArgs.init_new
            # when random.seed is being set during training
            "port": sglang_port,
            "nccl_port": sglang_port + 1,
            # NOTE(Chenyang): if you want to debug the SGLang engine output
            # please set the following parameters
            # Otherwise, it will make the engine run too slow
            "log_level": "info",
            # "log_level": "error",
            # log_requests=True,
            # log_requests_level=2,
            # NOTE(Chenyang): turn on max_running_requests to set the max concurrent running requests
            # max_running_requests=1,
            "mm_attention_backend": backend,
            "attention_backend": backend,
            # In async mode, we want token in token out.
            "skip_tokenizer_init": self.config.skip_tokenizer_init,
            "dist_timeout": 1800,
        }

        if is_server_mode:
            # add server specific args
            args["first_rank_in_node"] = first_rank_in_node
            args["timeout"] = self.config.server["timeout"]
            args["max_attempts"] = self.config.server["max_attempts"]
            args["retry_delay"] = self.config.server["retry_delay"]
            args["max_connections"] = self.config.server["max_connections"]
            args["max_start_wait_time"] = self.config.server["max_start_wait_time"]
            self._engine = AsyncHttpServerAdapter(**args)
        else:
            self._engine = AsyncEngine(**args)
            
            launch_server()
        
        
        
    def launch_server(self, rank:int):
        
        
        
        
        

        