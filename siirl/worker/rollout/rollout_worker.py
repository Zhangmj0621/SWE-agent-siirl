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

import os
import torch
import torch.distributed as dist

from loguru import logger

from siirl.engine.rollout.sglang_engine import SglangEngine
from siirl.params.training_args import SiiRLArguments
from siirl.utils.backend.device import get_nccl_backend,get_device_name
from siirl.utils.backend.net import get_free_port, get_net_interface_ip

class RolloutWorker:
    """RolloutWorker class"""
    def __init__(self, config:SiiRLArguments) -> None:
        self.config = config
        # Initial worker
    
    def init_engine(self, rank: int, dist_init_addr:str, ip = None, port = None, nccl_port = None):
        # todo: support vllm
        if self.config.rollout.name == 'sglang':
            self.engine = SglangEngine(rank, self.config, dist_init_addr, ip, port, nccl_port)
    
    def get_ip(self):
        return get_net_interface_ip()
    
    def get_ip_port(self):
        host = get_net_interface_ip()
        return f"{host}:{get_free_port(host)}"
    
    def get_free_port(self):
        host = get_net_interface_ip()
        return get_free_port(host)