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
import threading
import importlib
import asyncio

from loguru import logger

from siirl.engine.rollout.sglang_engine import SglangEngine
from siirl.params.training_args import SiiRLArguments
from siirl.utils.backend.device import get_nccl_backend,get_device_name
from siirl.utils.net_utils.net import get_free_port, get_net_interface_ip

def async_run_wrapper(executor):
    """
    Wrapper function to run asynchronous executor in a dedicated event loop.
    Creates a new asyncio event loop, sets it as the current loop, and runs the executor's main coroutine.
    Ensures proper cleanup of the event loop after execution completes or errors occur.
    
    Args:
        executor: The async executor instance with a `run()` coroutine method to execute
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(executor.run())
    finally:
        loop.close()

class RolloutWorker:
    """
    RolloutWorker class manages the rollout process execution in a distributed training environment.
    It handles engine initialization, executor thread management, and network configuration for rollout tasks.
    """
    def __init__(self, config:SiiRLArguments) -> None:
        """
        Initialize RolloutWorker with configuration parameters.
        
        Args:
            config: SiiRLArguments configuration object containing all training/rollout settings
        """
        self.config = config
        self.ip = None  # Network IP address for the worker
        self.port = None  # Network port for the worker
        self.executor = None  # Rollout executor instance
        self.rollout_thread = None  # Thread for running the async rollout executor
        # Initial worker
    
    def init_engine(self, rank: int, dist_init_addr:str, ip = None, port = None, nccl_port = None):
        """
        Initialize the rollout engine (currently supports SglangEngine only).
        
        Args:
            rank: Distributed training rank of the worker
            dist_init_addr: Initialization address for distributed communication
            ip: Optional IP address for the engine
            port: Optional port number for the engine
            nccl_port: Optional port for NCCL backend communication
        
        Todo:
            Add support for vLLM engine
        """
        # todo: support vllm
        if self.config.rollout.name == 'sglang':
            self.engine = SglangEngine(rank, self.config, dist_init_addr, ip, port, nccl_port)
            self.ip = ip
            self.port = port
    
    def start_rollout(self, router_address, data_coordinator, num_engine):
        """
        Start the rollout process in a dedicated background thread with async executor.
        
        Steps:
            1. Dynamically load the specified executor class
            2. Initialize executor with configuration and dependencies
            3. Start executor in a daemon thread with async event loop wrapper
        
        Args:
            router_address: Address of the router for engine communication
            data_coordinator: Data coordinator instance for data management
            num_engine: Number of engine instances to use
        """
        #create executor thread
        # 1. init executor
        executor_path = self.config.rollout.executor_module
        if executor_path == "naive":
            from siirl.execution.rollout.agent_executor.naive_executor import NaiveExecutor
            Executor = NaiveExecutor
        else:
            # Dynamically import custom executor module
            module_path, name = executor_path.rsplit('.', 1)
            mod = importlib.import_module(module_path)
            Executor = getattr(mod, name)
        executor = Executor(self.config, data_coordinator, self.engine, self.config.data.train_batch_size // num_engine)
        self.rollout_thread = threading.Thread(target=async_run_wrapper, args=(executor,), daemon=True)
        self.rollout_thread.start()

    def stop_rollout(self):
        """
        Stop the rollout process and clean up resources.
        Stops the executor and waits for the rollout thread to complete.
        """
        self.executor.stop()        
        self.rollout_thread.join()  

    def set_router(self, router_address):
        """
        Update the router address for the engine.
        
        Args:
            router_address: New router address to set for engine communication
        """
        self.engine.set_router(router_address)    

    def get_ip(self):
        """
        Get the network interface IP address of the current worker.
        
        Returns:
            str: Network interface IP address
        """
        return get_net_interface_ip()

    def get_ip_port(self):
        """
        Get a formatted string of IP address and a free port for the worker.
        
        Returns:
            str: Formatted string in "ip:port" format with a free port
        """
        host = get_net_interface_ip()
        return f"{host}:{get_free_port(host)}"

    def get_free_port(self):
        """
        Get a free network port on the current worker's network interface.
        
        Returns:
            int: Available free port number
        """
        host = get_net_interface_ip()
        return get_free_port(host)
    
    def init_param_sync_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        return self.engine.init_param_sync_group(master_address, master_port, rank_offset, world_size, group_name, backend)

    def param_sync_from_distributed(
        self, names, dtypes, shapes, group_name, flush_cache=False, weight_version: str | None = None
    ):
        return self.engine.sync_param_from_distributed(names, dtypes, shapes, group_name, flush_cache, weight_version)
    
    def destroy_weights_update_group(self, group_name):
        return self.engine.destroy_weights_update_group(group_name)
    
    def flush_cache(self):
        self.engine.flush_cache()

    def pause_generation(self):
        self.engine.pause_generation()

    def continue_generation(self):
        return self.engine.continue_generation()