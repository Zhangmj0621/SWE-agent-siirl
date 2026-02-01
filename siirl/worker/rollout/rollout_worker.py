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
import importlib
import os
import threading

from loguru import logger

from siirl.engine.rollout.sglang_engine import SglangEngine
from siirl.params.training_args import SiiRLArguments
from siirl.utils.net_utils.net import get_free_port, get_free_port_with_socket, get_net_interface_ip
from siirl.worker.rollout.validator import aggregate_and_log_validation_metrics


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

    def __init__(self, config: SiiRLArguments, global_dp_size=1, metric_worker=None) -> None:
        """
        Initialize RolloutWorker with configuration parameters.

        Args:
            config: SiiRLArguments configuration object containing all training/rollout settings
        """
        # Configure logging for this Ray actor process
        # (worker_process_setup_hook only works for task workers, not actors)
        from siirl.utils.logger.logging_utils import set_basic_config

        set_basic_config()
        self.rank = int(os.environ.get("RANK"))
        self.config = config
        self.global_dp_size = global_dp_size
        self.metric_worker = metric_worker
        self.ip = None  # Network IP address for the worker
        self.port = None  # Network port for the worker
        self.executor = None  # Rollout executor instance
        self.rollout_thread = None  # Thread for running the async rollout executor
        # Socket holders for port reservation (verl-style)
        self._port_sock = None
        self._nccl_sock = None

    def find_free_port(self, start_port: int = 15000) -> int:
        """
        Find a free port on this worker's node starting from start_port.

        This method is called remotely by RolloutManager to allocate ports
        on the correct node where the worker runs.

        Args:
            start_port: Starting port number for search

        Returns:
            int: Available port number
        """
        return get_free_port(get_net_interface_ip(), start_port=start_port)

    def find_free_port_with_hold(self, start_port: int = 15000, slot: str = "port") -> int:
        """
        Find a free port and hold the socket to prevent race conditions.

        Combines slime's sequential allocation with verl's socket holding.
        The socket is held until launch_server() is called.

        Args:
            start_port: Starting port number for search
            slot: Which slot to store socket ("port" or "nccl")

        Returns:
            int: Available port number
        """
        port, sock = get_free_port_with_socket(get_net_interface_ip(), start_port=start_port)
        if slot == "port":
            self._port_sock = sock
        elif slot == "nccl":
            self._nccl_sock = sock
        return port

    def init_engine(
        self,
        rank: int,
        dist_init_addr: str,
        ip: str,
        port: int,
        nccl_port: int,
        base_gpu_id: int,
        node_rank: int,
        nnodes: int,
    ):
        """
        Initialize the rollout engine with explicit GPU placement parameters.

        Args:
            rank: Global rank of this engine instance (TP0 rank within rollout workers).
            dist_init_addr: Initialization address for distributed communication.
                            Used for cross-node TP communication (ip:port format).
            ip: IP address for the engine HTTP server.
            port: Port number for the engine HTTP server.
            nccl_port: Port number for NCCL backend communication.
            base_gpu_id: Starting CUDA device ID for this TP group.
            node_rank: Rank of this node within the TP group.
            nnodes: Number of nodes participating in this TP group.

        Todo:
            Add support for vLLM engine
        """

        # TODO: Support vLLM engine
        if self.config.rollout.name == "sglang":
            self.engine = SglangEngine(
                rank=rank,
                config=self.config,
                dist_init_addr=dist_init_addr,
                ip=ip,
                port=port,
                nccl_port=nccl_port,
                base_gpu_id=base_gpu_id,
                node_rank=node_rank,
                nnodes=nnodes,
            )
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
        # create executor thread
        # 1. init executor
        executor_path = self.config.rollout.executor_module
        if executor_path == "naive":
            from siirl.execution.rollout.agent_executor.naive_executor import NaiveExecutor

            Executor = NaiveExecutor
        else:
            # Dynamically import custom executor module
            module_path, name = executor_path.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            Executor = getattr(mod, name)
        executor = Executor(
            self.config,
            data_coordinator,
            self.engine,
            self.config.data.train_batch_size // num_engine,
        )
        self.executor = executor
        self.rollout_thread = threading.Thread(target=async_run_wrapper, args=(executor,), daemon=True)
        self.rollout_thread.start()

    def stop_rollout(self):
        """
        Stop the rollout process and clean up resources.
        Stops the executor and waits for the rollout thread to complete.
        """
        self.executor.stop()
        self.rollout_thread.join()

    def _release_port_sockets(self):
        """Release held port sockets just before server binds."""
        if self._port_sock:
            self._port_sock.close()
            self._port_sock = None
        if self._nccl_sock:
            self._nccl_sock.close()
            self._nccl_sock = None

    def launch_server(self, extra_server_args: dict = None):
        """
        Launch the SGLang server, releasing held port sockets first.

        This method should be called after init_engine() to actually start the server.
        Sockets are released just before SGLang binds (verl-style minimal race window).

        Args:
            extra_server_args: Optional extra arguments for the server
        """
        self._release_port_sockets()
        self.engine.launch_server(extra_server_args)

    def get_port(self) -> int:
        """
        Get the allocated port number for this worker's engine.

        Returns:
            int: The allocated port number
        """
        return self.port

    def set_router(self, router_address):
        """
        Update the router address for the engine.

        Args:
            router_address: New router address to set for engine communication
        """
        self.engine.set_router(router_address)

    async def validate(self, val_batch_size, global_step):
        rank = int(os.environ.get("RANK"))
        # start_time = time.perf_counter()
        if rank == 0:
            logger.info("=" * 60)
            logger.info(f"Starting Validation @ Global Step {global_step}...")
            logger.info("=" * 60)
        samples, val_time_metrics = await self.executor.validate(val_batch_size)
        validate_samples = []
        for sample in samples:
            if sample.extra_info and isinstance(sample.extra_info, dict) and sample.extra_info.get("padded_duplicate", None):
                continue
            validate_samples.append(sample)
        val_metrics = aggregate_and_log_validation_metrics(validate_samples)
        await self.metric_worker.submit_metric.remote(val_metrics, self.global_dp_size)

    def get_ip(self):
        """
        Get the network interface IP address of the current worker.

        Returns:
            str: Network interface IP address
        """
        return get_net_interface_ip()

    def get_ip_port(self, start_port: int = 15000):
        """
        Get a formatted string of IP address and a free port for the worker.

        Args:
            start_port: Starting port number for search

        Returns:
            str: Formatted string in "ip:port" format with a free port
        """
        host = get_net_interface_ip()
        port = get_free_port(host, start_port=start_port)
        return f"{host}:{port}"

    def init_param_sync_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        return self.engine.init_param_sync_group(master_address, master_port, rank_offset, world_size, group_name, backend)

    def param_sync_from_distributed(
        self,
        names,
        dtypes,
        shapes,
        group_name,
        flush_cache=False,
        weight_version: str | None = None,
    ):
        return self.engine.param_sync_from_distributed(names, dtypes, shapes, group_name, flush_cache, weight_version)

    def destroy_weights_update_group(self, group_name):
        return self.engine.destroy_weights_update_group(group_name)

    def flush_cache(self):
        self.engine.flush_cache()

    def pause_generation(self):
        self.engine.pause_generation()

    def continue_generation(self):
        return self.engine.continue_generation()

    def weight_version(self):
        return self.engine.weight_version
