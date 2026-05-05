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
import contextlib
import errno
import importlib
import os
import threading
import time
import traceback

from loguru import logger

from siirl.engine.rollout.sglang_engine import SglangEngine
from siirl.params.training_args import SiiRLArguments
from siirl.utils.net_utils.net import get_free_port, get_net_interface_ip


def _is_port_conflict(exc: BaseException) -> bool:
    """Check whether *exc* (or any chained cause) is an EADDRINUSE error."""
    cur: BaseException | None = exc
    while cur is not None:
        if isinstance(cur, OSError):
            if cur.errno == errno.EADDRINUSE:
                return True
            cur_text = str(cur).lower()
            if "address already in use" in cur_text or "eaddrinuse" in cur_text:
                return True
        cur = cur.__cause__ or cur.__context__
    return False


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
        self.engine = None  # SGLang engine instance
        self._validate_progress = {
            "total": 0,
            "done": 0,
            "active": False,
            "start_time": 0.0,
            "updated_at": 0.0,
        }

    def _build_executor(self, data_coordinator, num_engine):
        executor_path = self.config.rollout.executor_module
        if executor_path == "naive":
            from siirl.execution.rollout.agent_executor.naive_executor import NaiveExecutor

            Executor = NaiveExecutor
        else:
            module_path, name = executor_path.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            Executor = getattr(mod, name)
        return Executor(
            self.config,
            data_coordinator,
            self.engine,
            self.config.data.train_batch_size // num_engine,
            dp_rank=self.rank,
        )

    def allocate_ports(self, start_port: int, count: int) -> list[int]:
        """
        Allocate multiple free ports sequentially on this node.

        This method ensures no race conditions by allocating all ports
        in a single call. Each port is verified free before moving to the next.

        Args:
            start_port: Starting port number for search
            count: Number of ports to allocate

        Returns:
            List of allocated port numbers
        """
        host = get_net_interface_ip()
        ports = []
        current = start_port
        for _ in range(count):
            port = get_free_port(host, start_port=current)
            ports.append(port)
            current = port + 1
        return ports

    def init_engine(
        self,
        rank: int,
        dist_init_addr: str,
        ip: str,
        port: int,
        base_gpu_id: int,
        node_rank: int,
        nnodes: int,
    ):
        """
        Initialize the rollout engine with explicit GPU placement parameters.

        Note: nccl_port is not passed - SGLang will auto-allocate it internally,
        eliminating port race conditions for NCCL communication.

        Args:
            rank: Global rank of this engine instance (TP0 rank within rollout workers).
            dist_init_addr: Initialization address for distributed communication.
                            Used for cross-node TP communication (ip:port format).
            ip: IP address for the engine HTTP server.
            port: Port number for the engine HTTP server.
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
                base_gpu_id=base_gpu_id,
                node_rank=node_rank,
                nnodes=nnodes,
            )
            self.ip = ip
            self.port = port

    def launch_server(self, max_retries: int = 3, reserved_ports: list[int] | None = None):
        """
        Launch the SGLang server with retry mechanism for port conflicts.

        If the server fails to start (e.g., due to port conflict), it will
        retry with a new port up to max_retries times.

        Args:
            max_retries: Maximum number of retry attempts (default: 3)
            reserved_ports: Optional reserved ports for this worker. When provided,
                retries are limited to this list to avoid cross-worker port stealing.
        """
        candidate_ports = []
        if reserved_ports:
            seen_ports = set()
            for port in reserved_ports:
                if port in seen_ports:
                    continue
                seen_ports.add(port)
                candidate_ports.append(port)

        attempt_budget = min(max_retries, len(candidate_ports)) if candidate_ports else max_retries

        for attempt in range(attempt_budget):
            if candidate_ports:
                target_port = candidate_ports[attempt]
            elif attempt == 0:
                target_port = self.port
            else:
                target_port = get_free_port(get_net_interface_ip(), start_port=self.port + 1)

            self.port = target_port
            self.engine.port = target_port
            try:
                self.engine.launch_server()
                return  # Success
            except Exception as e:
                with contextlib.suppress(Exception):
                    self.engine.shutdown()
                if not _is_port_conflict(e):
                    raise
                if attempt < attempt_budget - 1:
                    logger.warning(f"Port {self.port} conflict (attempt {attempt + 1}/{attempt_budget}), retrying: {e}")
                else:
                    logger.error(f"Failed to start server after {attempt_budget} attempts")
                    raise

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
        self.executor = self._build_executor(data_coordinator, num_engine)
        self.rollout_thread = threading.Thread(target=async_run_wrapper, args=(self.executor,), daemon=True)
        self.rollout_thread.start()

    def init_validate_executor(self, data_coordinator, num_engine):
        if self.executor is None:
            self.executor = self._build_executor(data_coordinator, num_engine)

    def stop_rollout(self):
        """
        Stop the rollout process and clean up resources.
        Stops the executor and waits for the rollout thread to complete.
        """
        self.executor.stop()
        self.rollout_thread.join()

    def shutdown_engine(self):
        """Shutdown the SGLang engine process."""
        if self.engine:
            self.engine.shutdown()

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

    def _filter_validate_samples(self, samples):
        validate_samples = []
        for sample in samples:
            if sample.extra_info and isinstance(sample.extra_info, dict) and sample.extra_info.get("padded_duplicate", None):
                continue
            validate_samples.append(sample)
        return validate_samples

    def _start_validate_progress(self, total: int):
        now = time.time()
        self._validate_progress = {
            "total": int(total),
            "done": 0,
            "active": True,
            "start_time": now,
            "updated_at": now,
        }

    def _update_validate_progress(self, delta: int = 1):
        self._validate_progress["done"] += int(delta)
        self._validate_progress["updated_at"] = time.time()

    def _finish_validate_progress(self):
        total = int(self._validate_progress.get("total", 0))
        done = int(self._validate_progress.get("done", 0))
        self._validate_progress["done"] = max(done, total)
        self._validate_progress["active"] = False
        self._validate_progress["updated_at"] = time.time()

    def get_validate_progress(self) -> dict:
        progress = dict(self._validate_progress)
        start_time = float(progress.get("start_time", 0.0))
        progress["elapsed"] = max(time.time() - start_time, 0.0) if start_time > 0 else 0.0
        return progress

    async def validate(self, val_batch_size, global_step):
        logger.debug(f"[RolloutWorker rank={self.rank}] Starting validation @ global step {global_step}")
        samples, val_time_metrics = await self.executor.validate(val_batch_size)
        return self._filter_validate_samples(samples), val_time_metrics

    async def validate_assigned(self, val_samples, global_step):
        logger.debug(f"[RolloutWorker rank={self.rank}] Starting assigned validation @ global step {global_step}")
        self._start_validate_progress(len(val_samples))
        samples = []
        val_time_metrics = {}
        try:
            samples, val_time_metrics = await self.executor.validate_samples(
                val_samples,
                use_router=False,
                progress_callback=self._update_validate_progress,
            )
        finally:
            self._finish_validate_progress()
        return self._filter_validate_samples(samples), val_time_metrics

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

    def param_sync_from_tensor(
        self,
        serialized_named_tensors,
        flush_cache=False,
        weight_version: str | None = None,
        load_format: str | None = None,
        trace_id: str | None = None,
        bucket_idx: int | None = None,
        part_idx: int | None = None,
        part_count: int | None = None,
        sync_key: str | None = None,
        lane_idx: int | None = None,
        route_epoch: int | None = None,
    ):
        trace_id = trace_id or "na"
        payload_bytes = 0
        payload_min = None
        payload_max = None
        if isinstance(serialized_named_tensors, list):
            for item in serialized_named_tensors:
                if isinstance(item, (bytes, bytearray, str)):
                    item_len = len(item)
                    payload_bytes += item_len
                    payload_min = item_len if payload_min is None else min(payload_min, item_len)
                    payload_max = item_len if payload_max is None else max(payload_max, item_len)
        start = time.monotonic()
        try:
            result = self.engine.param_sync_from_tensor(
                serialized_named_tensors,
                flush_cache=flush_cache,
                weight_version=weight_version,
                load_format=load_format,
                trace_id=trace_id,
                bucket_idx=bucket_idx,
                part_idx=part_idx,
                part_count=part_count,
                sync_key=sync_key,
                lane_idx=lane_idx,
                route_epoch=route_epoch,
            )
            return result
        except Exception:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.error(
                "[RolloutWorker] param_sync_from_tensor_failed trace_id={} rank={} "
                "weight_version={} load_format={} bucket_idx={} part_idx={} part_count={} elapsed_ms={} "
                "payload_bytes={} payload_min={} payload_max={} err={} debug_state={}",
                trace_id,
                self.rank,
                weight_version,
                load_format or "tensor",
                bucket_idx if bucket_idx is not None else -1,
                part_idx if part_idx is not None else -1,
                part_count if part_count is not None else -1,
                round(elapsed_ms, 2),
                payload_bytes,
                payload_min if payload_min is not None else -1,
                payload_max if payload_max is not None else -1,
                traceback.format_exc(),
                self.get_debug_state(),
            )
            raise

    def destroy_weights_update_group(self, group_name):
        return self.engine.destroy_weights_update_group(group_name)

    def flush_cache(self):
        self.engine.flush_cache()

    def pause_generation(self):
        self.engine.pause_generation()

    def continue_generation(self):
        return self.engine.continue_generation()

    def abort_generation(self, timeout_s: int | None = None):
        timeout_s = timeout_s or self.engine._rpc_timeout_s()
        self.engine.pause_generation_dispatch()
        if self.executor is not None:
            self.executor.pause_dispatch()
        result = self.engine.abort_generation(abort_all=True)
        if not self.engine.wait_for_no_inflight_generation(timeout_s=timeout_s):
            raise TimeoutError(
                f"Timed out waiting for in-flight generation to abort on rollout worker rank={self.rank}"
            )
        return result

    def resume_generation(self):
        if self.executor is not None:
            self.executor.resume_dispatch()
        self.engine.resume_generation_dispatch()
        return True

    def offload_memory(self, tags: list[str] | None = None):
        """Release GPU memory occupation for colocated mode."""
        try:
            return self.engine.release_memory_occupation(tags)
        except Exception:
            logger.error(
                "[COLOCATE_DEBUG][RolloutWorker rank={}] offload_memory failed tags={} debug_state={}",
                self.rank,
                tags,
                self.get_debug_state(),
            )
            raise

    def onload_memory(self, tags: list[str] | None = None):
        """Resume GPU memory occupation for colocated mode."""
        try:
            return self.engine.resume_memory_occupation(tags)
        except Exception:
            logger.error(
                "[COLOCATE_DEBUG][RolloutWorker rank={}] onload_memory failed tags={} debug_state={}",
                self.rank,
                tags,
                self.get_debug_state(),
            )
            raise

    def _collect_cuda_debug_stats(self) -> dict:
        stats = {}
        try:
            import torch
        except Exception as e:
            stats["torch_import_error"] = repr(e)
            return stats

        try:
            if not torch.cuda.is_available():
                stats["cuda_available"] = False
                return stats

            device = torch.cuda.current_device()
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            stats.update(
                {
                    "cuda_available": True,
                    "device": int(device),
                    "free_gb": round(free_bytes / (1024**3), 3),
                    "total_gb": round(total_bytes / (1024**3), 3),
                    "allocated_gb": round(torch.cuda.memory_allocated(device) / (1024**3), 3),
                    "reserved_gb": round(torch.cuda.memory_reserved(device) / (1024**3), 3),
                    "max_allocated_gb": round(torch.cuda.max_memory_allocated(device) / (1024**3), 3),
                    "max_reserved_gb": round(torch.cuda.max_memory_reserved(device) / (1024**3), 3),
                }
            )
            return stats
        except Exception as e:
            stats["cuda_stats_error"] = repr(e)
            return stats

    def get_debug_state(self) -> dict:
        process = getattr(self.engine, "process", None)
        return {
            "rank": self.rank,
            "ip": self.ip,
            "port": self.port,
            "engine_pid": getattr(process, "pid", None),
            "engine_alive": bool(process and process.is_alive()),
            "engine_exitcode": getattr(process, "exitcode", None),
            "cuda": self._collect_cuda_debug_stats(),
            "ts": round(time.time(), 3),
        }

    def weight_version(self):
        return self.engine._weight_version
