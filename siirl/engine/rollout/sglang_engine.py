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
import multiprocessing
import time

import requests
from loguru import logger
from requests.exceptions import RequestException


from sglang.utils import async_stream_and_merge, stream_and_merge
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs



from siirl.params.training_args import SiiRLArguments
from siirl.utils.backend.net import get_net_interface_ip


def wait_until_ok(
    url: str,
    *,
    process: "multiprocessing.Process" = None,
    max_wait: int = 3000,
    interval: int = 2,
    timeout: int = 3000,
    extra_headers: dict | None = None,
) -> None:
    """Block until `url` returns 200 or time-out."""
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        if process and not process.is_alive():
            raise RuntimeError(f"Server process terminated unexpectedly. {process} {process.is_alive()}")
        try:
            if requests.get(url, timeout=timeout, headers=extra_headers or {}).status_code == 200:
                return
        except RequestException as exc:
            logger.debug("Request failed: %s", exc)
        time.sleep(interval)

    raise RuntimeError(f"Health check failed after {max_wait} seconds.")


class SglangEngine:
    def __init__(self, rank: int, config: SiiRLArguments, dist_init_addr: str, ip: str, port: int , nccl_port: int):
        self.rank = rank
        self.config = config
        self.dist_init_addr = dist_init_addr
        self.port = port
        self.nccl_port = nccl_port
        self.ip = ip
        self.launch_server()
        
    def get_sglang_params(self, base_gpu_id, node_rank, nnodes):
        config = self.config.rollout
        print(f"model_path ", self.config.actor_rollout_ref.model.path)
        
        args = {
            "model_path": self.config.actor_rollout_ref.model.path,
            "dtype": config.dtype,
            "random_seed": self.config.rollout.seed + self.rank,
            "mem_fraction_static": config.gpu_memory_utilization,
            "enable_memory_saver": True,
            "base_gpu_id": base_gpu_id,
            "gpu_id_step": 1,
            "tp_size": config.tensor_model_parallel_size,
            "node_rank": node_rank,
            "load_format": "auto",
            "dist_init_addr": self.dist_init_addr,
            "nnodes": nnodes,
            "trust_remote_code": config.trust_remote_code,
            "max_running_requests": config.max_num_seqs,
            "host": self.ip,
            "port": self.port,
            "nccl_port": self.nccl_port,
            "log_level": "info",
            "mm_attention_backend": "fa3",
            "attention_backend": "fa3",
            # In async mode, we want token in token out.
            "skip_tokenizer_init": False,
            "dist_timeout": 1800,
            "skip_server_warmup": True,
        }
        return args
        
    def launch_server(self):   
        base_gpu_id = self.rank % self.config.trainer.n_gpus_per_node
        node_rank = self.rank // self.config.trainer.n_gpus_per_node
        nnodes = max(1, self.config.rollout.tensor_model_parallel_size // self.config.trainer.n_gpus_per_node)
        
        
        args = self.get_sglang_params(base_gpu_id, node_rank, nnodes)
        sgl_args = ServerArgs(**args)
        print(f"Launch SglangHttpServer at: {get_net_interface_ip()}:{self.port}")
        multiprocessing.set_start_method("spawn", force=True)
        self.process = multiprocessing.Process(target=launch_server, args=(sgl_args,))
        self.process.start()
        base_url = sgl_args.url()
        wait_until_ok(
            f"{base_url}/health_generate" if sgl_args.is_embedding else f"{base_url}/health",
            process=self.process,
            extra_headers={"Authorization": f"Bearer {sgl_args.api_key}"},
        )
        # Ensure cache is ready
        wait_until_ok(
            f"{base_url}/flush_cache",
            process=self.process,
            extra_headers={"Authorization": f"Bearer {sgl_args.api_key}"},
        )

        