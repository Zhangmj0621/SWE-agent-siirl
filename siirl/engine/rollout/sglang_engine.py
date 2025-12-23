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
from urllib3.exceptions import NewConnectionError

import requests
from loguru import logger
from typing import List, Dict

from sglang.utils import async_stream_and_merge, stream_and_merge
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs


from siirl.models.loader import load_tokenizer
from siirl.params.training_args import SiiRLArguments
from siirl.utils.net_utils.net import get_net_interface_ip
from siirl.utils.net_utils.http_utils import GlobalAsyncHTTPClient, wait_until_ok

global_engine_process = None

class SglangEngine:
    def __init__(self, rank: int, config: SiiRLArguments, dist_init_addr: str, ip: str, port: int , nccl_port: int):
        self.rank = rank
        self.config = config
        self.dist_init_addr = dist_init_addr
        self.port = port
        self.nccl_port = nccl_port
        self.ip = ip
        self.tokenizer = load_tokenizer(path = config.actor_ref.model.path, model_args = config.actor_ref.model)
        self.launch_server()
        
    def get_sglang_params(self, base_gpu_id, node_rank, nnodes):
        config = self.config.rollout
        print(f"model_path ", self.config.actor_ref.model.path)
        
        args = {
            "model_path": self.config.actor_ref.model.path,
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
        self.sgl_args = ServerArgs(**args)
        print(f"Launch SglangHttpServer at: {get_net_interface_ip()}:{self.port}")
        multiprocessing.set_start_method("spawn", force=True)
        self.process = multiprocessing.Process(target=launch_server, args=(self.sgl_args,))
        self.process.start()
        base_url = self.sgl_args.url()
        wait_until_ok(
            f"{base_url}/health_generate" if self.sgl_args.is_embedding else f"{base_url}/health",
            process=self.process,
            extra_headers={"Authorization": f"Bearer {self.sgl_args.api_key}"},
        )
        # Ensure cache is ready
        wait_until_ok(
            f"{base_url}/flush_cache",
            process=self.process,
            extra_headers={"Authorization": f"Bearer {self.sgl_args.api_key}"},
        )

    def set_router(self, router_address):
        self.router_address = router_address
    
    async def generate(self, input_ids:List[int], sampling_params:Dict):
        # url = f"http://{self.ip}:{self.port}/generate"
        url = f"http://{self.router_address}/generate"
        # Prepare payload for sglang server
        payload = {
            "sampling_params": sampling_params,
            "return_logprob": True,
        }
        payload["input_ids"] = input_ids
        output = await GlobalAsyncHTTPClient.make_request(url, payload, "POST")
        responses = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        rollout_log_prob = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        text = output['text'] 
        return text, responses, rollout_log_prob

    def flush_cache(self):
        """Flush the cache of the server."""
        if self.rank != 0:
            return
        # flush cache will not return status_code 200 when there are pending requests
        for _ in range(60):
            try:
                response = requests.get(f"{self.sgl_args.url()}/flush_cache")
                if response.status_code == 200:
                    break
            except NewConnectionError as e:
                raise e
            except Exception as e:
                logger.info(f"Error flushing cache: {e}")
                time.sleep(1)
                continue
        else:
            raise TimeoutError("Timeout while flushing cache.")

    def pause_generation(self):
        response = requests.post(f"{self.sgl_args.url()}/pause_generation", json={})
        response.raise_for_status()
        return response

    def continue_generation(self):
        response = requests.post(f"{self.sgl_args.url()}/continue_generation", json={})
        response.raise_for_status()
        return response

    def _make_request(self, endpoint: str, payload: dict | None = None):
        """Make a POST request to the specified endpoint with the given payload.

        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)

        Returns:
            The JSON response from the server
        """
        if self.sgl_args.node_rank != 0:
            return

        url = f"{self.sgl_args.url()}/{endpoint}"
        response = requests.post(url, json=payload or {})
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            e.add_note(f"{response.text=}")
            raise
        return response.json()


    def init_param_sync_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        return self._make_request(
            "init_weights_update_group",
            {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
                "group_name": group_name,
                "backend": backend,
            },
        )

    def sync_param_from_distributed(
        self, names, dtypes, shapes, group_name, flush_cache=False, weight_version: str | None = None
    ):
        payload = {
            "names": names,
            "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
            "shapes": shapes,
            "group_name": group_name,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._make_request(
            "update_weights_from_distributed",
            payload,
        )

    def destroy_weights_update_group(self, group_name):
        try:
            return self._make_request(
                "destroy_weights_update_group",
                {
                    "group_name": group_name,
                },
            )
        except requests.exceptions.RequestException:
            # catch the case there the engine is just created and does not have the group.
            pass


