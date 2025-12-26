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
    """
    SGLang inference engine wrapper for distributed rollout.
    
    Manages the SGLang server process lifecycle and provides HTTP interface
    for text generation, cache management, and parameter synchronization.
    """
    
    def __init__(
        self, 
        rank: int, 
        config: SiiRLArguments, 
        dist_init_addr: str, 
        ip: str, 
        port: int, 
        nccl_port: int,
        base_gpu_id: int,
        node_rank: int,
        nnodes: int,
        extra_server_args:dict = {},
    ):
        """
        Initialize SGLang engine with explicit GPU placement parameters.
        
        Args:
            rank: Global rank of this engine instance (TP0 rank within rollout workers).
            config: SiiRLArguments configuration object.
            dist_init_addr: Address for distributed initialization (ip:port format).
                            Used for cross-node TP communication.
            ip: IP address to bind the SGLang HTTP server.
            port: Port number for the SGLang HTTP server.
            nccl_port: Port number for NCCL backend communication.
            base_gpu_id: Starting CUDA device ID for this TP group (from GPUResources).
            node_rank: Rank of this node within the TP group (0 for single-node TP).
            nnodes: Number of nodes participating in this TP group (1 for single-node TP).
        """
        self.rank = rank
        self.config = config
        self.dist_init_addr = dist_init_addr
        self.port = port
        self.nccl_port = nccl_port
        self.ip = ip
        self.weight_version = 0
        # GPU placement parameters (directly passed, not calculated)
        self.base_gpu_id = base_gpu_id
        self.node_rank = node_rank
        self.nnodes = nnodes
        
        # init some local_parms
        self.tokenizer = load_tokenizer(
            path=config.actor_ref.model.path, 
            model_args=config.actor_ref.model
        )
        self.max_model_len = config.rollout.max_model_len if config.rollout.max_model_len else config.data.max_prompt_length + config.data.max_response_length
        self.max_response_length = config.data.max_response_length
        self.launch_server(extra_server_args)
        
    def _build_server_args(self) -> dict:
        """
        Build SGLang server arguments using pre-computed GPU placement info.
        
        Returns:
            Dictionary of arguments for SGLang ServerArgs.
        """
        config = self.config.rollout
        logger.info(f"Building SGLang server args: model_path={self.config.actor_ref.model.path}, "
                    f"base_gpu_id={self.base_gpu_id}, node_rank={self.node_rank}, nnodes={self.nnodes}")
        
        return {
            "model_path": self.config.actor_ref.model.path,
            "dtype": config.dtype,
            "random_seed": config.seed + self.rank,
            "mem_fraction_static": config.gpu_memory_utilization,
            "enable_memory_saver": True,
            # GPU placement parameters (directly from RolloutManager)
            "base_gpu_id": self.base_gpu_id,
            "gpu_id_step": 1,
            "tp_size": config.tensor_model_parallel_size,
            "node_rank": self.node_rank,
            "nnodes": self.nnodes,
            # Distributed communication
            "load_format": "auto",
            "dist_init_addr": self.dist_init_addr,
            # Network configuration
            "host": self.ip,
            "port": self.port,
            "nccl_port": self.nccl_port,
            # Server settings
            "trust_remote_code": config.trust_remote_code,
            "max_running_requests": config.max_num_seqs,
            "log_level": "warning",
            "mm_attention_backend": "fa3",
            "attention_backend": "fa3",
            "skip_tokenizer_init": False,
            "dist_timeout": 1800,
            "skip_server_warmup": True,
        }

    def launch_server(self, extra_server_args = {}):
        """
        Launch the SGLang HTTP server in a separate process.
        
        Uses pre-computed GPU placement parameters (base_gpu_id, node_rank, nnodes)
        instead of calculating them internally, ensuring correct GPU assignment
        in complex multi-node and cross-node TP scenarios.
        """
        args = self._build_server_args()
        args.update(extra_server_args)
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
        sampling_params['max_new_tokens'] = min(self.max_model_len - len(input_ids), self.max_response_length)
        url = f"http://{self.ip}:{self.port}/generate"
        # url = f"http://{self.router_address}/generate"
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
    
    async def generate_from_text(self, input_text:"str", sampling_params:Dict, use_sglang_router = False):
        url = f"http://{self.ip}:{self.port}/generate"
        if use_sglang_router:
            url = f"http://{self.router_address}/generate"
        # Prepare payload for sglang server
        payload = {
            "sampling_params": sampling_params,
            "return_logprob": True,
        }
        payload["text"] = input_text
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

    def param_sync_from_distributed(
        self, names, dtypes, shapes, group_name, flush_cache=True, weight_version: str | None = None
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
        result = self._make_request(
            "update_weights_from_distributed",
            payload,
        )
        if weight_version:
            self.weight_version = int(weight_version)
        else:
            self.weight_version += 1
        return result

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


    