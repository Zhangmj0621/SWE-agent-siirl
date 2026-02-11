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
import copy
import multiprocessing
import os
import time
from collections.abc import Callable

import requests
from loguru import logger
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from urllib3.exceptions import NewConnectionError

from siirl.execution.rollout.concurrency import resolve_rollout_concurrency
from siirl.models.loader import load_tokenizer
from siirl.params.training_args import SiiRLArguments
from siirl.utils.net_utils.http_utils import GlobalAsyncHTTPClient, wait_until_ok
from siirl.utils.net_utils.net import get_net_interface_ip


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
        base_gpu_id: int,
        node_rank: int,
        nnodes: int,
        nccl_port: int | None = None,
        extra_server_args: dict | None = None,
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
            base_gpu_id: Starting CUDA device ID for this TP group (from GPUResources).
            node_rank: Rank of this node within the TP group (0 for single-node TP).
            nnodes: Number of nodes participating in this TP group (1 for single-node TP).
            nccl_port: Port for NCCL communication. If None, SGLang auto-allocates.
        """
        if extra_server_args is None:
            extra_server_args = {}
        self.rank = rank
        self.config = config
        self.dist_init_addr = dist_init_addr
        self.port = port
        self.nccl_port = nccl_port
        self.ip = ip
        self.router_address = None
        self._weight_version = 0
        # GPU placement parameters (directly passed, not calculated)
        self.base_gpu_id = base_gpu_id
        self.node_rank = node_rank
        self.nnodes = nnodes

        # init some local_parms
        self.tokenizer = load_tokenizer(path=config.actor_ref.model.path, model_args=config.actor_ref.model)
        self.max_model_len = (
            config.rollout.max_model_len
            if config.rollout.max_model_len
            else config.data.max_prompt_length + config.data.max_response_length
        )
        self.max_response_length = config.data.max_response_length
        # Sampling parameters for text generation (LLM inference config)
        self.sampling_params = dict(
            temperature=config.rollout.temperature,
            top_p=config.rollout.top_p,
            top_k=config.rollout.top_k,
            repetition_penalty=1.0,
        )
        self._extra_server_args = extra_server_args
        self.process = None  # Server process, started by launch_server()

    def _build_server_args(self) -> dict:
        """
        Build SGLang server arguments using pre-computed GPU placement info.

        Returns:
            Dictionary of arguments for SGLang ServerArgs.
        """
        config = self.config.rollout
        logger.info(
            f"Building SGLang server args: model_path={self.config.actor_ref.model.path}, "
            f"base_gpu_id={self.base_gpu_id}, node_rank={self.node_rank}, nnodes={self.nnodes}"
        )

        args = {
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
            # Server settings
            "trust_remote_code": config.trust_remote_code,
            "max_running_requests": config.max_num_seqs,
            "log_level": "warning",
            "mm_attention_backend": "fa3",
            "attention_backend": "fa3",
            "skip_tokenizer_init": False,
            "dist_timeout": 1800,
        }

        # Only set nccl_port if explicitly provided (None = SGLang auto-allocates)
        if self.nccl_port is not None:
            args["nccl_port"] = self.nccl_port

        return args

    def launch_server(self, extra_server_args: dict | None = None):
        """
        Launch the SGLang HTTP server in a separate process.

        Uses pre-computed GPU placement parameters (base_gpu_id, node_rank, nnodes)
        instead of calculating them internally, ensuring correct GPU assignment
        in complex multi-node and cross-node TP scenarios.
        """
        if extra_server_args is None:
            extra_server_args = self._extra_server_args
        args = self._build_server_args()
        args.update(extra_server_args)
        self.sgl_args = ServerArgs(**args)
        logger.info(f"Launch SglangHttpServer at: {get_net_interface_ip()}:{self.port}")
        ctx = multiprocessing.get_context("spawn")
        self.process = ctx.Process(target=launch_server, args=(self.sgl_args,))
        self.process.start()
        base_url = self.sgl_args.url()
        wait_until_ok(
            (f"{base_url}/health_generate" if self.sgl_args.is_embedding else f"{base_url}/health"),
            process=self.process,
            extra_headers={"Authorization": f"Bearer {self.sgl_args.api_key}"},
        )
        # Ensure cache is ready
        wait_until_ok(
            f"{base_url}/flush_cache",
            process=self.process,
            extra_headers={"Authorization": f"Bearer {self.sgl_args.api_key}"},
        )
        # Verify scheduler is alive by testing actual generation
        self._verify_ready(base_url)

    def _verify_ready(self, base_url: str, timeout: int = 60):
        """Verify server can generate (catches scheduler crashes like port conflicts)."""
        payload = {"text": "Hi", "sampling_params": {"max_new_tokens": 1}}
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if self.process and not self.process.is_alive():
                raise RuntimeError("SGLang server process died. Check logs for port conflicts (EADDRINUSE).")
            try:
                if requests.post(f"{base_url}/generate", json=payload, timeout=10).ok:
                    logger.info(f"SGLang server at {base_url} is ready.")
                    return
            except requests.RequestException:
                pass
            time.sleep(2)

        raise RuntimeError(f"SGLang server at {base_url} not ready after {timeout}s.")

    def shutdown(self):
        """Terminate the SGLang server process gracefully."""
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=5)
            logger.info(f"SGLang server process at {self.ip}:{self.port} terminated.")
        self.process = None

    def set_router(self, router_address):
        self.router_address = router_address

    def _rpc_timeout_s(self) -> int:
        return max(1, int(getattr(self.config.trainer, "param_sync_rpc_timeout_s", 120)))

    def _get_sampling_params(self, is_validate: bool, input_len: int | None = None) -> dict:
        """Get sampling parameters based on mode (train/validate)."""
        params = copy.deepcopy(self.sampling_params)
        if input_len is not None:
            params["max_new_tokens"] = min(self.max_model_len - input_len, self.max_response_length)
        if is_validate:
            params.update(
                {
                    "top_k": self.config.rollout.val_kwargs.top_k,
                    "top_p": self.config.rollout.val_kwargs.top_p,
                    "temperature": self.config.rollout.val_kwargs.temperature,
                }
            )
        return params

    def _get_generate_url(self, use_router: bool = False) -> str:
        """Get generation endpoint URL."""
        if use_router and self.router_address:
            return f"http://{self.router_address}/generate"
        return f"http://{self.ip}:{self.port}/generate"

    async def generate(self, input_ids: list[int], is_validate: bool, use_router: bool = False):
        """Single sample generation with optional router load balancing."""
        sampling_params = self._get_sampling_params(is_validate, len(input_ids))
        url = self._get_generate_url(use_router=use_router)

        payload = {
            "input_ids": input_ids,
            "sampling_params": sampling_params,
            "return_logprob": True,
        }
        output = await GlobalAsyncHTTPClient.make_request(url, payload, "POST")
        responses = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        rollout_log_prob = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        return output["text"], responses, rollout_log_prob

    def _resolve_batch_concurrency(self, use_router: bool) -> int:
        limits = resolve_rollout_concurrency(self.config, phase="validate", use_router=use_router)
        logger.debug(
            "Batch concurrency: "
            f"phase={limits['phase']}, use_router={bool(limits['use_router'])}, "
            f"base_key={limits['base_key']}, base={limits['base']}, "
            f"num_engines={limits['num_engines']}, resolved={limits['resolved']}, "
            f"max_num_seqs={limits['max_num_seqs']}, effective={limits['effective']}"
        )
        return int(limits["effective"])

    async def generate_batch(
        self,
        batch_input_ids: list[list[int]],
        is_validate: bool = True,
        use_router: bool = True,
        show_progress: bool = True,
        progress_desc: str = "Validate",
        sort_by_length: bool = True,
        progress_callback: Callable[[int], None] | None = None,
    ) -> list[tuple[str, list[int], list[float]]]:
        """
        Batch generation for single-turn scenarios (no multi-turn/tool calls).
        Uses router for load balancing across multiple engines.

        Phase 2 optimization: sort by sequence length to reduce tail latency.

        Args:
            batch_input_ids: List of input token id lists.
            is_validate: Whether this is validation mode.
            use_router: Whether to use router for load balancing.
            show_progress: Whether to show progress bar.
            progress_desc: Description for progress bar.
            sort_by_length: If True, sort requests by prompt length to optimize batching.

        Returns:
            List of (text, response_ids, log_probs) tuples for each input.
        """
        if not batch_input_ids:
            return []

        log_enabled = os.environ.get("SIIRL_SGLANG_BATCH_LOG", "0") == "1"
        start_time = time.perf_counter()
        prompt_tokens = sum(len(ids) for ids in batch_input_ids)

        url = self._get_generate_url(use_router=use_router)

        # Phase 2: Sort by sequence length to reduce tail latency
        # Longer sequences take more time; processing them together reduces variance
        if sort_by_length:
            # Create index mapping for sorting and later restoration
            indexed_inputs = [(i, ids) for i, ids in enumerate(batch_input_ids)]
            # Sort by length (descending: longer first for better GPU utilization)
            indexed_inputs.sort(key=lambda x: len(x[1]), reverse=True)
            sorted_indices = [idx for idx, _ in indexed_inputs]
            sorted_input_ids = [ids for _, ids in indexed_inputs]
        else:
            sorted_indices = list(range(len(batch_input_ids)))
            sorted_input_ids = batch_input_ids

        # Use semaphore to control concurrency (prevent overwhelming the server).
        # Router path: scale by num_engines (router distributes across all engines).
        # Local/validate path: use validate_server_concurrency (single-engine scope).
        max_concurrent = self._resolve_batch_concurrency(use_router)
        semaphore = asyncio.Semaphore(max_concurrent)

        # Create concurrent tasks for each sample (SGLang handles batching internally)
        async def _generate_one(input_ids: list[int]):
            async with semaphore:
                sampling_params = self._get_sampling_params(is_validate, len(input_ids))
                payload = {
                    "input_ids": input_ids,
                    "sampling_params": sampling_params,
                    "return_logprob": True,
                }
                output = await GlobalAsyncHTTPClient.make_request(url, payload, "POST")
            responses = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
            if progress_callback is not None:
                progress_callback(1)
            return output["text"], responses, log_probs

        # SGLang handles continuous batching internally
        tasks = [_generate_one(ids) for ids in sorted_input_ids]

        if show_progress:
            from tqdm.asyncio import tqdm_asyncio

            total = len(tasks)
            logger.info(f"{progress_desc}: Starting {total} samples...")

            sorted_results = await tqdm_asyncio.gather(
                *tasks,
                desc=progress_desc,
                unit="sample",
                dynamic_ncols=True,
                mininterval=2.0,
                miniters=50,
            )

            logger.info(f"{progress_desc}: Completed {total} samples")
        else:
            sorted_results = await asyncio.gather(*tasks)

        # Phase 2: Restore original order
        if sort_by_length:
            results = [None] * len(batch_input_ids)
            for sorted_idx, original_idx in enumerate(sorted_indices):
                results[original_idx] = sorted_results[sorted_idx]
        else:
            results = sorted_results

        if log_enabled:
            total_response_tokens = sum(len(item[1]) for item in results)
            elapsed = time.perf_counter() - start_time
            total_tokens = prompt_tokens + total_response_tokens
            tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
            logger.info(
                f"SGLang batch stats: batch={len(batch_input_ids)}, prompt_tokens={prompt_tokens}, "
                f"response_tokens={total_response_tokens}, total_tokens={total_tokens}, "
                f"elapsed={elapsed:.2f}s, tokens_per_sec={tokens_per_sec:.2f}"
            )

        return results

    async def generate_from_text(self, input_text: "str", sampling_params: dict, use_sglang_router=False):
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
        text = output["text"]
        return text, responses, rollout_log_prob

    def flush_cache(self):
        """Flush the cache of the server."""
        if self.rank != 0:
            return
        timeout_s = self._rpc_timeout_s()
        # flush cache will not return status_code 200 when there are pending requests
        for _ in range(60):
            try:
                response = requests.get(f"{self.sgl_args.url()}/flush_cache", timeout=timeout_s)
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
        response = requests.post(f"{self.sgl_args.url()}/pause_generation", json={}, timeout=self._rpc_timeout_s())
        response.raise_for_status()
        return response

    def continue_generation(self):
        response = requests.post(f"{self.sgl_args.url()}/continue_generation", json={}, timeout=self._rpc_timeout_s())
        response.raise_for_status()
        return response

    def release_memory_occupation(self, tags: list[str] | None = None):
        """Release GPU memory occupation (weights/kv_cache) for colocated mode.

        Tells the SGLang server to free specified GPU memory regions so that the
        trainer can load its model onto the same GPU without OOM.

        Args:
            tags: Memory region tags to release. Supported: ["weights", "kv_cache", "cuda_graph"].
                  If None, releases all regions.
        """
        payload = {"tags": tags} if tags is not None else {}
        return self._make_request("release_memory_occupation", payload)

    def resume_memory_occupation(self, tags: list[str] | None = None):
        """Resume GPU memory occupation (weights/kv_cache) after colocated training.

        Tells the SGLang server to re-allocate specified GPU memory regions after
        the trainer has offloaded its model back to CPU.

        Args:
            tags: Memory region tags to resume. Supported: ["weights", "kv_cache", "cuda_graph"].
                  If None, resumes all regions.
        """
        payload = {"tags": tags} if tags is not None else {}
        return self._make_request("resume_memory_occupation", payload)

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
        response = requests.post(url, json=payload or {}, timeout=self._rpc_timeout_s())
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            logger.error(f"[ERROR] HTTP {response.status_code}: {response.text[:500]}")
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
        self,
        names,
        dtypes,
        shapes,
        group_name,
        flush_cache=True,
        weight_version: str | None = None,
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
            self._weight_version = int(weight_version)
        else:
            self._weight_version += 1
        return result

    def param_sync_from_tensor(
        self,
        serialized_named_tensors,
        flush_cache=True,
        weight_version: str | None = None,
        load_format: str | None = None,
    ):
        import base64

        # HTTP JSON boundary: bytes must be base64-encoded; str passes through.
        encoded = []
        for item in serialized_named_tensors:
            if isinstance(item, (bytes, bytearray)):
                encoded.append(base64.b64encode(item).decode("ascii"))
            else:
                encoded.append(item)

        payload = {
            "serialized_named_tensors": encoded,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        if load_format is not None:
            payload["load_format"] = load_format
        result = self._make_request("update_weights_from_tensor", payload)
        if weight_version:
            self._weight_version = int(weight_version)
        else:
            self._weight_version += 1
        return result

    def destroy_weights_update_group(self, group_name):
        if self.sgl_args.node_rank != 0:
            return

        url = f"{self.sgl_args.url()}/destroy_weights_update_group"
        response = requests.post(url, json={"group_name": group_name}, timeout=self._rpc_timeout_s())
        if response.status_code < 400:
            return response.json()
        if "does not exist" in response.text:
            return
        response.raise_for_status()
