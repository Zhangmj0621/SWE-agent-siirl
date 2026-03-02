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

from siirl.execution.rollout.concurrency import resolve_max_num_seqs, resolve_rollout_concurrency
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
            "max_running_requests": resolve_max_num_seqs(self.config),
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

    def _control_plane_timeout_s(self) -> int:
        ray_timeout = max(1, int(getattr(self.config.trainer, "colocate_timeout_s", 60)))
        # Keep a small buffer so the outer Ray wait is still the hard deadline.
        return ray_timeout if ray_timeout <= 5 else ray_timeout - 5

    def _get_sampling_params(
        self,
        is_validate: bool,
        input_len: int | None = None,
        request_seed: int | None = None,
    ) -> dict:
        """Get sampling parameters based on mode (train/validate)."""
        params = copy.deepcopy(self.sampling_params)
        if input_len is not None:
            params["max_new_tokens"] = min(self.max_model_len - input_len, self.max_response_length)
        if is_validate:
            val_kwargs = self.config.rollout.val_kwargs
            do_sample = bool(getattr(val_kwargs, "do_sample", False))
            params.update(
                {
                    "n": max(1, int(getattr(val_kwargs, "n", 1))),
                    "top_k": int(getattr(val_kwargs, "top_k", -1)),
                    "top_p": float(getattr(val_kwargs, "top_p", 1.0)),
                    "temperature": float(getattr(val_kwargs, "temperature", 0.0)),
                }
            )
            if not do_sample:
                params.update(
                    {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "top_k": 1,
                        "n": 1,
                    }
                )
        if request_seed is not None and ((not is_validate) or bool(getattr(self.config.rollout.val_kwargs, "do_sample", False))):
            params["random_seed"] = int(request_seed)
        return params

    def _get_generate_url(self, use_router: bool = False) -> str:
        """Get generation endpoint URL."""
        if use_router and self.router_address:
            return f"http://{self.router_address}/generate"
        return f"http://{self.ip}:{self.port}/generate"

    async def generate(
        self,
        input_ids: list[int],
        is_validate: bool,
        use_router: bool = False,
        request_seed: int | None = None,
    ):
        """Single sample generation with optional router load balancing."""
        sampling_params = self._get_sampling_params(
            is_validate,
            len(input_ids),
            request_seed=request_seed,
        )
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
        request_seeds: list[int] | None = None,
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

        if request_seeds is not None and len(request_seeds) != len(batch_input_ids):
            raise ValueError(f"request_seeds length mismatch: expected {len(batch_input_ids)}, got {len(request_seeds)}")
        if request_seeds is None:
            sorted_request_seeds = [None] * len(sorted_input_ids)
        elif sort_by_length:
            sorted_request_seeds = [request_seeds[idx] for idx in sorted_indices]
        else:
            sorted_request_seeds = request_seeds

        # Use semaphore to control concurrency (prevent overwhelming the server).
        # Router path: scale by num_engines (router distributes across all engines).
        # Local/validate path: use train_server_concurrency.
        max_concurrent = self._resolve_batch_concurrency(use_router)
        semaphore = asyncio.Semaphore(max_concurrent)

        # Create concurrent tasks for each sample (SGLang handles batching internally)
        async def _generate_one(input_ids: list[int], request_seed: int | None = None):
            async with semaphore:
                sampling_params = self._get_sampling_params(
                    is_validate,
                    len(input_ids),
                    request_seed=request_seed,
                )
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
        tasks = [_generate_one(ids, request_seed=seed) for ids, seed in zip(sorted_input_ids, sorted_request_seeds, strict=False)]

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
        if self.sgl_args.node_rank != 0:
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

    _CONTROL_PLANE_ENDPOINTS = frozenset(
        {
            "release_memory_occupation",
            "resume_memory_occupation",
        }
    )

    _RETRYABLE_ENDPOINTS = frozenset(
        {
            "update_weights_from_tensor",
            "update_weights_from_distributed",
            "release_memory_occupation",
            "resume_memory_occupation",
        }
    )

    def _control_plane_retry_params(self) -> tuple[int, int]:
        """Use a single long control-plane call instead of short retries."""
        return self._control_plane_timeout_s(), 0

    def _process_debug_state(self) -> dict:
        process = getattr(self, "process", None)
        return {
            "pid": getattr(process, "pid", None),
            "alive": bool(process and process.is_alive()),
            "exitcode": getattr(process, "exitcode", None),
        }

    def _probe_server_health(self, timeout_s: int = 2) -> str:
        url = f"{self.sgl_args.url()}/health"
        try:
            response = requests.get(url, timeout=timeout_s)
            return f"ok:{response.status_code}"
        except Exception as e:
            return f"error:{type(e).__name__}:{e}"

    def _make_request(self, endpoint: str, payload: dict | None = None):
        """Make a POST request with endpoint-aware timeout and retry strategy."""
        if self.sgl_args.node_rank != 0:
            return

        url = f"{self.sgl_args.url()}/{endpoint}"
        is_control = endpoint in self._CONTROL_PLANE_ENDPOINTS
        if is_control:
            timeout_s, max_retries = self._control_plane_retry_params()
        else:
            timeout_s = self._rpc_timeout_s()
            max_retries = 2 if endpoint in self._RETRYABLE_ENDPOINTS else 0
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            process = getattr(self, "process", None)
            if process is not None and not process.is_alive():
                raise RuntimeError(
                    f"SGLang process on {self.ip}:{self.port} is dead (exitcode={process.exitcode}); " f"cannot call /{endpoint}"
                )
            try:
                request_start = time.monotonic()
                response = requests.post(url, json=payload or {}, timeout=timeout_s)
                response.raise_for_status()
                if is_control:
                    elapsed_ms = (time.monotonic() - request_start) * 1000
                    if elapsed_ms >= 5_000:
                        logger.warning(
                            "[COLOCATE_DEBUG][SglangEngine] slow control request endpoint={} timeout_s={} elapsed_ms={:.1f} process={}",
                            endpoint,
                            timeout_s,
                            elapsed_ms,
                            self._process_debug_state(),
                        )
                return response.json()
            except requests.exceptions.HTTPError:
                logger.error(f"[ERROR] HTTP {response.status_code}: {response.text[:500]}")
                raise
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_error = e
                if is_control:
                    elapsed_ms = (time.monotonic() - request_start) * 1000
                    health = self._probe_server_health(timeout_s=min(3, timeout_s))
                    logger.error(
                        "[COLOCATE_DEBUG][SglangEngine] control request failed endpoint={} attempt={}/{} timeout_s={} "
                        "elapsed_ms={:.1f} process={} health_probe={} payload={}",
                        endpoint,
                        attempt + 1,
                        max_retries + 1,
                        timeout_s,
                        elapsed_ms,
                        self._process_debug_state(),
                        health,
                        payload,
                    )
                if attempt >= max_retries:
                    raise
                delay_s = min(1.0 * (2**attempt), 4.0)
                logger.warning(
                    f"[SglangEngine] request /{endpoint} failed on {self.ip}:{self.port}, "
                    f"retry {attempt + 1}/{max_retries} in {delay_s:.1f}s: {e}"
                )
                time.sleep(delay_s)

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Unexpected request failure for endpoint /{endpoint}")

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
        trace_id: str | None = None,
        bucket_idx: int | None = None,
        part_idx: int | None = None,
        part_count: int | None = None,
        sync_key: str | None = None,
        lane_idx: int | None = None,
        route_epoch: int | None = None,
    ):
        import base64

        trace_id = trace_id or "na"
        start = time.monotonic()
        # HTTP JSON boundary: bytes must be base64-encoded; str passes through.
        encoded = []
        raw_payload_bytes = 0
        raw_payload_min = None
        raw_payload_max = None
        for item in serialized_named_tensors:
            if isinstance(item, (bytes, bytearray)):
                item_len = len(item)
                raw_payload_bytes += item_len
                raw_payload_min = item_len if raw_payload_min is None else min(raw_payload_min, item_len)
                raw_payload_max = item_len if raw_payload_max is None else max(raw_payload_max, item_len)
                encoded.append(base64.b64encode(item).decode("ascii"))
            else:
                if isinstance(item, str):
                    item_len = len(item)
                    raw_payload_bytes += item_len
                    raw_payload_min = item_len if raw_payload_min is None else min(raw_payload_min, item_len)
                    raw_payload_max = item_len if raw_payload_max is None else max(raw_payload_max, item_len)
                encoded.append(item)
        encoded_payload_chars = sum(len(item) for item in encoded if isinstance(item, str))

        payload = {
            "serialized_named_tensors": encoded,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        if load_format is not None:
            payload["load_format"] = load_format
        if sync_key is not None:
            payload["sync_key"] = sync_key
        if lane_idx is not None:
            payload["lane_idx"] = lane_idx
        if route_epoch is not None:
            payload["route_epoch"] = route_epoch
        try:
            result = self._make_request("update_weights_from_tensor", payload)
        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.error(
                "[SglangEngine] param_sync_from_tensor_failed trace_id={} "
                "rank={} node_rank={} weight_version={} load_format={} bucket_idx={} part_idx={} part_count={} "
                "elapsed_ms={} payload_bytes={} payload_min={} payload_max={} encoded_chars={} err={} process={}",
                trace_id,
                self.rank,
                self.sgl_args.node_rank if hasattr(self, "sgl_args") else -1,
                weight_version,
                load_format or "tensor",
                bucket_idx if bucket_idx is not None else -1,
                part_idx if part_idx is not None else -1,
                part_count if part_count is not None else -1,
                round(elapsed_ms, 2),
                raw_payload_bytes,
                raw_payload_min if raw_payload_min is not None else -1,
                raw_payload_max if raw_payload_max is not None else -1,
                encoded_payload_chars,
                repr(e),
                self._process_debug_state(),
            )
            raise
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
