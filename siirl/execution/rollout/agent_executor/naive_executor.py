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
import contextvars
import copy
import hashlib
import importlib
import os
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import ray
import torch
from loguru import logger

from siirl.data_coordinator.sample import Sample, SampleInfo
from siirl.execution.rollout.concurrency import resolve_rollout_concurrency
from siirl.execution.rollout.utils import RolloutGenerationAborted
from siirl.params.training_args import SiiRLArguments
from siirl.utils.model_utils.model import compute_position_id_with_mask
from siirl.utils.net_utils.http_utils import GlobalAsyncHTTPClient
from siirl.utils.timer import Timer


class NaiveExecutor:
    """
    NaiveExecutor used in synchronous training workflows.
    Manages asynchronous sample generation, pre/post processing, and data coordination
    for rollout processes in reinforcement learning with large language models.
    """

    def __init__(self, config: SiiRLArguments, data_coordinator, engine, train_batch_size, dp_rank=0):
        """
        Initialize NaiveExecutor with core configuration and dependencies.

        Args:
            config: SiiRLArguments containing all training/rollout hyperparameters
            data_coordinator: Ray handle to data coordinator for sample management
            engine: Inference engine instance (e.g., SglangEngine) for text generation
            train_batch_size: Batch size for rollout sample generation
            dp_rank: Data parallel rank (0 to dp_size-1), used for logging/progress
        """
        self.config = config
        self.data_coordinator = data_coordinator  # Ray actor handle to data coordinator
        self.running = False  # Flag to control executor main loop
        self.engine = engine  # Inference engine for text generation
        # TODO: use validation arguments here
        self.train_batch_size = train_batch_size  # Target batch size for rollout samples
        target_concurrency = max(1, int(train_batch_size) * int(config.rollout.n))
        train_limits = resolve_rollout_concurrency(config, phase="train", use_router=False)
        self.max_concurrency_size = min(target_concurrency, int(train_limits["effective"]))
        self._train_target_concurrency = target_concurrency
        self._train_concurrency_limits = train_limits
        self.tasks: set[asyncio.Task] = set()  # Track active generation tasks for cleanup
        
        # Used for partial rollout
        self._dispatch_paused = False
        self._dispatch_condition: asyncio.Condition | None = None
        self._dispatch_loop: asyncio.AbstractEventLoop | None = None

        # Semaphore to control concurrent generation tasks (limit to batch size)
        # Use ContextVar to store semaphore per-async-context (per-event-loop)
        # This avoids cross-event-loop binding issues without any locks
        self._semaphore_ctx: contextvars.ContextVar = contextvars.ContextVar("semaphore")
        self.reward_fn = None  # Custom reward function (optional)
        self.rollout_flow = None  # Rollout flow function for sample generation
        self._rank = int(os.environ.get("RANK"))
        self._dp_rank = dp_rank  # Data parallel rank for logging
        self._verbose_validate_logs = os.environ.get("SIIRL_VERBOSE_VALIDATE_LOGS", "0") == "1"
        # Load custom reward function if configured
        if config.custom_reward_function.path:
            from siirl.utils.reward_score.custom_reward import load_custom_reward_function

            self.reward_fn = load_custom_reward_function(config=config)

        # Load rollout flow function (naive or custom)
        flow_path = config.rollout.flow_function
        if flow_path == "naive":
            from siirl.execution.rollout.agent_flow.naive_flow import NaiveFlow

            self.rollout_flow = NaiveFlow(self.config, self.engine)
        elif flow_path == "agent":
            import yaml

            from siirl.execution.rollout.agent_flow.agent_flow import build_agentflow

            with open(config.rollout.flow_config) as f:
                flow_config = yaml.safe_load(f)
            self.rollout_flow = build_agentflow(flow_config, engine)
        else:
            # Dynamically import custom rollout flow function
            module_path, name = flow_path.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            self.rollout_flow = getattr(mod, name)

    def pause_dispatch(self):
        # Atomic write under GIL; no need to hop to the dispatch loop just to flip a bool.
        # The dispatch loop checks the flag on each iteration via _wait_dispatch_resumed.
        self._dispatch_paused = True

    def resume_dispatch(self):
        loop = self._dispatch_loop
        if loop is None or loop.is_closed():
            self._dispatch_paused = False
            return
        # Resume has to run on the dispatch loop because it must notify the Condition
        # to wake a coroutine that is currently awaiting on it.
        asyncio.run_coroutine_threadsafe(self._resume_dispatch(), loop).result()

    async def _resume_dispatch(self):
        self._dispatch_paused = False
        if self._dispatch_condition is None:
            return
        async with self._dispatch_condition:
            self._dispatch_condition.notify_all()

    def _init_dispatch_condition(self):
        self._dispatch_loop = asyncio.get_running_loop()
        self._dispatch_condition = asyncio.Condition()

    async def _wait_dispatch_resumed(self):
        if self._dispatch_condition is None:
            self._init_dispatch_condition()
        condition = self._dispatch_condition
        async with condition:
            while self._dispatch_paused:
                await condition.wait()

    async def init_sample(self):
        # need_replenish = self.max_concurrency_size
        pass

    async def get_sample(self):
        """
        Get one new sample to replenish the batch.

        Returns at most one sample per call. The dispatch semaphore in ``run``
        already guarantees we only reach here when this worker has a free
        inference slot, so there is no need to double-gate on len(self.tasks).

        Priority: shared partial queue (DataCoordinator) > fresh replica
        from dataloader. Partial samples are pulled first so in-progress
        trajectories finish before fresh ones start.
        """
        # 1. Prefer a partial from the shared cancel queue (any worker can take any partial).
        partials = await self.data_coordinator.get_partial.remote(1)
        if partials:
            return partials

        # 2. Pull one replica from the dataloader (already expanded to rollout_n copies).
        return await self.data_coordinator.get_dataloader.remote(1)

    def _manual_pad(self, ids, max_length, padding_side="right"):
        """
        Manually pad token IDs and create corresponding attention mask.
        Handles both left and right padding for sequence length standardization.

        Args:
            ids: 2D list of token IDs (batch_size=1, [List[int]])
            max_length: Target sequence length after padding
            padding_side: Direction for padding ("left" or "right", default: "right")

        Returns:
            Tuple of (padded_ids, attention_mask) as torch tensors (shape: [1, max_length])
        """
        # Convert list to tensor (batch_size=1, seq_len)
        ids = torch.tensor(ids, dtype=torch.long)
        pad_length = max_length - ids.shape[1]

        # No padding needed - truncate to max length and create all-ones attention mask
        if pad_length <= 0:
            padded_ids = ids[:, :max_length]
            attention_mask = torch.ones_like(padded_ids, dtype=torch.long)
            return padded_ids, attention_mask

        # Create padding tensor with pad token ID
        pad_tensor = torch.full(
            (ids.shape[0], pad_length),
            self.engine.tokenizer.pad_token_id,
            dtype=torch.long,
        )

        # Left padding: pad first, then original sequence
        if padding_side == "left":
            padded_ids = torch.cat([pad_tensor, ids], dim=1)
            attention_mask = torch.cat([torch.zeros_like(pad_tensor), torch.ones_like(ids)], dim=1)
        # Right padding: original sequence first, then pad
        else:  # right
            padded_ids = torch.cat([ids, pad_tensor], dim=1)
            attention_mask = torch.cat([torch.ones_like(ids), torch.zeros_like(pad_tensor)], dim=1)

        return padded_ids, attention_mask

    def _pre_process(self, sample: Sample, is_validate=False):
        """
        Preprocess single sample before generation.
        Maps raw prompt IDs to prompts field for consistency.

        Args:
            sample: Sample object to preprocess

        Returns:
            Preprocessed sample with prompts field set
        """
        sample.prompts = sample.raw_prompt_ids
        if is_validate:
            sample.prompt_texts = self.engine.tokenizer.decode(sample.prompts)
        return sample

    def _post_process(self, sample: Sample):
        """
        Postprocess generated sample with padding, sequence concatenation, and reward formatting.
        Standardizes prompt/response lengths, creates attention masks, and formats reward tensors.

        Args:
            sample: Generated sample to postprocess

        Returns:
            Postprocessed sample with standardized tensor fields
        """
        # Pad prompts to max prompt length (left padding for prompt sequences)
        prompt_ids, prompt_attention_mask = self._manual_pad(
            [sample.prompts],
            max_length=self.config.data.max_prompt_length,
            padding_side="left",
        )

        # Pad responses to max response length (right padding for generated text)
        response_ids, response_attention_mask = self._manual_pad(
            [sample.responses],
            max_length=self.config.data.max_response_length,
            padding_side="right",
        )

        # Pad and combine response mask with attention mask (filter padding tokens)
        response_mask, _ = self._manual_pad(
            [sample.response_mask],
            max_length=self.config.data.max_response_length,
            padding_side="right",
        )
        response_mask = response_mask * response_attention_mask

        # Pad rollout_log_prob to match response length (for monitoring metrics)
        if sample.rollout_log_prob is not None:
            max_len = self.config.data.max_response_length
            current_len = len(sample.rollout_log_prob)
            if current_len < max_len:
                pad_len = max_len - current_len
                sample.rollout_log_prob = np.pad(
                    sample.rollout_log_prob,
                    (0, pad_len),
                    mode="constant",
                    constant_values=0.0,
                ).astype(np.float32)
            elif current_len > max_len:
                sample.rollout_log_prob = sample.rollout_log_prob[:max_len].astype(np.float32)

        # Pad rollout_routed_experts to match padded sequence length [max_seq_len, moe_dim].
        # Left-pad for prompt (matching left-padded prompt_ids) and right-pad for response.
        if sample.rollout_routed_experts is not None:
            max_prompt_len = self.config.data.max_prompt_length
            max_response_len = self.config.data.max_response_length
            max_seq_len = max_prompt_len + max_response_len
            actual_prompt_len = len(sample.prompts) if hasattr(sample.prompts, "__len__") else 0
            actual_response_len = len(sample.responses) if hasattr(sample.responses, "__len__") else 0
            routing_data = sample.rollout_routed_experts  # [n_real_tokens, moe_dim]
            if routing_data.ndim == 2:
                moe_dim = routing_data.shape[1]
                padded = np.zeros((max_seq_len, moe_dim), dtype=routing_data.dtype)
                # Place real routing data at the correct position (after left-pad)
                prompt_start = max_prompt_len - actual_prompt_len
                n_real = min(routing_data.shape[0], actual_prompt_len + actual_response_len)
                padded[prompt_start : prompt_start + n_real] = routing_data[:n_real]
                sample.rollout_routed_experts = padded
            else:
                sample.rollout_routed_experts = None

        # Validate tensor shape consistency
        assert (
            response_ids.shape == response_mask.shape
        ), f"mismatch in response_ids and response_mask shape: {response_ids.shape} vs {response_mask.shape}"

        # Concatenate prompt and response sequences for model input
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)

        # Create position IDs (account for padding in attention mask)
        # Initialize position_ids with the same shape as attention_mask
        # position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
        # # Get the attention mask for this sample
        # valid_indices = attention_mask [0].nonzero(as_tuple=False).squeeze(-1)
        # seq_length = attention_mask.size(1)
        # if len(valid_indices) > 0:
        #     first_valid = valid_indices[0].item()
        #     # Left padding positions remain 0
        #     # Valid positions get incremental IDs starting from 0
        #     # Right padding positions continue the sequence

        #     # Assign position IDs for the entire sequence starting from first valid position
        #     position_sequence = torch.arange(seq_length - first_valid)
        #     position_ids[0, first_valid:] = position_sequence

        # position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        position_ids = compute_position_id_with_mask(attention_mask)

        # Format reward tensor (place reward value at last valid response token position)
        reward_tensor = torch.zeros_like(response_ids[0], dtype=torch.float32)
        prompt_length = prompt_ids[0].shape[-1]
        valid_response_length = attention_mask[0][prompt_length:].sum()
        reward_tensor[valid_response_length - 1] = sample.rewards
        # Clean up and set processed fields in sample
        sample.token_level_rewards = reward_tensor.numpy()
        sample.token_level_scores = copy.deepcopy(reward_tensor.numpy())
        sample.prompts = prompt_ids[0].numpy()
        sample.responses = response_ids[0].numpy()
        sample.response_mask = response_mask[0].numpy()
        sample.input_ids = input_ids[0].numpy()
        sample.attention_mask = attention_mask[0].numpy()
        sample.position_ids = position_ids[0].numpy()

        return sample

    async def put_data(self, sample, loop):
        sample_ref = await loop.run_in_executor(None, ray.put, sample)

        # Create sample metadata for tracking
        sample_info = SampleInfo(
            sum_tokens=getattr(sample, "sum_tokens", int(sample.attention_mask.sum())),
            prompt_length=getattr(sample, "prompt_length", 0),
            response_length=getattr(sample, "response_length", 0),
            uid=str(sample.uid),
            replica_index=sample.replica_index,
            weight_version=self.engine._weight_version,
            dict_info={
                "key": "Actor",
            },
        )
        # DataCoordinator.put buffers replicas per uid and releases the full group
        # to _sample_queue once all rollout_n replicas arrive.
        await self.data_coordinator.put.remote(sample_info, sample_ref)

        return sample

    async def generate(
        self,
        sample,
        is_validate=False,
        validate_request_seed: int | None = None,
    ):
        """
        Asynchronous sample generation pipeline: preprocess → rollout → postprocess → data coordination.
        Dispatch concurrency is gated at the caller (``run`` or the validate path's
        validate_semaphore); this function itself does not throttle.

        Args:
            sample: Raw sample from data coordinator

        Returns:
            Postprocessed sample with generated response and formatted tensors, or None if failed
        """
        # Record timing information for performance analysis
        timing_info = {
            "rollout_start_at": time.time(),
        }
        sample_uid = getattr(sample, "uid", "unknown")

        try:
            loop = asyncio.get_running_loop()
            # 1. Preprocess sample (CPU-bound, offload to executor)
            sample = await loop.run_in_executor(None, self._pre_process, sample, is_validate)

            # 2. Execute rollout flow (LLM generation with reward calculation)
            if validate_request_seed is None:
                sample = await self.rollout_flow(sample, self.reward_fn, is_validate)
            else:
                try:
                    sample = await self.rollout_flow(
                        sample,
                        self.reward_fn,
                        is_validate,
                        request_seed=validate_request_seed,
                    )
                except TypeError as exc:
                    if "request_seed" not in str(exc):
                        raise
                    sample = await self.rollout_flow(sample, self.reward_fn, is_validate)

            # 3. Postprocess sample (CPU-bound padding and tensor formatting)
            sample = await loop.run_in_executor(None, self._post_process, sample)

            # 4. Collect timing information from rollout flow
            timing_info["rollout_end_at"] = time.time()
            timing_info["rollout_duration"] = timing_info["rollout_end_at"] - timing_info["rollout_start_at"]
            timing_info["generation_duration"] = getattr(sample, "_generation_duration", 0)
            timing_info["reward_duration"] = getattr(sample, "_reward_duration", 0)
            sample.timing_info = timing_info

            # 5. Store processed sample in Ray object store and notify data coordinator
            if not is_validate:
                await self.put_data(sample=sample, loop=loop)
            return sample

        except RolloutGenerationAborted as aborted:
            if not is_validate:
                # Hand the partial sample back to the shared cancel queue so any
                # worker can pick it up after weight sync resumes dispatch.
                await self.data_coordinator.put_partial.remote(aborted.sample)
                logger.debug(
                    "[NaiveExecutor.generate] Sample uid={} aborted by weight sync; "
                    "handed back to shared cancel queue",
                    sample_uid,
                )
                return None
            raise

        except Exception as e:
            import traceback

            logger.error(f"[NaiveExecutor.generate] Sample uid={sample_uid} failed: {e}")
            logger.error(f"[NaiveExecutor.generate] Traceback:\n{traceback.format_exc()}")
            raise

    async def run(self):
        """
        Main execution loop for the executor.
        Continuously replenishes samples, creates generation tasks, and maintains batch size.
        Runs until self.running is set to False.

        Dispatch is gated by ``self._semaphore_ctx`` — we only pull a new sample
        from the DataCoordinator once this worker has a free inference slot.
        This prevents a slow worker from draining the shared queue into its own
        task set while peers sit idle. The slot is released in the task's
        done-callback so generate() itself does not need its own semaphore.
        """
        # Create semaphore for this event loop if not already exists
        # Check if ContextVar has been set to avoid creating multiple semaphores
        # when switching between run() and validate() in the same event loop
        try:
            semaphore = self._semaphore_ctx.get()
        except LookupError:
            # ContextVar not set yet - create new semaphore for this event loop
            semaphore = asyncio.Semaphore(self.max_concurrency_size)
            self._semaphore_ctx.set(semaphore)
        self._init_dispatch_condition()

        self.running = True
        stats_task = None
        if self._dp_rank == 0:
            limits = self._train_concurrency_limits
            logger.info(
                "Rollout train concurrency: "
                f"target={self._train_target_concurrency}, "
                f"base_key={limits['base_key']}, base={limits['base']}, "
                f"max_num_seqs={limits['max_num_seqs']}, effective={self.max_concurrency_size}"
            )
            stats_task = asyncio.create_task(self.rollout_status())

        while self.running:
            await self._wait_dispatch_resumed()
            # Block until this worker has an inference slot before touching the
            # shared queue — if we are already saturated, let peers pull instead.
            await semaphore.acquire()
            samples = await self.get_sample()
            if not samples:
                # Nothing to dispatch right now; give the slot back and retry.
                semaphore.release()
                await asyncio.sleep(0.001)
                continue
            for sample in samples:
                task = asyncio.create_task(self.generate(sample))
                self.tasks.add(task)
                # Release both the task-set slot and the semaphore slot when the
                # task finishes (including on exception / cancellation).
                task.add_done_callback(self.tasks.discard)
                task.add_done_callback(lambda t, sem=semaphore: sem.release())

            # Debug code (commented out) - save batch for inspection
            # samples = await asyncio.gather(*tasks)
            # batch = Samples2Dict(samples=samples)
            # torch.save(batch, f"save_dict/{os.environ.get('RANK')}_batch.pt")
            # break

            # Yield control to event loop (non-blocking sleep)
            await asyncio.sleep(0)
        if self._dp_rank == 0:
            stats_task.cancel()
            await asyncio.gather(stats_task, return_exceptions=True)

    def _is_single_turn(self) -> bool:
        """Check if current config is single-turn (no tool/env calls)."""
        return not self.config.rollout.multiturn.env_type

    async def _load_val_data(self, val_batch_size: int) -> list[Sample]:
        """Load validation data with async optimization."""
        val_samples = []
        while True:
            data = await self.data_coordinator.get_dataloader.remote(batch_size=val_batch_size, is_validate=True)
            if not data:
                break
            val_samples.extend(data)
        return val_samples

    def _compute_reward(self, sample: Sample, response_text: str):
        """Compute reward for a sample using custom or default reward function.

        Shared by _validate_single_turn and NaiveFlow to avoid logic duplication.
        """
        from siirl.utils.reward_score import default_compute_score

        if self.reward_fn:
            return self.reward_fn(
                data_source=sample.data_source,
                solution_str=response_text,
                ground_truth=sample.reward_model["ground_truth"],
            )
        return default_compute_score(
            data_source=sample.data_source,
            solution_str=response_text,
            ground_truth=sample.reward_model["ground_truth"],
        )

    def _get_validate_request_seed(self, sample: Sample) -> int:
        base_seed = int(getattr(self.config.rollout, "seed", 0))
        uid = getattr(sample, "uid", None)
        if uid is not None:
            key = f"uid:{uid}"
        else:
            raw_prompt_ids = getattr(sample, "raw_prompt_ids", None)
            if raw_prompt_ids is None:
                key = f"prompt:{getattr(sample, 'prompt_texts', '')}"
            elif hasattr(raw_prompt_ids, "tolist"):
                key = "ids:" + ",".join(map(str, raw_prompt_ids.tolist()))
            else:
                key = "ids:" + ",".join(map(str, list(raw_prompt_ids)))
        offset = int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), byteorder="little")
        return (base_seed + offset) % (2**31 - 1)

    async def _validate_single_turn(
        self,
        samples: list[Sample],
        use_router: bool = True,
        progress_callback: Callable[[int], None] | None = None,
    ) -> list[Sample]:
        """
        Optimized validation for single-turn scenarios.
        Uses batch generation with router load balancing and sort-by-length.
        """
        # 1. Preprocess (direct loop, microsecond-level)
        with Timer("preprocess") as preprocess_time:
            for sample in samples:
                sample.prompts = sample.raw_prompt_ids
                sample.prompt_texts = self.engine.tokenizer.decode(sample.prompts)

        # 2. Batch generate with router load balancing (the only expensive operation)
        with Timer("generate") as generate_time:

            def to_list(ids):
                return ids.tolist() if hasattr(ids, "tolist") else list(ids)

            batch_input_ids = [to_list(sample.raw_prompt_ids) for sample in samples]
            request_seeds = [self._get_validate_request_seed(sample) for sample in samples]
            results = await self.engine.generate_batch(
                batch_input_ids,
                is_validate=True,
                use_router=use_router,
                show_progress=False,
                progress_desc="Validate",
                request_seeds=request_seeds,
                progress_callback=progress_callback,
            )

        # 3. Reward + postprocess (direct loop, millisecond-level)
        with Timer("reward_and_postprocess") as reward_time:
            for sample, (_, response_ids, log_probs, _routed) in zip(samples, results, strict=False):
                sample.responses = response_ids
                sample.response_mask = [1] * len(response_ids)
                sample.rollout_log_prob = np.array(log_probs, dtype=np.float32)
                response_text = self.engine.tokenizer.decode(response_ids)
                sample.rewards = self._compute_reward(sample, response_text)
                sample = self._post_process(sample)

        # Log timing breakdown (rank 0 only)
        if self._dp_rank == 0:
            message = (
                f"Validate timing: preprocess={preprocess_time.elapsed:.2f}s, "
                f"generate={generate_time.elapsed:.2f}s, "
                f"reward_postprocess={reward_time.elapsed:.2f}s"
            )
            if self._verbose_validate_logs:
                logger.info(message)
            else:
                logger.debug(message)

        return samples

    async def _indexed_generate(
        self,
        idx: int,
        sample: Sample,
        request_seed: int | None = None,
    ):
        """Wrapper that returns (index, result) for correct ordering with as_completed."""
        result = await self.generate(sample, is_validate=True, validate_request_seed=request_seed)
        return idx, result

    def _resolve_validate_concurrency(self, use_router: bool) -> int:
        limits = resolve_rollout_concurrency(self.config, phase="validate", use_router=use_router)
        logger.debug(
            "Validate multi-turn concurrency: "
            f"use_router={use_router}, base_key={limits['base_key']}, base={limits['base']}, "
            f"num_engines={limits['num_engines']}, resolved={limits['resolved']}, "
            f"max_num_seqs={limits['max_num_seqs']}, effective={limits['effective']}"
        )
        return int(limits["effective"])

    async def _validate_multi_turn(
        self,
        samples: list[Sample],
        use_router: bool = True,
        progress_callback: Callable[[int], None] | None = None,
    ) -> list[Sample]:
        """
        Validation for multi-turn scenarios.
        Uses high concurrency with router load balancing.
        Streaming collection via as_completed to reduce tail latency.
        """
        # Enable router on rollout_flow for validate duration
        self.rollout_flow.use_router = use_router
        max_concurrent = self._resolve_validate_concurrency(use_router)
        validate_semaphore = asyncio.Semaphore(max_concurrent)

        async def _indexed_generate_limited(idx: int, sample: Sample):
            async with validate_semaphore:
                return await self._indexed_generate(idx, sample, request_seed=self._get_validate_request_seed(sample))

        tasks: list[asyncio.Task] = []
        try:
            results = [None] * len(samples)
            tasks = [asyncio.create_task(_indexed_generate_limited(i, s)) for i, s in enumerate(samples)]

            for task in asyncio.as_completed(tasks):
                idx, result = await task
                results[idx] = result
                if progress_callback is not None:
                    progress_callback(1)

            return results
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self.rollout_flow.use_router = False

    async def validate(self, val_batch_size: int) -> tuple[list[Sample], dict]:
        """
        Optimized validation with A+B strategy:
        - Single-turn: batch generation + router load balancing
        - Multi-turn: high concurrency + router load balancing
        """
        with Timer("get_val_data") as val_get_time:
            val_samples = await self._load_val_data(val_batch_size)
        return await self.validate_samples(val_samples, val_get_time=val_get_time.elapsed)

    async def validate_samples(
        self,
        val_samples: list[Sample],
        val_get_time: float = 0.0,
        use_router: bool = True,
        progress_callback: Callable[[int], None] | None = None,
    ) -> tuple[list[Sample], dict]:
        logger.debug(
            f"RANK_{self._rank} start validate, batch_size:{len(val_samples)}, "
            f"mode:{'single-turn' if self._is_single_turn() else 'multi-turn'}"
        )

        if not val_samples:
            return [], {"val_get_time": val_get_time, "val_generate_time": 0.0}

        with Timer("val_generate") as val_generate_time:
            if self._is_single_turn():
                result = await self._validate_single_turn(
                    val_samples,
                    use_router=use_router,
                    progress_callback=progress_callback,
                )
            else:
                result = await self._validate_multi_turn(
                    val_samples,
                    use_router=use_router,
                    progress_callback=progress_callback,
                )

        val_get_time_sec = val_get_time.elapsed if hasattr(val_get_time, "elapsed") else float(val_get_time)
        metrics = {
            "val_get_time": val_get_time_sec,
            "val_generate_time": val_generate_time.elapsed,
        }
        return result, metrics

    async def rollout_status(self, interval: float = 10.0):
        last_status = 0
        while True:
            await asyncio.sleep(interval)
            current_status = len(self.tasks)
            http_metrics = GlobalAsyncHTTPClient.drain_metrics()
            attempts = int(http_metrics.get("attempts", 0))
            timeouts = int(http_metrics.get("timeouts", 0))
            timeout_rate = (timeouts / attempts) if attempts > 0 else 0.0

            if last_status != current_status or timeouts > 0:
                try:
                    cancel_queue_size = await self.data_coordinator.get_partial_size.remote()
                except Exception:
                    cancel_queue_size = -1
                message = (
                    f"rank_{self._rank} active generate tasks: {current_status}, "
                    f"{cancel_queue_size} left in shared cancel_queue, "
                    f"sem_limit={self.max_concurrency_size}, "
                    f"http_attempts={attempts}, http_timeouts={timeouts}, http_timeout_rate={timeout_rate:.3f}"
                )
                if timeouts > 0:
                    logger.warning(message)
                elif self._verbose_validate_logs:
                    logger.info(message)
                else:
                    logger.debug(message)
                last_status = current_status

    async def stop(self):
        """
        Stop executor and clean up active tasks.
        Sets running flag to False and waits for all active generation tasks to complete.
        Partials live on the shared DataCoordinator queue and survive across workers,
        so they are NOT touched here.
        """
        self.running = False
        # Wait for all remaining tasks to finish before exiting
        await asyncio.gather(*self.tasks)
