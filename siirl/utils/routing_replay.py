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

"""
Routing Replay for MoE models during RL training.

In Mixture-of-Experts (MoE) models, router networks select which experts process
each token. During RL training (PPO/GRPO), multiple forward/backward passes over
the same batch require consistent routing decisions:

  1. compute_log_prob (forward-only): Record routing decisions OR replay from rollout
  2. update_actor (forward+backward): Replay the recorded routing decisions

Without routing replay, different routing decisions across passes cause gradient
inconsistency and training instability, because the loss computed in the forward
pass targets different expert outputs than those used during backpropagation.

Architecture:
  - RoutingReplayStage: Enum controlling the current routing behavior
  - RoutingReplayCache: Per-router cache storing expert indices in CPU pinned memory
  - RoutingReplayManager: Singleton managing global state and Megatron patching
  - stage(): Context manager for safe stage transitions

Usage in the training loop:
    manager = RoutingReplayManager.get()
    manager.install()  # Patches Megatron's TopKRouter and compute_topk

    # During compute_log_prob:
    with manager.stage(RoutingReplayStage.RECORD):
        actor_worker.compute_log_prob(batch)

    # During ref forward (different model, don't replay):
    with manager.stage(RoutingReplayStage.FALLTHROUGH):
        ref_worker.compute_ref_log_prob(batch)

    # During update_actor backward:
    with manager.stage(RoutingReplayStage.REPLAY_BACKWARD):
        actor_worker.update_actor(batch)

    manager.clear_all()
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from enum import Enum
from typing import TYPE_CHECKING

import torch
from loguru import logger

if TYPE_CHECKING:
    from torch import nn


class RoutingReplayStage(Enum):
    """Stage controlling how MoE routing decisions are handled."""

    DISABLED = "disabled"  # Routing replay not active, use original compute_topk
    RECORD = "record"  # Record routing decisions from this forward pass
    REPLAY_FORWARD = "replay_forward"  # Replay recorded decisions (forward-only)
    REPLAY_BACKWARD = "replay_backward"  # Replay recorded decisions (forward+backward)
    FALLTHROUGH = "fallthrough"  # Temporarily bypass replay (e.g., for ref model)


class RoutingReplayCache:
    """
    Per-router cache for MoE expert routing indices.

    Stores top_indices from compute_topk in CPU pinned memory to minimize
    GPU memory overhead while maintaining fast H2D transfer for replay.

    Each MoE layer's TopKRouter gets its own RoutingReplayCache instance,
    registered via a forward pre-hook that sets the active cache before
    each router's forward pass.
    """

    def __init__(self, layer_id: int = -1):
        self.layer_id = layer_id
        self._forward_idx: int = 0
        self._backward_idx: int = 0
        self._entries: list[torch.Tensor] = []

    def record(self, top_indices: torch.Tensor) -> None:
        """Store routing indices in CPU pinned memory."""
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self._entries.append(buf)

    def pop_forward(self) -> torch.Tensor:
        """Retrieve next cached indices for forward replay."""
        t = self._entries[self._forward_idx]
        self._forward_idx += 1
        return t.to(torch.cuda.current_device(), non_blocking=True)

    def pop_backward(self) -> torch.Tensor:
        """Retrieve next cached indices for backward replay."""
        t = self._entries[self._backward_idx]
        self._backward_idx += 1
        return t.to(torch.cuda.current_device(), non_blocking=True)

    def reset_forward(self) -> None:
        """Reset forward replay index (for re-reading same data)."""
        self._forward_idx = 0

    def reset_backward(self) -> None:
        """Reset backward replay index."""
        self._backward_idx = 0

    def clear(self) -> None:
        """Release all cached entries and reset indices."""
        self._forward_idx = 0
        self._backward_idx = 0
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return (
            f"RoutingReplayCache(layer={self.layer_id}, entries={len(self._entries)}, "
            f"fwd_idx={self._forward_idx}, bwd_idx={self._backward_idx})"
        )


class RoutingReplayManager:
    """
    Singleton manager for MoE routing replay.

    Coordinates stage transitions and manages all per-router caches.
    Thread-safe for Ray actor processes (each process has its own singleton).

    The manager patches Megatron's TopKRouter and compute_topk function at
    install() time. This is done via runtime monkey-patching rather than
    requiring external patch files, making deployment simpler.
    """

    _instance: RoutingReplayManager | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._stage = RoutingReplayStage.DISABLED
        self._caches: list[RoutingReplayCache] = []
        self._installed = False
        # The currently active cache (set by per-router forward pre-hooks)
        self._active_cache: RoutingReplayCache | None = None

    @classmethod
    def get(cls) -> RoutingReplayManager:
        """Get or create the process-local singleton."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.clear_all()
            cls._instance = None

    # --- Stage management ---

    @property
    def current_stage(self) -> RoutingReplayStage:
        return self._stage

    def set_stage(self, stage: RoutingReplayStage) -> None:
        self._stage = stage

    @contextmanager
    def stage(self, target: RoutingReplayStage):
        """Context manager for temporary stage transition."""
        prev = self._stage
        self._stage = target
        try:
            yield
        finally:
            self._stage = prev

    # --- Cache management ---

    @property
    def caches(self) -> list[RoutingReplayCache]:
        return self._caches

    def register_cache(self, cache: RoutingReplayCache) -> None:
        self._caches.append(cache)

    def set_active_cache(self, cache: RoutingReplayCache) -> None:
        """Set the currently active cache (called by per-router pre-hook)."""
        self._active_cache = cache

    @property
    def active_cache(self) -> RoutingReplayCache | None:
        return self._active_cache

    def clear_all(self) -> None:
        """Clear all caches and reset to disabled."""
        for cache in self._caches:
            cache.clear()

    def reset_all_forward(self) -> None:
        """Reset forward indices on all caches (for re-reading cached routing)."""
        for cache in self._caches:
            cache.reset_forward()

    def reset_all_backward(self) -> None:
        """Reset backward indices on all caches."""
        for cache in self._caches:
            cache.reset_backward()

    def reset_all_indices(self) -> None:
        """Reset both forward and backward indices on all caches."""
        for cache in self._caches:
            cache.reset_forward()
            cache.reset_backward()

    # --- Megatron patching ---

    def install(self) -> None:
        """
        Monkey-patch Megatron's MoE routing to support routing replay.

        Patches TopKRouter:
        1. __init__: Adds a RoutingReplayCache + forward pre-hook to each router instance
        2. routing(): Wraps the routing method to intercept top_indices for record/replay

        The routing() patch is version-adaptive: works regardless of whether Megatron
        uses topk_routing_with_score_function or inline topk logic.

        Safe to call multiple times (idempotent).
        """
        if self._installed:
            return
        self._patch_topk_router_init()
        self._installed = True
        logger.info("[RoutingReplay] Installed Megatron MoE patches")

    def _patch_topk_router_init(self) -> None:
        """
        Patch TopKRouter to support routing replay.

        Two patches per router instance:
        1. __init__: Creates a RoutingReplayCache and registers a forward pre-hook
           to set the active cache before each forward pass.
        2. routing(): Wraps the routing method to intercept (scores, top_indices).
           In RECORD mode, caches top_indices. In REPLAY mode, substitutes cached
           top_indices and recomputes scores via gather.
        """
        try:
            from megatron.core.transformer.moe.router import TopKRouter
        except ImportError:
            logger.warning("[RoutingReplay] megatron.core.transformer.moe.router not found, skipping")
            return

        original_init = TopKRouter.__init__
        original_routing = TopKRouter.routing
        manager = self

        def patched_init(self_router, *args, **kwargs):
            original_init(self_router, *args, **kwargs)
            # Create and register a cache for this router instance
            layer_id = len(manager.caches)
            cache = RoutingReplayCache(layer_id=layer_id)
            self_router._routing_replay_cache = cache
            manager.register_cache(cache)

            # Register forward pre-hook to set this cache as active before each forward
            def _pre_hook(module, inputs):
                manager.set_active_cache(module._routing_replay_cache)

            self_router.register_forward_pre_hook(_pre_hook)

        def patched_routing(self_router, logits):
            """
            Wraps TopKRouter.routing() to intercept expert routing decisions.

            TopKRouter.routing() returns (scores, routing_map) where:
            - scores: expert probabilities (may be 2D or 3D depending on Megatron version)
            - routing_map: token-to-expert assignment (binary mask or index map)

            We cache routing_map (the routing decision) and replay it to ensure
            consistent expert assignment across forward/backward passes. Scores are
            recomputed from current logits in replay mode to maintain gradient flow
            through the router weights.

            Gradient checkpointing compatibility:
            With recompute_granularity="full", each layer's forward runs twice per
            micro-batch during update_actor:
              1. Checkpointed forward: inside torch.no_grad() → is_grad_enabled()=False
              2. Backward recompute: inside torch.enable_grad() → is_grad_enabled()=True
            Both calls must return the SAME cached entry. We use separate indices
            (forward_idx for #1, backward_idx for #2) that advance in lockstep.
            Without checkpointing, only the grad-enabled path (#2) is taken.
            """
            stage = manager.current_stage
            cache = manager.active_cache

            # Fast path: delegate to original
            if (
                stage == RoutingReplayStage.DISABLED
                or stage == RoutingReplayStage.FALLTHROUGH
                or cache is None
            ):
                return original_routing(self_router, logits)

            if stage == RoutingReplayStage.RECORD:
                scores, routing_map = original_routing(self_router, logits)
                cache.record(routing_map)
                return scores, routing_map

            # REPLAY_FORWARD or REPLAY_BACKWARD: use cached routing_map
            if stage == RoutingReplayStage.REPLAY_FORWARD:
                routing_map = cache.pop_forward()
            elif stage == RoutingReplayStage.REPLAY_BACKWARD:
                # Distinguish checkpointed forward vs backward recompute:
                # - Checkpointed forward runs inside torch.no_grad() → pop_forward
                # - Backward recompute runs inside torch.enable_grad() → pop_backward
                # - Non-checkpointed forward has grad enabled → pop_backward (no recompute follows)
                # Both indices advance in lockstep, returning the same entry[i] for micro-batch i.
                if not torch.is_grad_enabled():
                    routing_map = cache.pop_forward()
                else:
                    routing_map = cache.pop_backward()
            else:
                return original_routing(self_router, logits)

            # Recompute scores from current logits for gradient flow.
            # Must replicate the exact score computation that Megatron's
            # topk_routing_with_score_function performs, so that expert outputs
            # are weighted identically to the original (non-replay) forward pass.
            score_function = getattr(self_router, "routing_score_function", None) or getattr(
                self_router.config, "moe_router_score_function", "softmax"
            )
            use_pre_softmax = getattr(self_router.config, "moe_router_pre_softmax", False)

            # Flatten logits to 2D [num_tokens, num_experts] to match routing_map shape.
            orig_logits = logits
            if logits.dim() == 3:
                logits = logits.view(-1, logits.shape[-1])

            if logits.shape[0] != routing_map.shape[0]:
                raise RuntimeError(
                    f"[RoutingReplay] Token count mismatch in layer {cache.layer_id}: "
                    f"logits {logits.shape} vs routing_map {routing_map.shape}. "
                    f"Stage={stage.value}, grad_enabled={torch.is_grad_enabled()}, "
                    f"fwd_idx={cache._forward_idx}, bwd_idx={cache._backward_idx}, "
                    f"entries={len(cache)}. This usually means micro-batch formation "
                    f"differs between RECORD and REPLAY phases."
                )

            top_mask = routing_map.bool()

            if score_function == "softmax":
                if use_pre_softmax:
                    # Pre-softmax: softmax over ALL experts, then mask to selected.
                    scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(orig_logits)
                    scores = scores * top_mask.to(scores.dtype)
                else:
                    # Post-softmax: softmax only among selected top-k experts.
                    # Megatron selects top-k logits then applies softmax to
                    # normalize among them. We replicate this by masking
                    # non-selected positions to -inf before softmax.
                    masked_logits = logits.masked_fill(~top_mask, float('-inf'))
                    scores = torch.softmax(masked_logits, dim=-1, dtype=torch.float32).type_as(orig_logits)
            elif score_function == "sigmoid":
                scores = torch.sigmoid(logits).type_as(orig_logits)
                scores = scores * top_mask.to(scores.dtype)
                # Megatron normalizes sigmoid scores among selected experts.
                score_sum = scores.sum(dim=-1, keepdim=True) + 1e-20
                scores = scores / score_sum
            else:
                scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(orig_logits)
                scores = scores * top_mask.to(scores.dtype)

            # Apply optional scaling factor (e.g. DeepSeek-V3 style).
            scaling_factor = getattr(self_router.config, "moe_router_topk_scaling_factor", None)
            if scaling_factor:
                scores = scores * scaling_factor

            return scores, routing_map

        TopKRouter.__init__ = patched_init
        TopKRouter.routing = patched_routing

    # --- Rollout routing replay support ---

    def fill_from_rollout(
        self,
        rollout_routed_experts: torch.Tensor,
        micro_batches: list,
        model_modules: nn.ModuleList,
        sequence_parallel: bool = False,
    ) -> None:
        """
        Fill routing caches from rollout-captured expert indices.

        This is the "rollout routing replay" path: during inference, SGLang captures
        which experts were routed for each token. These decisions are then replayed
        during training to ensure exact consistency with the rollout.

        Args:
            rollout_routed_experts: Tensor of shape [batch_size, num_tokens, num_moe_layers, topk]
                Expert indices captured during rollout inference.
            micro_batches: List of micro-batches (TensorDicts) for iteration count.
            model_modules: The Megatron model module list (for VPP stage iteration).
            sequence_parallel: Whether sequence parallel is enabled (requires TP slicing).
        """
        from megatron.core import parallel_state as mpu

        try:
            from megatron.core.transformer.transformer_block import get_num_layers_to_build
            from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
        except ImportError:
            logger.error("[RoutingReplay] Cannot import Megatron layer utilities")
            return

        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        # Process each micro-batch's routing data
        for mb_idx in range(len(micro_batches)):
            # rollout_routed_experts: [seq_len, num_layers, topk] per micro-batch
            # This tensor comes from the rollout engine with shape per sample
            mb_routing = rollout_routed_experts[mb_idx]
            if not isinstance(mb_routing, torch.Tensor):
                mb_routing = torch.tensor(mb_routing, dtype=torch.int64)

            # Handle sequence parallel: slice along sequence dimension
            if sequence_parallel and tp_size > 1:
                seq_len = mb_routing.shape[0]
                assert seq_len % tp_size == 0, f"seq_len {seq_len} not divisible by tp_size {tp_size}"
                chunk_size = seq_len // tp_size
                mb_routing = mb_routing[tp_rank * chunk_size : (tp_rank + 1) * chunk_size]

            # Distribute to per-layer caches following VPP stage ordering
            cache_offset = 0
            for vp_stage, model_chunk in enumerate(model_modules):
                config = model_chunk.module.config if hasattr(model_chunk, "module") else model_chunk.config
                num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
                offset = get_transformer_layer_offset(config, vp_stage=vp_stage)

                for layer_id in range(offset, offset + num_layers_to_build):
                    # Skip non-MoE layers
                    moe_layer_freq = getattr(config, "moe_layer_freq", 1)
                    if isinstance(moe_layer_freq, int):
                        if moe_layer_freq > 0 and layer_id % moe_layer_freq != 0:
                            continue
                    elif isinstance(moe_layer_freq, list):
                        if moe_layer_freq[layer_id] == 0:
                            continue

                    layer_routing = mb_routing[:, layer_id]  # [seq_len, topk]
                    self._caches[cache_offset].record(layer_routing)
                    cache_offset += 1

            assert cache_offset == len(self._caches), (
                f"[RoutingReplay] Cache count mismatch: filled {cache_offset}, "
                f"registered {len(self._caches)}"
            )
