# Copyright 2025, Shanghai Innovation Institute. All rights reserved.
"""Memory-optimized logits processing for PPO training."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from loguru import logger
from megatron.core import parallel_state as mpu

from siirl.utils.megatron.tensor_parallel import vocab_parallel_entropy, vocab_parallel_log_probs_from_logits

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class LogitsProcessorConfig:
    """Configuration for logits processing."""

    chunk_size: int = 4096
    use_checkpoint: bool = True
    use_fused: bool = False
    debug: bool = False

    @classmethod
    def from_actor_config(
        cls,
        chunk_size: int = 4096,
        use_checkpoint: bool = True,
        use_fused: bool = False,
    ) -> LogitsProcessorConfig:
        """Create config from ActorArguments, with env var overrides for debug."""
        # Allow env var override for chunk_size (useful for tuning)
        chunk_size_env = os.environ.get("SIIRL_CE_CHUNK_SIZE", "").strip()
        return cls(
            chunk_size=int(chunk_size_env) if chunk_size_env else chunk_size,
            use_checkpoint=use_checkpoint,
            use_fused=use_fused,
            debug=os.environ.get("SIIRL_LOGPROB_DEBUG", "0") == "1",
        )

    @classmethod
    def from_env(cls) -> LogitsProcessorConfig:
        """Create config from environment variables (fallback)."""
        chunk_size_env = os.environ.get("SIIRL_CE_CHUNK_SIZE", "").strip()
        return cls(
            chunk_size=int(chunk_size_env) if chunk_size_env else 4096,
            use_checkpoint=os.environ.get("SIIRL_USE_CHECKPOINT", "1") == "1",
            use_fused=os.environ.get("SIIRL_USE_FUSED_LOGPROB", "0") == "1",
            debug=os.environ.get("SIIRL_LOGPROB_DEBUG", "0") == "1",
        )


def _get_fused_cross_entropy() -> Callable | None:
    """Try to import fused cross entropy kernel."""
    try:
        from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

        return fused_vocab_parallel_cross_entropy
    except ImportError:
        return None


def compute_log_probs_chunk(
    logits: Tensor,
    labels: Tensor,
    use_fused: bool = True,
    fused_fn: Callable | None = None,
) -> Tensor:
    """Compute log probabilities for a chunk of logits."""
    if use_fused and fused_fn is not None:
        return -fused_fn(
            logits.unsqueeze(1),
            labels.unsqueeze(1),
            mpu.get_tensor_model_parallel_group(),
        ).squeeze(1)
    return vocab_parallel_log_probs_from_logits(logits, labels)


def compute_entropy_chunk(logits: Tensor) -> Tensor:
    """Compute entropy for a chunk of logits."""
    return vocab_parallel_entropy(logits)


def collect_response_indices(
    attention_mask: Tensor,
    cu_seqlens: Tensor,
    label_mask: Tensor,
) -> tuple[list[int], list[int], list[tuple[int, int, int]]]:
    """Collect response token indices from packed sequences."""
    batch_size = attention_mask.shape[0]
    seq_lens = attention_mask.sum(dim=1).tolist()
    cu_seqlens_cpu = cu_seqlens.tolist()
    packed_mask = label_mask.squeeze(0).to(torch.bool)

    all_logits_indices = []
    all_label_indices = []
    sample_boundaries = []

    output_pos = 0
    for i in range(batch_size):
        sample_start = cu_seqlens_cpu[i]
        sample_len = seq_lens[i]

        if sample_len < 2:
            sample_boundaries.append((i, output_pos, 0))
            continue

        sample_end = sample_start + sample_len
        sample_mask = packed_mask[sample_start:sample_end]
        sample_resp_indices = sample_mask.nonzero(as_tuple=False).squeeze(-1)

        if sample_resp_indices.numel() == 0:
            sample_boundaries.append((i, output_pos, 0))
            continue

        local_start = sample_resp_indices[0].item()
        local_end = sample_resp_indices[-1].item() + 1

        if local_start == 0:
            local_start = 1
            if local_end <= local_start:
                sample_boundaries.append((i, output_pos, 0))
                continue

        actual_resp_len = local_end - local_start
        logits_start = sample_start + local_start - 1
        label_start = sample_start + local_start

        for j in range(actual_resp_len):
            all_logits_indices.append(logits_start + j)
            all_label_indices.append(label_start + j)

        sample_boundaries.append((i, output_pos, actual_resp_len))
        output_pos += actual_resp_len

    return all_logits_indices, all_label_indices, sample_boundaries


def process_slice_mode(
    packed_logits: Tensor,
    packed_label: Tensor,
    attention_mask: Tensor,
    cu_seqlens: Tensor,
    label_mask: Tensor,
    response_length: int,
    calculate_entropy: bool,
    config: LogitsProcessorConfig,
) -> dict:
    """Process logits using zero-copy slicing (inference mode)."""
    fused_fn = _get_fused_cross_entropy() if config.use_fused else None
    use_fused = fused_fn is not None

    batch_size = attention_mask.shape[0]
    seq_lens = attention_mask.sum(dim=1).tolist()
    cu_seqlens_cpu = cu_seqlens.tolist()
    packed_mask = label_mask.squeeze(0).to(torch.bool)

    log_probs = packed_logits.new_zeros((batch_size, response_length), dtype=torch.float32)
    entropy = packed_logits.new_zeros((batch_size, response_length), dtype=torch.float32) if calculate_entropy else None

    if config.debug:
        logits_gb = packed_logits.numel() * packed_logits.element_size() / (1024**3)
        logger.debug(f"[LogProb] SLICE: {tuple(packed_logits.shape)} ({logits_gb:.2f}GB)")

    for i in range(batch_size):
        sample_start = cu_seqlens_cpu[i]
        sample_len = seq_lens[i]

        if sample_len < 2:
            continue

        sample_mask = packed_mask[sample_start : sample_start + sample_len]
        sample_resp_indices = sample_mask.nonzero(as_tuple=False).squeeze(-1)

        if sample_resp_indices.numel() == 0:
            continue

        local_start = sample_resp_indices[0].item()
        local_end = sample_resp_indices[-1].item() + 1

        if local_start == 0:
            local_start = 1
            if local_end <= local_start:
                continue

        actual_resp_len = local_end - local_start
        logits_start = sample_start + local_start - 1
        label_start = sample_start + local_start
        pad_len = response_length - actual_resp_len

        logits_chunk = packed_logits[logits_start : logits_start + actual_resp_len]
        labels_chunk = packed_label[label_start : label_start + actual_resp_len]

        # Entropy first, then log_probs (clone logits for log_probs when computing entropy)
        if calculate_entropy:
            logits_for_log_probs = logits_chunk.clone()
            entropy[i, pad_len:] = compute_entropy_chunk(logits_chunk).to(torch.float32)
        else:
            logits_for_log_probs = logits_chunk

        log_probs[i, pad_len:] = compute_log_probs_chunk(logits_for_log_probs, labels_chunk, use_fused, fused_fn).to(torch.float32)

    result = {"log_probs": log_probs, "_response_only": True}
    if calculate_entropy:
        result["entropy"] = entropy
    return result


def process_batched_mode(
    packed_logits: Tensor,
    packed_label: Tensor,
    attention_mask: Tensor,
    cu_seqlens: Tensor,
    label_mask: Tensor,
    response_length: int,
    calculate_entropy: bool,
    config: LogitsProcessorConfig,
) -> dict:
    """Process logits with gradient checkpointing (training mode)."""
    fused_fn = _get_fused_cross_entropy() if config.use_fused else None
    use_fused = fused_fn is not None

    batch_size = attention_mask.shape[0]
    all_logits_indices, all_label_indices, sample_boundaries = collect_response_indices(attention_mask, cu_seqlens, label_mask)
    total_response_tokens = len(all_logits_indices)

    log_probs = packed_logits.new_zeros((batch_size, response_length), dtype=torch.float32)
    entropy = packed_logits.new_zeros((batch_size, response_length), dtype=torch.float32) if calculate_entropy else None

    if total_response_tokens == 0:
        result = {"log_probs": log_probs, "_response_only": True}
        if calculate_entropy:
            result["entropy"] = entropy
        return result

    if config.debug:
        logits_gb = packed_logits.numel() * packed_logits.element_size() / (1024**3)
        logger.debug(f"[LogProb] BATCHED: tokens={total_response_tokens} ({logits_gb:.2f}GB)")

    logits_idx = torch.tensor(all_logits_indices, device=packed_logits.device, dtype=torch.long)
    label_idx = torch.tensor(all_label_indices, device=packed_logits.device, dtype=torch.long)

    all_log_probs, all_entropy = _compute_chunked_log_probs(
        packed_logits=packed_logits,
        packed_label=packed_label,
        logits_idx=logits_idx,
        label_idx=label_idx,
        calculate_entropy=calculate_entropy,
        use_fused=use_fused,
        fused_fn=fused_fn,
        config=config,
    )

    for sample_idx, start_pos, length in sample_boundaries:
        if length == 0:
            continue
        pad_len = response_length - length
        log_probs[sample_idx, pad_len:] = all_log_probs[start_pos : start_pos + length].to(torch.float32)
        if calculate_entropy and all_entropy is not None:
            entropy[sample_idx, pad_len:] = all_entropy[start_pos : start_pos + length].to(torch.float32)

    result = {"log_probs": log_probs, "_response_only": True}
    if calculate_entropy:
        result["entropy"] = entropy
    return result


def _compute_chunked_log_probs(
    packed_logits: Tensor,
    packed_label: Tensor,
    logits_idx: Tensor,
    label_idx: Tensor,
    calculate_entropy: bool,
    use_fused: bool,
    fused_fn: Callable | None,
    config: LogitsProcessorConfig,
) -> tuple[Tensor, Tensor | None]:
    """Compute log_probs with optional gradient checkpointing."""
    total_tokens = logits_idx.shape[0]

    if total_tokens <= config.chunk_size:
        all_logits = packed_logits.index_select(0, logits_idx)
        all_labels = packed_label.index_select(0, label_idx)

        if calculate_entropy:
            logits_for_log_probs = all_logits.clone()
            all_entropy = compute_entropy_chunk(all_logits)
        else:
            logits_for_log_probs = all_logits
            all_entropy = None

        all_log_probs = compute_log_probs_chunk(logits_for_log_probs, all_labels, use_fused, fused_fn)
        del all_logits, all_labels
        return all_log_probs, all_entropy

    num_chunks = (total_tokens - 1) // config.chunk_size + 1
    logits_idx_chunks = logits_idx.chunk(num_chunks, dim=0)
    label_idx_chunks = label_idx.chunk(num_chunks, dim=0)

    log_probs_list = []
    entropy_list = [] if calculate_entropy else None

    if config.use_checkpoint:
        from torch.utils.checkpoint import checkpoint as torch_checkpoint

        def _checkpoint_fn(packed_logits_in, packed_label_in, logits_idx_chunk, label_idx_chunk):
            logits_chunk = packed_logits_in.index_select(0, logits_idx_chunk)
            labels_chunk = packed_label_in.index_select(0, label_idx_chunk)

            if calculate_entropy:
                logits_for_log_probs = logits_chunk.clone()
                entropy_chunk = compute_entropy_chunk(logits_chunk)
                return compute_log_probs_chunk(logits_for_log_probs, labels_chunk, use_fused, fused_fn), entropy_chunk

            return compute_log_probs_chunk(logits_chunk, labels_chunk, use_fused, fused_fn)

        for logits_idx_chunk, label_idx_chunk in zip(logits_idx_chunks, label_idx_chunks, strict=False):
            result = torch_checkpoint(
                _checkpoint_fn,
                packed_logits,
                packed_label,
                logits_idx_chunk,
                label_idx_chunk,
                use_reentrant=False,
            )
            if calculate_entropy:
                log_probs_list.append(result[0])
                entropy_list.append(result[1])
            else:
                log_probs_list.append(result)
    else:
        for logits_idx_chunk, label_idx_chunk in zip(logits_idx_chunks, label_idx_chunks, strict=False):
            logits_chunk = packed_logits.index_select(0, logits_idx_chunk)
            labels_chunk = packed_label.index_select(0, label_idx_chunk)

            if calculate_entropy:
                logits_for_log_probs = logits_chunk.clone()
                entropy_list.append(compute_entropy_chunk(logits_chunk))
            else:
                logits_for_log_probs = logits_chunk

            log_probs_list.append(compute_log_probs_chunk(logits_for_log_probs, labels_chunk, use_fused, fused_fn))
            del logits_chunk, labels_chunk

    all_log_probs = torch.cat(log_probs_list, dim=0)
    all_entropy = torch.cat(entropy_list, dim=0) if calculate_entropy else None

    return all_log_probs, all_entropy


def process_fallback_mode(
    packed_logits: Tensor,
    packed_label: Tensor,
    label_mask: Tensor,
    calculate_entropy: bool,
    config: LogitsProcessorConfig,
) -> dict:
    """Fallback mode matching original master code behavior."""
    if config.debug:
        logits_gb = packed_logits.numel() * packed_logits.element_size() / (1024**3)
        logger.debug(f"[LogProb] Fallback: {tuple(packed_logits.shape)} ({logits_gb:.2f}GB)")

    ret = {}
    if calculate_entropy:
        logits_bak = packed_logits.clone()
        ret["entropy"] = vocab_parallel_entropy(packed_logits).unsqueeze(0)
    else:
        logits_bak = packed_logits

    log_probs = vocab_parallel_log_probs_from_logits(logits_bak, packed_label)
    ret["log_probs"] = log_probs.masked_fill(~label_mask.squeeze(0), 0.0).unsqueeze(0)
    ret["_response_only"] = False
    return ret
