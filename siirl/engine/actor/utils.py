"""Simplified utility functions for actor training"""

import os
import hashlib
import tempfile
from typing import Dict, Optional

import torch
import numpy as np
from siirl.utils.model_utils.torch_functional import masked_mean
from siirl.utils.backend.device import get_torch_device



def append_to_dict(data: Dict, new_data: Dict):
    """Append values from new_data to lists in data.

    For each key in new_data, this function appends the corresponding value to a list
    stored under the same key in data. If the key doesn't exist in data, a new list is created.

    Args:
        data: Target dictionary containing lists as values
        new_data: Source dictionary with values to append

    Example:
        >>> metrics = {}
        >>> append_to_dict(metrics, {"loss": 0.5})
        >>> append_to_dict(metrics, {"loss": 0.3})
        >>> metrics
        {'loss': [0.5, 0.3]}
    """
    for key, val in new_data.items():
        if key not in data:
            data[key] = []
        data[key].append(val)



def copy_to_local(
    src: str,
    cache_dir: Optional[str] = None,
    filelock: str = ".file.lock",
    verbose: bool = False,
    always_recopy: bool = False,
    use_shm: bool = False
) -> str:
    """Copy files/directories from remote to local cache (simplified version).

    This is a simplified version that primarily handles local paths and basic HDFS support.
    For full HDFS functionality, consider using the original implementation.

    Args:
        src: Source path - can be HDFS (hdfs://...) or local filesystem path
        cache_dir: Local directory for cached files. Uses system tempdir if None
        filelock: Base name for file lock (unused in simplified version)
        verbose: Enable copy operation logging
        always_recopy: Force fresh copy ignoring cache
        use_shm: Enable shared memory copy (unused in simplified version)

    Returns:
        Local filesystem path to the resource

    Example:
        >>> path = copy_to_local("/path/to/model")
        >>> path
        '/path/to/model'
    """
    # Check if it's a remote HDFS path
    if src.startswith("hdfs://"):
        # For HDFS paths, download to local cache
        if cache_dir is None:
            cache_dir = tempfile.gettempdir()
        os.makedirs(cache_dir, exist_ok=True)

        # Create unique local path based on HDFS path hash
        src_hash = hashlib.md5(src.encode()).hexdigest()
        temp_dir = os.path.join(cache_dir, src_hash)
        os.makedirs(temp_dir, exist_ok=True)
        local_path = os.path.join(temp_dir, os.path.basename(src))

        # Check if already cached
        if not always_recopy and os.path.exists(local_path):
            if verbose:
                print(f"Using cached copy at {local_path}")
            return local_path

        # For actual HDFS copy, you need hdfs_io module
        try:
            from hdfs_io import copy
            if verbose:
                print(f"Copying from {src} to {local_path}")
            copy(src, local_path)
        except ImportError:
            raise ImportError(
                "hdfs_io module not available. "
                "For HDFS support, install hdfs_io or use local paths only."
            )

        return local_path
    else:
        # For local paths, return as-is
        return src


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """Aggregate loss matrix into a scalar"""
    if loss_agg_mode == "token-mean":
        loss = masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.mean(seq_losses)
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
        loss = torch.mean(seq_losses)
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]
    else:
        raise ValueError(f"Unsupported loss_agg_mode: {loss_agg_mode}")
    return loss


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty: str) -> torch.FloatTensor:
    """Compute KL divergence penalty"""
    forward_score = kl_penalty_forward(logprob, ref_logprob, kl_penalty)
    if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):
        return forward_score

    # Straight through trick for unbiased KL gradient
    backward_score = 0.5 * (logprob - ref_logprob).square()
    return backward_score - backward_score.detach() + forward_score.detach()


def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty: str) -> torch.FloatTensor:
    """Compute KL divergence forward pass"""
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    raise ValueError(f"Unsupported kl_penalty: {kl_penalty}")


def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[object] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute vanilla PPO clipped policy loss"""
    assert config is not None

    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if hasattr(config, 'clip_ratio_low') and config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if hasattr(config, 'clip_ratio_high') and config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.clip_ratio_c if hasattr(config, 'clip_ratio_c') else 3.0

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)

    # PPO KL divergence
    ppo_kl = masked_mean(-negative_approx_kl, response_mask)

    # Clipped policy gradient loss
    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    pg_loss_max = torch.max(pg_losses, pg_losses2)

    # Apply rollout importance weights if provided
    if rollout_is_weights is not None:
        pg_loss_max = pg_loss_max * rollout_is_weights

    pg_loss = agg_loss(loss_mat=pg_loss_max, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    # Clip fraction
    pg_clipfrac = masked_mean((pg_losses2 > pg_losses).to(torch.float32), response_mask)

    # Lower clip fraction
    pg_losses_low = -advantages * torch.clamp(ratio, 1.0 - cliprange_low * clip_ratio_c, 1.0 + cliprange_high * clip_ratio_c)
    pg_clipfrac_lower = masked_mean((pg_losses_low > pg_losses).to(torch.float32), response_mask)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    """Compute clipped value function loss for PPO"""
    vpredclipped = torch.clamp(
        vpreds,
        values - cliprange_value,
        values + cliprange_value,
    )

    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss_max = torch.max(vf_losses1, vf_losses2)
    vf_loss = agg_loss(loss_mat=vf_loss_max, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    vf_clipfrac = masked_mean((vf_losses2 > vf_losses1).to(torch.float32), response_mask)

    return vf_loss, vf_clipfrac


# Policy loss registry
POLICY_LOSS_REGISTRY = {
    "vanilla": compute_policy_loss_vanilla,
}

def get_policy_loss_fn(name: str):
    """Get policy loss function by name"""
    if name not in POLICY_LOSS_REGISTRY:
        raise ValueError(f"Unsupported loss mode: {name}. Supported: {list(POLICY_LOSS_REGISTRY.keys())}")
    return POLICY_LOSS_REGISTRY[name]

def set_random_seed(seed):
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if get_torch_device().device_count() > 0:
        from megatron.core import tensor_parallel
        tensor_parallel.model_parallel_cuda_manual_seed(seed)
