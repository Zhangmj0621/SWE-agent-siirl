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


import torch

from siirl.utils.model_utils.torch_functional import masked_mean


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


def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: object | None = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute vanilla PPO clipped policy loss (Dual-clip PPO).

    Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
    """
    assert config is not None

    clip_ratio = config.clip_ratio
    clip_ratio_low = (
        config.clip_ratio_low if hasattr(config, "clip_ratio_low") and config.clip_ratio_low is not None else clip_ratio
    )
    clip_ratio_high = (
        config.clip_ratio_high
        if hasattr(config, "clip_ratio_high") and config.clip_ratio_high is not None
        else clip_ratio
    )
    clip_ratio_c = config.clip_ratio_c if hasattr(config, "clip_ratio_c") else 3.0

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)

    # PPO KL divergence
    ppo_kl = masked_mean(-negative_approx_kl, response_mask)

    # Clipped policy gradient loss (Dual-clip PPO)
    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    # Dual-clip: additional lower bound for negative advantages
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    # Select based on advantage sign
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # Apply rollout importance weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


# Policy loss registry - must be defined after functions
POLICY_LOSS_REGISTRY = {
    "vanilla": compute_policy_loss_vanilla,
}


def get_policy_loss_fn(name: str):
    """Get policy loss function by name"""
    if name not in POLICY_LOSS_REGISTRY:
        raise ValueError(f"Unsupported loss mode: {name}. Supported: {list(POLICY_LOSS_REGISTRY.keys())}")
    return POLICY_LOSS_REGISTRY[name]
