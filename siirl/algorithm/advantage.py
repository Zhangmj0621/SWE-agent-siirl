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

from collections import defaultdict
from enum import Enum
from typing import Any

import numpy as np
import torch
from loguru import logger
from tensordict import TensorDict

import siirl.utils.model_utils.torch_functional as siirl_F
from siirl.params.model_args import AlgorithmArguments


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    PPO = "ppo"
    GRPO = "grpo"


ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register_adv_est(name_or_enum: str | AdvantageEstimator) -> Any:
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}")
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def compute_response_mask(data: TensorDict):
    """Compute the attention mask for the response part of the sequence.

    Handles both 2D responses (NLP) and 3D responses (Embodied AI).

    Returns:
        torch.Tensor: The attention mask for the response tokens (always 2D).
    """
    responses = data["responses"]
    attention_mask = data["attention_mask"]
    batch_size = responses.size(0)

    # Handle 3D responses (Embodied AI): (batch_size, traj_len, action_token_len)
    if responses.ndim == 3:
        traj_len = responses.size(1)
        action_token_len = responses.size(2)

        # Check if attention_mask is also 3D
        if attention_mask.ndim == 3:
            # attention_mask: (batch_size, traj_len, tot_pad_len)
            # Extract response part from last dimension: (batch_size, traj_len, action_token_len)
            response_mask = attention_mask[:, :, -action_token_len:]
            # Flatten to 2D: (batch_size, traj_len * action_token_len)
            response_mask = response_mask.reshape(batch_size, -1)
        else:
            # attention_mask is 2D: (batch_size, total_length)
            # Calculate flattened response_length and slice
            response_length = traj_len * action_token_len
            response_mask = attention_mask[:, -response_length:]
    # Handle 2D responses (NLP): (batch_size, response_length)
    elif responses.ndim == 2:
        response_length = responses.size(1)
        response_mask = attention_mask[:, -response_length:]
    else:
        raise ValueError(f"Unexpected responses shape: {responses.shape}, ndim={responses.ndim}")

    return response_mask


@register_adv_est(AdvantageEstimator.PPO)
def compute_ppo_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]
        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam

            # skip values and TD-error on observation tokens
            nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam

            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = siirl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: AlgorithmArguments | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgorithmArguments])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            # Convert tensor index to Python int for use as dict key
            idx_key = int(index[i].item()) if isinstance(index[i], torch.Tensor) else int(index[i])
            id2score[idx_key].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            # Convert tensor index to Python int for dict lookup
            idx_key = int(index[i].item()) if isinstance(index[i], torch.Tensor) else int(index[i])
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[idx_key]) / (id2std[idx_key] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[idx_key]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_advantage(
    data: TensorDict,
    adv_estimator,
    gamma=1.0,
    lam=1.0,
    norm_adv_by_std_in_grpo=True,
    weight_factor_in_cpgd="STD_weight",
    **kwargs,
):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, CPGD, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (TensorDict): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++, CPGD).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.
        weight_factor_in_cpgd (str, optional): whether to use the STD weight as GRPO or
            clip_filter_like_weight. choices: {STD_weight, clip_filter_like_weight, naive}

    Returns:
        TensorDict: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.PPO:
        advantages, returns = compute_ppo_advantage_return(
            token_level_rewards=data["token_level_rewards"],
            values=data["values"],
            response_mask=data["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data["advantages"] = advantages
        data["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        grpo_calculation_mask = data["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=data["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data["advantages"] = advantages
        data["returns"] = returns
        # Store the mask for consistent metrics calculation
        data["response_mask"] = grpo_calculation_mask
        logger.debug("[GRPO] Stored response_mask in batch for consistent metrics")
    else:
        raise NotImplementedError
    return data
