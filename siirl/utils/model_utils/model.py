# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Utilities to create common models from huggingface
"""

import os
import re
import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from loguru import logger
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
    MistralForSequenceClassification,
    PreTrainedModel,
    AutoConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast


class LambdaLayer(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def squeeze(x):
    return torch.squeeze(x, dim=-1)


def update_model_config(module_config, override_config_kwargs):
    """Update the module config with the override_config_kwargs.
    Args:
        module_config: The module config from Huggingface Transformers.
        override_config_kwargs: The kwargs to override the module config.
    """
    for key, val in override_config_kwargs.items():
        if isinstance(val, dict):
            update_model_config(getattr(module_config, key), val)
        else:
            setattr(module_config, key, val)

def get_huggingface_actor_config(model_name: str, override_config_kwargs=None, trust_remote_code=False) -> Dict:
    if override_config_kwargs is None:
        override_config_kwargs = {}
    assert isinstance(override_config_kwargs, Dict), (
        f"override_config_kwargs must be a dict, got {type(override_config_kwargs)}"
    )
    module_config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    update_model_config(module_config, override_config_kwargs)

    return module_config

def get_generation_config(
    model: str,
    trust_remote_code: bool = False,
) -> Optional[GenerationConfig]:
    try:
        return GenerationConfig.from_pretrained(model)
    except OSError:  # Not found
        try:
            config = get_huggingface_actor_config(
                model,
                trust_remote_code=trust_remote_code,
            )
            return GenerationConfig.from_model_config(config)
        except OSError:  # Not found
            return None


def get_model_size(model: nn.Module, scale="auto"):
    n_params = sum(p.numel() for p in model.parameters())

    if scale == "auto":
        if n_params > 1e9:
            scale = "B"
        elif n_params > 1e6:
            scale = "M"
        elif n_params > 1e3:
            scale = "K"
        else:
            scale = ""

    if scale == "B":
        n_params = n_params / 1e9
    elif scale == "M":
        n_params = n_params / 1e6
    elif scale == "K":
        n_params = n_params / 1e3
    elif scale == "":
        pass
    else:
        raise NotImplementedError(f"Unknown scale {scale}")

    return n_params, scale


def print_model_size(model: nn.Module, name: str = None):
    n_params, scale = get_model_size(model, scale="auto")
    if name is None:
        name = model.__class__.__name__
    logger.info(f"{name} contains {n_params:.2f}{scale} parameters")


def normalize_model_name(name, pp_rank, vpp_rank, transformer_config, layer_name="layers"):
    """
    Transform the model name in each model_chunk in each pp stage into the name in inference engine
    """
    from siirl.utils.megatron.megatron_utils import get_transformer_layer_offset

    layer_offset = get_transformer_layer_offset(pp_rank, vpp_rank, transformer_config)

    if layer_name in name:  # belong to an intermediate layer
        split_name = name.split(".")
        # find the num next to split_name
        for i, name in enumerate(split_name):
            if name == layer_name:
                break
        layer_num_idx = i + 1
        # check the name
        assert len(split_name) >= layer_num_idx + 1, f"split_name = {split_name}"
        assert split_name[layer_num_idx].isdigit(), f"split_name = {split_name}"
        # increment layer_num_idx by layer_offset
        split_name[layer_num_idx] = str(int(split_name[layer_num_idx]) + layer_offset)
        name = ".".join(split_name)  # weight name in inference_tp_model
    return name


def _load_hf_model(config, model_config, is_value_model, local_cache_path):
    """Helper function containing the loading hf model logic"""
    from accelerate import init_empty_weights
    from megatron.core import parallel_state as mpu

    from siirl.models.mcore.saver import _megatron_calc_global_rank

    assert hasattr(model_config, "architectures"), "architectures cannot be empty when load weight!"
    architectures = getattr(model_config, "architectures", [])
    local_cache_path = os.path.expanduser(local_cache_path)

    if config.model.path.startswith("hdfs:"):
        from siirl.engine.actor.utils import copy_to_local

        logger.info(f"start download from {config.model.path}")
        local_model_path = copy_to_local(
            src=config.model.path, cache_dir=local_cache_path, use_shm=config.model.get("use_shm", False)
        )
        logger.info("finish download")
    else:
        local_model_path = config.model.path
        logger.info(f"load from local dir {local_model_path}")

    src_rank = _megatron_calc_global_rank(tp_rank=0, dp_rank=0, pp_rank=0, cp_rank=mpu.get_context_parallel_rank())
    cpu_init_weights = lambda: torch.device("cpu")
    init_context = init_empty_weights if torch.distributed.get_rank() != src_rank else cpu_init_weights
    with init_context(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # TODO: to find a better way to load mistral7b-rm lm_head
        if "mistral7b-rm" in config.model.path:
            model = MistralForSequenceClassification.from_pretrained(
                local_model_path,
                torch_dtype="auto",
                # device_map="auto",  # disable auto device_map, the HF weight is only loaded to CPU in src_rank
                # low_cpu_mem_usage=True
            )  # use score head instead of lm_head
            state_dict = model.state_dict()
            state_dict["lm_head.weight"] = state_dict["score.weight"]
            state_dict["model.embed_tokens.weight"] = state_dict["model.embed_tokens.weight"][
                :32000
            ]  # workaround, 32001 -> 32000
            is_value_model = True
        else:
            model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                torch_dtype="auto",
                # device_map="auto", # disable auto device_map, the HF weight is only loaded to CPU in src_rank
                # low_cpu_mem_usage=True
            )
            state_dict = model.state_dict()

    return architectures, model, state_dict, is_value_model


def get_hf_model_path(config, local_cache_path="~/.cache/siirl/rlhf"):
    local_cache_path = os.path.expanduser(local_cache_path)
    if config.model.path.startswith("hdfs:"):
        from siirl.engine.actor.utils import copy_to_local

        local_model_path = copy_to_local(
            src=config.model.path, cache_dir=local_cache_path, use_shm=config.model.use_shm
        )
    else:
        local_model_path = config.model.path
    return local_model_path


def load_megatron_gptmodel_weights(
    config, model_config, parallel_model, params_dtype, is_value_model=False, local_cache_path="~/.cache/siirl/rlhf"
):
    """Load weights for mcore GPT model."""
    _, model, state_dict, is_value_model = _load_hf_model(config, model_config, is_value_model, local_cache_path)

    from siirl.models.mcore.loader import load_state_dict_to_megatron_gptmodel

    load_state_dict_to_megatron_gptmodel(
        state_dict=state_dict,
        wrapped_models=parallel_model,
        config=model.config,
        params_dtype=params_dtype,
        is_value_model=is_value_model,
    )
    del state_dict, model


def convert_weight_keys(state_dict: Dict[str, torch.Tensor], model: PreTrainedModel):
    # convert state dict keys: https://github.com/huggingface/transformers/pull/38385
    if not hasattr(model, "_checkpoint_conversion_mapping"):
        return state_dict

    reverse_key_mapping = {v: k for k, v in model._checkpoint_conversion_mapping.items()}
    original_weights = {}
    for key, value in state_dict.items():
        for pattern, replacement in reverse_key_mapping.items():
            replacement = replacement.lstrip("^")  # strip off un-needed chars and patterns
            replacement = re.sub(r"\(.*\)", "", replacement)
            key, n_replace = re.subn(pattern, replacement, key)
            # Early exit of the loop
            if n_replace > 0:
                break

        original_weights[key] = value

    return original_weights

def compute_position_id_with_mask(mask):
    return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)

@dataclass
class CausalLMOutputForPPO(CausalLMOutputWithPast):
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None
