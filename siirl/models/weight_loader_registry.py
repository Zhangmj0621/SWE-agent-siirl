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
Registry for model weight savers.

This module provides a registry mechanism for different model architectures
to register their weight saving functions. The weight saver functions are
responsible for converting Megatron distributed model weights into
HuggingFace format.

Usage:
    from siirl.models.weight_loader_registry import get_weight_saver

    weight_saver = get_weight_saver("llama")
    state_dict = weight_saver(model, hf_config, dtype=torch.bfloat16)
"""

from collections.abc import Callable

from loguru import logger

# Global registry for weight savers
_WEIGHT_SAVERS: dict[str, Callable] = {}


def register_weight_saver(arch: str, saver: Callable) -> None:
    """
    Register a weight saver function for a model architecture.

    Args:
        arch: Architecture name (e.g., "llama", "qwen2").
        saver: Function that converts Megatron weights to HuggingFace format.
               Expected signature: saver(model, hf_config, dtype, is_value_model, **kwargs) -> dict
    """
    arch_lower = arch.lower()
    if arch_lower in _WEIGHT_SAVERS:
        logger.warning(f"Overwriting existing weight saver for architecture: {arch}")
    _WEIGHT_SAVERS[arch_lower] = saver
    logger.debug(f"Registered weight saver for architecture: {arch}")


def get_weight_saver(arch: str) -> Callable:
    """
    Get the weight saver function for the given architecture.

    Args:
        arch: Architecture name (e.g., "llama", "qwen2").

    Returns:
        The weight saver function for the architecture.

    Raises:
        ValueError: If the architecture is not supported.
    """
    arch_lower = arch.lower()

    # Lazy registration of built-in weight savers
    _ensure_builtin_savers_registered()

    if arch_lower not in _WEIGHT_SAVERS:
        raise ValueError(
            f"Unsupported architecture for weight saving: {arch}. "
            f"Supported architectures: {list(_WEIGHT_SAVERS.keys())}. "
            f"You can register a custom saver using register_weight_saver()."
        )
    return _WEIGHT_SAVERS[arch_lower]


def list_supported_architectures() -> list:
    """
    List all supported architectures for weight saving.

    Returns:
        List of supported architecture names.
    """
    _ensure_builtin_savers_registered()
    return list(_WEIGHT_SAVERS.keys())


def _ensure_builtin_savers_registered() -> None:
    """
    Ensure built-in weight savers are registered.
    This is called lazily to avoid import issues.
    """
    if _WEIGHT_SAVERS:
        return

    try:
        from siirl.models.llama.megatron.checkpoint_utils.llama_saver import (
            merge_megatron_ckpt_llama,
        )

        # Register Llama-based architectures
        # Llama and Qwen2 use the same saver since they have similar architectures
        _WEIGHT_SAVERS["llama"] = merge_megatron_ckpt_llama
        _WEIGHT_SAVERS["llama2"] = merge_megatron_ckpt_llama
        _WEIGHT_SAVERS["llama3"] = merge_megatron_ckpt_llama
        _WEIGHT_SAVERS["qwen2"] = merge_megatron_ckpt_llama
        _WEIGHT_SAVERS["qwen3"] = merge_megatron_ckpt_llama

        logger.debug("Registered built-in weight savers for: llama, llama2, llama3, qwen2, qwen3")
    except ImportError as e:
        logger.warning(f"Failed to register built-in weight savers: {e}")
