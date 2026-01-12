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
Megatron checkpoint manager with HuggingFace format support.

This module provides checkpoint management for Megatron-LM distributed training,
with support for saving checkpoints in both Megatron distributed format and
HuggingFace format.
"""

import glob
import os
import shutil
import warnings

import torch
import torch.distributed as dist
from loguru import logger
from megatron.core import mpu
from transformers import PreTrainedTokenizer, ProcessorMixin

from siirl.params.model_args import CheckpointArguments
from siirl.utils.checkpoint.base_checkpoint_manager import BaseCheckpointManager
from siirl.utils.megatron.dist_checkpointing import load_dist_checkpointing, save_dist_checkpointing


class MegatronCheckpointManager(BaseCheckpointManager):
    """
    Megatron checkpoint manager with configurable save/load contents.

    This manager handles saving and loading of Megatron distributed checkpoints,
    with optional support for converting and saving in HuggingFace format.

    Key features:
        - Distributed checkpoint saving/loading using Megatron's dist_checkpointing
        - Optional HuggingFace format model saving (when 'hf_model' in save_contents)
        - Support for tensor parallel, pipeline parallel configurations
        - Automatic cleanup of old checkpoints

    Example:
        ```python
        # Create manager with HF saving enabled
        checkpoint_config = CheckpointArguments(
            save_contents=["model", "optimizer", "extra", "hf_model"]
        )
        manager = MegatronCheckpointManager(
            model=megatron_model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            processing_class=tokenizer,
            checkpoint_config=checkpoint_config,
            hf_config=hf_config,
            model_path="/path/to/model",
            arch="llama",
        )

        # Save checkpoint (will also save HF format)
        manager.save_checkpoint(local_path="checkpoints/actor", global_step=100)
        ```
    """

    def __init__(
        self,
        model,
        optimizer,
        lr_scheduler=None,
        processing_class: PreTrainedTokenizer | ProcessorMixin = None,
        checkpoint_config: CheckpointArguments = None,
        hf_config=None,
        model_path: str = None,
        arch: str = None,
        param_dtype: torch.dtype = torch.bfloat16,
        is_value_model: bool = False,
        share_embeddings_and_output_weights: bool = False,
    ):
        """
        Initialize Megatron checkpoint manager.

        Args:
            model: Megatron model or list of models (for virtual pipeline).
            optimizer: Megatron optimizer.
            lr_scheduler: Learning rate scheduler (optional).
            processing_class: Tokenizer or processor for HF format saving.
            checkpoint_config: Configuration for save/load contents.
            hf_config: HuggingFace model config for HF format saving.
            model_path: Path to the original model (for HF saving).
            arch: Model architecture name (e.g., "llama", "qwen2").
            param_dtype: Parameter dtype for HF model saving.
            is_value_model: Whether this is a value/critic model.
            share_embeddings_and_output_weights: Whether embeddings are shared.
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_config=checkpoint_config,
        )

        # Ensure model is a list for virtual pipeline support
        self.model = model if isinstance(model, list) else [model]

        # HF model saving related attributes
        self.hf_config = hf_config
        self.model_path = model_path
        self.arch = arch
        self.param_dtype = param_dtype
        self.is_value_model = is_value_model
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: str = None,
        global_step: int = 0,
        max_ckpt_to_keep: int = None,
    ) -> None:
        """
        Save checkpoint using Megatron distributed checkpointing.

        This method saves the model, optimizer, and scheduler states in Megatron's
        distributed checkpoint format. If 'hf_model' is in save_contents, it will
        also save the model in HuggingFace format.

        Args:
            local_path: Local directory path to save the checkpoint.
            hdfs_path: HDFS path (not implemented, reserved for future use).
            global_step: Current global training step.
            max_ckpt_to_keep: Maximum number of step_* subdirectories to keep.
                             Note: global_step_* cleanup is handled by upper-level manager.
        """
        self.previous_global_step = global_step

        dist_checkpoint_path = os.path.join(local_path, f"step_{global_step}")

        # Create directory on all ranks before saving
        os.makedirs(dist_checkpoint_path, exist_ok=True)
        dist.barrier()

        # Generate and save state dict
        state_dict = self._generate_state_dict()

        save_dist_checkpointing(
            sharded_state_dict=state_dict,
            ckpt_path=dist_checkpoint_path,
            async_save=False,
        )

        dist.barrier()

        if self.rank == 0:
            logger.info(f"Saved Megatron checkpoint to {dist_checkpoint_path}")

        # Save HuggingFace format model if configured
        if self.should_save_hf_model:
            self._save_hf_model(local_path, global_step)

        # Save tokenizer and config (rank 0 only)
        if self.rank == 0 and self.processing_class is not None:
            self._save_tokenizer_and_config(local_path)

        dist.barrier()

        # Cleanup old step_* subdirectories within this local_path
        if self.rank == 0 and max_ckpt_to_keep and max_ckpt_to_keep > 0:
            self._cleanup_old_step_checkpoints(local_path, max_ckpt_to_keep)

        # Track this path for potential future cleanup
        self.previous_saved_paths.append(local_path)

    def load_checkpoint(
        self,
        local_path: str,
        hdfs_path: str = None,
        del_local_after_load: bool = False,
    ) -> None:
        """
        Load checkpoint using Megatron distributed checkpointing.

        Args:
            local_path: Local directory path to load the checkpoint from.
            hdfs_path: HDFS path (not implemented, reserved for future use).
            del_local_after_load: Whether to delete local files after loading.
        """
        step_dirs = glob.glob(os.path.join(local_path, "step_*"))
        if not step_dirs:
            logger.warning(f"No checkpoint found at {local_path}")
            return

        # Find the latest step directory
        latest_step_dir = max(step_dirs, key=lambda x: int(x.split("step_")[-1]))

        # Generate state dict template for loading
        state_dict = self._generate_state_dict()

        # Load checkpoint
        load_dist_checkpointing(
            sharded_state_dict=state_dict,
            ckpt_dir=latest_step_dir,
        )

        # Load lr_scheduler state
        self._load_state_dict(state_dict)

        dist.barrier()
        logger.info(f"Loaded checkpoint from {latest_step_dir}")

        # Optionally delete local checkpoint after loading
        if del_local_after_load and self.rank == 0:
            try:
                shutil.rmtree(local_path, ignore_errors=True)
                logger.info(f"Deleted local checkpoint: {local_path}")
            except Exception as e:
                logger.warning(f"Failed to delete local checkpoint {local_path}: {e}")

    def _generate_state_dict(self) -> dict:
        """
        Generate sharded state dict for save/load operations.

        Returns:
            Dictionary containing model, optimizer, and lr_scheduler states.
        """
        state_dict = {}

        # Generate model state dict for each virtual pipeline stage
        for vpp_rank, model in enumerate(self.model):
            if len(self.model) > 1:
                mpu.set_virtual_pipeline_model_parallel_rank(vpp_rank)
                key = f"model{vpp_rank}"
            else:
                key = "model"

            # Unwrap DDP wrapper if present
            if hasattr(model, "module"):
                model = model.module

            state_dict[key] = model.sharded_state_dict()

        dist.barrier()

        # Generate optimizer state dict
        optimizer_sharded_states = self.optimizer.sharded_state_dict(state_dict)
        state_dict["optimizer"] = optimizer_sharded_states

        # Add lr_scheduler state
        if self.lr_scheduler is not None:
            lr_state_dict = self.lr_scheduler.state_dict()
            state_dict["lr_scheduler"] = lr_state_dict

        return state_dict

    def _load_state_dict(self, state_dict: dict) -> None:
        """
        Load state dict into lr_scheduler.

        Note: Model and optimizer states are loaded automatically by load_dist_checkpointing.

        Args:
            state_dict: Dictionary containing loaded states.
        """
        if self.lr_scheduler is not None and "lr_scheduler" in state_dict:
            self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
            logger.debug("Loaded lr_scheduler state")

    def _save_hf_model(self, local_path: str, global_step: int) -> None:
        """
        Convert Megatron weights to HuggingFace format and save.

        This method uses the registered weight saver for the model architecture
        to merge sharded parameters and save in HuggingFace format.

        Args:
            local_path: Base path for saving the HF model.
            global_step: Current global step (used in path).
        """
        if self.arch is None:
            logger.warning("Cannot save HF model: 'arch' not specified. " "Set arch parameter to enable HF model saving.")
            return

        if self.hf_config is None:
            logger.warning("Cannot save HF model: 'hf_config' not specified. " "Set hf_config parameter to enable HF model saving.")
            return

        try:
            from siirl.models.weight_loader_registry import get_weight_saver

            weight_saver = get_weight_saver(self.arch)
        except (ImportError, ValueError) as e:
            logger.warning(f"Cannot save HF model: {e}")
            return

        logger.info("Converting Megatron checkpoint to HuggingFace format...")

        # Merge sharded weights into a single state dict
        state_dict = weight_saver(
            self.model,
            self.hf_config,
            dtype=self.param_dtype,
            is_value_model=self.is_value_model,
            tie_word_embeddings=self.share_embeddings_and_output_weights,
        )

        dist.barrier()

        # Only rank 0 saves the HF model
        if self.rank == 0:
            hf_model_path = os.path.join(local_path, "hf_model")
            os.makedirs(hf_model_path, exist_ok=True)

            try:
                from accelerate import init_empty_weights
                from transformers import AutoModelForCausalLM

                # Use init_empty_weights to avoid loading full model into memory
                with init_empty_weights(), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype="auto",
                        trust_remote_code=True,
                    )

                # Save model with merged state dict
                model.save_pretrained(hf_model_path, state_dict=state_dict)
                logger.info(f"Saved HuggingFace model to {hf_model_path}")

            except Exception as e:
                logger.error(f"Failed to save HuggingFace model: {e}")
                raise

        dist.barrier()

    def _save_tokenizer_and_config(self, local_path: str) -> None:
        """
        Save tokenizer and HF config to the checkpoint directory.

        Args:
            local_path: Base path for saving.
        """
        if self.processing_class is None:
            return

        try:
            # Save to hf_model subdirectory if HF model is being saved
            if self.should_save_hf_model:
                save_path = os.path.join(local_path, "hf_model")
            else:
                save_path = local_path

            os.makedirs(save_path, exist_ok=True)

            # Save tokenizer
            self.processing_class.save_pretrained(save_path)
            logger.debug(f"Saved tokenizer to {save_path}")

            # Save HF config
            if self.hf_config is not None:
                self.hf_config.save_pretrained(save_path)
                logger.debug(f"Saved HF config to {save_path}")

            # Try to save generation config if available
            if self.hf_config is not None and hasattr(self.hf_config, "name_or_path"):
                try:
                    from transformers import GenerationConfig

                    generation_config = GenerationConfig.from_pretrained(self.hf_config.name_or_path)
                    generation_config.save_pretrained(save_path)
                except Exception:
                    # Generation config may not be available for all models
                    pass

        except Exception as e:
            logger.warning(f"Failed to save tokenizer/config: {e}")

    def _cleanup_old_step_checkpoints(self, local_path: str, max_ckpt_to_keep: int) -> None:
        """
        Remove old step_* subdirectories to save disk space.

        Note: This only cleans up step_* directories within the local_path.
        The cleanup of global_step_* directories is handled by the upper-level
        CheckpointManager.

        Args:
            local_path: Base path containing step_* directories.
            max_ckpt_to_keep: Maximum number of step_* directories to keep.
        """
        step_dirs = sorted(
            glob.glob(os.path.join(local_path, "step_*")),
            key=lambda x: int(x.split("step_")[-1]),
        )

        if len(step_dirs) > max_ckpt_to_keep:
            for old_dir in step_dirs[:-max_ckpt_to_keep]:
                try:
                    shutil.rmtree(old_dir)
                    logger.info(f"Removed old step checkpoint: {old_dir}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {old_dir}: {e}")
