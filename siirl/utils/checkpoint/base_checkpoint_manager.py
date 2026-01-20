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

"""Base checkpoint manager with configurable save/load contents."""

import os
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed
from filelock import FileLock
from loguru import logger
from transformers import PreTrainedTokenizer, ProcessorMixin

from siirl.params.model_args import CheckpointArguments


class BaseCheckpointManager:
    """
    Base checkpoint manager that provides common functionality for saving and loading checkpoints.

    This class manages the save/load content configuration and provides utility methods
    for checkpoint operations. Subclasses should implement the actual save_checkpoint
    and load_checkpoint methods.

    Attributes:
        model: The model to checkpoint.
        optimizer: The optimizer instance.
        lr_scheduler: The learning rate scheduler.
        processing_class: Tokenizer or processor for saving HF format.
        checkpoint_save_contents: List of content types to save.
        checkpoint_load_contents: List of content types to load.
        previous_saved_paths: List of previously saved checkpoint paths for cleanup.

    Supported content types:
        - "model": Model weights
        - "optimizer": Optimizer states
        - "extra": Extra states (RNG, lr_scheduler, etc.)
        - "hf_model": HuggingFace format model
    """

    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer = None,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        processing_class: PreTrainedTokenizer | ProcessorMixin = None,
        checkpoint_config: CheckpointArguments = None,
    ):
        """
        Initialize the base checkpoint manager.

        Args:
            model: The model to checkpoint.
            optimizer: The optimizer instance (optional).
            lr_scheduler: The learning rate scheduler (optional).
            processing_class: Tokenizer or processor for HF format saving (optional).
            checkpoint_config: Configuration for save/load contents (optional).
        """
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.processing_class = processing_class
        self.checkpoint_config = checkpoint_config

        # Initialize save/load contents from config or use defaults
        if checkpoint_config is not None:
            self.checkpoint_save_contents = checkpoint_config.save_contents
            self.checkpoint_load_contents = checkpoint_config.load_contents
        else:
            self.checkpoint_save_contents = ["model", "optimizer", "extra"]
            self.checkpoint_load_contents = ["model", "optimizer", "extra"]

        # Track previous saved paths for cleanup
        self.previous_saved_paths: list[str] = []
        self.previous_global_step: int | None = None

        # Get distributed info
        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    @property
    def should_save_model(self) -> bool:
        """Returns True if 'model' is in checkpoint_save_contents."""
        return "model" in self.checkpoint_save_contents

    @property
    def should_save_optimizer(self) -> bool:
        """Returns True if 'optimizer' is in checkpoint_save_contents."""
        return "optimizer" in self.checkpoint_save_contents

    @property
    def should_save_extra(self) -> bool:
        """Returns True if 'extra' is in checkpoint_save_contents."""
        return "extra" in self.checkpoint_save_contents

    @property
    def should_save_hf_model(self) -> bool:
        """Returns True if 'hf_model' is in checkpoint_save_contents."""
        return "hf_model" in self.checkpoint_save_contents

    @property
    def should_load_model(self) -> bool:
        """Returns True if 'model' is in checkpoint_load_contents."""
        return "model" in self.checkpoint_load_contents

    @property
    def should_load_optimizer(self) -> bool:
        """Returns True if 'optimizer' is in checkpoint_load_contents."""
        return "optimizer" in self.checkpoint_load_contents

    @property
    def should_load_extra(self) -> bool:
        """Returns True if 'extra' is in checkpoint_load_contents."""
        return "extra" in self.checkpoint_load_contents

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load: bool = False):
        """
        Load checkpoint from the specified path.

        Args:
            local_path: Local path to load checkpoint from.
            hdfs_path: Optional HDFS path (not implemented in base class).
            del_local_after_load: Whether to delete local files after loading.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement load_checkpoint")

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: str = None,
        global_step: int = 0,
        max_ckpt_to_keep: int = None,
    ):
        """
        Save checkpoint to the specified path.

        Args:
            local_path: Local path to save checkpoint to.
            hdfs_path: Optional HDFS path (not implemented in base class).
            global_step: Current global training step.
            max_ckpt_to_keep: Maximum number of checkpoints to keep.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement save_checkpoint")

    def remove_previous_save_local_path(self, paths: str | list[str]) -> None:
        """
        Remove old checkpoint directories to save disk space.

        This method will delete entire global_step_* directories if the path
        is within such a directory structure.

        Args:
            paths: Single path or list of paths to remove.
        """
        if isinstance(paths, str):
            paths = [paths]

        for path in paths:
            abs_path = os.path.abspath(path)
            if not os.path.exists(abs_path):
                continue

            # Check if parent directory is a global_step_* folder
            global_step_path = Path(path).parent
            delete_path = abs_path

            if "global_step_" in str(global_step_path) and os.path.exists(global_step_path):
                delete_path = str(global_step_path)

            if self.rank == 0:
                logger.info(f"Removing old checkpoint: {delete_path}")
                try:
                    shutil.rmtree(delete_path, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {delete_path}: {e}")

    @staticmethod
    def local_mkdir(path: str) -> str:
        """
        Create a local directory with file locking for thread safety.

        Args:
            path: Directory path to create.

        Returns:
            The absolute path of the created directory.
        """
        if not os.path.isabs(path):
            working_dir = os.getcwd()
            path = os.path.join(working_dir, path)

        # Use hash of path as lock file name to avoid long file names
        lock_filename = f"ckpt_{hash(path) & 0xFFFFFFFF:08x}.lock"
        lock_path = os.path.join(tempfile.gettempdir(), lock_filename)

        try:
            with FileLock(lock_path, timeout=60):
                os.makedirs(path, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to acquire lock for {path}: {e}")
            # Try to create directory even without lock
            os.makedirs(path, exist_ok=True)

        return path

    @staticmethod
    def get_rng_state() -> dict:
        """
        Get current RNG states for reproducibility.

        Returns:
            Dictionary containing CPU, NumPy, Python random, and GPU RNG states.
        """
        rng_state = {
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }

        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state()

        return rng_state

    @staticmethod
    def load_rng_state(rng_state: dict) -> None:
        """
        Restore RNG states from a saved checkpoint.

        Args:
            rng_state: Dictionary containing saved RNG states.
        """
        torch.set_rng_state(rng_state["cpu"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])

        if torch.cuda.is_available() and "cuda" in rng_state:
            torch.cuda.set_rng_state(rng_state["cuda"])
