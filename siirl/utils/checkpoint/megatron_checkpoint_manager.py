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

import os
import torch.distributed as dist
from loguru import logger
from megatron.core import mpu

from siirl.utils.megatron.dist_checkpointing import save_dist_checkpointing, load_dist_checkpointing


class MegatronCheckpointManager:
    """Simplified Megatron checkpoint manager for actor/critic models."""

    def __init__(self, model, optimizer, lr_scheduler=None):
        """
        Args:
            model: Megatron model or ModuleList
            optimizer: Megatron optimizer
            lr_scheduler: Learning rate scheduler (optional)
        """
        self.model = model if isinstance(model, list) else [model]
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.rank = dist.get_rank()

    def save_checkpoint(self, local_path, global_step=0, max_ckpt_to_keep=None):
        """Save checkpoint using Megatron distributed checkpointing."""
        dist_checkpoint_path = os.path.join(local_path, f"step_{global_step}")

        # Create directory on all ranks before saving
        os.makedirs(dist_checkpoint_path, exist_ok=True)
        dist.barrier()

        state_dict = self._generate_state_dict()

        save_dist_checkpointing(
            sharded_state_dict=state_dict,
            ckpt_path=dist_checkpoint_path,
            async_save=False
        )

        dist.barrier()

        if self.rank == 0:
            logger.info(f"Saved checkpoint to {dist_checkpoint_path}")

            if max_ckpt_to_keep and max_ckpt_to_keep > 0:
                self._cleanup_old_checkpoints(local_path, max_ckpt_to_keep)

    def load_checkpoint(self, local_path):
        """Load checkpoint using Megatron distributed checkpointing."""
        import glob

        step_dirs = glob.glob(os.path.join(local_path, "step_*"))
        if not step_dirs:
            logger.warning(f"No checkpoint found at {local_path}")
            return

        latest_step_dir = max(step_dirs, key=lambda x: int(x.split('step_')[-1]))

        state_dict = self._generate_state_dict()

        load_dist_checkpointing(
            sharded_state_dict=state_dict,
            ckpt_dir=latest_step_dir
        )

        self._load_state_dict(state_dict)

        dist.barrier()
        logger.info(f"Loaded checkpoint from {latest_step_dir}")

    def _generate_state_dict(self):
        """Generate sharded state dict for save/load."""
        state_dict = {}

        for vpp_rank, model in enumerate(self.model):
            if len(self.model) > 1:
                mpu.set_virtual_pipeline_model_parallel_rank(vpp_rank)
                key = f"model{vpp_rank}"
            else:
                key = "model"

            if hasattr(model, "module"):
                model = model.module

            state_dict[key] = model.sharded_state_dict()

        dist.barrier()
        optimizer_sharded_states = self.optimizer.sharded_state_dict(state_dict)
        state_dict["optimizer"] = optimizer_sharded_states

        if self.lr_scheduler is not None:
            lr_state_dict = self.lr_scheduler.state_dict()
            state_dict["lr_scheduler"] = lr_state_dict

        return state_dict

    def _load_state_dict(self, state_dict):
        """Load state dict into model and optimizer."""
        # Optimizer state is loaded automatically by load_dist_checkpointing
        # We only need to handle lr_scheduler
        if self.lr_scheduler is not None and "lr_scheduler" in state_dict:
            self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])

    def _cleanup_old_checkpoints(self, local_path, max_ckpt_to_keep):
        """Remove old checkpoints to save disk space."""
        import glob
        import shutil

        step_dirs = sorted(
            glob.glob(os.path.join(local_path, "step_*")),
            key=lambda x: int(x.split('step_')[-1])
        )

        if len(step_dirs) > max_ckpt_to_keep:
            for old_dir in step_dirs[:-max_ckpt_to_keep]:
                shutil.rmtree(old_dir)
                logger.info(f"Removed old checkpoint {old_dir}")
