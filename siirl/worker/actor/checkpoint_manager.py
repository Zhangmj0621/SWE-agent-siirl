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

import glob
import os
import shutil

import torch
import torch.distributed as dist
from loguru import logger

from siirl.params import SiiRLArguments
from siirl.utils.checkpoint.checkpoint_utils import find_latest_ckpt_path


class CheckpointManager:
    """Manages distributed checkpoint save/load."""

    def __init__(
        self,
        config: SiiRLArguments,
        rank: int,
        world_size: int,
        actor_worker,
        ref_worker,
        critic_worker,
        data_coordinator,
        dp_rank: int,
        dp_world_size: int,
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.actor_worker = actor_worker
        self.ref_worker = ref_worker
        self.critic_worker = critic_worker
        self.data_coordinator = data_coordinator
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size

    def save_checkpoint(self, global_steps: int) -> None:
        """
        Save checkpoint atomically across all ranks.

        This method saves model states (actor/critic), dataloader state,
        and commits the checkpoint by writing a tracker file. After committing,
        it cleans up old global_step_* directories based on max_ckpt_to_keep.

        Args:
            global_steps: Current global training step number.
        """
        step_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{global_steps}")
        os.makedirs(step_dir, exist_ok=True)
        dist.barrier()

        logger.info(f"Rank {self.rank}: Saving checkpoint for global_step {global_steps}")

        self._save_model_states(global_steps, step_dir)
        self._save_dataloader_state(step_dir)

        dist.barrier()

        if self.rank == 0:
            self._commit_checkpoint(global_steps)
            # Clean up old global_step_* directories after successful commit
            self._cleanup_old_global_steps()

        dist.barrier()
        logger.info(f"Rank {self.rank}: Checkpoint saved for step {global_steps}")

    def _save_model_states(self, global_steps: int, step_dir: str) -> None:
        """Save actor and critic model states."""
        actor_path = os.path.join(step_dir, "actor")
        max_actor_ckpt = self.config.trainer.max_actor_ckpt_to_keep

        self.actor_worker.save_checkpoint(
            local_path=actor_path,
            global_step=global_steps,
            max_ckpt_to_keep=max_actor_ckpt,
        )
        logger.debug(f"Rank {self.rank}: Saved actor checkpoint to {actor_path}")

        if self.critic_worker is not None:
            critic_path = os.path.join(step_dir, "critic")
            max_critic_ckpt = self.config.trainer.max_critic_ckpt_to_keep

            self.critic_worker.save_checkpoint(
                local_path=critic_path,
                global_step=global_steps,
                max_ckpt_to_keep=max_critic_ckpt,
            )
            logger.debug(f"Rank {self.rank}: Saved critic checkpoint to {critic_path}")

    def _save_dataloader_state(self, step_dir: str) -> None:
        """Save dataloader state (only rank 0)."""
        if self.rank != 0:
            return

        if self.data_coordinator is None:
            return

        import ray

        dataloader_state = ray.get(self.data_coordinator.save_dataloader_state.remote())
        if dataloader_state is not None:
            dataloader_path = os.path.join(step_dir, "dataloader_state.pt")
            torch.save(dataloader_state, dataloader_path)
            logger.debug(f"Rank {self.rank}: Saved dataloader state to {dataloader_path}")

    def _commit_checkpoint(self, global_steps: int) -> None:
        """Atomically commit checkpoint by writing tracker file."""
        tracker_file = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(tracker_file, "w") as f:
            f.write(str(global_steps))
        logger.info(f"Rank 0: Checkpoint {global_steps} committed")

    def _cleanup_old_global_steps(self) -> None:
        """
        Remove old global_step_* directories based on max_ckpt_to_keep config.

        This method is called after checkpoint commit on rank 0 only.
        It uses the minimum of max_actor_ckpt_to_keep and max_critic_ckpt_to_keep
        to determine how many global_step_* directories to retain.

        The cleanup happens at the global_step_* level, meaning entire checkpoint
        directories (including actor/, critic/, and dataloader_state.pt) are removed.
        """
        checkpoint_dir = self.config.trainer.default_local_dir

        # Use the minimum of actor and critic limits for global directory cleanup
        max_actor_keep = self.config.trainer.max_actor_ckpt_to_keep
        max_critic_keep = self.config.trainer.max_critic_ckpt_to_keep

        # If critic doesn't exist, just use actor limit
        max_keep = max_actor_keep if self.critic_worker is None else min(max_actor_keep, max_critic_keep)

        # Skip cleanup if max_keep is not set or <= 0
        if max_keep <= 0:
            return

        # Find all global_step_* directories
        global_step_pattern = os.path.join(checkpoint_dir, "global_step_*")
        global_step_dirs = glob.glob(global_step_pattern)

        if not global_step_dirs:
            return

        # Sort by step number (ascending)
        global_step_dirs = sorted(
            global_step_dirs,
            key=lambda x: int(os.path.basename(x).split("global_step_")[-1]),
        )

        # Remove oldest directories if exceeding the limit
        if len(global_step_dirs) > max_keep:
            dirs_to_remove = global_step_dirs[:-max_keep]
            for old_dir in dirs_to_remove:
                try:
                    shutil.rmtree(old_dir, ignore_errors=True)
                    logger.info(f"Rank 0: Removed old checkpoint directory: {old_dir}")
                except Exception as e:
                    logger.warning(f"Rank 0: Failed to remove checkpoint {old_dir}: {e}")

    def load_checkpoint(self) -> int:
        """Load checkpoint and return global step to resume from."""
        if self.config.trainer.resume_mode == "disable":
            if self.rank == 0:
                logger.info("Checkpoint loading disabled, starting from scratch")
            return 0

        checkpoint_path = self._determine_checkpoint_path()

        checkpoint_path_container = [checkpoint_path]
        dist.broadcast_object_list(checkpoint_path_container, src=0)
        global_step_folder = checkpoint_path_container[0]

        if global_step_folder is None:
            if self.rank == 0:
                logger.info("No checkpoint found, starting from step 0")
            dist.barrier()
            return 0

        global_steps = int(os.path.basename(global_step_folder).split("global_step_")[-1])
        logger.info(f"Rank {self.rank}: Resuming from checkpoint step {global_steps}")

        self._load_model_states(global_step_folder)
        self._load_dataloader_state(global_step_folder)

        dist.barrier()
        logger.info(f"Rank {self.rank}: Checkpoint loaded")

        return global_steps

    def _determine_checkpoint_path(self) -> str | None:
        """Determine checkpoint path (rank 0 only)."""
        if self.rank != 0:
            return None

        checkpoint_dir = self.config.trainer.default_local_dir
        resume_from_path = self.config.trainer.resume_from_path
        path_to_load = None

        if self.config.trainer.resume_mode == "auto":
            latest_path = find_latest_ckpt_path(checkpoint_dir)
            if latest_path:
                logger.info(f"Rank 0: Found latest checkpoint at {latest_path}")
                path_to_load = latest_path
        elif self.config.trainer.resume_mode == "resume_path" and resume_from_path:
            logger.info(f"Rank 0: Loading from {resume_from_path}")
            path_to_load = resume_from_path

        if path_to_load and os.path.exists(path_to_load):
            return path_to_load
        else:
            logger.warning(f"Rank 0: Checkpoint path not found: '{path_to_load}'")
            return None

    def _load_model_states(self, global_step_folder: str) -> None:
        """Load actor and critic model states."""
        actor_path = os.path.join(global_step_folder, "actor")

        if os.path.exists(actor_path):
            self.actor_worker.load_checkpoint(local_path=actor_path)
            logger.debug(f"Rank {self.rank}: Loaded actor checkpoint from {actor_path}")
        else:
            logger.warning(f"Rank {self.rank}: Actor checkpoint not found at {actor_path}")

        if self.critic_worker is not None:
            critic_path = os.path.join(global_step_folder, "critic")

            if os.path.exists(critic_path):
                self.critic_worker.load_checkpoint(local_path=critic_path)
                logger.debug(f"Rank {self.rank}: Loaded critic checkpoint from {critic_path}")
            else:
                logger.warning(f"Rank {self.rank}: Critic checkpoint not found at {critic_path}")

    def _load_dataloader_state(self, global_step_folder: str) -> None:
        """Load dataloader state (only rank 0)."""
        if self.rank != 0:
            return

        if self.data_coordinator is None:
            return

        import ray

        dataloader_path = os.path.join(global_step_folder, "dataloader_state.pt")

        if os.path.exists(dataloader_path):
            dataloader_state = torch.load(dataloader_path, map_location="cpu")
            ray.get(self.data_coordinator.load_dataloader_state.remote(dataloader_state))
            logger.debug(f"Rank {self.rank}: Loaded dataloader state from {dataloader_path}")
        else:
            logger.warning(f"Rank {self.rank}: Dataloader checkpoint not found at {dataloader_path}")
