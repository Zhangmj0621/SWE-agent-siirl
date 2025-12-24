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


import time
import ray

from siirl.params import SiiRLArguments, log_dict_formatted, parse_config
from siirl.utils.logger.logging_utils import set_basic_config
from siirl.data_coordinator.data_buffer import init_data_coordinator
from siirl.worker.ray_utils import allocate_resources
from siirl.worker.rollout.rollout_manager import RolloutManager
from siirl.worker.actor.trainer_group import TrainerGroup


# --- Constants ---
RAY_RUNTIME_ENV_VARS = {
    "TOKENIZERS_PARALLELISM": "true",
    "NCCL_DEBUG": "WARN",
    "VLLM_LOGGING_LEVEL": "WARN",
}

MAIN_RUNNER_CPU_RESERVATION = 5


@ray.remote(num_cpus=MAIN_RUNNER_CPU_RESERVATION)
class MainRunner:
    """
    A Ray actor responsible for orchestrating the entire RL training workflow.

    This actor handles loading configurations, scheduling task graphs, initializing
    process groups, and launching the distributed Ray trainers. Isolating this
    orchestration logic in a dedicated actor ensures the main process remains clean
    and that the setup process is managed within the Ray cluster.
    """

    def run(self, config: SiiRLArguments) -> None:
        """
        Executes the main training workflow.

        Args:
            config: A SiiRLArguments object containing all parsed configurations.
        """
        # NOTE: Do not call set_basic_config() before creating Ray actors!
        # The file logger creates file handles that cannot be serialized by Ray.
        # We use basic loguru logging first, then configure file logging after actors are created.
        from loguru import logger

        logger.info("MainRunner started. Beginning workflow setup...")
        start_time = time.time()

        # === 1. Allocate GPU Resources ===
        logger.info("Allocating GPU resources...")
        resources = allocate_resources(config)
        is_colocated = config.trainer.colocate

        # === 2. Initialize DataCoordinator ===
        # Determine number of actor GPUs based on allocation mode
        num_actor_gpus = resources["shared"].num_gpus if is_colocated else resources["actor"].num_gpus
        
        logger.info(f"Initializing DataCoordinator with {config.trainer.nnodes} distributed DataBuffers...")
        data_coordinator = init_data_coordinator(
            num_buffers=config.trainer.nnodes,
            ppo_mini_batch_size=config.actor_ref.actor.ppo_mini_batch_size,
            world_size=num_actor_gpus
        )
        
        # Initialize dataloader in DataCoordinator
        ray.get(data_coordinator.init_dataloader.remote(config))
        
        # Get training info from DataCoordinator and update config
        # NOTE: Ray actors modify their local copy of config, so we must fetch the calculated values
        total_training_steps, batches_per_epoch = ray.get(data_coordinator.epoch_info.remote())
        config.actor_ref.actor.optim.total_training_steps = total_training_steps
        config.critic.optim.total_training_steps = total_training_steps
        logger.success(f"DataCoordinator initialized: {batches_per_epoch} batches/epoch, {total_training_steps} total steps")

        # === 3. Initialize Components (RolloutManager & TrainerGroup) ===
        if is_colocated:
            # Colocated mode: training and rollout share the same GPUs
            shared_resources = resources["shared"]
            logger.info(f"Initializing components in colocated mode with {shared_resources.num_gpus} shared GPUs...")
            
            rollout_manager = RolloutManager.remote(config, shared_resources, data_coordinator)
            trainer_group = TrainerGroup(config, shared_resources, data_coordinator, rollout_manager)
        else:
            # Separated mode: training and rollout use different GPUs
            actor_resources = resources["actor"]
            rollout_resources = resources["rollout"]
            logger.info(f"Initializing components in separated mode: {actor_resources.num_gpus} training GPUs, {rollout_resources.num_gpus} rollout GPUs...")
            
            rollout_manager = RolloutManager.remote(config, rollout_resources, data_coordinator)
            trainer_group = TrainerGroup(config, actor_resources, data_coordinator, rollout_manager)

        # Initialize trainer actors (creates Trainer Ray actors with models)
        trainer_group.init_actors()
        
        # Now that all Ray actors are created, we can safely configure file logging
        # (file handles won't be serialized anymore)
        set_basic_config()
        
        router_address = ray.get(rollout_manager.get_router_address.remote()) if rollout_manager else "N/A"
        logger.success(f"RolloutManager initialized. Router at: {router_address}")
        logger.success(f"TrainerGroup initialized with {len(trainer_group.trainers)} trainers")
        
        init_time = time.time() - start_time
        logger.info(f"Initialization completed in {init_time:.1f}s")

        # === 4. Async Training Loop ===
        logger.info("Starting async training loop...")
        rollout_manager.run_dataloader.remote()
        ray.get(rollout_manager.next_rollout.remote())
        total_epochs = config.trainer.total_epochs
        trainer_group.train(num_epochs=total_epochs)
        
        # TODO: Add the following functionality
        # - Weight synchronization: trainer_group.put_weight() -> rollout_manager
        # - Model saving: trainer_group.save_model()
        # - Evaluation: rollout_manager.eval()
        
        total_time = time.time() - start_time
        logger.success(f"Training completed! Total time: {total_time:.1f}s")


def main() -> None:
    """
    Main entry point for launching the PPO DAG training job.

    This function initializes Ray, parses configurations using Hydra, and
    starts the MainRunner actor to orchestrate the distributed training workflow.
    """
    # Import logger locally to avoid Ray serialization issues
    from loguru import logger

    start_time = time.time()

    # Initialize Ray cluster if not already running
    if not ray.is_initialized():
        logger.info("Initializing local Ray cluster...")
        ray.init(runtime_env={"env_vars": RAY_RUNTIME_ENV_VARS}, num_cpus=None)
    logger.success(f"Ray is initialized. Time cost: {(time.time() - start_time) * 1000:.2f} ms")

    # Parse the complete configuration into a structured object
    siirl_args = parse_config()
    log_dict_formatted(siirl_args.to_dict(), "SiiRLArguments")

    # Launch the main orchestration actor and wait for it to complete.
    logger.info("Starting MainRunner actor to orchestrate the job.")
    runner = MainRunner.remote()
    # This is a blocking call that waits for the remote `run` method to finish.
    ray.get(runner.run.remote(siirl_args))

    logger.success("MainRunner has completed its execution. Shutting down.")


if __name__ == "__main__":
    main()
