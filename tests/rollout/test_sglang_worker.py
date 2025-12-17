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
import time
import ray


from siirl.params import SiiRLArguments, log_dict_formatted, parse_config
from siirl.utils.logger.logging_utils import set_basic_config
from siirl.data_coordinator.data_buffer import init_data_coordinator
from siirl.worker.ray_utils import create_placement_groups
from siirl.worker.rollout.rollout_manager import RolloutManager
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

    def run(self, siirl_args: SiiRLArguments) -> None:
        """
        Executes the main training workflow.

        Args:
            siirl_args: A SiiRLArguments object containing all parsed configurations.
        """
        set_basic_config()
        from loguru import logger

        logger.info("MainRunner started. Beginning workflow setup...")
        start_time = time.time()

        # 1. Init DataBuffer
        logger.info(f"Initializing DataCoordinator with {siirl_args.trainer.nnodes} distributed DataBuffers...")
        # In the new architecture, the number of buffers is typically the number of nodes.
        # We pass force_local=False to enable distributed deployment.
        data_coordinator_handle = init_data_coordinator(
            num_buffers=siirl_args.trainer.nnodes, ppo_mini_batch_size = siirl_args.actor_rollout_ref.actor.ppo_mini_batch_size,
            world_size=siirl_args.trainer.nnodes * siirl_args.trainer.n_gpus_per_node
        )
        dataloader_fut = data_coordinator_handle.init_dataloader.remote(siirl_args)
        # 2. initialize pg
        pgs = create_placement_groups(siirl_args)
        print(f"[pgs] {pgs}")
        # 3. Initialize rollout worker
        rollout_pgs = pgs
        rollout_worker = RolloutManager(siirl_args, rollout_pgs, data_coordinator_handle)
        ray.get(dataloader_fut)
        total_training_steps, num_train_batches = ray.get(data_coordinator_handle.epoch_info.remote())
        global_steps = 0
        
        if num_train_batches > 0:
            start_epoch = global_steps // num_train_batches
            batches_to_skip = global_steps % num_train_batches
        for epoch in range(start_epoch, siirl_args.trainer.total_epochs):
            for batch_idx in range(num_train_batches):
                for _ in range(siirl_args.trainer.async_factor):
                    ray.get(data_coordinator_handle.run_dataloader.remote(epoch))
                batch_idx += siirl_args.trainer.async_factor
                time.sleep(10)
        # 4. Initialize Actor worker
        
        
        
        # 5. start rollout and actor worker




def main() -> None:
    """
    Main entry point for launching the PPO DAG training job.

    This function initializes Ray, parses configurations using Hydra, and
    starts the MainRunner actor to orchestrate the distributed training workflow.

    Args:
        siirl_config: The configuration object provided by Hydra.
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
    siirl_args.actor_rollout_ref.model.path = '/inspire/hdd/project/qianghuaxuexi/public/models/Qwen3-1.7B'
    siirl_args.rollout.tensor_model_parallel_size = 2
    siirl_args.data.train_files = ['/inspire/hdd/project/qianghuaxuexi/public/datasets/deepscaler/train.parquet']
    siirl_args.data.val_files = ['/inspire/hdd/project/qianghuaxuexi/public/datasets/deepscaler/test.parquet']
    siirl_args.data.max_prompt_length=2048
    siirl_args.data.max_response_length=4096
    siirl_args.data.filter_overlong_prompts=True
    siirl_args.rollout.n=8
    # siirl_args.data.train_batch_size = siirl_args.data.train_batch_size // 2
    log_dict_formatted(siirl_args.to_dict(), "SiiRLArguments")

    # Launch the main orchestration actor and wait for it to complete.
    logger.info("Starting MainRunner actor to orchestrate the job.")
    runner = MainRunner.remote()
    # This is a blocking call that waits for the remote `run` method to finish.
    ray.get(runner.run.remote(siirl_args))

    logger.success("MainRunner has completed its execution. Shutting down.")


if __name__ == "__main__":
    main()