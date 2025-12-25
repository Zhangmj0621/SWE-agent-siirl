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


import sys
import time
import traceback
import ray

from siirl.params import SiiRLArguments, log_dict_formatted, parse_config
from siirl.utils.task_coordinator import create_coordinator
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


def _worker_logging_setup():
    """
    Called automatically in each Ray worker process at startup.
    
    This is the most elegant way to configure logging across all Ray actors:
    - Only configured once in ray.init()
    - Automatically applies to all worker processes (RolloutManager, Trainer, RolloutWorker, etc.)
    - No need to manually call set_basic_config() in each actor's __init__
    
    See: https://docs.ray.io/en/latest/ray-observability/user-guides/configure-logging.html
    """
    from siirl.utils.logger.logging_utils import set_basic_config
    set_basic_config()


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
        # NOTE: Logging is automatically configured when siirl is imported (see siirl/__init__.py)
        # All Ray actors inherit this configuration as they import siirl modules.
        from loguru import logger

        logger.info("MainRunner started. Beginning workflow setup...")
        start_time = time.time()

        # === 0. Create Task Coordinator ===
        # Coordinator manages task lifecycle: graceful shutdown, failure propagation
        coordinator = create_coordinator()
        logger.info("TaskCoordinator created for lifecycle management")

        # === 1. Allocate GPU Resources (Separated Mode) ===
        logger.info("Allocating GPU resources...")
        resources = allocate_resources(config)
        actor_resources = resources["actor"]
        rollout_resources = resources["rollout"]

        # === 2. Initialize DataCoordinator ===
        logger.info(f"Initializing DataCoordinator with {config.trainer.nnodes} distributed DataBuffers...")
        data_coordinator = init_data_coordinator(
            num_buffers=config.trainer.nnodes,
            ppo_mini_batch_size=config.actor_ref.actor.ppo_mini_batch_size,
            world_size=actor_resources.num_gpus
        )
        
        # Initialize dataloader in DataCoordinator
        ray.get(data_coordinator.init_dataloader.remote(config))
        
        # Get training info from DataCoordinator and update config
        # NOTE: Ray actors modify their local copy of config, so we must fetch the calculated values
        total_training_steps, batches_per_epoch = ray.get(data_coordinator.epoch_info.remote())
        config.actor_ref.actor.optim.total_training_steps = total_training_steps
        config.critic.optim.total_training_steps = total_training_steps
        logger.success(f"DataCoordinator initialized: {batches_per_epoch} batches/epoch, {total_training_steps} total steps")

        # === 3. Initialize MetricWorker ===
        # Note: MetricTracker is created inside Trainer (only rank=0) for cleaner lifecycle management
        from siirl.utils.metrics import MetricWorker
        logger.info("Initializing MetricWorker...")
        metric_worker = MetricWorker.remote()
        ray.get(metric_worker.start.remote())
        logger.success("MetricWorker initialized")

        # === 4. Initialize Components (RolloutManager & TrainerGroup) ===
        rollout_manager = None
        trainer_group = None

        try:
            logger.info(f"Initializing components: {actor_resources.num_gpus} training GPUs, {rollout_resources.num_gpus} rollout GPUs...")
            
            rollout_manager = RolloutManager.remote(config, rollout_resources, data_coordinator, coordinator)
            trainer_group = TrainerGroup(
                config, 
                actor_resources, 
                data_coordinator, 
                rollout_manager, 
                coordinator,
                metric_worker=metric_worker,
            )

            # Initialize trainer actors (creates Trainer Ray actors with models)
            trainer_group.init_actors()

            router_address = ray.get(rollout_manager.get_router_address.remote()) if rollout_manager else "N/A"
            logger.success(f"RolloutManager initialized. Router at: {router_address}")
            logger.success(f"TrainerGroup initialized with {len(trainer_group.trainers)} trainers")

            # Load checkpoint if resume mode is enabled
            if config.trainer.resume_mode != "disable":
                logger.info("Loading checkpoint...")
                trainer_group.load_checkpoint()
                logger.success("Checkpoint loaded successfully")

            init_time = time.time() - start_time
            logger.info(f"Initialization completed in {init_time:.1f}s")

            # === 5. Async Training Loop ===
            logger.info("Starting async training loop...")
            rollout_manager.run_dataloader.remote()
            ray.get(rollout_manager.next_rollout.remote())
            trainer_group.train()

            # === 6. Wait for completion or failure ===
            self._wait_for_completion(coordinator, logger)
            
            # === 7. Check final status and raise if failed ===
            final_status = ray.get(coordinator.get_status.remote())
            if final_status == "failed":
                failure_reason = ray.get(coordinator.get_failure_reason.remote())
                raise RuntimeError(f"Training failed: {failure_reason}")

        except Exception as e:
            logger.error(f"Training failed with exception: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            # Only report if not already reported (avoid duplicate reports)
            current_status = ray.get(coordinator.get_status.remote())
            if current_status == "running":
                ray.get(coordinator.report_failure.remote("main_runner", str(e)))
            raise
        
        finally:
            # === 8. Cleanup and summary ===
            # Note: MetricTracker cleanup is handled inside Trainer (rank=0)
            self._cleanup_and_report(coordinator, trainer_group, rollout_manager, start_time, logger)

    def _wait_for_completion(self, coordinator, logger, check_interval: float = 5.0):
        """
        Wait for training to complete, fail, or shutdown.
        
        Polls coordinator status periodically until task ends.
        """
        logger.info("Monitoring task status...")

        while True:
            status = ray.get(coordinator.get_status.remote())
            if status != "running":
                break
            time.sleep(check_interval)

        # Log final status
        summary = ray.get(coordinator.get_summary.remote())
        logger.info(f"Task ended with status: {summary['status']}")
        if summary['failure_reason']:
            logger.info(f"Reason: {summary['failure_reason']}")

    def _cleanup_and_report(self, coordinator, trainer_group, rollout_manager, start_time, logger):
        """
        Cleanup resources and report final summary.
        """
        total_time = time.time() - start_time

        # Get final summary
        summary = ray.get(coordinator.get_summary.remote())
        events = ray.get(coordinator.get_events.remote())

        logger.info("=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Final Status: {summary['status']}")
        logger.info(f"Total Duration: {total_time:.1f}s")
        if summary['failure_reason']:
            logger.info(f"Reason: {summary['failure_reason']}")
        logger.info(f"Total Events: {summary['event_count']}")

        # Log recent events for debugging
        if events:
            logger.info("Recent Events:")
            for event in events[-5:]:  # Last 5 events
                logger.info(f"  [{event['source']}] {event['event_type']}: {event['message']}")

        logger.info("=" * 60)

        # Determine exit status
        if summary['status'] == "failed":
            logger.error("Training FAILED")
        elif summary['status'] == "completed":
            logger.success("Training COMPLETED successfully")
        else:
            logger.info(f"Training ended with status: {summary['status']}")


def main() -> None:
    """
    Main entry point for launching the PPO DAG training job.

    This function initializes Ray, parses configurations using Hydra, and
    starts the MainRunner actor to orchestrate the distributed training workflow.
    """
    # Import logger locally to avoid Ray serialization issues
    from loguru import logger

    start_time = time.time()
    exit_code = 0

    try:
        # Initialize Ray cluster if not already running
        if not ray.is_initialized():
            logger.info("Initializing local Ray cluster...")
            ray.init(
                runtime_env={
                    "env_vars": RAY_RUNTIME_ENV_VARS,
                    "worker_process_setup_hook": _worker_logging_setup,
                },
                num_cpus=None,
            )
        logger.success(f"Ray is initialized. Time cost: {(time.time() - start_time) * 1000:.2f} ms")

        # Parse the complete configuration into a structured object
        siirl_args = parse_config()
        log_dict_formatted(siirl_args.to_dict(), "SiiRLArguments")

        # Launch the main orchestration actor and wait for it to complete.
        logger.info("Starting MainRunner actor to orchestrate the job.")
        runner = MainRunner.remote()

        # This is a blocking call that waits for the remote `run` method to finish.
        ray.get(runner.run.remote(siirl_args))
        logger.success("MainRunner has completed its execution.")

    except KeyboardInterrupt:
        logger.warning("Received keyboard interrupt, shutting down...")
        exit_code = 130  # Standard exit code for SIGINT

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        exit_code = 1

    finally:
        # Always cleanup Ray
        logger.info("Shutting down Ray cluster...")
        ray.shutdown()
        logger.info("Ray shutdown complete.")

        if exit_code == 0:
            logger.success("Training finished successfully!")
        else:
            logger.error(f"Exiting with code {exit_code}")

        sys.exit(exit_code)


if __name__ == "__main__":
    main()
