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

import ray
import os
from loguru import logger
from typing import List, Optional
import torch.distributed as dist
import os
from siirl.utils.enums import DistributedEnv

from siirl.params.training_args import SiiRLArguments
from siirl.utils.enums import DistributedEnv
from siirl.engine.actor.megatron_actor import ActorWorker, ReferenceWorker, CriticWorker
from siirl.algorithm.advantage import compute_advantage
from siirl.engine.actor.utils import get_master_info


class Trainer:
    """
    Manages a single training unit consisting of:
    - One Actor model
    - One Reference model
    - One Critic model (optional, only for PPO)

    Each Trainer instance corresponds to one GPU and handles the training logic
    for its set of models. The Trainer itself will be wrapped as a Ray Actor by TrainerGroup.
    """

    def __init__(
        self,
        config,
        rank: int,
        local_rank: int,
        world_size: int,
        master_addr: str,
        master_port_actor: str,
        master_port_critic: str,
        master_port_ref: str,
        use_critic: bool = False,
    ):
        """
        Initialize a Trainer and create its models (actor, ref, optionally critic).

        Args:
            config: Training configuration
            rank: Global rank for this trainer
            local_rank: Local rank on the node
            world_size: Total number of trainers
            master_addr: Master address for distributed training
            master_port_actor: Port for actor process group
            master_port_critic: Port for critic process group (if used)
            master_port_ref: Port for ref process group
            use_critic: Whether to create a critic model (PPO vs GRPO)
        """

        self.config = config
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.use_critic = use_critic

        # Store distributed training info
        self.master_addr = master_addr
        self.master_port_actor = master_port_actor
        self.master_port_critic = master_port_critic
        self.master_port_ref = master_port_ref

        # Initialize models (will be created in init_models method)
        self.actor_worker = None
        self.ref_worker = None
        self.critic_worker = None

    def init_models(self):
        """
        Initialize actor, ref, and optionally critic workers.
        This method creates the actual worker instances and initializes their models.
        """

        # Create Actor worker with actor port
        logger.info(f"Trainer[{self.rank}]: Creating ActorWorker...")
        os.environ[DistributedEnv.MASTER_PORT.value] = self.master_port_actor
        self.actor_worker = ActorWorker(config=self.config)

        # Create Reference worker with ref port
        logger.info(f"Trainer[{self.rank}]: Creating ReferenceWorker...")
        os.environ[DistributedEnv.MASTER_PORT.value] = self.master_port_ref
        self.ref_worker = ReferenceWorker(config=self.config)

        # Create Critic worker (only for PPO) with critic port
        if self.use_critic:
            logger.info(f"Trainer[{self.rank}]: Creating CriticWorker...")
            os.environ[DistributedEnv.MASTER_PORT.value] = self.master_port_critic
            self.critic_worker = CriticWorker(config=self.config)

        # Initialize models for all workers
        logger.info(f"Trainer[{self.rank}]: Initializing models...")
        os.environ[DistributedEnv.MASTER_PORT.value] = self.master_port_actor
        self.actor_worker.init_model()

        os.environ[DistributedEnv.MASTER_PORT.value] = self.master_port_ref
        self.ref_worker.init_model()

        if self.use_critic:
            os.environ[DistributedEnv.MASTER_PORT.value] = self.master_port_critic
            self.critic_worker.init_model()

        logger.success(f"Trainer[{self.rank}]: All models initialized successfully")

    def has_critic(self):
        """Check if this trainer has a critic worker."""
        return self.critic_worker is not None

    def compute_log_prob(self, data):
        """Compute log probabilities using the actor."""
        return self.actor_worker.compute_log_prob(data)

    def compute_ref_log_prob(self, data):
        """Compute reference log probabilities."""
        return self.ref_worker.compute_ref_log_prob(data)

    def compute_values(self, data):
        """Compute values using the critic (PPO only)."""
        if not self.has_critic():
            raise RuntimeError("Critic not available for this trainer")
        return self.critic_worker.compute_values(data)

    def update_actor(self, data):
        """Update the actor policy."""
        return self.actor_worker.update_actor(data)

    def update_critic(self, data):
        """Update the critic (PPO only)."""
        if not self.has_critic():
            raise RuntimeError("Critic not available for this trainer")
        return self.critic_worker.update_critic(data)


class TrainerGroup:
    """
    Manages a group of Trainers for distributed RL training.
    Each Trainer manages one actor, one ref, and optionally one critic (for PPO).
    Supports multiple algorithms (PPO, GRPO) and training backends (megatron, fsdp, etc.)

    Model requirements by algorithm:
    - PPO: actor + critic + ref (per Trainer)
    - GRPO: actor + ref (per Trainer)
    """

    def __init__(
        self,
        config: SiiRLArguments,
        data_coordinator,
        num_gpus: int,
        placement_groups: Optional[List] = None,
    ) -> None:
        """
        Initialize TrainerGroup with configuration and resource handles.

        Args:
            config: Training configuration
            data_coordinator: Ray handle to DataCoordinator
            num_gpus: Number of GPUs to use (creates one Trainer per GPU)
            placement_groups: Ray placement groups for resource allocation
        """
        self.config = config
        self.data_coordinator = data_coordinator
        self.num_gpus = num_gpus
        self.placement_groups = placement_groups

        self.trainers: List[Trainer] = []

        self.use_critic = self.config.actor_ref.algo.adv_estimator == "ppo"

        self.master_addr, base_port = get_master_info()
        base_port_int = int(base_port)
        #TODO: add roubust port access
        self.master_ports = {
            "actor": str(base_port_int),
            "critic": str(base_port_int + 1),
            "ref": str(base_port_int + 2),
        }

    def init_actors(self):
        """
        Initialize trainers by creating Trainer instances and wrapping them as Ray Actors.
        Each Trainer manages its own actor, ref, and optionally critic models.
        """
        n_gpus_per_node = self.config.trainer.n_gpus_per_node

        # Create Trainer Ray Actors
        for rank in range(self.num_gpus):
            node_idx = rank // n_gpus_per_node
            local_rank = rank % n_gpus_per_node
            pg = self.placement_groups[node_idx] if self.placement_groups else None
            bundle_index = local_rank

            # Set up environment variables for this trainer
            env_vars = {
                DistributedEnv.WORLD_SIZE.value: str(self.num_gpus),
                DistributedEnv.RANK.value: str(rank),
                DistributedEnv.LOCAL_RANK.value: str(local_rank),
                DistributedEnv.MASTER_ADDR.value: self.master_addr,
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            }

            if os.getenv('GLOO_SOCKET_IFNAME'):
                env_vars['GLOO_SOCKET_IFNAME'] = os.getenv('GLOO_SOCKET_IFNAME')

            # Create Trainer as Ray Actor
            TrainerActor = ray.remote(Trainer)

            trainer_options = {
                "runtime_env": {"env_vars": env_vars},
                "name": f"trainer_{rank}",
                "num_gpus": 1,
            }

            if pg:
                trainer_handle = TrainerActor.options(
                    **trainer_options,
                    placement_group=pg,
                    placement_group_bundle_index=bundle_index,
                ).remote(
                    config=self.config.actor_ref,
                    rank=rank,
                    local_rank=local_rank,
                    world_size=self.num_gpus,
                    master_addr=self.master_addr,
                    master_port_actor=self.master_ports["actor"],
                    master_port_critic=self.master_ports["critic"],
                    master_port_ref=self.master_ports["ref"],
                    use_critic=self.use_critic,
                )
            else:
                trainer_handle = TrainerActor.options(**trainer_options).remote(
                    config=self.config.actor_ref,
                    rank=rank,
                    local_rank=local_rank,
                    world_size=self.num_gpus,
                    master_addr=self.master_addr,
                    master_port_actor=self.master_ports["actor"],
                    master_port_critic=self.master_ports["critic"],
                    master_port_ref=self.master_ports["ref"],
                    use_critic=self.use_critic,
                )

            self.trainers.append(trainer_handle)

        # Wait for all Trainer __init__ to complete
        ray.get([trainer.__ray_ready__.remote() for trainer in self.trainers])
        logger.success(f"All {len(self.trainers)} Trainer Ray Actors __init__ completed")

        # Initialize models on all trainers
        futures = [trainer.init_models.remote() for trainer in self.trainers]
        ray.get(futures)

        logger.success(f"Successfully initialized {len(self.trainers)} trainers with their models")

    def get_batch(self, batch_size: int, dp_rank: int = 0):
        """
        Fetch a batch of training data from DataCoordinator.

        Args:
            batch_size: Number of samples to fetch
            dp_rank: Data parallel rank

        Returns:
            Batch data references from DataCoordinator
        """
        logger.debug(f"Fetching batch of size {batch_size} from DataCoordinator")
        batch_refs = ray.get(
            self.data_coordinator.get_batch.remote(
                batch_size=batch_size,
                dp_rank=dp_rank,
                balance_partitions=self.num_gpus,
            )
        )
        return batch_refs

    def train(self, num_epochs: int = 1, use_critic: Optional[bool] = None):
        """
        Execute training loop for both PPO and GRPO algorithms.

        Args:
            num_epochs: Number of training epochs
            use_critic: Whether to use critic model. If None, uses self.use_critic
                       (True for PPO, False for GRPO)
        """
        if use_critic is None:
            use_critic = self.use_critic


        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            batch_size = self.config.actor_ref.actor.ppo_mini_batch_size
            batch_refs = self.get_batch(batch_size, dp_rank=0)

            if not batch_refs:
                logger.warning("No data available, skipping training step")
                continue

            batch_data = ray.get(batch_refs)

            metrics = self._train_step(batch_data, use_critic=use_critic)

            logger.info(f"Epoch {epoch + 1} metrics: {metrics}")

        logger.success(f"Training completed for {num_epochs} epochs")

    def _train_step(self, batch_data, use_critic: bool = True):
        """
        Execute a single training step for PPO or GRPO using Trainer abstraction.

        Args:
            batch_data: Training batch data
            use_critic: Whether to use critic model (True for PPO, False for GRPO)

        Returns:
            Aggregated metrics
        """

        logger.debug("Computing actor log probabilities")
        actor_futures = []
        for i, trainer in enumerate(self.trainers):
            actor_data = batch_data[i] if isinstance(batch_data, list) else batch_data
            actor_futures.append(trainer.compute_log_prob.remote(actor_data))
        actor_data_with_logprobs = ray.get(actor_futures)

        logger.debug("Computing reference log probabilities")
        ref_futures = []
        for i, trainer in enumerate(self.trainers):
            ref_data = actor_data_with_logprobs[i] if isinstance(actor_data_with_logprobs, list) else actor_data_with_logprobs
            ref_futures.append(trainer.compute_ref_log_prob.remote(ref_data))
        data_with_ref = ray.get(ref_futures)

        if use_critic:
            logger.debug("Computing critic values")
            critic_futures = []
            for i, trainer in enumerate(self.trainers):
                critic_data = data_with_ref[i] if isinstance(data_with_ref, list) else data_with_ref
                critic_futures.append(trainer.compute_values.remote(critic_data))
            data_with_values = ray.get(critic_futures)
        else:
            data_with_values = data_with_ref

        logger.debug("Computing advantages")
        adv_estimator = self.config.actor_ref.algo.adv_estimator
        gamma = self.config.algorithm.gamma
        lam = self.config.algorithm.lam

        data_for_update = []
        for i, data in enumerate(data_with_values if isinstance(data_with_values, list) else [data_with_values]):
            data_with_adv = compute_advantage(
                data=data,
                adv_estimator=adv_estimator,
                gamma=gamma,
                lam=lam,
            )
            data_for_update.append(data_with_adv)

        if not isinstance(data_with_values, list):
            data_for_update = data_for_update[0]

        logger.debug("Updating actor policy")
        actor_update_futures = []
        for i, trainer in enumerate(self.trainers):
            update_data = data_for_update[i] if isinstance(data_for_update, list) else data_for_update
            actor_update_futures.append(trainer.update_actor.remote(update_data))
        actor_results = ray.get(actor_update_futures)

        if use_critic:
            logger.debug("Updating critic")
            critic_update_futures = []
            for i, trainer in enumerate(self.trainers):
                update_data = data_for_update[i] if isinstance(data_for_update, list) else data_for_update
                critic_update_futures.append(trainer.update_critic.remote(update_data))
            critic_results = ray.get(critic_update_futures)
            all_results = actor_results + critic_results
        else:
            all_results = actor_results

        metrics = self._aggregate_metrics(all_results)

        return metrics

    def _aggregate_metrics(self, results: List):
        """
        Aggregate training metrics from all trainers.

        Args:
            results: List of result dicts from trainers

        Returns:
            Aggregated metrics dict
        """
        if not results:
            return {}

        aggregated = {}
        for result in results:
            if "metrics" in result:
                metrics = result["metrics"]
                for key, value in metrics.items():
                    if key not in aggregated:
                        aggregated[key] = []
                    aggregated[key].append(value)

        for key in aggregated:
            if isinstance(aggregated[key], list) and aggregated[key]:
                first_elem = aggregated[key][0]
                try:
                    if isinstance(first_elem, (int, float)):
                        aggregated[key] = sum(aggregated[key]) / len(aggregated[key])
                    elif hasattr(first_elem, 'item'):
                        values = [v.item() if hasattr(v, 'item') else v for v in aggregated[key]]
                        aggregated[key] = sum(values) / len(values)
                    else:
                        aggregated[key] = aggregated[key][0]
                except (TypeError, AttributeError):
                    aggregated[key] = aggregated[key][0]

        return aggregated

    def put_weight(self):
        """
        Extract trained model parameters from actor and update to RolloutManager.
        Supports model weight synchronization for rollout/inference.

        Note: Weight extraction strategy depends on specific backend implementation.
        For megatron backend, this typically involves extracting state_dict and
        synchronizing across workers.
        """
        logger.info("Extracting model weights from actor workers")

        if not self.trainers:
            logger.warning("No trainers available for weight extraction")
            return

        # Extract weights from the first trainer's actor (assuming all actors have synchronized weights)
        # In data parallel training, all actors should have identical weights after training
        logger.info("Preparing to update weights to RolloutManager")

        # Placeholder for actual weight extraction and update logic
        # Actual implementation would involve:
        # 1. Extract state_dict from actor_module on first trainer's actor
        # 2. Save weights to shared storage or serialize
        # 3. Notify or directly update RolloutManager workers
        # 4. RolloutManager loads new weights into inference engines

        # Example pseudo-code:
        # weights = ray.get(self.trainers[0].actor_handle.get_model_weights.remote())
        # ray.get(self.rollout_manager.update_weights.remote(weights))

        logger.warning(
            "Weight update functionality is a placeholder. "
            "Implement based on specific requirements and backend."
        )

