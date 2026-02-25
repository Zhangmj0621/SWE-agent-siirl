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

from dataclasses import asdict, dataclass, field
from typing import Any

from .data_args import DataArguments
from .model_args import ActorRefArguments, CriticArguments, RolloutArguments


@dataclass
class CustomRewardArguments:
    """Configuration for custom reward function."""

    path: str | None = field(default=None, metadata={"help": "Path to custom reward function file"})
    name: str = field(
        default="reward_function",
        metadata={"help": "Function name in the custom reward file"},
    )
    reward_kwargs: dict[str, Any] = field(default_factory=dict, metadata={"help": "Keyword arguments for reward function"})


@dataclass
class TrainingArguments:
    total_epochs: int = field(default=30, metadata={"help": "Total training epochs"})
    total_training_steps: int | None = field(default=None, metadata={"help": "Override training steps"})
    project_name: str = field(default="siirl_examples", metadata={"help": "Project name"})
    experiment_name: str = field(default="gsm8k", metadata={"help": "Experiment name"})
    logger: list[str] = field(
        default_factory=lambda: ["console", "wandb"],
        metadata={"help": "Logging backends"},
    )
    log_val_generations: int = field(default=0, metadata={"help": "Validation samples to log"})
    nnodes: int = field(default=1, metadata={"help": "Number of nodes"})
    n_gpus_per_node: int = field(default=8, metadata={"help": "GPUs per node"})
    save_freq: int = field(default=-1, metadata={"help": "Checkpoint frequency"})
    resume_mode: str = field(
        default="auto",
        metadata={"help": "Resume training mode: auto/disable/resume_path"},
    )
    resume_from_path: str | None = field(default=None, metadata={"help": "Resume from specific path"})
    test_freq: int = field(default=-1, metadata={"help": "Testing frequency"})
    critic_warmup: int = field(default=0, metadata={"help": "Critic warmup steps"})
    default_local_dir: str = field(
        default="checkpoints/siirl_examples/gsm8k",
        metadata={"help": "Checkpoint directory"},
    )
    seed: int = field(default=1, metadata={"help": "Train seed param"})
    val_before_train: bool = field(default=True, metadata={"help": "Whether or not to validate before train"})
    val_only: bool = field(default=False, metadata={"help": "Whether or not just eval only"})
    max_actor_ckpt_to_keep: int = field(default=100, metadata={"help": "Maximum number of actor ckpts."})
    max_critic_ckpt_to_keep: int = field(default=100, metadata={"help": "Maximum number of critic ckpts."})
    validation_data_dir: str | None = field(default=None, metadata={"help": "Validation data directory."})
    device: str | None = field(default="cuda", metadata={"help": "Training device."})
    async_factor: int = field(default=1, metadata={"help": "Control async speed"})
    param_sync_buffer_size: int = field(
        default=512 * 1024**2,
        metadata={
            "help": "buffer size for param_sync, in bytes. This is used for updating weights by chunk and should be useful for MoE models."
        },
    )

    # === Off Policy Configuration ===
    off_policy_step: int = field(
        default=0,
        metadata={
            "help": "Number of version steps allowed for off-policy data. "
            "0 means on-policy only (strict current version). "
            "N means accept data from versions [current - N, current]."
        },
    )
    off_policy_strategy: str = field(
        default="fifo",
        metadata={"help": "Strategy for dispatching off-policy data. Options: 'fifo' (default), 'oldest_first'"},
    )

    # === Resource Allocation Configuration ===
    actor_gpus: int = field(default=2, metadata={"help": "Number of GPUs for training (Actor/Ref/Critic)"})
    rollout_gpus: int = field(default=6, metadata={"help": "Number of GPUs for rollout/inference"})
    colocate: bool = field(
        default=False,
        metadata={"help": "Share GPUs between training and rollout (colocated mode)"},
    )
    validate_reuse_train_gpus: bool = field(
        default=False,
        metadata={"help": "Reuse training GPUs to scale validation in separated mode"},
    )
    validate_reuse_begin_timeout_s: int = field(
        default=30,
        metadata={"help": "Timeout in seconds for trainer ranks to rendezvous before validate-reuse sync"},
    )
    param_sync_rpc_timeout_s: int = field(
        default=120,
        metadata={"help": "Timeout in seconds for param sync RPC calls to rollout workers"},
    )
    colocate_timeout_s: int = field(
        default=60,
        metadata={"help": "Ray-level timeout for colocated offload/resume lifecycle operations"},
    )
    tensor_model_parallel_size: int = field(default=1, metadata={"help": "Tensor parallelism size"})
    pipeline_model_parallel_size: int = field(default=1, metadata={"help": "Pipeline parallelism size"})
    context_parallel_size: int = field(default=1, metadata={"help": "Context parallelism size"})
    expert_model_parallel_size: int = field(default=1, metadata={"help": "Expert model parallelism size"})
    expert_tensor_parallel_size: int = field(default=1, metadata={"help": "Expert tensor parallelism size"})
    virtual_pipeline_model_parallel_size: int | None = field(default=None, metadata={"help": "Virtual pipeline model parallel size"})
    sequence_parallel: bool = field(default=False, metadata={"help": "Whether the sequence parallel is enabled."})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SiiRLArguments:
    data: DataArguments = field(default_factory=DataArguments)
    actor_ref: ActorRefArguments = field(default_factory=ActorRefArguments)
    rollout: RolloutArguments = field(default_factory=RolloutArguments)
    critic: CriticArguments = field(default_factory=CriticArguments)
    trainer: TrainingArguments = field(default_factory=TrainingArguments)
    custom_reward_function: CustomRewardArguments = field(default_factory=CustomRewardArguments)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
