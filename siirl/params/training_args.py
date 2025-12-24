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
from typing import Optional, Dict, List, Any
from .data_args import DataArguments
from .model_args import (
    ActorRefArguments,
    CriticArguments,
    RolloutArguments
)

@dataclass
class TrainingArguments:
    total_epochs: int = field(default=30, metadata={"help": "Total training epochs"})
    total_training_steps: Optional[int] = field(default=None, metadata={"help": "Override training steps"})
    project_name: str = field(default="siirl_examples", metadata={"help": "Project name"})
    experiment_name: str = field(default="gsm8k", metadata={"help": "Experiment name"})
    logger: List[str] = field(
        default_factory=lambda: ["console", "wandb"],
        metadata={"help": "Logging backends"},
    )
    log_val_generations: int = field(default=0, metadata={"help": "Validation samples to log"})
    nnodes: int = field(default=1, metadata={"help": "Number of nodes"})
    n_gpus_per_node: int = field(default=8, metadata={"help": "GPUs per node"})
    save_freq: int = field(default=-1, metadata={"help": "Checkpoint frequency"})
    resume_mode: str = field(default="auto", metadata={"help": "Resume training mode"})
    resume_from_path: bool = field(default=False, metadata={"help": "Resume from specific path"})
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
    validation_data_dir: Optional[str] = field(default=None, metadata={"help": "Validation data directory."})
    device: Optional[str] = field(default="cuda", metadata={"help": "Training device."})
    async_factor: int = field(default=1, metadata={"help": "Control async speed"})
    param_sync_buffer_size: int = field(default=512 * 1024**2,metadata={"help":"buffer size for param_sync, in bytes. This is used for updating weights by chunk and should be useful for MoE models."})

    # === Resource Allocation Configuration ===
    actor_gpus: int = field(
        default=2,
        metadata={"help": "Number of GPUs for training (Actor/Ref/Critic)"}
    )
    rollout_gpus: int = field(
        default=6,
        metadata={"help": "Number of GPUs for rollout/inference"}
    )
    colocate: bool = field(
        default=False,
        metadata={"help": "Share GPUs between training and rollout (colocated mode)"}
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SiiRLArguments:
    data: DataArguments = field(default_factory=DataArguments)
    actor_ref: ActorRefArguments = field(default_factory=ActorRefArguments)
    rollout: RolloutArguments = field(default_factory=RolloutArguments)
    critic: CriticArguments = field(default_factory=CriticArguments)
    trainer: TrainingArguments = field(default_factory=TrainingArguments)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
