# Copyright 2025, Shanghai Innovation Institute. All rights reserved.
# Copyright 2025, Infrawaves. All rights reserved.
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


@dataclass
class MegatronArguments:
    use_distributed_optimizer: bool = field(
        default=True,
        metadata={"help": "Whether the distributed optimizer is enabled."},
    )
    param_dtype: str = field(default="bfloat16", metadata={"help": "parameter data dtype"})
    seed: int = field(default=1, metadata={"help": "The random seed"})
    param_offload: bool = field(default=False, metadata={"help": "Offload parameters to CPU"})
    grad_offload: bool = field(default=False, metadata={"help": "Offload gradients to CPU"})
    optimizer_offload: bool = field(default=False, metadata={"help": "Offload optimizer states to CPU"})
    override_transformer_config: dict[str, Any] = field(default_factory=dict, metadata={"help": "Override transformer config"})
    override_ddp_config: dict[str, Any] = field(default_factory=dict, metadata={"help": "Override ddp config"})
    use_mbridge: bool = field(default=True, metadata={"help": "Whether to use mbridge"})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizerArguments:
    lr: float = field(default=1e-6, metadata={"help": "Learning rate"})
    lr_warmup_steps_ratio: float = field(default=0.0, metadata={"help": "Warmup steps ratio"})
    min_lr: float = field(default=0.0, metadata={"help": "Min learning rate"})
    lr_warmup_init: float = field(default=0.0, metadata={"help": "Learning rate warmup init"})
    lr_decay_steps: int | None = field(default=None, metadata={"help": "Learning rate decay steps"})
    lr_decay_style: str = field(default="linear", metadata={"help": "Learning rate decay style"})
    weight_decay_incr_style: str = field(default="constant", metadata={"help": "Weight decay increase style"})
    lr_wsd_decay_style: str = field(default="exponential", metadata={"help": "Learning rate warmup decay style"})
    lr_wsd_decay_steps: int | None = field(default=None, metadata={"help": "Learning rate warmup decay steps"})
    use_checkpoint_opt_param_scheduler: bool = field(
        default=False,
        metadata={"help": "Whether to use checkpoint opt param scheduler"},
    )
    total_training_steps: int = field(default=-1, metadata={"help": "Total training steps"})
    weight_decay: float = field(default=1e-2, metadata={"help": "Weight decay params of Optimizer"})
    lr_warmup_steps: int = field(
        default=-1,
        metadata={"help": "Prioritized. Negative values mean delegating to lr_warmup_steps_ratio."},
    )
    clip_grad: float = field(default=1.0, metadata={"help": "gradient clip"})
    override_optimizer_config: dict | None = field(default=None, metadata={"help": "Override optimizer config"})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelArguments:
    path: str = field(
        default="~/models/deepseek-llm-7b-chat",
        metadata={"help": "Model path or identifier"},
    )
    override_config: dict[str, Any] = field(default_factory=dict, metadata={"help": "Model config overrides"})
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust the execution of code from datasets/models defined on the Hub or not."},
    )
    megatron: MegatronArguments = field(default_factory=MegatronArguments, metadata={"help": "Megatron settings"})

    # used for dataloader
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether or not to use one of the fast tokenizer (backed by the tokenizers library)."},
    )
    split_special_tokens: bool = field(
        default=False,
        metadata={"help": "Whether or not the special tokens should be split during the tokenization process."},
    )

    def __post_init__(self):
        if self.path is None:
            raise ValueError("Please provide `path`.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActorArguments:
    train_backend: str = field(default="megatron", metadata={"help": "Backend for training"})
    ppo_mini_batch_size: int = field(default=256, metadata={"help": "PPO mini-batch size"})
    ppo_micro_batch_size_per_gpu: int | None = field(default=None, metadata={"help": "Per-GPU micro-batch size"})
    loss_mode: str = field(default="vanilla", metadata={"help": "loss_mode for loss compute"})

    # Dynamic batching config
    use_dynamic_batch: bool = field(default=False, metadata={"help": "Enable dynamic batching (token-based instead of fixed batch size)"})
    max_tokens_per_gpu: int = field(default=4096, metadata={"help": "Max tokens per GPU when dynamic batching is enabled"})
    use_workload_balance: bool = field(default=True, metadata={"help": "Use FLOPs-based balancing (otherwise sequence length based)"})
    denominator_scope: str = field(
        default="local",
        metadata={"help": "Loss denominator scope for dynamic batch: local or dp_global"},
    )
    loss_scale_factor: float | None = field(
        default=None,
        metadata={"help": "Optional fixed denominator for seq-mean-token-sum-norm"},
    )
    clip_ratio: float = field(default=0.2, metadata={"help": "Clipping ratio"})
    clip_ratio_low: float = field(default=0.2, metadata={"help": "Min value for clip ratio"})
    clip_ratio_c: float = field(
        default=3.0,
        metadata={"help": "lower bound of the value for Dual-clip PPO from https://arxiv.org/pdf/1912.09729"},
    )
    clip_ratio_high: float = field(default=0.2, metadata={"help": "Max value for clip ratio"})
    entropy_coeff: float = field(default=0, metadata={"help": "Entropy coefficient"})
    use_kl_loss: bool = field(default=False, metadata={"help": "Enable KL loss"})
    kl_loss_coef: float = field(default=0.001, metadata={"help": "KL loss coefficient"})
    kl_loss_type: str = field(default="low_var_kl", metadata={"help": "KL loss type"})
    ppo_epochs: int = field(default=1, metadata={"help": "PPO epochs"})
    optim: OptimizerArguments = field(default_factory=OptimizerArguments, metadata={"help": "Optimizer settings"})
    megatron: MegatronArguments = field(default_factory=MegatronArguments, metadata={"help": "Megatron settings"})
    load_weight: bool = field(default=True)
    loss_agg_mode: str = field(
        default="token-mean",
        metadata={"help": "seq-mean-token-sum, seq-mean-token-mean"},
    )
    temperature: float = field(default=1.0, metadata={"help": "Sampling temperature"})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalSamplingArguments:
    top_k: int = field(default=-1, metadata={"help": "0 for hf rollout, -1 for vllm rollout"})
    top_p: float = field(default=1.0)
    temperature: int = field(default=0)
    n: int = field(default=1)
    do_sample: bool = field(default=False)


@dataclass
class MultiturnArguments:
    env_type: str = field(default=None, metadata={"help": "env type: tool_env, vla_env ..."})
    max_env_turns: int = field(default=1, metadata={"help": "max env turns"})
    max_assistant_turns: int = field(default=1, metadata={"help": "max model generate turns"})
    env_path: str = field(default=None, metadata={"help": "env yaml config path"})
    env_kwargs: dict[str, Any] = field(default_factory=lambda: {})
    max_parallel_calls: int = field(default=1, metadata={"help": "Max parallel env"})
    max_env_response_length: int = field(default=256, metadata={"help": "Max env response"})
    env_response_truncate_side: str = field(
        default="middle",
        metadata={"help": "Truncate side of Env response: left, middle, right"},
    )


@dataclass
class RolloutArguments:
    name: str = field(default="sglang", metadata={"help": "Rollout engine"})
    temperature: float = field(default=1.0, metadata={"help": "Sampling temperature"})
    top_k: int = field(default=-1, metadata={"help": "Top-k sampling"})
    top_p: float = field(default=1.0, metadata={"help": "Top-p sampling"})
    dtype: str = field(default="bfloat16", metadata={"help": "Compute dtype"})
    gpu_memory_utilization: float = field(default=0.5, metadata={"help": "GPU memory usage"})
    ignore_eos: bool = field(default=False, metadata={"help": "Ignore EOS tokens"})
    enforce_eager: bool = field(default=True, metadata={"help": "Eager execution"})
    free_cache_engine: bool = field(default=True, metadata={"help": "Free GPU cache"})
    load_format: str = field(default="dummy_dtensor", metadata={"help": "Weight loading format"})
    tensor_model_parallel_size: int = field(default=1, metadata={"help": "Tensor parallelism"})
    colocate_param_sync_backend: str = field(
        default="tensor",
        metadata={"help": "Colocated param sync backend: tensor or flattened_bucket"},
    )
    colocate_release_weights_during_sync: bool = field(
        default=False,
        metadata={
            "help": "Release rollout weights (not just KV cache) during colocated offload. "
            "Frees more GPU memory but requires onload before IPC weight sync."
        },
    )
    max_num_batched_tokens: int = field(default=8192, metadata={"help": "Max batched tokens"})
    max_model_len: int | None = field(default=None, metadata={"help": "Max model length"})
    max_num_seqs: int = field(default=1024, metadata={"help": "Max concurrent sequences"})
    server_concurrency: int = field(
        default=512,
        metadata={"help": "Max concurrent client requests per rollout engine during batch generation"},
    )
    train_server_concurrency: int = field(
        default=256,
        metadata={"help": "Max concurrent client requests per worker during training rollout"},
    )
    validate_server_concurrency: int = field(
        default=256,
        metadata={"help": "Max concurrent client requests per worker during validation (local, no router)"},
    )
    validate_chunk_size: int = field(
        default=1024,
        metadata={"help": "Number of samples per validation chunk to limit peak concurrency"},
    )
    do_sample: bool = field(default=True, metadata={"help": "Enable sampling"})
    n: int = field(default=1, metadata={"help": "Number of responses"})
    enable_chunked_prefill: bool = field(default=True, metadata={"help": "Whether or not enable chunked prefill"})
    trust_remote_code: bool = field(default=False, metadata={"help": "trust the code or not."})
    val_kwargs: EvalSamplingArguments = field(default_factory=EvalSamplingArguments)
    seed: int = field(default=0, metadata={"help": "The random seed"})
    calculate_log_probs: bool = field(default=True, metadata={"help": "Whether rollout calculate log probs"})
    multi_stage_wake_up: bool = field(
        default=False,
        metadata={"help": "# Whether to wake up inference engine in multi-stage. (Wake up model weights first, then resume kv cache)"},
    )
    router_ip: str = field(default=None, metadata={"help": "Rollout Router IP"})
    router_port: str = field(default=None, metadata={"help": "Rollout Router Port"})
    executor_module: str = field(default="naive", metadata={"help": "Batch rollout Generate Executor"})
    flow_function: str = field(default="naive", metadata={"help": "Sample rollout Generate Executor"})
    flow_config: str = field(
        default="config.yaml",
        metadata={"help": "Sample rollout Generate Executor config path"},
    )
    multiturn: MultiturnArguments = field(default_factory=MultiturnArguments)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RefArguments:
    megatron: MegatronArguments = field(default_factory=MegatronArguments, metadata={"help": "Megatron settings"})
    log_prob_micro_batch_size: int | None = field(default=None, metadata={"help": "[Deprecated] Log prob batch size"})
    log_prob_micro_batch_size_per_gpu: int | None = field(default=None, metadata={"help": "Per-GPU log prob batch size"})
    use_remove_padding: bool = field(default=False, metadata={"help": "Padding removal optimization"})
    ppo_micro_batch_size_per_gpu: int | None = field(default=None, metadata={"help": "Per-GPU micro-batch size"})
    param_offload: bool = field(default=False, metadata={"help": "Enable param offload or not"})
    load_weight: bool = field(default=True)
    temperature: float = field(default=1.0, metadata={"help": "Sampling temperature"})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AlgorithmArguments:
    adv_estimator: str = field(default="grpo", metadata={"help": "Advantage estimator"})
    gamma: float = field(default=1.0, metadata={"help": "Discount factor"})
    lam: float = field(default=1.0, metadata={"help": "GAE lambda"})
    kl_penalty: str = field(default="kl", metadata={"help": "KL penalty type"})
    norm_adv_by_std_in_grpo: bool = field(default=True, metadata={"help": "Whether to scale the GRPO advantage"})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CheckpointArguments:
    """Configuration for checkpoint save/load contents.

    Supported content types:
        - "model": Model weights (Megatron distributed format)
        - "optimizer": Optimizer states
        - "extra": Extra states (RNG states, lr_scheduler, etc.)
        - "hf_model": HuggingFace format model (converted from Megatron)

    Example:
        To enable HF model saving, add "hf_model" to save_contents:
        ```yaml
        checkpoint:
            save_contents: ["model", "optimizer", "extra", "hf_model"]
        ```
    """

    contents: list[str] = field(
        default_factory=lambda: ["model", "optimizer", "extra"],
        metadata={"help": "Default contents to save and load in the checkpoint."},
    )
    save_contents: list[str] = field(
        default_factory=lambda: ["model", "optimizer", "extra"],
        metadata={"help": "Contents to save: model, optimizer, extra, hf_model"},
    )
    load_contents: list[str] = field(
        default_factory=lambda: ["model", "optimizer", "extra"],
        metadata={"help": "Contents to load: model, optimizer, extra"},
    )
    async_save: bool = field(default=False, metadata={"help": "Enable async checkpoint save (experimental)"})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActorRefArguments:
    model: ModelArguments = field(default_factory=ModelArguments, metadata={"help": "Base model settings"})
    actor: ActorArguments = field(default_factory=ActorArguments, metadata={"help": "Actor configuration"})
    ref: RefArguments = field(default_factory=RefArguments, metadata={"help": "Reference model settings"})
    algorithm: AlgorithmArguments = field(default_factory=AlgorithmArguments, metadata={"help": "Algorithm settings"})
    checkpoint: CheckpointArguments = field(
        default_factory=CheckpointArguments,
        metadata={"help": "Checkpoint save/load configuration"},
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CriticArguments:
    optim: OptimizerArguments = field(
        default_factory=lambda: OptimizerArguments(lr=1e-5),
        metadata={"help": "Optimizer settings"},
    )
    model: ModelArguments = field(
        default_factory=lambda: ModelArguments(path="~/models/deepseek-llm-7b-chat"),
        metadata={"help": "Critic model"},
    )
    megatron: MegatronArguments = field(default_factory=MegatronArguments, metadata={"help": "Megatron settings"})
    ppo_mini_batch_size: int = field(default=256, metadata={"help": "PPO mini-batch size"})
    ppo_micro_batch_size_per_gpu: int | None = field(default=None, metadata={"help": "Per-GPU micro-batch size"})
    ppo_epochs: int = field(default=1, metadata={"help": "PPO epochs"})
    cliprange_value: float = field(default=0.5, metadata={"help": "Value clipping range"})
    load_weight: bool = field(default=True)
    loss_agg_mode: str = field(
        default="token-mean",
        metadata={"help": "token-mean, seq-mean-token-sum, seq-mean-token-mean"},
    )
    checkpoint: CheckpointArguments = field(
        default_factory=CheckpointArguments,
        metadata={"help": "Checkpoint save/load configuration"},
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
