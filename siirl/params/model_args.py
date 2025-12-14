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
from typing import Any, Dict, List, Optional

@dataclass
class MegatronArguments:
    tensor_model_parallel_size: int = field(default=1, metadata={"help": "Tensor parallelism size"})
    pipeline_model_parallel_size: int = field(default=1, metadata={"help": "Pipeline parallelism size"})
    context_parallel_size: int = field(default=1, metadata={"help": "Context parallelism size"})
    expert_model_parallel_size: int = field(default=1, metadata={"help": "Expert model parallelism size"})
    expert_tensor_parallel_size: int = field(default=1, metadata={"help": "Expert tensor parallelism size"})
    virtual_pipeline_model_parallel_size: Optional[int] = field(default=None, metadata={"help": "Virtual pipeline model parallel size"})
    sequence_parallel: bool = field(default=False, metadata={"help": "Whether the sequence parallel is enabled."})
    use_distributed_optimizer: bool = field(
        default=True,
        metadata={"help": "Whether the distributed optimizer is enabled."},
    )
    param_dtype: str = field(default="bfloat16", metadata={"help": "parameter data dtype"})
    seed: int = field(default=1, metadata={"help": "The random seed"})
    param_offload: bool = field(default=True, metadata={"help": "Offload parameters to CPU"})
    grad_offload: bool = field(default=False, metadata={"help": "Offload gradients to CPU"})
    optimizer_offload: bool = field(default=False, metadata={"help": "Offload optimizer states to CPU"})
    extra: Dict[str, Any] = field(default_factory=dict, metadata={"help": "Extra settings"})
    override_transformer_config: Dict[str, Any] = field(default_factory=dict, metadata={"help": "Override transformer config"})
    use_dist_checkpointing: bool = field(default=False, metadata={"help": "Whether to use distributed checkpointing"})
    dist_checkpointing_path: str = field(default="", metadata={"help": "Path to save distributed checkpointing"})
    override_ddp_config: Dict[str, Any] = field(default_factory=dict, metadata={"help": "Override ddp config"})
    use_mbridge: bool = field(default=False, metadata={"help": "Whether to use mbridge"})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizerArguments:
    lr: float = field(default=1e-6, metadata={"help": "Learning rate"})
    lr_warmup_steps_ratio: float = field(default=0.0, metadata={"help": "Warmup steps ratio"})
    min_lr: float = field(default=0.0, metadata={"help": "Min learning rate"})
    min_lr_ratio: Optional[float] = field(default=0.0, metadata={"help": "Min learning rate ratio"})
    warmup_style: str = field(default="constant", metadata={"help": "Warmup strategy"})
    lr_warmup_init: float = field(default=0.0, metadata={"help": "Learning rate warmup init"})
    lr_decay_steps: Optional[int] = field(default=None, metadata={"help": "Learning rate decay steps"})
    lr_decay_style: str = field(default="linear", metadata={"help": "Learning rate decay style"})
    weight_decay_incr_style: str = field(default="constant", metadata={"help": "Weight decay increase style"})
    lr_wsd_decay_style: str = field(default="exponential", metadata={"help": "Learning rate warmup decay style"})
    lr_wsd_decay_steps: Optional[int] = field(default=None, metadata={"help": "Learning rate warmup decay steps"})
    use_checkpoint_opt_param_scheduler: bool = field(default=False, metadata={"help": "Whether to use checkpoint opt param scheduler"})
    total_training_steps: int = field(default=-1, metadata={"help": "Total training steps"})
    betas: tuple[float, float] = field(default=(0.9, 0.999), metadata={"help": "Beta params Of Optimizer"})
    weight_decay: float = field(default=1e-2, metadata={"help": "Weight decay params of Optimizer"})
    lr_warmup_steps: int = field(
        default=-1,
        metadata={"help": "Prioritized. Negative values mean delegating to lr_warmup_steps_ratio."},
    )
    clip_grad: float = field(default=1.0, metadata={"help": "gradient clip"})
    num_cycles: float = field(default=0.5, metadata={"help": "num cycles"})
    override_optimizer_config: Optional[dict] = field(default=None, metadata={"help": "Override optimizer config"})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelArguments():
    path: str = field(
        default="~/models/deepseek-llm-7b-chat",
        metadata={"help": "Model path or identifier"},
    )
    external_lib: Optional[str] = field(default=None, metadata={"help": "External model library"})
    override_config: Dict[str, Any] = field(default_factory=dict, metadata={"help": "Model config overrides"})
    enable_gradient_checkpointing: bool = field(default=True, metadata={"help": "Gradient checkpointing"})
    gradient_checkpointing_kwargs: Dict[str, Any] = field(default_factory=dict, metadata={"help": "Gradient checkpointing kwargs"})
    use_remove_padding: bool = field(default=False, metadata={"help": "Padding removal optimization"})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Download from hugging face, modelscope, openmind local cache dir"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust the execution of code from datasets/models defined on the Hub or not."},
    )
    hf_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Hugging Face Hub."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether or not to use one of the fast tokenizer (backed by the tokenizers library)."},
    )
    split_special_tokens: bool = field(
        default=False,
        metadata={"help": "Whether or not the special tokens should be split during the tokenization process."},
    )
    model_max_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum input length for model, derived from `cutoff_len`. Do not specify it."},
    )
    new_special_tokens: Optional[str] = field(
        default=None,
        metadata={"help": "Special tokens to be added into the tokenizer. Use commas to separate multiple tokens."},
    )
    resize_vocab: bool = field(
        default=False,
        metadata={"help": "Whether or not to resize the tokenizer vocab and the embedding layers."},
    )
    megatron: MegatronArguments = field(default_factory=MegatronArguments, metadata={"help": "Megatron settings"})
    input_tokenizer: Optional[str] = field(default=None, metadata={"help": "input tokenizer path"})
    rm_tokenizer: Optional[str] = field(default=None, metadata={"help": "rmokenizer path"})
    lora_rank: int = field(default=0, metadata={"help": "set to positive value to enable LoRA (e.g., 32)"})
    lora_alpha: float = field(default=16, metadata={"help": "LoRA scaling factor"})
    target_modules: str = field(default="all-linear", metadata={"help": "all-linear or [q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]"})
    use_shm: bool = field(default=False)
    enable_activation_offload: bool = field(default=False, metadata={"help": "enable activation offload"})

    def __post_init__(self):
        if self.path is None:
            raise ValueError("Please provide `path`.")

        if self.split_special_tokens and self.use_fast_tokenizer:
            raise ValueError("`split_special_tokens` is only supported for slow tokenizers.")

        if self.new_special_tokens is not None:  # support multiple special tokens
            self.new_special_tokens = [token.strip() for token in self.new_special_tokens.split(",")]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CheckpointArguments:
    contents: List[str] = field(
        default_factory=lambda: ["model", "hf_model", "optimizer", "extra"], 
        metadata={"help": "The contents to save and load in the checkpoint."}
    )
    save_contents: List[str] = field(
        default_factory=lambda: ["model", "optimizer", "extra"],
        metadata={"help": "The contents to save in the checkpoint."}
    )
    load_contents: List[str] = field(
        default_factory=lambda: ["model", "optimizer", "extra"],
        metadata={"help": "The contents to load in the checkpoint."}
    )
    async_save: bool = field(default=False, metadata={"help": "Async checkpoint save mode"})


@dataclass
class PolicyLossArguments:
    loss_mode: str = field(default="vanilla", metadata={"help": "Loss function mode. Options: 'vanilla', 'clip-cov', 'kl-cov', 'gpg'."})
    clip_cov_ratio: float = field(default=0.0002, metadata={"help": "Ratio of tokens to be clipped for clip-cov loss."})
    clip_cov_lb: float = field(default=1.0, metadata={"help": "Lower bound for clip-cov loss."})
    clip_cov_ub: float = field(default=5.0, metadata={"help": "Upper bound for clip-cov loss."})
    kl_cov_ratio: float = field(default=0.0002, metadata={"help": "Ratio of tokens to be applied KL penalty for kl-cov loss."})
    ppo_kl_coef: float = field(default=0.1, metadata={"help": "KL divergence penalty coefficient."})


@dataclass
class ActorArguments:
    ppo_mini_batch_size: int = field(default=256, metadata={"help": "PPO mini-batch size"})
    ppo_micro_batch_size: Optional[int] = field(default=None, metadata={"help": "[Deprecated] Micro-batch size"})
    ppo_micro_batch_size_per_gpu: Optional[int] = field(default=None, metadata={"help": "Per-GPU micro-batch size"})
    ppo_max_token_len_per_gpu: int = field(default=16384, metadata={"help": "Max tokens per GPU"})
    grad_clip: float = field(default=1.0, metadata={"help": "Gradient clipping"})
    clip_ratio: float = field(default=0.2, metadata={"help": "Clipping ratio"})
    clip_ratio_low: float = field(default=0.2, metadata={"help": "Min value for clip ratio"})
    clip_ratio_high: float = field(default=0.2, metadata={"help": "Max value for clip ratio"})
    clip_ratio_c: float = field(default=3.0, metadata={"help": "lower bound of the value for Dual-clip PPO from https://arxiv.org/pdf/1912.09729"})
    entropy_coeff: float = field(default=0, metadata={"help": "Entropy coefficient"})
    use_kl_loss: bool = field(default=False, metadata={"help": "Enable KL loss"})
    kl_loss_coef: float = field(default=0.001, metadata={"help": "KL loss coefficient"})
    kl_loss_type: str = field(default="low_var_kl", metadata={"help": "KL loss type"})
    ppo_epochs: int = field(default=1, metadata={"help": "PPO epochs"})
    shuffle: bool = field(default=False, metadata={"help": "Data shuffling"})
    policy_loss: PolicyLossArguments = field(default_factory=PolicyLossArguments, metadata={"help": "Policy loss settings"})
    tis_imp_ratio_cap: float = field(default=-1, metadata={"help": "Truncated importance sampling ratio cap"})
    optim: OptimizerArguments = field(default_factory=OptimizerArguments, metadata={"help": "Optimizer settings"})
    megatron: MegatronArguments = field(default_factory=MegatronArguments, metadata={"help": "Megatron settings"})
    use_remove_padding: bool = field(default=False, metadata={"help": "Padding removal optimization"})
    use_torch_compile: bool = field(default=True, metadata={"help": "Whether or not use torch compile"})
    checkpoint: CheckpointArguments = field(default_factory=CheckpointArguments, metadata={"help": "Checkpoint configuration"})
    param_offload: bool = field(default=False, metadata={"help": "Enable param offload or not"})
    grad_offload: bool = field(default=False, metadata={"help": "Enable grad offload or not"})
    optimizer_offload: bool = field(default=False, metadata={"help": "Enable optimizer offload or not"})
    load_weight: bool = field(default=True)
    loss_agg_mode: str = field(default="token-mean", metadata={"help": "seq-mean-token-sum, seq-mean-token-mean"})
    recompute_old_log_prob: bool = field(default=True, metadata={"help": "recompute old log prob"})
    data_loader_seed: Optional[int] = field(default=None, metadata={"help": "Data loader seed"})
    n: int = field(default=1, metadata={"help": "Number of responses per prompt"})
    log_prob_micro_batch_size_per_gpu: Optional[int] = field(default=None, metadata={"help": "Per-GPU log prob micro-batch size"})
    log_prob_max_token_len_per_gpu: int = field(default=16384, metadata={"help": "Max tokens per GPU for log prob"})
    temperature: float = field(default=1.0, metadata={"help": "Sampling temperature"})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvalSamplingArguments:
    top_k: int = field(default=-1, metadata={"help": "0 for hf rollout, -1 for vllm rollout"})
    top_p: float = field(default=1.0)
    temperature: int = field(default=0)
    n: int = field(default=1)
    do_sample: bool = field(default=False)


@dataclass
class LayerNameMapArguments:
    qkv_layer_name: str = field(default="qkv", metadata={"help": "QKV layer name map"})
    gate_proj_layer_name: str = field(default="linear_fc1.weight", metadata={"help": "Gate projection layer name map"})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class EngineArguments:
    vllm: Dict[str, Any] = field(default_factory=lambda: {})
    sglang: Dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class RolloutArguments:
    name: str = field(default="vllm", metadata={"help": "Rollout engine"})
    temperature: float = field(default=1.0, metadata={"help": "Sampling temperature"})
    top_k: int = field(default=-1, metadata={"help": "Top-k sampling"})
    top_p: float = field(default=1.0, metadata={"help": "Top-p sampling"})
    use_fire_sampling: bool = field(default=False, metadata={"help": "Fire sampling optimization"})
    prompt_length: int = field(default=None, metadata={"help": "Prompt length"})
    response_length: int = field(default=None, metadata={"help": "Response length"})
    dtype: str = field(default="bfloat16", metadata={"help": "Compute dtype"})
    gpu_memory_utilization: float = field(default=0.5, metadata={"help": "GPU memory usage"})
    ignore_eos: bool = field(default=False, metadata={"help": "Ignore EOS tokens"})
    enforce_eager: bool = field(default=True, metadata={"help": "Eager execution"})
    free_cache_engine: bool = field(default=True, metadata={"help": "Free GPU cache"})
    load_format: str = field(default="dummy_dtensor", metadata={"help": "Weight loading format"})
    tensor_model_parallel_size: int = field(default=1, metadata={"help": "Tensor parallelism"})
    max_num_batched_tokens: int = field(default=8192, metadata={"help": "Max batched tokens"})
    max_model_len: Optional[int] = field(default=None, metadata={"help": "Max model length"})
    max_num_seqs: int = field(default=1024, metadata={"help": "Max concurrent sequences"})
    limit_images: Optional[int] = field(default=None, metadata={"help": "support for multi-image data"})
    do_sample: bool = field(default=True, metadata={"help": "Enable sampling"})
    n: int = field(default=1, metadata={"help": "Number of responses"})
    log_prob_micro_batch_size: Optional[int] = field(default=None, metadata={"help": "[Deprecated] Log prob batch size"})
    log_prob_micro_batch_size_per_gpu: Optional[int] = field(default=None, metadata={"help": "Per-GPU log prob batch size"})
    log_prob_max_token_len_per_gpu: int = field(default=16384, metadata={"help": "Max tokens per GPU"})
    disable_log_stats: bool = field(default=True, metadata={"help": "Whether or not disable log stats"})
    enable_chunked_prefill: bool = field(default=True, metadata={"help": "Whether or not enable chunked prefill"})
    trust_remote_code: bool = field(default=False, metadata={"help": "trust the code or not."})
    val_kwargs: EvalSamplingArguments = field(default_factory=EvalSamplingArguments)
    layer_name_map: LayerNameMapArguments = field(default_factory=LayerNameMapArguments)
    seed: int = field(default=0, metadata={"help": "The random seed"})
    engine_kwargs: EngineArguments = field(default_factory=EngineArguments)
    multi_stage_wake_up: bool = field(default=False, metadata={"help": "# Whether to wake up inference engine in multi-stage. (Wake up model weights first, then resume kv cache)"})
    router_ip: str = field(default="None", metadata={"help": "Rollout Router IP"})
    router_port: str = field(default="None", metadata={"help": "Rollout Router Port"})
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RefArguments:
    strategy: str = field(default="fsdp", metadata={"help": "Parallel strategy"})
    megatron: MegatronArguments = field(default_factory=MegatronArguments, metadata={"help": "Megatron settings"})
    log_prob_micro_batch_size: Optional[int] = field(default=None, metadata={"help": "[Deprecated] Log prob batch size"})
    log_prob_micro_batch_size_per_gpu: Optional[int] = field(default=None, metadata={"help": "Per-GPU log prob batch size"})
    log_prob_max_token_len_per_gpu: int = field(default=16384, metadata={"help": "Max tokens per GPU"})
    ulysses_sequence_parallel_size: int = field(default=1, metadata={"help": "Sequence parallel size"})
    use_remove_padding: bool = field(default=False, metadata={"help": "Padding removal optimization"})
    use_torch_compile: bool = field(default=True, metadata={"help": "Whether or not use torch compile"})
    ppo_micro_batch_size: Optional[int] = field(default=None, metadata={"help": "[Deprecated] Micro-batch size"})
    ppo_micro_batch_size_per_gpu: Optional[int] = field(default=None, metadata={"help": "Per-GPU micro-batch size"})
    param_offload: bool = field(default=False, metadata={"help": "Enable param offload or not"})
    grad_offload: bool = field(default=False, metadata={"help": "Enable grad offload or not"})
    optimizer_offload: bool = field(default=False, metadata={"help": "Enable optimizer offload or not"})
    load_weight: bool = field(default=True)
    shuffle: bool = field(default=False, metadata={"help": "Data shuffling"})
    data_loader_seed: Optional[int] = field(default=None, metadata={"help": "Data loader seed"})
    recompute_old_log_prob: bool = field(default=True, metadata={"help": "recompute old log prob"})
    temperature: float = field(default=1.0, metadata={"help": "Sampling temperature"})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActorRefArguments:
    hybrid_engine: bool = field(default=True, metadata={"help": "Hybrid engine mode"})
    model: ModelArguments = field(default_factory=ModelArguments, metadata={"help": "Base model settings"})
    actor: ActorArguments = field(default_factory=ActorArguments, metadata={"help": "Actor configuration"})
    ref: RefArguments = field(default_factory=RefArguments, metadata={"help": "Reference model settings"})
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CriticArguments:
    strategy: str = field(default="fsdp", metadata={"help": "Parallel strategy"})
    optim: OptimizerArguments = field(
        default_factory=lambda: OptimizerArguments(lr=1e-5),
        metadata={"help": "Optimizer settings"},
    )
    model: ModelArguments = field(
        default_factory=lambda: ModelArguments(path="~/models/deepseek-llm-7b-chat", enable_gradient_checkpointing=True),
        metadata={"help": "Critic model"},
    )
    megatron: MegatronArguments = field(default_factory=MegatronArguments, metadata={"help": "Megatron settings"})
    ppo_mini_batch_size: int = field(default=256, metadata={"help": "PPO mini-batch size"})
    ppo_micro_batch_size: Optional[int] = field(default=None, metadata={"help": "[Deprecated] Micro-batch size"})
    ppo_micro_batch_size_per_gpu: Optional[int] = field(default=None, metadata={"help": "Per-GPU micro-batch size"})
    ppo_epochs: int = field(default=1, metadata={"help": "PPO epochs"})
    shuffle: bool = field(default=False, metadata={"help": "Data shuffling"})
    grad_clip: float = field(default=1.0, metadata={"help": "Gradient clipping"})
    cliprange_value: float = field(default=0.5, metadata={"help": "Value clipping range"})
    forward_max_token_len_per_gpu: int = field(default=32768, metadata={"help": "Forward max token length in per gpu"})
    load_weight: bool = field(default=True)
    rollout_n: int = field(default=1, metadata={"help": "rollout n"})
    checkpoint: CheckpointArguments = field(default_factory=CheckpointArguments, metadata={"help": "Checkpoint configuration"})
    ppo_max_token_len_per_gpu: int = field(default=32768, metadata={"help": "Max tokens per GPU"})
    loss_agg_mode: str = field(default="token-mean", metadata={"help": "token-mean, seq-mean-token-sum, seq-mean-token-mean"})
    data_loader_seed: Optional[int] = field(default=None, metadata={"help": "Data loader seed"})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KLCtrlArguments:
    type: str = field(default="fixed", metadata={"help": "Type of KL Ctrl, fixed or adaptive"})
    kl_coef: float = field(default=0.001, metadata={"help": "Coef of KL"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "Target KL value"})
    horizon: Optional[float] = field(default=1000, metadata={"help": "Horizon of KL"})


@dataclass
class AlgorithmArguments:
    gamma: float = field(default=1.0, metadata={"help": "Discount factor"})
    lam: float = field(default=1.0, metadata={"help": "GAE lambda"})
    adv_estimator: str = field(default="gae", metadata={"help": "Advantage estimator"})
    kl_penalty: str = field(default="kl", metadata={"help": "KL penalty type"})
    kl_ctrl: KLCtrlArguments = field(default_factory=KLCtrlArguments)
    use_kl_in_reward: bool = field(default=False, metadata={"help": "Use KL In-Reward"})
    share_reward_in_agent: bool = field(default=True, metadata={"help": "Shard Reward in Reward"})
    norm_adv_by_std_in_grpo: bool = field(default=True, metadata={"help": "Whether to scale the GRPO advantage"})
    weight_factor_in_cpgd: str = field(default="STD_weight", metadata={"help": "The weighting methods for advantage {STD_weight, clip_filter_like_weight, naive}"})
    algorithm_name: str = field(default="grpo", metadata={"help": "Algorithm name, e.g., grpo, ppo, dapo"})
    use_pf_ppo: bool = field(default=False, metadata={"help": "Whether to enable preference feedback PPO."})
    pf_ppo: dict[str, Any] = field(default_factory=dict, metadata={"help": " Preference feedback PPO settings."})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
