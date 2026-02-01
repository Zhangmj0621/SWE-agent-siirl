import os
from functools import partial

# Disable Transformer Engine to avoid ABI compatibility issues
# This is needed when transformer_engine is compiled for a different PyTorch version
os.environ.setdefault("NVTE_FRAMEWORK", "none")

import torch  # noqa: E402
import torch.distributed  # noqa: E402
from loguru import logger  # noqa: E402
from megatron.core import parallel_state as mpu  # noqa: E402
from megatron.core.optimizer import DistributedOptimizer  # noqa: E402
from megatron.core.pipeline_parallel import get_forward_backward_func  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from tensordict import NonTensorData, TensorDict  # noqa: E402
from torch import nn  # noqa: E402

from siirl.algorithm.kl_penalty import kl_penalty  # noqa: E402
from siirl.algorithm.loss import agg_loss, compute_value_loss, get_policy_loss_fn  # noqa: E402
from siirl.engine.actor.utils import append_to_dict  # noqa: E402
from siirl.params import SiiRLArguments  # noqa: E402
from siirl.utils.backend.device import get_device_id, get_device_name, get_torch_device  # noqa: E402
from siirl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager  # noqa: E402
from siirl.utils.megatron.megatron_utils import (  # noqa: E402
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
)
from siirl.utils.megatron.pipeline_parallel import make_batch_generator  # noqa: E402
from siirl.utils.model_utils.flops_counter import FlopsCounter  # noqa: E402
from siirl.utils.model_utils.model import get_hf_model_path, load_megatron_gptmodel_weights  # noqa: E402
from siirl.utils.model_utils.torch_dtypes import PrecisionType  # noqa: E402
from siirl.utils.model_utils.torch_functional import broadcast_dict_tensor, masked_mean  # noqa: E402
from siirl.utils.timer import Timer  # noqa: E402


class ActorWorker:
    def __init__(self, config: SiiRLArguments):
        assert isinstance(config, SiiRLArguments)
        self.rank = 0
        self.hf_config = None
        self.tf_config = None
        self.bridge = None
        self.tokenizer = None
        self.processor = None
        self.architectures = None
        self.share_embeddings_and_output_weights = False

        self.config = config
        self.actor_ref_config = config.actor_ref
        # global_initialize_model_parallel(self.config.actor)

        self._is_offload_param = self.actor_ref_config.actor.megatron.param_offload
        self._is_offload_grad = self.actor_ref_config.actor.megatron.grad_offload
        self._is_offload_optimizer = self.actor_ref_config.actor.megatron.optimizer_offload

    def _init_hf_config_and_tf_config(
        self,
        model_path,
        tokenizer_or_path,
        dtype,
        override_model_config,
        override_transformer_config,
        trust_remote_code=False,
        use_mbridge=False,
    ):
        """Initialize HuggingFace and Transformer configs"""
        from transformers import AutoConfig

        from siirl.models.loader import load_tokenizer
        from siirl.models.mcore import hf_to_mcore_config
        from siirl.utils.model_utils.model import update_model_config

        # Initialize tokenizer
        self.local_path = model_path
        if tokenizer_or_path is None:
            self.tokenizer = load_tokenizer(path=model_path)
        elif isinstance(tokenizer_or_path, str):
            self.tokenizer = load_tokenizer(path=tokenizer_or_path)
        else:
            self.tokenizer = tokenizer_or_path

        # Get HuggingFace config
        hf_config = AutoConfig.from_pretrained(self.local_path, trust_remote_code=trust_remote_code)

        # Override config
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config.get("model_config", {}))
        self.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)
        update_model_config(hf_config, override_config_kwargs=override_config_kwargs)
        self.architectures = getattr(hf_config, "architectures", None)

        # Convert to Megatron config
        tf_config = hf_to_mcore_config(hf_config, dtype, **override_transformer_config)

        # Handle mbridge if needed
        if use_mbridge:
            try:
                from mbridge import AutoBridge
            except ImportError:
                print("mbridge package not found. Please install mbridge with `pip install verl[mcore]` or `pip install mbridge`")

            bridge = AutoBridge.from_config(hf_config)
            # Default activation recomputation config for memory optimization
            # These can be overridden by override_transformer_config
            recompute_defaults = {
                "recompute_granularity": "full",
                "recompute_method": "uniform",
                "recompute_num_layers": 1,
            }
            # Merge defaults with user overrides (user overrides take precedence)
            merged_config = {**recompute_defaults, **override_transformer_config}
            bridge.set_extra_args(**merged_config)
            tf_config = bridge.config

            # Comprehensive logging for memory optimization debugging
            logger.warning("=" * 60)
            logger.warning("[Memory Optimization] ActorWorker TransformerConfig")
            logger.warning("=" * 60)
            logger.warning(f"  Model: {getattr(hf_config, 'model_type', 'unknown')}")
            logger.warning(f"  num_layers: {getattr(tf_config, 'num_layers', 'N/A')}")
            logger.warning(f"  hidden_size: {getattr(tf_config, 'hidden_size', 'N/A')}")
            logger.warning("-" * 60)
            logger.warning("[Recompute/Activation Checkpointing Config] <<<< CRITICAL >>>>")
            logger.warning(f"  recompute_granularity: {getattr(tf_config, 'recompute_granularity', 'NOT SET')}")
            logger.warning(f"  recompute_method: {getattr(tf_config, 'recompute_method', 'NOT SET')}")
            logger.warning(f"  recompute_num_layers: {getattr(tf_config, 'recompute_num_layers', 'NOT SET')}")
            logger.warning("-" * 60)
            logger.warning(f"[mbridge merged_config]: {merged_config}")
            logger.warning("=" * 60)

            self.bridge = bridge
        else:
            self.bridge = None
            # Log config when not using mbridge
            logger.warning("=" * 60)
            logger.warning("[Memory Optimization] ActorWorker (non-mbridge mode)")
            logger.warning("=" * 60)
            logger.warning(f"  Model: {getattr(hf_config, 'model_type', 'unknown')}")
            logger.warning(f"  num_layers: {getattr(tf_config, 'num_layers', 'N/A')}")
            logger.warning("-" * 60)
            logger.warning("[Recompute Config] <<<< CRITICAL >>>>")
            logger.warning(f"  recompute_granularity: {getattr(tf_config, 'recompute_granularity', 'NOT SET')}")
            logger.warning(f"  recompute_method: {getattr(tf_config, 'recompute_method', 'NOT SET')}")
            logger.warning(f"  recompute_num_layers: {getattr(tf_config, 'recompute_num_layers', 'NOT SET')}")
            logger.warning("=" * 60)

        self.hf_config = hf_config
        self.tf_config = tf_config

    def _build_actor_model_optimizer(
        self,
        model_path,
        optim_config,
        override_model_config,
        override_transformer_config,
        override_ddp_config,
    ):
        from siirl.engine.actor.optimizer import get_megatron_optimizer, get_megatron_optimizer_param_scheduler, init_megatron_optim_config
        from siirl.utils.megatron.megatron_utils import McoreModuleWrapperConfig, make_megatron_module

        self._init_hf_config_and_tf_config(
            model_path,
            model_path,
            self.dtype,
            override_model_config,
            override_transformer_config,
            self.actor_ref_config.model.trust_remote_code,
            self.actor_ref_config.actor.megatron.use_mbridge,
        )

        wrap_config = McoreModuleWrapperConfig(
            is_value_model=False,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            wrap_with_ddp=True,
            use_distributed_optimizer=self.actor_ref_config.actor.megatron.use_distributed_optimizer,
        )

        actor_module = make_megatron_module(
            wrap_config=wrap_config,
            tf_config=self.tf_config,
            hf_config=self.hf_config,
            bridge=self.bridge,
            override_model_config=override_model_config,
            override_ddp_config=override_ddp_config,
        )

        if self.actor_ref_config.actor.load_weight:
            if self.bridge is not None:
                local_model_path = get_hf_model_path(self.actor_ref_config)
                self.bridge.load_weights(actor_module, local_model_path)
            else:
                load_megatron_gptmodel_weights(
                    self.actor_ref_config,
                    self.hf_config,
                    actor_module,
                    params_dtype=self.dtype,
                    is_value_model=False,
                )

        optim_megatron_config = init_megatron_optim_config(optim_config)
        actor_optimizer = get_megatron_optimizer(model=actor_module, config=optim_megatron_config)
        actor_optimizer_scheduler = get_megatron_optimizer_param_scheduler(optimizer=actor_optimizer, config=optim_config)

        return (
            actor_module,
            actor_optimizer,
            actor_optimizer_scheduler,
            self.hf_config,
            optim_config,
        )

    def init_model(self):
        override_model_config = self.actor_ref_config.model.override_config
        override_transformer_config = self.actor_ref_config.actor.megatron.override_transformer_config or OmegaConf.create()
        override_ddp_config = self.actor_ref_config.actor.megatron.override_ddp_config or OmegaConf.create()

        self.param_dtype = torch.bfloat16
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        optim_config = self.actor_ref_config.actor.optim
        (
            self.actor_module,
            self.actor_optimizer,
            self.actor_optimizer_scheduler,
            self.actor_model_config,
            self.actor_optim_config,
        ) = self._build_actor_model_optimizer(
            model_path=self.actor_ref_config.model.path,
            optim_config=optim_config,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
            override_ddp_config=override_ddp_config,
        )

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)

        self.actor = MegatronPPOActor(
            config=self.config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            actor_module=self.actor_module,
            actor_optimizer=self.actor_optimizer,
        )

        self.checkpoint_manager = MegatronCheckpointManager(
            model=self.actor_module,
            optimizer=self.actor_optimizer,
            lr_scheduler=self.actor_optimizer_scheduler,
        )

        # Initialize FlopsCounter for MFU calculation
        self.flops_counter = FlopsCounter(self.hf_config, forward_only=False)

        get_torch_device().empty_cache()

    def update_actor(self, data: TensorDict):
        # Import memory profiler for debugging
        from siirl.utils.memory_profiler import log_memory, memory_trace

        log_memory("Before update_actor", reset_peak=True)

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
            log_memory("After load_megatron_model_to_gpu")

        if self._is_offload_optimizer:
            load_megatron_optimizer(self.actor_optimizer)
            log_memory("After load_megatron_optimizer")

        data = data.to(get_device_name())
        micro_batch_size = self.actor_ref_config.actor.ppo_micro_batch_size_per_gpu
        data["micro_batch_size"] = NonTensorData(micro_batch_size)
        data["temperature"] = NonTensorData(self.actor_ref_config.actor.temperature)
        log_memory("Before update_policy (forward+backward)")

        # Time the update_policy call for MFU calculation
        with Timer("update_policy") as timer, memory_trace("update_policy (forward+backward)"):
            metrics = self.actor.update_policy(data=data)
        delta_time = timer.elapsed

        log_memory("After update_policy")

        # Calculate MFU (Model FLOPs Utilization)
        # Note: flops_counter calculates FLOPs for the entire model, but with TP each GPU
        # only computes 1/TP of the model FLOPs. So we need to divide by TP world size.
        if "global_token_num" in data:
            global_token_num = data["global_token_num"]
            if hasattr(global_token_num, "data"):
                global_token_num = global_token_num.data
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_token_num, delta_time)
            if promised_flops > 0:
                tp_world_size = mpu.get_tensor_model_parallel_world_size()
                metrics["perf/mfu/actor"] = estimated_flops / promised_flops / tp_world_size

        metrics["perf/delta_time/actor"] = delta_time

        # Add GPU memory metrics
        metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
        metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)

        data["metrics"] = NonTensorData(metrics)
        data = data.to("cpu")

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)

        return data

    def compute_log_prob(self, data: TensorDict):
        from siirl.utils.memory_profiler import log_memory, memory_trace

        log_memory("Before compute_log_prob", reset_peak=True)

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module, load_grad=False)
            log_memory("After load model for log_prob")

        data["micro_batch_size"] = NonTensorData(self.actor_ref_config.actor.ppo_micro_batch_size_per_gpu)
        data["temperature"] = NonTensorData(self.actor_ref_config.actor.temperature)
        data = data.to(get_device_id())
        log_memory("Before forward (compute_log_prob)")

        with memory_trace("compute_log_prob forward"):
            output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)

        data["old_log_probs"] = output
        data["entropys"] = entropys
        data = data.to("cpu")

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
        return data

    def save_checkpoint(self, local_path, global_step=0, max_ckpt_to_keep=None):
        """Save actor checkpoint using Megatron distributed checkpointing."""
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
        )

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)

    def load_checkpoint(self, local_path):
        """Load actor checkpoint using Megatron distributed checkpointing."""
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)

        self.checkpoint_manager.load_checkpoint(local_path=local_path)

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)


class ReferenceWorker:
    def __init__(self, config: SiiRLArguments):
        assert isinstance(config, SiiRLArguments)
        self.rank = 0
        self.hf_config = None
        self.tf_config = None
        self.bridge = None
        self.tokenizer = None
        self.processor = None
        self.architectures = None
        self.share_embeddings_and_output_weights = False

        self.config = config
        self.actor_ref_config = config.actor_ref
        ref_config = self.actor_ref_config.ref
        # global_initialize_model_parallel(self.config.actor)

        # Normalize config
        if ref_config.log_prob_micro_batch_size:
            ref_config.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
            ref_config.log_prob_micro_batch_size_per_gpu = ref_config.log_prob_micro_batch_size
        else:
            assert ref_config.log_prob_micro_batch_size_per_gpu is not None

        self._ref_is_offload_param = ref_config.megatron.param_offload

    def _init_hf_config_and_tf_config(
        self,
        model_path,
        tokenizer_or_path,
        dtype,
        override_model_config,
        override_transformer_config,
        trust_remote_code=False,
        use_mbridge=False,
    ):
        """Initialize HuggingFace and Transformer configs"""
        from transformers import AutoConfig

        from siirl.models.loader import load_tokenizer
        from siirl.models.mcore import hf_to_mcore_config
        from siirl.utils.model_utils.model import update_model_config

        # Initialize tokenizer
        self.local_path = model_path
        if tokenizer_or_path is None:
            self.tokenizer = load_tokenizer(path=model_path)
        elif isinstance(tokenizer_or_path, str):
            self.tokenizer = load_tokenizer(path=tokenizer_or_path)
        else:
            self.tokenizer = tokenizer_or_path

        # Get HuggingFace config
        hf_config = AutoConfig.from_pretrained(self.local_path, trust_remote_code=trust_remote_code)

        # Override config
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config.get("model_config", {}))
        self.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)
        update_model_config(hf_config, override_config_kwargs=override_config_kwargs)
        self.architectures = getattr(hf_config, "architectures", None)

        # Convert to Megatron config
        tf_config = hf_to_mcore_config(hf_config, dtype, **override_transformer_config)

        # Handle mbridge if needed
        if use_mbridge:
            try:
                from mbridge import AutoBridge
            except ImportError:
                logger.warning("mbridge package not found. Please install mbridge with `pip install verl[mcore]` or `pip install mbridge`")

            bridge = AutoBridge.from_config(hf_config)
            # Default activation recomputation config for memory optimization (ReferenceWorker)
            recompute_defaults = {
                "recompute_granularity": "full",
                "recompute_method": "uniform",
                "recompute_num_layers": 1,
            }
            merged_config = {**recompute_defaults, **override_transformer_config}
            bridge.set_extra_args(**merged_config)
            tf_config = bridge.config
            self.bridge = bridge
        else:
            self.bridge = None

        self.hf_config = hf_config
        self.tf_config = tf_config

    def _build_ref_model(self, model_path, override_model_config, override_transformer_config):
        from siirl.utils.megatron.megatron_utils import McoreModuleWrapperConfig, make_megatron_module

        self._init_hf_config_and_tf_config(
            model_path,
            model_path,
            self.dtype,
            override_model_config,
            override_transformer_config,
            self.actor_ref_config.model.trust_remote_code,
            self.actor_ref_config.actor.megatron.use_mbridge,
        )

        wrap_config = McoreModuleWrapperConfig(
            is_value_model=False,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            wrap_with_ddp=False,
            use_distributed_optimizer=self.actor_ref_config.ref.megatron.use_distributed_optimizer,
        )

        ref_module = make_megatron_module(
            wrap_config=wrap_config,
            tf_config=self.tf_config,
            hf_config=self.hf_config,
            bridge=self.bridge,
            override_model_config=override_model_config,
        )

        if self.actor_ref_config.ref.load_weight:
            assert self.actor_ref_config.actor.load_weight == self.actor_ref_config.ref.load_weight
            if self.bridge is not None:
                local_model_path = get_hf_model_path(self.actor_ref_config)
                self.bridge.load_weights(ref_module, local_model_path)
            else:
                load_megatron_gptmodel_weights(
                    self.actor_ref_config,
                    self.hf_config,
                    ref_module,
                    params_dtype=self.dtype,
                    is_value_model=False,
                )

        return ref_module, self.hf_config

    def init_model(self):
        override_model_config = self.actor_ref_config.model.override_config
        override_transformer_config = self.actor_ref_config.ref.megatron.override_transformer_config or OmegaConf.create()

        self.param_dtype = torch.bfloat16
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        self.ref_module, self.ref_model_config = self._build_ref_model(
            model_path=self.actor_ref_config.model.path,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
        )

        self.ref_policy = MegatronPPOActor(
            config=self.config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            actor_module=self.ref_module,
            actor_optimizer=None,
        )

        if self._ref_is_offload_param:
            offload_megatron_model_to_cpu(self.ref_module)

        get_torch_device().empty_cache()

    def compute_ref_log_prob(self, data: TensorDict):
        if self._ref_is_offload_param:
            load_megatron_model_to_gpu(self.ref_module, load_grad=False)

        micro_batch_size = self.actor_ref_config.ref.log_prob_micro_batch_size_per_gpu
        data["micro_batch_size"] = NonTensorData(micro_batch_size)
        data["temperature"] = NonTensorData(self.actor_ref_config.ref.temperature)
        data = data.to(get_device_id())

        output, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)

        data["ref_log_prob"] = output
        data = data.to("cpu")

        if self._ref_is_offload_param:
            offload_megatron_model_to_cpu(self.ref_module)

        return data


class CriticWorker:
    def __init__(self, config: SiiRLArguments):
        self.rank = 0
        self.hf_config = None
        self.tf_config = None
        self.bridge = None
        self.tokenizer = None
        self.processor = None
        self.architectures = None
        self.share_embeddings_and_output_weights = False
        self.config = config
        self.critic_config = config.critic

        self._is_offload_param = self.critic_config.megatron.param_offload
        self._is_offload_optimizer = self.critic_config.megatron.optimizer_offload

    def _init_hf_config_and_tf_config(
        self,
        model_path,
        tokenizer_or_path,
        dtype,
        override_model_config,
        override_transformer_config,
        trust_remote_code=False,
        use_mbridge=False,
    ):
        """Initialize HuggingFace and Transformer configs"""
        from transformers import AutoConfig

        from siirl.models.loader import load_tokenizer
        from siirl.models.mcore import hf_to_mcore_config
        from siirl.utils.model_utils.model import update_model_config

        self.local_path = model_path
        if tokenizer_or_path is None:
            self.tokenizer = load_tokenizer(path=model_path)
        elif isinstance(tokenizer_or_path, str):
            self.tokenizer = load_tokenizer(path=tokenizer_or_path)
        else:
            self.tokenizer = tokenizer_or_path

        # Get HuggingFace config
        hf_config = AutoConfig.from_pretrained(self.local_path, trust_remote_code=trust_remote_code)

        # Override config
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config.get("model_config", {}))
        self.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)
        update_model_config(hf_config, override_config_kwargs=override_config_kwargs)
        self.architectures = getattr(hf_config, "architectures", None)

        # Convert to Megatron config
        tf_config = hf_to_mcore_config(hf_config, dtype, **override_transformer_config)

        # Handle mbridge if needed
        if use_mbridge:
            try:
                from mbridge import AutoBridge
            except ImportError:
                logger.warning("mbridge package not found. Please install mbridge with `pip install verl[mcore]` or `pip install mbridge`")

            bridge = AutoBridge.from_config(hf_config)
            # Default activation recomputation config for memory optimization (CriticWorker)
            recompute_defaults = {
                "recompute_granularity": "full",
                "recompute_method": "uniform",
                "recompute_num_layers": 1,
            }
            merged_config = {**recompute_defaults, **override_transformer_config}
            bridge.set_extra_args(**merged_config)
            tf_config = bridge.config

            logger.warning("=" * 60)
            logger.warning("[Memory Optimization] CriticWorker TransformerConfig")
            logger.warning("=" * 60)
            logger.warning("[Recompute Config] <<<< CRITICAL >>>>")
            logger.warning(f"  recompute_granularity: {getattr(tf_config, 'recompute_granularity', 'NOT SET')}")
            logger.warning(f"  recompute_method: {getattr(tf_config, 'recompute_method', 'NOT SET')}")
            logger.warning(f"  recompute_num_layers: {getattr(tf_config, 'recompute_num_layers', 'NOT SET')}")
            logger.warning("=" * 60)

            self.bridge = bridge
        else:
            self.bridge = None

        self.hf_config = hf_config
        self.tf_config = tf_config

    def _build_critic_model_optimizer(
        self,
        model_path,
        optim_config,
        override_model_config,
        override_transformer_config,
        override_ddp_config,
    ):
        from siirl.engine.actor.optimizer import get_megatron_optimizer, get_megatron_optimizer_param_scheduler, init_megatron_optim_config
        from siirl.utils.megatron.megatron_utils import McoreModuleWrapperConfig, make_megatron_module

        self._init_hf_config_and_tf_config(
            model_path,
            model_path,
            self.dtype,
            override_model_config,
            override_transformer_config,
            self.critic_config.model.trust_remote_code,
            self.critic_config.megatron.use_mbridge,
        )

        wrap_config = McoreModuleWrapperConfig(
            is_value_model=True,
            share_embeddings_and_output_weights=False,
            wrap_with_ddp=True,
            use_distributed_optimizer=self.critic_config.megatron.use_distributed_optimizer,
        )

        critic_module = make_megatron_module(
            wrap_config=wrap_config,
            tf_config=self.tf_config,
            hf_config=self.hf_config,
            bridge=self.bridge,
            override_model_config=override_model_config,
            override_ddp_config=override_ddp_config,
        )

        if self.critic_config.load_weight:
            if self.bridge is not None:
                local_model_path = get_hf_model_path(self.critic_config)
                self.bridge.load_weights(critic_module, local_model_path)
            else:
                load_megatron_gptmodel_weights(
                    self.critic_config,
                    self.hf_config,
                    critic_module,
                    params_dtype=self.dtype,
                    is_value_model=True,
                )

        optim_config_megatron = init_megatron_optim_config(optim_config)
        critic_optimizer = get_megatron_optimizer(model=critic_module, config=optim_config_megatron)
        critic_optimizer_scheduler = get_megatron_optimizer_param_scheduler(optimizer=critic_optimizer, config=optim_config)

        get_torch_device().empty_cache()
        return (
            critic_module,
            critic_optimizer,
            critic_optimizer_scheduler,
            self.hf_config,
            optim_config,
        )

    def init_model(self):
        override_model_config = self.critic_config.model.override_config
        override_transformer_config = self.critic_config.megatron.override_transformer_config or OmegaConf.create()
        override_ddp_config = self.critic_config.megatron.override_ddp_config or OmegaConf.create()

        self.param_dtype = torch.bfloat16
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        (
            self.critic_module,
            self.critic_optimizer,
            self.critic_optimizer_scheduler,
            self.critic_model_config,
            critic_optimizer_config,
        ) = self._build_critic_model_optimizer(
            model_path=self.critic_config.model.path,
            optim_config=self.critic_config.optim,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
            override_ddp_config=override_ddp_config,
        )

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.critic_optimizer)

        self.critic = MegatronPPOCritic(
            config=self.config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            critic_module=self.critic_module,
            critic_optimizer=self.critic_optimizer,
            critic_optimizer_config=critic_optimizer_config,
        )

        # Initialize FlopsCounter for MFU calculation
        self.flops_counter = FlopsCounter(self.hf_config, forward_only=False)

        self.checkpoint_manager = MegatronCheckpointManager(
            model=self.critic_module,
            optimizer=self.critic_optimizer,
            lr_scheduler=self.critic_optimizer_scheduler,
        )

    def compute_values(self, data: TensorDict):
        micro_batch_size = self.critic_config.ppo_micro_batch_size_per_gpu
        data["micro_batch_size"] = NonTensorData(micro_batch_size)
        data = data.to(get_device_id())

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)

        values = self.critic.compute_values(data=data)
        data["values"] = values
        data = data.to("cpu")

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)

        return data

    def update_critic(self, data: TensorDict):
        data = data.to(get_device_id())

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_megatron_optimizer(self.critic_optimizer)

        # Time the update_critic call for MFU calculation
        with Timer("update_critic") as timer:
            metrics = self.critic.update_critic(data=data)
        delta_time = timer.elapsed

        # Calculate MFU (Model FLOPs Utilization)
        # Note: flops_counter calculates FLOPs for the entire model, but with TP each GPU
        # only computes 1/TP of the model FLOPs. So we need to divide by TP world size.
        if "global_token_num" in data:
            global_token_num = data["global_token_num"]
            if hasattr(global_token_num, "data"):
                global_token_num = global_token_num.data
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_token_num, delta_time)
            if promised_flops > 0:
                tp_world_size = mpu.get_tensor_model_parallel_world_size()
                metrics["perf/mfu/critic"] = estimated_flops / promised_flops / tp_world_size

        metrics["perf/delta_time/critic"] = delta_time

        data["metrics"] = NonTensorData(metrics)
        data = data.to("cpu")

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.critic_optimizer)

        return data

    def save_checkpoint(self, local_path, global_step=0, max_ckpt_to_keep=None):
        """Save critic checkpoint using Megatron distributed checkpointing."""
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
        )

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)

    def load_checkpoint(self, local_path):
        """Load critic checkpoint using Megatron distributed checkpointing."""
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)

        self.checkpoint_manager.load_checkpoint(local_path=local_path)

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)


class MegatronPPOActor:
    """Core PPO Actor implementation with Megatron backend"""

    def __init__(
        self,
        config: SiiRLArguments,
        hf_config,
        tf_config,
        actor_module: nn.ModuleList,
        actor_optimizer: DistributedOptimizer,
    ):
        self.config = config
        self.actor_config = config.actor_ref.actor
        self.hf_config = hf_config
        self.tf_config = tf_config
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

    def compute_log_prob(self, data: TensorDict, calculate_entropy=False):
        """Compute log probability and optionally entropy"""
        micro_batch_size = data["micro_batch_size"]

        assert micro_batch_size is not None

        entropys = torch.Tensor()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(*select_keys)
        input_ids = data["input_ids"]
        batch_size = input_ids.size(0)
        response = batch["responses"]
        temperature = data["temperature"]
        response_length = response.size(1)

        with torch.no_grad():
            output = self.forward_backward_batch(
                batch,
                temperature=temperature,
                forward_only=True,
                calculate_entropy=calculate_entropy,
                micro_batch_size=micro_batch_size,
            )

            def _unwrap_output_item(item):
                if isinstance(item, dict):
                    return item
                if isinstance(item, tuple):
                    for elem in item:
                        if isinstance(elem, dict):
                            return elem
                raise TypeError(f"Unexpected output item type: {type(item)}")

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                output_items = [_unwrap_output_item(o) for o in output["output"]]
                log_probs = [o["log_probs"] for o in output_items]
                log_probs = torch.cat(log_probs, dim=0).to(torch.float32)

                if calculate_entropy:
                    entropys = torch.cat([o["entropy"] for o in output_items], dim=0).to(torch.float32)

            else:
                log_probs = torch.empty(
                    size=(batch_size, response_length),
                    dtype=torch.float32,
                    device=input_ids.device,
                )
                if calculate_entropy:
                    entropys = torch.empty(
                        size=(batch_size, response_length),
                        dtype=torch.float32,
                        device=input_ids.device,
                    )

            # Broadcast across pipeline ranks
            log_probs = log_probs.to(get_device_id())
            torch.distributed.broadcast(
                tensor=log_probs,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=mpu.get_pipeline_model_parallel_group(),
                async_op=False,
            )
            log_probs = log_probs.to("cpu")

            if calculate_entropy:
                entropys = entropys.to(get_device_id())
                torch.distributed.broadcast(
                    tensor=entropys,
                    src=mpu.get_pipeline_model_parallel_last_rank(),
                    group=mpu.get_pipeline_model_parallel_group(),
                    async_op=False,
                )
                entropys = entropys.to("cpu")

        get_torch_device().empty_cache()
        return log_probs, entropys

    def compute_ppo_loss(self, model_output, data):
        """Compute PPO loss including policy gradient, entropy, and KL"""
        log_prob = model_output["log_probs"]
        entropy = model_output.get("entropy", None)
        metrics = {}

        response_mask = data["response_mask"].to(bool)
        old_log_prob = data["old_log_probs"]
        advantages = data["advantages"]
        loss_agg_mode = self.actor_config.loss_agg_mode
        loss_mode = self.actor_config.loss_mode

        # Policy gradient loss
        policy_loss_fn = get_policy_loss_fn(loss_mode)
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=self.actor_config,
        )

        metrics.update(
            {
                "actor/pg_loss": pg_loss.detach().item(),
                "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                "actor/ppo_kl": ppo_kl.detach().item(),
                "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
            }
        )
        policy_loss = pg_loss

        # Entropy loss
        if entropy is not None:
            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
            policy_loss -= self.actor_config.entropy_coeff * entropy_loss

        # KL loss
        if self.actor_config.use_kl_loss:
            ref_log_prob = data["ref_log_prob"]
            kld = kl_penalty(
                logprob=log_prob,
                ref_logprob=ref_log_prob,
                kl_penalty=self.actor_config.kl_loss_type,
            )
            kl_loss = agg_loss(
                loss_mat=kld,
                loss_mask=response_mask,
                loss_agg_mode=self.actor_config.loss_agg_mode,
            )
            policy_loss += kl_loss * self.actor_config.kl_loss_coef
            metrics["actor/kl_loss"] = kl_loss.detach().item()
            metrics["actor/kl_coef"] = self.actor_config.kl_loss_coef

        return policy_loss, metrics

    def forward_backward_batch(
        self,
        data: TensorDict,
        temperature: float,
        forward_only=False,
        calculate_entropy=False,
        micro_batch_size=None,
    ):
        """Execute forward-backward pass through pipeline parallel stages"""
        debug = os.environ.get("SIIRL_MEMORY_DEBUG", "0") == "1"

        # Log TransformerConfig settings once
        if debug and not hasattr(self, "_logged_tf_config"):
            self._logged_tf_config = True
            tf = self.tf_config
            logger.warning("=" * 60)
            logger.warning("[MemDebug] TransformerConfig Memory-Related Settings:")
            logger.warning("  recompute_granularity: {}", getattr(tf, "recompute_granularity", "N/A"))
            logger.warning("  recompute_method: {}", getattr(tf, "recompute_method", "N/A"))
            logger.warning("  recompute_num_layers: {}", getattr(tf, "recompute_num_layers", "N/A"))
            logger.warning("  sequence_parallel: {}", getattr(tf, "sequence_parallel", "N/A"))
            logger.warning("  num_layers: {}", getattr(tf, "num_layers", "N/A"))
            logger.warning("  hidden_size: {}", getattr(tf, "hidden_size", "N/A"))
            logger.warning("  num_attention_heads: {}", getattr(tf, "num_attention_heads", "N/A"))
            logger.warning("  context_parallel_size: {}", getattr(tf, "context_parallel_size", "N/A"))
            logger.warning("  tensor_model_parallel_size: {}", getattr(tf, "tensor_model_parallel_size", "N/A"))
            logger.warning("=" * 60)

        # Broadcast data across pipeline ranks
        data.to(get_device_id())
        data = data.contiguous()
        mini_batch = data
        broadcast_dict_tensor(
            mini_batch,
            src=mpu.get_pipeline_model_parallel_last_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
        )
        mini_batch.to("cpu")

        mini_batch["attention_mask"] = mini_batch["attention_mask"].to(bool)

        assert micro_batch_size is not None
        micro_batches = mini_batch.split(micro_batch_size)

        if debug:
            total_batch_size = data["input_ids"].shape[0]
            seq_len = data["input_ids"].shape[1]
            response_len = data["responses"].shape[1]
            logger.warning("[MemDebug] Batch Info:")
            logger.warning("  total_batch_size: {}", total_batch_size)
            logger.warning("  micro_batch_size: {}", micro_batch_size)
            logger.warning("  n_micro_batches: {}", len(micro_batches))
            logger.warning("  seq_len: {}", seq_len)
            logger.warning("  response_len: {}", response_len)
            logger.warning("  forward_only: {}", forward_only)

        n_micro_batch = len(micro_batches)
        forward_backward_func = get_forward_backward_func()

        def loss_func(output, data, non_loss_data=False):
            device = output["log_probs"].device
            responses = data["responses"]
            response_length = responses.size(1)

            # Check output format:
            # - Response-only: shape is [batch, response_length], use directly
            # - Packed/full: shape is [batch, seq_len], extract response part
            log_probs_shape = output["log_probs"].shape
            if log_probs_shape[-1] == response_length:
                # Response-only format: already [batch, response_length]
                log_prob = output["log_probs"].contiguous()
            else:
                # Full sequence format: [batch, seq_len], extract response part
                log_prob = output["log_probs"][:, -response_length - 1 : -1].contiguous()

            model_output = {"log_probs": log_prob}

            if calculate_entropy:
                entropy_shape = output["entropy"].shape
                if entropy_shape[-1] == response_length:
                    entropy = output["entropy"].contiguous()
                else:
                    entropy = output["entropy"][:, -response_length - 1 : -1].contiguous()
                model_output["entropy"] = entropy

            if forward_only or non_loss_data:
                return torch.tensor(1.0, device=device), model_output

            policy_loss, metrics = self.compute_ppo_loss(model_output, data)
            return policy_loss, metrics

        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            batch = batch.to(get_device_id())
            batch = batch.contiguous()

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"].to(bool)
            position_ids = batch["position_ids"]

            responses = batch["responses"]
            response_length = responses.size(1)
            label = position_ids.clone()
            label[:, -response_length - 1 : -1] = responses
            label_mask = attention_mask.clone()
            label_mask[:, : -response_length - 1] = False
            label_mask[:, -1] = False

            from siirl.models.mcore import get_mcore_forward_fn

            forward_fn = get_mcore_forward_fn(self.hf_config)
            if forward_only:
                try:
                    import torch._dynamo as _dynamo  # type: ignore

                    forward_fn = _dynamo.disable(forward_fn)
                except Exception:
                    pass

            # Create logits processor config from actor arguments
            actor_cfg = self.actor_config
            _logits_config = None
            from siirl.utils.logits_processing import LogitsProcessorConfig

            _logits_config = LogitsProcessorConfig.from_actor_config(
                chunk_size=getattr(actor_cfg, "logprob_chunk_size", 4096),
                use_checkpoint=getattr(actor_cfg, "use_logprob_checkpoint", True),
                use_fused=getattr(actor_cfg, "use_fused_logprob", False),
            )

            def logits_processor(
                logits, label, label_mask, _cu_seqlens=None, _attention_mask=None, _forward_only=None, _response_length=None
            ):
                """Memory-optimized logits processor."""
                from siirl.utils.logits_processing import process_batched_mode, process_fallback_mode, process_slice_mode

                # Apply temperature scaling
                logits.div_(temperature)
                packed_logits = logits.squeeze(0)  # [packed_len, vocab_size]
                packed_label = label.squeeze(0)  # [packed_len]

                # Use config from actor arguments (captured from outer scope)
                config = _logits_config

                # Disable dynamo for fused kernel compatibility
                import torch._dynamo as _dynamo

                orig_dynamo_disable = _dynamo.config.disable
                if config.use_fused:
                    _dynamo.config.disable = True
                    _dynamo.reset()

                try:
                    # Check if we can use response-only optimization
                    # Set SIIRL_LOGPROB_FALLBACK=1 to force fallback mode (useful for debugging)
                    use_fallback = os.environ.get("SIIRL_LOGPROB_FALLBACK", "0") == "1"
                    has_boundary_info = _cu_seqlens is not None and _attention_mask is not None and _response_length is not None

                    if use_fallback or not has_boundary_info:
                        # Fallback mode: Standard processing (matches original master)
                        return process_fallback_mode(
                            packed_logits=packed_logits,
                            packed_label=packed_label,
                            label_mask=label_mask,
                            calculate_entropy=calculate_entropy,
                            config=config,
                        )
                    elif _forward_only:
                        # SLICE mode: Zero-copy tensor views (inference only)
                        return process_slice_mode(
                            packed_logits=packed_logits,
                            packed_label=packed_label,
                            attention_mask=_attention_mask,
                            cu_seqlens=_cu_seqlens,
                            label_mask=label_mask,
                            response_length=_response_length,
                            calculate_entropy=calculate_entropy,
                            config=config,
                        )
                    else:
                        # BATCHED mode: Chunked cross entropy with gradient checkpointing
                        return process_batched_mode(
                            packed_logits=packed_logits,
                            packed_label=packed_label,
                            attention_mask=_attention_mask,
                            cu_seqlens=_cu_seqlens,
                            label_mask=label_mask,
                            response_length=_response_length,
                            calculate_entropy=calculate_entropy,
                            config=config,
                        )
                finally:
                    _dynamo.config.disable = orig_dynamo_disable

            try:
                import torch._dynamo as _dynamo  # type: ignore

                logits_processor = _dynamo.disable(logits_processor)
            except Exception:
                pass

            logits_processor_args = {
                "label": label,
                "label_mask": label_mask,
                "_forward_only": forward_only,
                "_response_length": response_length,
            }
            output = forward_fn(
                model,
                input_ids,
                attention_mask,
                position_ids,
                sequence_parallel=self.tf_config.sequence_parallel,
                logits_processor=logits_processor,
                logits_processor_args=logits_processor_args,
            )

            return output, partial(loss_func, data=batch)

        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))

        # Set model to eval mode for forward_only to disable dropout
        if forward_only:
            for model_chunk in self.actor_module:
                model_chunk.eval()

        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.actor_module,
            num_microbatches=n_micro_batch,
            seq_length=1,
            micro_batch_size=1,
            forward_only=forward_only,
            collect_non_loss_data=forward_only,  # Collect outputs without computing gradients
        )

        # Restore train mode after forward_only
        if forward_only:
            for model_chunk in self.actor_module:
                model_chunk.train()

        losses_reduced = {"output": losses_reduced}

        return losses_reduced

    def update_policy(self, data: TensorDict) -> dict:
        """Update policy using PPO algorithm"""
        metrics = {}
        temperature = data["temperature"]

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.actor_config.use_kl_loss:
            select_keys.append("ref_log_prob")

        batch = data.select(*select_keys)

        local_ppo_mini_batch_size = self.actor_config.ppo_mini_batch_size * self.config.rollout.n
        self.local_ppo_mini_batch_size = local_ppo_mini_batch_size // mpu.get_data_parallel_world_size(with_context_parallel=False)

        dataloader = batch.split(self.local_ppo_mini_batch_size)

        for data in dataloader:
            self.actor_optimizer.zero_grad()
            for chunk in self.actor_module:
                chunk.zero_grad_buffer()

            calculate_entropy = self.actor_config.entropy_coeff != 0
            micro_batch_size = data.get("micro_batch_size") or self.actor_config.ppo_micro_batch_size_per_gpu

            metric_micro_batch = self.forward_backward_batch(
                data,
                temperature=temperature,
                calculate_entropy=calculate_entropy,
                micro_batch_size=micro_batch_size,
            )

            metric_micro_batch = metric_micro_batch["output"]
            for metric in metric_micro_batch:
                append_to_dict(metrics, metric)

            update_successful, grad_norm, num_zeros_in_grad = self.actor_optimizer.step()
            learning_rate = self.actor_optimizer.param_groups[-1]["lr"]
            data = {"actor/grad_norm": grad_norm, "actor/lr": learning_rate}
            append_to_dict(metrics, data)

            if not update_successful:
                raise NotImplementedError

        get_torch_device().empty_cache()
        return metrics


class MegatronPPOCritic:
    """Core PPO Critic implementation with Megatron backend"""

    def __init__(
        self,
        config: SiiRLArguments,
        hf_config,
        tf_config,
        critic_module: nn.ModuleList,
        critic_optimizer: DistributedOptimizer,
        critic_optimizer_config,
    ):
        self.config = config
        self.critic_config = config.critic
        self.hf_config = hf_config
        self.tf_config = tf_config
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.critic_optimizer_config = critic_optimizer_config
        local_ppo_mini_batch_size = self.critic_config.ppo_mini_batch_size * config.rollout.n
        self.local_ppo_mini_batch_size = local_ppo_mini_batch_size // mpu.get_data_parallel_world_size(with_context_parallel=False)

    def compute_values(self, data: TensorDict):
        """Compute value predictions"""
        data.to(get_device_id())
        responses = data["responses"]
        micro_batch_size = data["micro_batch_size"]

        assert micro_batch_size is not None

        response_length = responses.size(1)

        with torch.no_grad():
            output = self.forward_backward_batch(data=data, forward_only=True, micro_batch_size=micro_batch_size)

            def _unwrap_output_item(item):
                if isinstance(item, dict):
                    return item
                if isinstance(item, tuple):
                    for elem in item:
                        if isinstance(elem, dict):
                            return elem
                raise TypeError(f"Unexpected output item type: {type(item)}")

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                output_items = [_unwrap_output_item(o) for o in output["output"]]
                values = [o["vpreds"] for o in output_items]
                values = torch.cat(values, dim=0).to(torch.float32)
            else:
                attention_mask = data["attention_mask"]
                values = torch.empty_like(attention_mask, dtype=torch.float32)

            values = values[:, -response_length - 1 : -1]
            response_mask = data["response_mask"]
            values = values * response_mask
            values = values.contiguous()

            torch.distributed.broadcast(
                tensor=values,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=mpu.get_pipeline_model_parallel_group(),
            )

        get_torch_device().empty_cache()
        return values

    def forward_backward_batch(self, data: TensorDict, forward_only=False, micro_batch_size=None):
        """Execute forward-backward pass for critic"""
        mini_batch = data
        mini_batch.to(get_device_id())
        mini_batch = mini_batch.contiguous()
        broadcast_dict_tensor(
            mini_batch,
            src=mpu.get_pipeline_model_parallel_last_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
        )

        mini_batch["attention_mask"] = mini_batch["attention_mask"].to(bool)

        assert micro_batch_size is not None
        micro_batches = mini_batch.split(micro_batch_size)
        seq_len = micro_batches[0]["input_ids"].shape[1]
        total_seqlen = micro_batch_size * seq_len

        n_micro_batch = len(micro_batches)
        forward_backward_func = get_forward_backward_func()

        def loss_func(output, data, non_loss_data=False):
            if forward_only or non_loss_data:
                return torch.tensor(1.0, device=output.device), {"vpreds": output}

            responses = data["responses"]
            values = data["values"]
            returns = data["returns"]
            response_length = responses.size(1)
            response_mask = data["response_mask"]
            cliprange_value = self.critic_config.cliprange_value

            vpreds = output[:, -response_length - 1 : -1]

            vf_loss, vf_clipfrac = compute_value_loss(
                vpreds=vpreds,
                values=values,
                returns=returns,
                response_mask=response_mask,
                cliprange_value=cliprange_value,
                loss_agg_mode=self.critic_config.loss_agg_mode,
            )

            stats = {
                "critic/vf_loss": vf_loss.detach().item(),
                "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
            }

            return vf_loss, stats

        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            position_ids = batch["position_ids"]

            from siirl.models.mcore import get_mcore_forward_fn

            forward_fn = get_mcore_forward_fn(self.hf_config)

            output = forward_fn(
                model,
                input_ids,
                attention_mask,
                position_ids,
                sequence_parallel=self.tf_config.sequence_parallel,
                value_model=True,
            )

            return output, partial(loss_func, data=batch)

        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.critic_module))

        # Set model to eval mode for forward_only to disable dropout
        if forward_only:
            for model_chunk in self.critic_module:
                model_chunk.eval()

        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.critic_module,
            num_microbatches=n_micro_batch,
            seq_length=total_seqlen,
            micro_batch_size=1,
            forward_only=forward_only,
            collect_non_loss_data=forward_only,  # Collect outputs without computing gradients
        )

        # Restore train mode after forward_only
        if forward_only:
            for model_chunk in self.critic_module:
                model_chunk.train()

        losses_reduced = {"output": losses_reduced}

        return losses_reduced

    def update_critic(self, data: TensorDict):
        """Update critic using value loss"""
        metrics = {}
        select_keys = [
            "input_ids",
            "responses",
            "attention_mask",
            "position_ids",
            "values",
            "returns",
            "response_mask",
        ]

        batch = data.select(*select_keys)
        dataloader = batch.split(self.local_ppo_mini_batch_size)

        for _ in range(self.critic_config.ppo_epochs):
            for _, data in enumerate(dataloader):
                self.critic_optimizer.zero_grad()
                for chunk in self.critic_module:
                    chunk.zero_grad_buffer()

                micro_batch_size = self.critic_config.ppo_micro_batch_size_per_gpu

                metric_micro_batch = self.forward_backward_batch(
                    data,
                    forward_only=False,
                    micro_batch_size=micro_batch_size,
                )

                metric_micro_batch = metric_micro_batch["output"]
                update_successful, grad_norm, num_zeros_in_grad = self.critic_optimizer.step()
                learning_rate = self.critic_optimizer.param_groups[-1]["lr"]

                data = {"critic/grad_norm": grad_norm, "critic/lr": learning_rate}
                append_to_dict(metrics, data)

                if not update_successful:
                    raise NotImplementedError

                for metric in metric_micro_batch:
                    append_to_dict(metrics, metric)

        get_torch_device().empty_cache()
        return metrics
