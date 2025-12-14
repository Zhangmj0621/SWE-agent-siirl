# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, Infrawaves. All rights reserved.
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
"""Simplified Megatron PPO Actor/Critic implementation"""

import os
import datetime
from functools import partial

import torch
import torch.distributed
from torch import nn
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict, NonTensorData

from megatron.core import parallel_state as mpu
from megatron.core.optimizer import DistributedOptimizer
from megatron.core.pipeline_parallel import get_forward_backward_func

# Local simplified modules
from siirl.engine.actor.utils import (
    agg_loss, get_policy_loss_fn, kl_penalty, compute_value_loss,
    append_to_dict, set_random_seed,
)
from siirl.params.model_args import ActorRolloutRefArguments

# Utilities
from siirl.utils.backend.device import get_device_id, get_device_name, get_nccl_backend, get_torch_device
from siirl.utils.model_utils.model import get_hf_model_path, load_mcore_dist_weights, load_megatron_gptmodel_weights
from siirl.utils.model_utils.torch_dtypes import PrecisionType
from siirl.utils.model_utils.torch_functional import broadcast_dict_tensor, masked_mean
from siirl.utils.megatron.megatron_utils import (
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
)
from siirl.utils.megatron.pipeline_parallel import make_batch_generator
from siirl.utils.megatron.tensor_parallel import vocab_parallel_entropy, vocab_parallel_log_probs_from_logits



def global_initialize_model_parallel(config: ActorRolloutRefArguments):
    """Initialize Megatron model parallel groups"""
    megatron_config = config.actor.megatron

    rank = int(os.environ["LOCAL_RANK"])
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend=get_nccl_backend(),
            timeout=datetime.timedelta(seconds=600),
            init_method=os.environ.get("DIST_INIT_METHOD", None),
        )
    get_torch_device().set_device(rank)

    if megatron_config.sequence_parallel:
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=megatron_config.tensor_model_parallel_size,
        pipeline_model_parallel_size=megatron_config.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=megatron_config.virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_split_rank=None,
        use_sharp=False,
        context_parallel_size=megatron_config.context_parallel_size,
        expert_model_parallel_size=megatron_config.expert_model_parallel_size,
        expert_tensor_parallel_size=megatron_config.expert_tensor_parallel_size,
        nccl_communicator_config_path=None,
    )
    set_random_seed(seed=megatron_config.seed)


class ActorWorker:
    """Dedicated worker for actor training"""

    def __init__(self, config: DictConfig):
        assert isinstance(config, ActorRolloutRefArguments)
        # Initialize attributes from MegatronWorker
        self.rank = 0
        self.hf_config = None
        self.tf_config = None
        self.bridge = None
        self.tokenizer = None
        self.processor = None
        self.architectures = None
        self.share_embeddings_and_output_weights = False

        self.config = config
        global_initialize_model_parallel(self.config)

        # Normalize config
        self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
        self.config.actor.ppo_mini_batch_size //= mpu.get_data_parallel_world_size()
        if self.config.actor.ppo_micro_batch_size:
            self.config.actor.ppo_micro_batch_size //= mpu.get_data_parallel_world_size()
            self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size

        self._is_offload_param = self.config.actor.megatron.param_offload
        self._is_offload_grad = self.config.actor.megatron.grad_offload
        self._is_offload_optimizer = self.config.actor.megatron.optimizer_offload

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
        from siirl.models.mcore import hf_to_mcore_config
        from siirl.models.loader import load_tokenizer
        from siirl.engine.actor.utils import copy_to_local
        from siirl.utils.model_utils.model import update_model_config

        # Initialize tokenizer
        self.local_path = copy_to_local(model_path)
        if tokenizer_or_path is None:
            tokenizer_processor = load_tokenizer(path=self.local_path)
            self.tokenizer = tokenizer_processor["tokenizer"]
            self.processor = tokenizer_processor["processor"]
        elif isinstance(tokenizer_or_path, str):
            tokenizer_processor = load_tokenizer(path=copy_to_local(tokenizer_or_path))
            self.tokenizer = tokenizer_processor["tokenizer"]
            self.processor = tokenizer_processor["processor"]
        else:
            self.tokenizer = tokenizer_or_path
            self.processor = tokenizer_or_path

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
            from siirl.models.mcore.mbridge import AutoBridge

            bridge = AutoBridge.from_config(hf_config)
            bridge.set_extra_args(**override_transformer_config)
            tf_config = bridge.config
            self.bridge = bridge
        else:
            self.bridge = None

        self.hf_config = hf_config
        self.tf_config = tf_config

    def _build_actor_model_optimizer(self, model_path, optim_config, override_model_config,
                                     override_transformer_config, override_ddp_config):
        from siirl.utils.megatron.megatron_utils import init_megatron_optim_config
        from siirl.utils.megatron.megatron_utils import McoreModuleWrapperConfig, make_megatron_module
        from siirl.utils.megatron.optimizer import get_megatron_optimizer, get_megatron_optimizer_param_scheduler

        self._init_hf_config_and_tf_config(
            model_path, model_path, self.dtype, override_model_config,
            override_transformer_config, self.config.model.trust_remote_code,
            self.config.actor.megatron.use_mbridge,
        )

        wrap_config = McoreModuleWrapperConfig(
            is_value_model=False,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            wrap_with_ddp=True,
            use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
        )

        actor_module = make_megatron_module(
            wrap_config=wrap_config, tf_config=self.tf_config, hf_config=self.hf_config,
            bridge=self.bridge, override_model_config=override_model_config,
            override_ddp_config=override_ddp_config,
        )

        if self.config.actor.load_weight:
            if self.config.actor.megatron.use_dist_checkpointing:
                load_mcore_dist_weights(actor_module, self.config.actor.megatron.dist_checkpointing_path, is_value_model=False)
            else:
                if self.bridge is not None:
                    local_model_path = get_hf_model_path(self.config)
                    self.bridge.load_weights(actor_module, local_model_path)
                else:
                    load_megatron_gptmodel_weights(self.config, self.hf_config, actor_module, params_dtype=self.dtype, is_value_model=False)

        optim_megatron_config = init_megatron_optim_config(optim_config)
        actor_optimizer = get_megatron_optimizer(model=actor_module, config=optim_megatron_config)
        actor_optimizer_scheduler = get_megatron_optimizer_param_scheduler(optimizer=actor_optimizer, config=optim_config)

        return actor_module, actor_optimizer, actor_optimizer_scheduler, self.hf_config, optim_config

    def init_model(self):

        override_model_config = self.config.model.override_config
        override_transformer_config = self.config.actor.megatron.override_transformer_config or OmegaConf.create()
        override_ddp_config = self.config.actor.megatron.override_ddp_config or OmegaConf.create()

        self.param_dtype = torch.bfloat16
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        optim_config = self.config.actor.optim
        self.actor_module, self.actor_optimizer, self.actor_optimizer_scheduler, self.actor_model_config, self.actor_optim_config = \
            self._build_actor_model_optimizer(
                model_path=self.config.model.path,
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
            config=self.config.actor,
            model_config=self.actor_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            actor_module=self.actor_module,
            actor_optimizer=self.actor_optimizer,
        )
        get_torch_device().empty_cache()

    def update_actor(self, data: TensorDict):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
        if self._is_offload_optimizer:
            load_megatron_optimizer(self.actor_optimizer)

        data = data.to(get_device_name())
        micro_batch_size = self.config.actor.ppo_micro_batch_size_per_gpu
        data["micro_batch_size"] = NonTensorData(micro_batch_size)

        metrics = self.actor.update_policy(data=data)
        data["metrics"] = NonTensorData(metrics)
        data = data.to("cpu")

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)

        return data

    def compute_log_prob(self, data: TensorDict):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module, load_grad=False)

        data["micro_batch_size"] = NonTensorData(self.config.rollout.log_prob_micro_batch_size_per_gpu)
        data["max_token_len"] = NonTensorData(self.config.rollout.log_prob_max_token_len_per_gpu)
        data["temperature"] = NonTensorData(self.config.rollout.temperature)
        data = data.to(get_device_id())

        output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)

        data["old_log_probs"] = output
        data["entropys"] = entropys
        data = data.to("cpu")

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
        return data


class ReferenceWorker:
    """Dedicated worker for reference policy"""

    def __init__(self, config: DictConfig, process_group=None):
        assert isinstance(config, ActorRolloutRefArguments)
        # Initialize attributes from MegatronWorker
        self.rank = 0
        self.hf_config = None
        self.tf_config = None
        self.bridge = None
        self.tokenizer = None
        self.processor = None
        self.architectures = None
        self.share_embeddings_and_output_weights = False

        self.config = config
        global_initialize_model_parallel(self.config)

        # Normalize config
        if self.config.ref.log_prob_micro_batch_size:
            self.config.ref.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size
        else:
            assert self.config.ref.log_prob_micro_batch_size_per_gpu is not None

        self._ref_is_offload_param = self.config.ref.megatron.param_offload

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
        from siirl.models.mcore import hf_to_mcore_config
        from siirl.models.loader import load_tokenizer
        from siirl.engine.actor.utils import copy_to_local
        from siirl.utils.model_utils.model import update_model_config

        # Initialize tokenizer
        self.local_path = copy_to_local(model_path)
        if tokenizer_or_path is None:
            tokenizer_processor = load_tokenizer(path=self.local_path)
            self.tokenizer = tokenizer_processor["tokenizer"]
            self.processor = tokenizer_processor["processor"]
        elif isinstance(tokenizer_or_path, str):
            tokenizer_processor = load_tokenizer(path=copy_to_local(tokenizer_or_path))
            self.tokenizer = tokenizer_processor["tokenizer"]
            self.processor = tokenizer_processor["processor"]
        else:
            self.tokenizer = tokenizer_or_path
            self.processor = tokenizer_or_path

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
            from siirl.utils.backend.device import is_npu_available
            if is_npu_available:
                from siirl.engine.base_worker.megatron import npu_mbridge_patch
            from siirl.models.mcore.mbridge import AutoBridge

            bridge = AutoBridge.from_config(hf_config)
            bridge.set_extra_args(**override_transformer_config)
            tf_config = bridge.config
            self.bridge = bridge
        else:
            self.bridge = None

        self.hf_config = hf_config
        self.tf_config = tf_config

    def _build_ref_model(self, model_path, override_model_config, override_transformer_config):
        from siirl.utils.megatron.megatron_utils import McoreModuleWrapperConfig, make_megatron_module

        self._init_hf_config_and_tf_config(
            model_path, model_path, self.dtype, override_model_config,
            override_transformer_config, self.config.model.trust_remote_code,
            self.config.actor.megatron.use_mbridge,
        )

        wrap_config = McoreModuleWrapperConfig(
            is_value_model=False,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            wrap_with_ddp=False,
            use_distributed_optimizer=self.config.ref.megatron.use_distributed_optimizer,
        )

        ref_module = make_megatron_module(
            wrap_config=wrap_config, tf_config=self.tf_config, hf_config=self.hf_config,
            bridge=self.bridge, override_model_config=override_model_config,
        )

        if self.config.ref.load_weight:
            assert self.config.actor.load_weight == self.config.ref.load_weight
            if self.config.ref.megatron.use_dist_checkpointing:
                load_mcore_dist_weights(ref_module, self.config.ref.megatron.dist_checkpointing_path, is_value_model=False)
            else:
                if self.bridge is not None:
                    local_model_path = get_hf_model_path(self.config)
                    self.bridge.load_weights(ref_module, local_model_path)
                else:
                    load_megatron_gptmodel_weights(self.config, self.hf_config, ref_module, params_dtype=self.dtype, is_value_model=False)

        return ref_module, self.hf_config

    def init_model(self):

        override_model_config = self.config.model.override_config
        override_transformer_config = self.config.ref.megatron.override_transformer_config or OmegaConf.create()

        self.param_dtype = torch.bfloat16
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        self.ref_module, self.ref_model_config = self._build_ref_model(
            model_path=self.config.model.path,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
        )

        self.ref_policy = MegatronPPOActor(
            config=self.config.ref,
            model_config=self.ref_model_config,
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

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data["micro_batch_size"] = NonTensorData(micro_batch_size)
        data["max_token_len"] = NonTensorData(self.config.ref.log_prob_max_token_len_per_gpu)
        data["temperature"] = NonTensorData(self.config.rollout.temperature)
        data = data.to(get_device_id())

        output, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)

        data["ref_log_prob"] = output
        data = data.to("cpu")

        if self._ref_is_offload_param:
            offload_megatron_model_to_cpu(self.ref_module)

        return data


class CriticWorker:
    """Dedicated worker for critic training"""

    def __init__(self, config, process_group=None):
        # Initialize attributes from MegatronWorker
        self.rank = 0
        self.hf_config = None
        self.tf_config = None
        self.bridge = None
        self.tokenizer = None
        self.processor = None
        self.architectures = None
        self.share_embeddings_and_output_weights = False

        self.config = config

        if not torch.distributed.is_initialized():
            rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(backend=get_nccl_backend())
            get_torch_device().set_device(rank)

            if self.config.megatron.sequence_parallel:
                os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.megatron.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=self.config.megatron.context_parallel_size,
                expert_model_parallel_size=self.config.megatron.expert_model_parallel_size,
                expert_tensor_parallel_size=self.config.megatron.expert_tensor_parallel_size,
                nccl_communicator_config_path=None,
            )

        set_random_seed(seed=self.config.megatron.seed)

        self._is_offload_param = self.config.megatron.param_offload
        self._is_offload_optimizer = self.config.megatron.optimizer_offload

        # Normalize config
        self.config.ppo_mini_batch_size *= self.config.rollout_n
        self.config.ppo_mini_batch_size //= mpu.get_data_parallel_world_size()
        if self.config.ppo_micro_batch_size:
            self.config.ppo_micro_batch_size //= mpu.get_data_parallel_world_size()
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size

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
        from siirl.models.mcore import hf_to_mcore_config
        from siirl.models.loader import load_tokenizer
        from siirl.engine.actor.utils import copy_to_local
        from siirl.utils.model_utils.model import update_model_config

        # Initialize tokenizer
        self.local_path = copy_to_local(model_path)
        if tokenizer_or_path is None:
            tokenizer_processor = load_tokenizer(path=self.local_path)
            self.tokenizer = tokenizer_processor["tokenizer"]
            self.processor = tokenizer_processor["processor"]
        elif isinstance(tokenizer_or_path, str):
            tokenizer_processor = load_tokenizer(path=copy_to_local(tokenizer_or_path))
            self.tokenizer = tokenizer_processor["tokenizer"]
            self.processor = tokenizer_processor["processor"]
        else:
            self.tokenizer = tokenizer_or_path
            self.processor = tokenizer_or_path

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
            from siirl.utils.backend.device import is_npu_available
            if is_npu_available:
                from siirl.engine.base_worker.megatron import npu_mbridge_patch
            from siirl.models.mcore.mbridge import AutoBridge

            bridge = AutoBridge.from_config(hf_config)
            bridge.set_extra_args(**override_transformer_config)
            tf_config = bridge.config
            self.bridge = bridge
        else:
            self.bridge = None

        self.hf_config = hf_config
        self.tf_config = tf_config

    def _build_critic_model_optimizer(self, model_path, optim_config, override_model_config,
                                      override_transformer_config, override_ddp_config):
        from siirl.utils.megatron.optimizer import get_megatron_optimizer, get_megatron_optimizer_param_scheduler
        from siirl.utils.megatron.megatron_utils import init_megatron_optim_config, McoreModuleWrapperConfig, make_megatron_module

        self._init_hf_config_and_tf_config(
            model_path, model_path, self.dtype, override_model_config,
            override_transformer_config, self.config.model.trust_remote_code,
            self.config.megatron.use_mbridge,
        )

        wrap_config = McoreModuleWrapperConfig(
            is_value_model=True,
            share_embeddings_and_output_weights=False,
            wrap_with_ddp=True,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
        )

        critic_module = make_megatron_module(
            wrap_config=wrap_config, tf_config=self.tf_config, hf_config=self.hf_config,
            bridge=self.bridge, override_model_config=override_model_config,
            override_ddp_config=override_ddp_config,
        )

        if self.config.load_weight:
            if self.config.megatron.use_dist_checkpointing:
                load_mcore_dist_weights(critic_module, self.config.megatron.dist_checkpointing_path, is_value_model=True)
            else:
                if self.bridge is not None:
                    local_model_path = get_hf_model_path(self.config)
                    self.bridge.load_weights(critic_module, local_model_path)
                else:
                    load_megatron_gptmodel_weights(self.config, self.hf_config, critic_module, params_dtype=self.dtype, is_value_model=True)

        optim_config_megatron = init_megatron_optim_config(optim_config)
        critic_optimizer = get_megatron_optimizer(model=critic_module, config=optim_config_megatron)
        critic_optimizer_scheduler = get_megatron_optimizer_param_scheduler(optimizer=critic_optimizer, config=optim_config)

        get_torch_device().empty_cache()
        return critic_module, critic_optimizer, critic_optimizer_scheduler, self.hf_config, optim_config

    def init_model(self):

        override_model_config = self.config.model.override_config
        override_transformer_config = self.config.megatron.override_transformer_config or OmegaConf.create()
        override_ddp_config = self.config.megatron.override_ddp_config or OmegaConf.create()

        self.param_dtype = torch.bfloat16
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        self.critic_module, self.critic_optimizer, self.critic_optimizer_scheduler, self.critic_model_config, critic_optimizer_config = \
            self._build_critic_model_optimizer(
                model_path=self.config.model.path,
                optim_config=self.config.optim,
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
            model_config=self.critic_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            critic_module=self.critic_module,
            critic_optimizer=self.critic_optimizer,
            critic_optimizer_config=critic_optimizer_config,
        )

    def compute_values(self, data: TensorDict):
        micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
        data["micro_batch_size"] = NonTensorData(micro_batch_size)
        data["max_token_len"] = NonTensorData(self.config.forward_max_token_len_per_gpu)
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

        metrics = self.critic.update_critic(data=data)
        data["metrics"] = NonTensorData(metrics)
        data = data.to("cpu")

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.critic_optimizer)

        return data


class MegatronPPOActor():
    """Core PPO Actor implementation with Megatron backend"""

    def __init__(self, config, model_config, hf_config, tf_config,
                 actor_module: nn.ModuleList, actor_optimizer: DistributedOptimizer):
        super().__init__(config)
        self._validate_config(config)
        self.model_config = model_config
        self.hf_config = hf_config
        self.tf_config = tf_config
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

    def _validate_config(self, config):
        if config.shuffle:
            assert config.data_loader_seed is not None
        if config.megatron.tensor_model_parallel_size == 1:
            config.megatron.sequence_parallel = False
        self.config = config

    def compute_log_prob(self, data: TensorDict, calculate_entropy=False):
        """Compute log probability and optionally entropy"""
        micro_batch_size = data["micro_batch_size"]
        max_token_len = data["max_token_len"]

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
                batch, temperature=temperature, forward_only=True,
                calculate_entropy=calculate_entropy, micro_batch_size=micro_batch_size, 
                max_token_len=max_token_len,
            )

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                log_probs = [o["log_probs"] for o in output["output"]]
                log_probs = torch.cat(log_probs, dim=0).to(torch.float32)

                if calculate_entropy:
                    entropys = torch.cat([o["entropy"] for o in output["output"]], dim=0).to(torch.float32)

            else:
                log_probs = torch.empty(size=(batch_size, response_length), dtype=torch.float32, device=input_ids.device)
                if calculate_entropy:
                    entropys = torch.empty(size=(batch_size, response_length), dtype=torch.float32, device=input_ids.device)

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
        loss_agg_mode = self.config.loss_agg_mode
        loss_mode = self.config.policy_loss.loss_mode

        # Policy gradient loss
        policy_loss_fn = get_policy_loss_fn(loss_mode)
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
            old_log_prob=old_log_prob, log_prob=log_prob, advantages=advantages,
            response_mask=response_mask, loss_agg_mode=loss_agg_mode, config=self.config,
        )

        metrics.update({
            "actor/pg_loss": pg_loss.detach().item(),
            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
            "actor/ppo_kl": ppo_kl.detach().item(),
            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        })
        policy_loss = pg_loss

        # Entropy loss
        if entropy is not None:
            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
            policy_loss -= self.config.entropy_coeff * entropy_loss

        # KL loss
        if self.config.use_kl_loss:
            ref_log_prob = data["ref_log_prob"]
            kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)
            policy_loss += kl_loss * self.config.kl_loss_coef
            metrics["actor/kl_loss"] = kl_loss.detach().item()
            metrics["actor/kl_coef"] = self.config.kl_loss_coef

        return policy_loss, metrics

    def forward_backward_batch(self, data: TensorDict, temperature: float, forward_only=False,
                               calculate_entropy=False, micro_batch_size=None):
        """Execute forward-backward pass through pipeline parallel stages"""
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

        n_micro_batch = len(micro_batches)
        forward_backward_func = get_forward_backward_func()

        def loss_func(output, data):
            device = output["log_probs"].device
            responses = data["responses"]
            response_length = responses.size(1)

            log_prob = output["log_probs"][:, -response_length - 1 : -1].contiguous()
            model_output = {"log_probs": log_prob}

            if calculate_entropy:
                entropy = output["entropy"][:, -response_length - 1 : -1].contiguous()
                model_output["entropy"] = entropy

            if forward_only:
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

            multi_modal_inputs = {}
            if "multi_modal_inputs" in batch:
                for key in batch["multi_modal_inputs"][0].keys():
                    idxs = batch["multi_modal_inputs_idx"]
                    mmi = batch["multi_modal_inputs"]
                    multi_modal_inputs[key] = torch.cat(
                        [mmi[idx].get(key) for idx in idxs if mmi[idx].get(key) is not None], dim=0
                    )

            responses = batch["responses"]
            response_length = responses.size(1)
            label = position_ids.clone()
            label[:, -response_length - 1 : -1] = responses
            label_mask = attention_mask.clone()
            label_mask[:, : -response_length - 1] = False
            label_mask[:, -1] = False

            from siirl.models.mcore import get_mcore_forward_fn

            forward_fn = get_mcore_forward_fn(self.hf_config)

            def logits_processor(logits, label, label_mask):
                logits.div_(temperature)
                ret = {}
                if calculate_entropy:
                    logits_bak = logits.clone()
                    entropy = vocab_parallel_entropy(logits)
                    ret["entropy"] = entropy
                else:
                    logits_bak = logits
                log_probs = vocab_parallel_log_probs_from_logits(logits_bak, label)
                log_probs = log_probs.masked_fill(~label_mask, 0.0)
                ret["log_probs"] = log_probs
                return ret

            logits_processor_args = {"label": label, "label_mask": label_mask}
            output = forward_fn(
                model, input_ids, attention_mask, position_ids,
                sequence_parallel=self.tf_config.sequence_parallel,
                multi_modal_inputs=multi_modal_inputs,
                logits_processor=logits_processor,
                logits_processor_args=logits_processor_args,
            )

            return output, partial(loss_func, data=batch)

        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))

        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.actor_module,
            num_microbatches=n_micro_batch,
            seq_length=1,
            micro_batch_size=1,
            forward_only=forward_only,
        )

        losses_reduced = {"output": losses_reduced}

        return losses_reduced

    def update_policy(self, data: TensorDict) -> dict:
        """Update policy using PPO algorithm"""
        metrics = {}
        temperature = data["temperature"]

        select_keys = ["responses", "response_mask", "input_ids", "attention_mask",
                      "position_ids", "old_log_probs", "advantages"]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        batch = data.select(*select_keys)

        dataloader = batch.split(self.config.ppo_mini_batch_size)

        for data in dataloader:
            self.actor_optimizer.zero_grad()
            for chunk in self.actor_module:
                chunk.zero_grad_buffer()

            calculate_entropy = self.config.entropy_coeff != 0
            micro_batch_size = data.get("micro_batch_size") or self.config.ppo_micro_batch_size_per_gpu
            max_token_len = None

            metric_micro_batch = self.forward_backward_batch(
                data, temperature=temperature, calculate_entropy=calculate_entropy,
                micro_batch_size=micro_batch_size, max_token_len=max_token_len,
            )

            metric_micro_batch = metric_micro_batch["output"]
            for metric in metric_micro_batch:
                append_to_dict(metrics, metric)

            update_successful, grad_norm, num_zeros_in_grad = self.actor_optimizer.step()
            data = {"actor/grad_norm": grad_norm}
            append_to_dict(metrics, data)

            if not update_successful:
                raise NotImplementedError

        get_torch_device().empty_cache()
        return metrics


class MegatronPPOCritic():
    """Core PPO Critic implementation with Megatron backend"""

    def __init__(self, config, model_config, hf_config, tf_config,
                 critic_module: nn.ModuleList, critic_optimizer: DistributedOptimizer,
                 critic_optimizer_config):
        super().__init__(config=config)
        self._validate_config(config)
        self.model_config = model_config
        self.hf_config = hf_config
        self.tf_config = tf_config
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.critic_optimizer_config = critic_optimizer_config

    def _validate_config(self, config):
        if config.shuffle:
            assert config.data_loader_seed is not None
        if config.megatron.tensor_model_parallel_size == 1:
            config.megatron.sequence_parallel = False
        self.config = config

    def compute_values(self, data: TensorDict):
        """Compute value predictions"""
        data.to(get_device_id())
        responses = data["responses"]
        micro_batch_size = data["micro_batch_size"]
        max_token_len = data["max_token_len"]

        assert micro_batch_size is not None

        response_length = responses.size(1)

        with torch.no_grad():
            output = self.forward_backward_batch(
                data=data, forward_only=True,
                micro_batch_size=micro_batch_size, max_token_len=max_token_len, mini_batch_size=None
            )

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                values = [o["vpreds"] for o in output["output"]]
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
            group=mpu.get_pipeline_model_parallel_group()
        )

        mini_batch["attention_mask"] = mini_batch["attention_mask"].to(bool)

        assert micro_batch_size is not None
        micro_batches = mini_batch.split(micro_batch_size)
        seq_len = micro_batches[0]["input_ids"].shape[1]
        total_seqlen = micro_batch_size * seq_len

        n_micro_batch = len(micro_batches)
        forward_backward_func = get_forward_backward_func()

        def loss_func(output, data):
            if forward_only:
                return torch.tensor(1.0, device=output.device), {"vpreds": output}

            responses = data["responses"]
            values = data["values"]
            returns = data["returns"]
            response_length = responses.size(1)
            response_mask = data["response_mask"]
            cliprange_value = self.config.cliprange_value

            vpreds = output[:, -response_length - 1 : -1]

            vf_loss, vf_clipfrac = compute_value_loss(
                vpreds=vpreds, values=values, returns=returns,
                response_mask=response_mask, cliprange_value=cliprange_value,
                loss_agg_mode=self.config.loss_agg_mode,
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
                model, input_ids, attention_mask, position_ids,
                sequence_parallel=self.tf_config.sequence_parallel,
                value_model=True,
            )

            return output, partial(loss_func, data=batch, meta_info={})

        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.critic_module))

        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.critic_module,
            num_microbatches=n_micro_batch,
            seq_length=total_seqlen,
            micro_batch_size=1,
            forward_only=forward_only,
        )

        losses_reduced = {"output": losses_reduced}

        return losses_reduced

    def update_critic(self, data: TensorDict):
        """Update critic using value loss"""
        metrics = {}
        select_keys = ["input_ids", "responses", "attention_mask", "position_ids",
                      "values", "returns", "response_mask"]

        batch = data.select(*select_keys)
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                self.critic_optimizer.zero_grad()
                for chunk in self.critic_module:
                    chunk.zero_grad_buffer()

                micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
                max_token_len = None

                metric_micro_batch = self.forward_backward_batch(
                    data, forward_only=False, micro_batch_size=micro_batch_size, 
                    max_token_len=max_token_len, mini_batch_size=self.config.ppo_mini_batch_size
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
