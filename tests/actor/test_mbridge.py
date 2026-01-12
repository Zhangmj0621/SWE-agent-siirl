import datetime
import os
from collections.abc import Sequence

import torch
from mbridge import AutoBridge
from megatron.core import parallel_state as mpu


def global_initialize_model_parallel():
    """Initialize Megatron model parallel groups"""

    rank = int(os.environ["LOCAL_RANK"])
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(seconds=600),
            init_method=os.environ.get("DIST_INIT_METHOD", None),
        )
        torch.cuda.set_device(rank)

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=1,
            pipeline_model_parallel_split_rank=None,
            use_sharp=False,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
            nccl_communicator_config_path=None,
        )
        from megatron.core import tensor_parallel

        tensor_parallel.model_parallel_cuda_manual_seed(1)


global_initialize_model_parallel()
from mbridge.core.bridge import Bridge  # noqa: E402
from mbridge.core.util import unwrap_model  # noqa: E402
from megatron.core import mpu as mpu_core  # noqa: E402, F811


def _named_params_and_buffers_global_in_current_pp_stage(self: Bridge, models: Sequence[torch.nn.Module]):
    models = [unwrap_model(model) for model in models]

    def get_model_chunk_generator():
        for model in models:
            existing_keys = set()
            for name, param in model.named_parameters():
                existing_keys.add(name)
                yield name, param

            # note
            # there is a bug in megatron GPTModel
            # decoder.layers[n].mlp.router.expert_bias" in GPTModel is not registered in named_parameter, but in state_dict().
            # for now we patch it by adding those keys to extra_keys.
            extra_keys = [
                x
                for x in model.state_dict().keys()
                if "_extra_state" not in x and "expert_bias" in x and x not in existing_keys
            ]
            for name in extra_keys:
                yield name, model.state_dict()[name].to(torch.cuda.current_device())

    weights_names = []
    for vpp_rank, model in enumerate(models):
        existing_keys = set()
        for name, param in model.named_parameters():
            existing_keys.add(name)
            weights_names.append((self.mpu.pp_rank, vpp_rank, name))
        extra_keys = [
            x
            for x in model.state_dict().keys()
            if "_extra_state" not in x and "expert_bias" in x and x not in existing_keys
        ]
        for name in extra_keys:
            weights_names.append((self.mpu.pp_rank, vpp_rank, name))

    # weights_names_all_pp = weights_names
    # torch.distributed.all_gather_object(
    #     object_list=weights_names_all_pp, obj=weights_names, group=self.mpu.pp_group
    # )
    # weights_names = sum(weights_names, [])
    model_chunk_generator = get_model_chunk_generator()
    local_to_global_maps = [
        self._weight_name_mapping_mcore_local_to_global(model, consider_ep=False) for model in models
    ]
    for iter_pp_rank, iter_vpp_rank, iter_name in weights_names:
        local_to_global_map = local_to_global_maps[iter_vpp_rank]
        try:
            name, param = next(model_chunk_generator)
        except StopIteration:
            name, param = None, None
        name = local_to_global_map[iter_name]

        # name = broadcast_str_from_megatron_pp(name)
        # broad_pp_param = broadcast_from_megatron_pp(param)

        # EP
        if ".mlp.experts.linear_fc" in name and self.mpu.ep_size > 1:
            num_experts = self.config.num_moe_experts
            num_experts_per_rank = num_experts // self.mpu.ep_size
            infer_params = [torch.empty_like(param) for _ in range(self.mpu.ep_size)]
            torch.distributed.all_gather(infer_params, param, group=self.mpu.ep_group)

            name_prefix, local_expert_id = name.split(".weight")
            local_expert_id = int(local_expert_id)
            global_expert_ids = [
                num_experts_per_rank * ep_rank + local_expert_id for ep_rank in range(self.mpu.ep_size)
            ]
            global_expert_names = [f"{name_prefix}.weight{expert_id}" for expert_id in global_expert_ids]

            for name, param in zip(global_expert_names, infer_params, strict=False):
                if self.mpu.etp_size > 1:
                    # gather etp
                    etp_params = [torch.empty_like(param) for _ in range(self.mpu.etp_size)]
                    torch.distributed.all_gather(etp_params, param, group=self.mpu.etp_group)
                    params = etp_params
                else:
                    params = [param]

                merge_params = self._weight_merge_across_tp(name, params, param)
                converted_names, converted_params = self._weight_to_hf_format(name, merge_params)
                yield from zip(converted_names, converted_params, strict=False)
            continue

        # TP
        if hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel:
            # allocate a new tensor with proper size
            if self.mpu.tp_size <= 1:
                infer_params = [param]
            else:
                infer_params = [torch.empty_like(param) for _ in range(self.mpu.tp_size)]
                torch.distributed.all_gather(infer_params, param, group=self.mpu.tp_group)
            infer_params = self._weight_merge_across_tp(name, infer_params, param)
        else:
            infer_params = param

        converted_names, converted_params = self._weight_to_hf_format(name, infer_params)

        yield from zip(converted_names, converted_params, strict=False)


Bridge.export_weights = _named_params_and_buffers_global_in_current_pp_stage


path = "/inspire/ssd/project/qianghuaxuexi/public/debug_models/Qwen3-0.6B"
bridge = AutoBridge.from_pretrained(path)
model = bridge.get_model(path, wrap_with_ddp=True, bf16=True)
generator = bridge.export_weights(model)
rank = int(os.environ["LOCAL_RANK"])
for key, param in generator:
    if mpu_core.get_tensor_model_parallel_rank() == 0:
        print(f"pp rank{mpu_core.get_pipeline_model_parallel_rank()}: {key}: {param.size()} {param.dtype}")
