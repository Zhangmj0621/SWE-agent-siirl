from collections.abc import Sequence

import torch
from loguru import logger
from mbridge.core.bridge import Bridge
from mbridge.core.util import unwrap_model


def _export_weights_in_current_pipeline_stage(self: Bridge, models: Sequence[torch.nn.Module]):
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


logger.debug("patching mbridge with _export_weights_in_current_pipeline_stage")
Bridge._export_weights_in_current_pipeline_stage = _export_weights_in_current_pipeline_stage
