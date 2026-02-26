from collections.abc import Sequence

import torch
from loguru import logger
from mbridge.core.bridge import Bridge
from mbridge.core.util import unwrap_model

COLOCATE_MEM_DEBUG_PREFIX = "[COLOCATE_MEM_DEBUG]"


def _cuda_export_snapshot() -> dict[str, float | int | str]:
    snapshot: dict[str, float | int | str] = {}
    if not torch.cuda.is_available():
        snapshot["cuda_available"] = 0
        return snapshot
    try:
        device = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        snapshot.update(
            {
                "cuda_available": 1,
                "device": int(device),
                "allocated_gb": round(torch.cuda.memory_allocated(device) / (1024**3), 3),
                "reserved_gb": round(torch.cuda.memory_reserved(device) / (1024**3), 3),
                "free_gb": round(free_bytes / (1024**3), 3),
                "total_gb": round(total_bytes / (1024**3), 3),
            }
        )
    except Exception as e:
        snapshot["snapshot_error"] = repr(e)
    return snapshot


def _log_export_mem(stage: str, **fields):
    merged = {
        "stage": stage,
        **_cuda_export_snapshot(),
        **fields,
    }
    payload = " ".join(f"{k}={v}" for k, v in merged.items())
    logger.info(f"{COLOCATE_MEM_DEBUG_PREFIX} {payload}")


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
            extra_keys = [x for x in model.state_dict() if "_extra_state" not in x and "expert_bias" in x and x not in existing_keys]
            for name in extra_keys:
                yield name, model.state_dict()[name].to(torch.cuda.current_device())

    weights_names = []
    for vpp_rank, model in enumerate(models):
        existing_keys = set()
        for name, _ in model.named_parameters():
            existing_keys.add(name)
            weights_names.append((self.mpu.pp_rank, vpp_rank, name))
        extra_keys = [x for x in model.state_dict() if "_extra_state" not in x and "expert_bias" in x and x not in existing_keys]
        for name in extra_keys:
            weights_names.append((self.mpu.pp_rank, vpp_rank, name))

    model_chunk_generator = get_model_chunk_generator()
    local_to_global_maps = [self._weight_name_mapping_mcore_local_to_global(model, consider_ep=False) for model in models]
    emitted_count = 0
    emitted_bytes = 0
    _log_export_mem(stage="mbridge_export_start", weight_name_count=len(weights_names))

    for idx, (_, iter_vpp_rank, iter_name) in enumerate(weights_names, start=1):
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
            global_expert_ids = [num_experts_per_rank * ep_rank + local_expert_id for ep_rank in range(self.mpu.ep_size)]
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
                for converted_name, converted_param in zip(converted_names, converted_params, strict=False):
                    emitted_count += 1
                    emitted_bytes += converted_param.numel() * converted_param.element_size()
                    if emitted_count % 256 == 0:
                        _log_export_mem(
                            stage="mbridge_export_progress",
                            source_index=idx,
                            emitted_count=emitted_count,
                            emitted_mb=round(emitted_bytes / (1024**2), 2),
                        )
                    yield converted_name, converted_param
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
        for converted_name, converted_param in zip(converted_names, converted_params, strict=False):
            emitted_count += 1
            emitted_bytes += converted_param.numel() * converted_param.element_size()
            if emitted_count % 256 == 0:
                _log_export_mem(
                    stage="mbridge_export_progress",
                    source_index=idx,
                    emitted_count=emitted_count,
                    emitted_mb=round(emitted_bytes / (1024**2), 2),
                )
            yield converted_name, converted_param

    _log_export_mem(
        stage="mbridge_export_done",
        emitted_count=emitted_count,
        emitted_mb=round(emitted_bytes / (1024**2), 2),
        weight_name_count=len(weights_names),
    )


logger.debug("patching mbridge with _export_weights_in_current_pipeline_stage")
Bridge._export_weights_in_current_pipeline_stage = _export_weights_in_current_pipeline_stage
