import os

import torch
from loguru import logger

from siirl.utils.megatron.megatron_utils import unwrap_model

from .util import postprocess_packed_seqs, preprocess_packed_seqs, recover_left_padding, remove_left_padding
from siirl.utils.backend.device import get_device_id, get_torch_device

def _log_memory(tag: str):
    """Log GPU memory usage at a specific point."""
    if os.environ.get("SIIRL_MEMORY_DEBUG", "0") != "1":
        return
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    peak_allocated = torch.cuda.max_memory_allocated() / (1024**3)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
    logger.warning(
        "[MemDebug] {} | Alloc: {:.2f}GB | Reserved: {:.2f}GB | Peak Alloc: {:.2f}GB | Peak Reserved: {:.2f}GB",
        tag,
        allocated,
        reserved,
        peak_allocated,
        peak_reserved,
    )


def _log_tensor(name: str, tensor):
    """Log tensor shape and memory usage."""
    if os.environ.get("SIIRL_MEMORY_DEBUG", "0") != "1":
        return
    if tensor is None:
        logger.warning("[TensorDebug] {} = None", name)
        return
    size_gb = tensor.numel() * tensor.element_size() / (1024**3)
    logger.warning(
        "[TensorDebug] {} | shape={} | dtype={} | size={:.3f}GB",
        name,
        tuple(tensor.shape),
        tensor.dtype,
        size_gb,
    )


def gptmodel_forward(
    model,
    input_ids,
    attention_mask,
    position_ids,
    sequence_parallel,
    value_model=False,
    pack_seqs=True,
    logits_processor=None,
    logits_processor_args: dict = None,
    **kwargs,
):
    """Default forward pass for GPT models with optional sequence packing."""
    debug = os.environ.get("SIIRL_MEMORY_DEBUG", "0") == "1"
    if debug:
        torch.cuda.reset_peak_memory_stats()
        logger.warning("=" * 60)
        logger.warning("[MemDebug] gptmodel_forward START")

    pre_process = unwrap_model(model).pre_process
    post_process = unwrap_model(model).post_process
    get_torch_device().empty_cache()

    if debug:
        _log_tensor("input_ids", input_ids)
        _log_tensor("attention_mask", attention_mask)
        _log_memory("Before model forward")

    if pack_seqs:
        batch_size, seq_len = attention_mask.shape[:2]
        input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=pre_process)
        input_ids_rmpad = input_ids_rmpad.contiguous()

        if debug:
            _log_tensor("input_ids_rmpad (packed)", input_ids_rmpad)
            packed_len = input_ids_rmpad.shape[1] if input_ids_rmpad.dim() > 1 else input_ids_rmpad.shape[0]
            logger.warning("[MemDebug] batch_size={} seq_len={} packed_len={}", batch_size, seq_len, packed_len)
            _log_memory("Before model()")

        output_orig = model(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
        )

        if debug:
            _log_memory("After model() - output_orig created")
            _log_tensor("output_orig (logits)", output_orig)

        if post_process and logits_processor is not None:
            # Separate special args (prefixed with _) from regular args that need packing
            regular_args = {k: v for k, v in logits_processor_args.items() if not k.startswith("_")}
            special_args = {k: v for k, v in logits_processor_args.items() if k.startswith("_")}
            args = {k: preprocess_packed_seqs(v, attention_mask, pre_process=True)[0] for k, v in regular_args.items()}
            # Pass boundary info for per-sample processing optimization
            args["_cu_seqlens"] = packed_seq_params.cu_seqlens_q_padded
            args["_attention_mask"] = attention_mask
            args.update(special_args)  # Add special args without preprocessing

            if debug:
                _log_memory("Before logits_processor()")

            output_dict = logits_processor(output_orig, **args)

            if debug:
                _log_memory("After logits_processor() - before del output_orig")
                for k, v in output_dict.items():
                    if isinstance(v, torch.Tensor):
                        _log_tensor(f"output_dict['{k}']", v)

            # Release output_orig immediately to free memory
            del output_orig
            torch.cuda.empty_cache()

            if debug:
                _log_memory("After del output_orig + empty_cache")

            # Check if logits_processor returned response-only format
            # If so, skip postprocess_packed_seqs (data is already [batch, response_len])
            is_response_only = output_dict.pop("_response_only", False)

            if is_response_only:
                # Response-only mode: output is already [batch, response_len], no unpack needed
                output = {k: v for k, v in output_dict.items() if isinstance(v, torch.Tensor)}
                if debug:
                    logger.warning("[MemDebug] Response-only mode: skipping postprocess_packed_seqs")
                    for k, v in output.items():
                        _log_tensor(f"output['{k}'] (response-only)", v)
            else:
                # Packed mode: need to unpack to [batch, seq_len]
                output = {}
                for k, v in output_dict.items():
                    if not isinstance(v, torch.Tensor):
                        continue
                    if debug:
                        _log_memory(f"Before postprocess_packed_seqs for '{k}'")
                    output[k] = postprocess_packed_seqs(
                        v,
                        packed_seq_params,
                        attention_mask,
                        batch_size,
                        seq_len,
                        post_process=post_process,
                    )
                    if debug:
                        _log_tensor(f"output['{k}'] (after postprocess)", output[k])
                        _log_memory(f"After postprocess_packed_seqs for '{k}'")

            # Release output_dict to free memory
            del output_dict
            torch.cuda.empty_cache()

            if debug:
                _log_memory("After del output_dict + empty_cache")
        else:
            output = postprocess_packed_seqs(
                output_orig,
                packed_seq_params,
                attention_mask,
                batch_size,
                seq_len,
                post_process=post_process,
            )
            if debug:
                _log_tensor("output (after postprocess)", output)
                _log_memory("After postprocess_packed_seqs")
    else:
        assert logits_processor is None, "logits_processor is not supported for non-packed sequence"
        batch_size, sequence_length = attention_mask.shape
        new_input_ids, new_attention_mask, new_position_ids = remove_left_padding(
            input_ids,
            attention_mask,
            position_ids,
            sequence_parallel,
            pre_process=pre_process,
        )
        output = model(
            input_ids=new_input_ids,
            attention_mask=new_attention_mask,
            position_ids=new_position_ids,
        )
        output = recover_left_padding(
            output,
            new_attention_mask,
            attention_mask,
            sequence_length,
            post_process=post_process,
        )
    if value_model and post_process:
        output = output[..., 0]

    if debug:
        _log_memory("gptmodel_forward END")
        logger.warning("=" * 60)

    return output
