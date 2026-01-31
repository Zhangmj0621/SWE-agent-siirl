"""GPT model forward pass with memory optimization."""

import os

import torch

from siirl.utils.megatron.megatron_utils import unwrap_model

from .util import postprocess_packed_seqs, preprocess_packed_seqs, recover_left_padding, remove_left_padding
from siirl.utils.backend.device import get_device_id, get_torch_device

def _is_debug_enabled() -> bool:
    """Check if memory debug mode is enabled."""
    return os.environ.get("SIIRL_MEMORY_DEBUG", "0") == "1"


def gptmodel_forward(
    model,
    input_ids,
    attention_mask,
    position_ids,
    sequence_parallel,
    value_model=False,
    pack_seqs=True,
    logits_processor=None,
    logits_processor_args: dict | None = None,
    **kwargs,
):
    """Forward pass for GPT models with optional sequence packing.

    Memory optimizations:
    - Releases logits tensor immediately after logits_processor
    - Supports response-only output format to skip unpacking
    - Passes boundary info for per-sample optimization
    """
    pre_process = unwrap_model(model).pre_process
    post_process = unwrap_model(model).post_process
    get_torch_device().empty_cache()

    if pack_seqs:
        batch_size, seq_len = attention_mask.shape[:2]
        input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=pre_process)
        input_ids_rmpad = input_ids_rmpad.contiguous()

        # Model forward pass
        output_orig = model(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
        )

        if post_process and logits_processor is not None:
            # Prepare arguments for logits_processor
            output = _process_logits(
                output_orig=output_orig,
                logits_processor=logits_processor,
                logits_processor_args=logits_processor_args,
                attention_mask=attention_mask,
                packed_seq_params=packed_seq_params,
                batch_size=batch_size,
                seq_len=seq_len,
                post_process=post_process,
            )
        else:
            output = postprocess_packed_seqs(
                output_orig,
                packed_seq_params,
                attention_mask,
                batch_size,
                seq_len,
                post_process=post_process,
            )
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

    return output


def _process_logits(
    output_orig,
    logits_processor,
    logits_processor_args,
    attention_mask,
    packed_seq_params,
    batch_size,
    seq_len,
    post_process,
):
    """Process logits with memory-optimized cleanup.

    Key optimizations:
    - Passes boundary info (_cu_seqlens, _attention_mask) for per-sample processing
    - Releases output_orig immediately after logits_processor
    - Supports response-only format to skip postprocess_packed_seqs
    """
    # Separate special args (prefixed with _) from regular args that need packing
    regular_args = {k: v for k, v in logits_processor_args.items() if not k.startswith("_")}
    special_args = {k: v for k, v in logits_processor_args.items() if k.startswith("_")}

    # Pack regular args and add boundary info
    args = {k: preprocess_packed_seqs(v, attention_mask, pre_process=True)[0] for k, v in regular_args.items()}
    args["_cu_seqlens"] = packed_seq_params.cu_seqlens_q_padded
    args["_attention_mask"] = attention_mask
    args.update(special_args)

    # Run logits processor
    output_dict = logits_processor(output_orig, **args)

    # Memory optimization: release output_orig immediately
    del output_orig
    torch.cuda.empty_cache()

    # Check if response-only format (skip unpacking)
    is_response_only = output_dict.pop("_response_only", False)

    if is_response_only:
        # Response-only: output is already [batch, response_len]
        output = {k: v for k, v in output_dict.items() if isinstance(v, torch.Tensor)}
    else:
        # Packed format: need to unpack to [batch, seq_len]
        output = {}
        for k, v in output_dict.items():
            if not isinstance(v, torch.Tensor):
                continue
            output[k] = postprocess_packed_seqs(
                v,
                packed_seq_params,
                attention_mask,
                batch_size,
                seq_len,
                post_process=post_process,
            )

    # Release output_dict
    del output_dict
    torch.cuda.empty_cache()

    return output
