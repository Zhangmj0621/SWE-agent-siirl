from typing import TYPE_CHECKING

from transformers import AutoTokenizer

from loguru import logger

if TYPE_CHECKING:
    from siirl.params import ModelArguments

def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.warning(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}")


def load_tokenizer(
    path: str = "",
    model_args: "ModelArguments" = None,
    correct_pad_token: bool = True,
):
    r"""
    Loads pretrained tokenizer and optionally loads processor.
    Note: including inplace operation of model_args.
    """
    init_kwargs = {}
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            use_fast=model_args.use_fast_tokenizer if model_args else True,
            split_special_tokens=model_args.split_special_tokens if model_args else False,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try the fast one
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            use_fast=True,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    if correct_pad_token:
        set_pad_token_id(tokenizer)

    return tokenizer
