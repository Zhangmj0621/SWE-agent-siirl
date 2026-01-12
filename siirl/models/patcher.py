from types import MethodType
from typing import TYPE_CHECKING

from loguru import logger
from transformers import PreTrainedTokenizerBase

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer

    from siirl.params import ModelArguments


def patch_tokenizer(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    config: "PretrainedConfig",
) -> None:
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

    if model_args.model_max_length is not None and tokenizer.model_max_length != model_args.model_max_length:
        tokenizer.model_max_length = model_args.model_max_length

    if model_args.new_special_tokens is not None:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=model_args.new_special_tokens),
            replace_additional_special_tokens=False,
        )
        logger.info("Add {} to special tokens.".format(",".join(model_args.new_special_tokens)))
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning("New tokens have been added, changed `resize_vocab` to True.")
