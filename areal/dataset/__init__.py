from __future__ import annotations

from typing import TYPE_CHECKING

from areal.api.cli_args import _DatasetConfig
from areal.utils import logging

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.processing_utils import ProcessorMixin
    from transformers.tokenization_utils import PreTrainedTokenizerBase

# Built-in dataset adapters. Text/chat Hugging Face Hub datasets can also use the
# generic fallback loader in ``areal.dataset.hf_text``.
VALID_DATASETS = [
    "gsm8k",
    "clevr_count_70k",
    "geometry3k",
    "virl39k",
    "hh-rlhf",
    "torl_data",
]

logger = logging.getLogger("Dataset")


def _get_builtin_dataset(
    path: str,
    type: str = "sft",
    split: str | None = None,
    max_length: int | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    processor: ProcessorMixin | None = None,
    **kwargs,
) -> Dataset | None:
    if "gsm8k" in path and type == "sft":
        from .gsm8k import get_gsm8k_sft_dataset

        return get_gsm8k_sft_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    if "gsm8k" in path and type == "rl":
        from .gsm8k import get_gsm8k_rl_dataset

        return get_gsm8k_rl_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    if "clevr_count_70k" in path and type == "sft":
        from .clevr_count_70k import get_clevr_count_70k_sft_dataset

        return get_clevr_count_70k_sft_dataset(
            path=path,
            split=split,
            processor=processor,
            max_length=max_length,
            **kwargs,
        )
    if "clevr_count_70k" in path and type == "rl":
        from .clevr_count_70k import get_clevr_count_70k_rl_dataset

        return get_clevr_count_70k_rl_dataset(
            path=path,
            split=split,
            processor=processor,
            max_length=max_length,
            **kwargs,
        )
    if "geometry3k" in path and type == "sft":
        from .geometry3k import get_geometry3k_sft_dataset

        return get_geometry3k_sft_dataset(
            path=path,
            split=split,
            processor=processor,
            max_length=max_length,
            **kwargs,
        )
    if "geometry3k" in path and type == "rl":
        from .geometry3k import get_geometry3k_rl_dataset

        return get_geometry3k_rl_dataset(
            path=path,
            split=split,
            processor=processor,
            max_length=max_length,
            **kwargs,
        )
    if "virl39k" in path.lower() and type == "rl":
        from .virl39k import get_virl39k_rl_dataset

        return get_virl39k_rl_dataset(
            path=path,
            split=split,
            processor=processor,
            max_length=max_length,
            **kwargs,
        )
    if "hh-rlhf" in path and type == "rw":
        from .hhrlhf import get_hhrlhf_rw_dataset

        return get_hhrlhf_rw_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    if "torl_data" in path and type == "rl":
        from .torl_data import get_torl_data_rl_dataset

        return get_torl_data_rl_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    return None


def _get_custom_dataset(
    path: str,
    type: str = "sft",
    split: str | None = None,
    max_length: int | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    processor: ProcessorMixin | None = None,
    config_name: str | None = None,
    messages_column: str | None = None,
    prompt_column: str | None = None,
    completion_column: str | None = None,
    **kwargs,
) -> Dataset:
    built_in_dataset = _get_builtin_dataset(
        path=path,
        type=type,
        split=split,
        max_length=max_length,
        tokenizer=tokenizer,
        processor=processor,
        **kwargs,
    )
    if built_in_dataset is not None:
        return built_in_dataset

    if type not in {"rl", "sft"}:
        raise ValueError(
            f"Dataset {path} with split {split} and training type {type} is not handled by a built-in adapter. "
            f"Built-in adapters are: {VALID_DATASETS}. "
            "Generic dataset loading is only available for type='rl' or type='sft' with either a 'messages' column "
            "or configured prompt_column/completion_column fields."
        )

    from .hf_text import get_hf_text_dataset

    logger.info(
        "Falling back to the generic Hugging Face text dataset loader for %s.",
        path,
    )
    dataset_config = _DatasetConfig(
        path=path,
        type=type,
        config_name=config_name,
        split=split,
        messages_column=messages_column,
        prompt_column=prompt_column,
        completion_column=completion_column,
        max_length=max_length,
    )
    return get_hf_text_dataset(
        dataset_config=dataset_config,
        split=split,
        tokenizer=tokenizer,
    )


def get_custom_dataset(
    split: str | None = None,
    dataset_config: _DatasetConfig | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    processor: ProcessorMixin | None = None,
    **kwargs,
) -> Dataset:
    if dataset_config is not None:
        resolved_split = dataset_config.split or split
        return _get_custom_dataset(
            path=dataset_config.path,
            type=dataset_config.type,
            split=resolved_split,
            max_length=dataset_config.max_length,
            tokenizer=tokenizer,
            processor=processor,
            config_name=dataset_config.config_name,
            messages_column=dataset_config.messages_column,
            prompt_column=dataset_config.prompt_column,
            completion_column=dataset_config.completion_column,
            **kwargs,
        )

    logger.warning("dataset_config is not provided")
    return _get_custom_dataset(
        split=split,
        tokenizer=tokenizer,
        processor=processor,
        **kwargs,
    )


__all__ = [
    "VALID_DATASETS",
    "get_custom_dataset",
]
