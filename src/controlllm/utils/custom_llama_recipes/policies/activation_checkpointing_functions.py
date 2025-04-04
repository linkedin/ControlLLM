# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from functools import partial

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)


def apply_fsdp_checkpointing(model, transformer_layer_names=[LlamaDecoderLayer]):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f"--> applying fsdp activation checkpointing on {transformer_layer_names} ...")

    check_fn = lambda submodule: isinstance(submodule, tuple(transformer_layer_names))

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )
