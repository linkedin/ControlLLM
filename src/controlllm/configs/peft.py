# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass, field
from typing import List
from peft.config import PeftConfig


@dataclass
class LoraConfig(PeftConfig):
    r: int = 8
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.05
    inference_mode: bool = False


@dataclass
class LlamaAdapterConfig(PeftConfig):
    adapter_len: int = 10
    adapter_layers: int = 30
    task_type: str = "CAUSAL_LM"


@dataclass
class PrefixConfig(PeftConfig):
    num_virtual_tokens: int = 30
    task_type: str = "CAUSAL_LM"
