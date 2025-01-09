# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import os
from dataclasses import dataclass, field

from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType


def default_sharding_group_size():
    return int(os.environ.get('LOCAL_WORLD_SIZE', "1"))


def default_replica_group_size():
    return int(int(os.environ.get("WORLD_SIZE", "1")) / int(os.environ.get('LOCAL_WORLD_SIZE', "1")))


@dataclass
class FsdpConfig:
    enable_mixed_precision: bool = True
    # mixed precision training for FSDP
    fp16: bool = False  # if mixed_precision is True but fp16 is False here, then it will use bf16(avoiding challenges of scaler accuracies of fp16) for mixed precision. So recommended to keep it False.
    # HYBRID_SHARD "Full Shard within a node DDP cross Nodes", SHARD_GRAD_OP "Shard only Gradients and Optimizer States", NO_SHARD "Similar to DDP".
    sharding_strategy: ShardingStrategy = ShardingStrategy.HYBRID_SHARD
    # Require HYBRID_SHARD to be set. This flag can extend the HYBRID_SHARD by allowing sharding a model on customized number of GPUs (Sharding_group)
    # and Replicas over Sharding_group. Similar to deepspeed ZeRO++.
    hsdp: bool = True
    # requires hsdp to be set. This specifies the sharding group size, number of GPUs that you model can fit into to form a replica of a model.
    sharding_group_size: int = field(default_factory=default_sharding_group_size)  # requires hsdp to be set. Specifies the sharding group size, number of GPUs that you model can fit into to form a replica of a model.
    replica_group_size: int = field(default_factory=default_replica_group_size)  # requires hsdp to be set. Specifies the replica group size, which is world_size/sharding_group_size, change this to the actual number of nodes!
    # alternatively can use SHARDED_STATE_DICT save one file per rank, and can resize the world-size.
    checkpoint_type: StateDictType = StateDictType.SHARDED_STATE_DICT
    fsdp_activation_checkpointing: bool = True
    # TBD: disable it per https://github.com/facebookresearch/llama-recipes/issues/360
    fsdp_cpu_offload: bool = False  # with this 70B can train on single node
    fsdp_cpu_ram_efficient_loading: bool = False  # low cpu fsdp is used to train 70B on single node 8 GPUs avoiding loading 8 times of 70B to cpu
    pure_bf16: bool = False  # no mixed precision if True
    optimizer: str = "AdamW"
