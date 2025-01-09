# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import packaging
import functools
import torch
from torch.distributed._tensor.device_mesh import init_device_mesh
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from accelerate.utils import is_xpu_available

from controlllm.utils.custom_llama_recipes import fpSixteen, bfSixteen


def fsdp_auto_wrap_policy(model, transformer_layer_names):
    import functools

    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=set(transformer_layer_names)
    )

    auto_wrap_policy = functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
    return auto_wrap_policy


def hsdp_device_mesh(fsdp_config, setup_config, device=None):
    """
     Initializes a device mesh for use with Hybrid Sharding strategy in FSDP (HSDP) training.

    This function requires explicit sizes for replica and sharding groups to accommodate models
    whose GPU fit is unknown, providing flexibility in distributed training setups.

    Args:
        replica_group_size (int): The size of each replica group. Must be provided to ensure
            the model fits within the available resources.
        sharding_group_size (int): The size of each sharding group that the model can fit. Must be provided to 
            ensure the correct distribution of model parameters.
        device (str, optional): The device to use (e.g., "cuda:0"). If None, defaults to "cuda"
            with the local rank as the device index.

    Returns:
        A device mesh object compatible with FSDP.

    Raises:
        ValueError: If replica_group_size or sharding_group_size are not provided, or if the
            world size is not evenly divisible by the sharding group size.
        RuntimeError: If a valid device mesh cannot be created.

    Usage:
        If your model fits on 4 GPUS, and you have 3 nodes of 8 GPUs, then:
        Sharding_Group_Size = 4
        Replica_Groups_Size = (24 total gpus, 4 per sharding group) = 6 Replica Groups
        >>> device_mesh = initialize_device_mesh(replica_group_size, sharding_group_size)
        >>> sharded_model = FSDP(model, device_mesh=device_mesh, ...)
    """

    if fsdp_config.replica_group_size is None or fsdp_config.sharding_group_size is None:
        raise ValueError("Both replica_group_size and sharding_group_size must be provided.")

    # local_rank = int(os.getenv("LOCAL_RANK", "0"))
    # world_size = int(os.getenv("WORLD_SIZE", "1"))
    # local_rank = setup_config.local_rank
    world_size = setup_config.world_size

    device = device or str(setup_config.device)

    if world_size % fsdp_config.sharding_group_size != 0:
        raise ValueError(f"World size {world_size} is not evenly divisible by "
                         f"sharding group size {fsdp_config.sharding_group_size}.")

    if (world_size // fsdp_config.sharding_group_size) % fsdp_config.replica_group_size != 0:
        raise ValueError(f"The calculated number of replica groups is not evenly divisible by "
                         f"replica_group_size {fsdp_config.replica_group_size}.")

    device_mesh = init_device_mesh(device, (fsdp_config.replica_group_size, fsdp_config.sharding_group_size))
    if device_mesh is None:
        raise RuntimeError("Failed to create a valid device mesh.")

    return device_mesh


def get_policies(fsdp_config, setup_config, transformer_layer_names):
    """Get the policies for mixed precision and fsdp wrapping"""

    verify_bfloat_support = ((
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    ) or
        (is_xpu_available()))

    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if fsdp_config.enable_mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not fsdp_config.fp16:
            mixed_precision_policy = bfSixteen
            if setup_config.rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif fsdp_config.fp16:
            mixed_precision_policy = fpSixteen
            if setup_config.rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")

    def get_transformer_wrapper():
        """we register our main layer class and use the fsdp transformer wrapping policy
        ensures embedding layers are in the root fsdp unit for shared access and that fsdp units map to transformer layers
        """
        # ====   use new transformer wrapper
        llm_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=set(transformer_layer_names),
        )

        return llm_auto_wrap_policy

    wrapping_policy = get_transformer_wrapper()
    return mixed_precision_policy, wrapping_policy