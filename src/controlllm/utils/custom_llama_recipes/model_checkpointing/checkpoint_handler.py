# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from pathlib import Path
from datetime import datetime
import logging
import torch
import time

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,  # general model non-sharded, non-flattened params
)
from torch.distributed._shard.checkpoint import (
    FileSystemReader,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
import torch.distributed._shard.checkpoint as dist_cp
import torch.distributed as dist

# from linkedin.dllib.common.checkpoint_manger import CheckpointManager


def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    logging.info(f"--> current date and time of run = {date_of_run}")
    return date_of_run


# create singleton saving policies to avoid making over and over
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)


def is_load_checkpoint_needed(cfg):
    # Check if load_checkpoint method needs to be called
    load_dir = Path.cwd() / cfg.output_dir / cfg.resume_checkpoint_folder

    # Check if peft_checkpoint.txt exists
    if (load_dir / "peft_checkpoint.txt").exists():
        logging.info(f"PEFT checkpoint found at {load_dir}")
        return True

    # Check if sharded_checkpoint.txt exists
    if (load_dir / "sharded_checkpoint.txt").exists():
        logging.info(f"Sharded checkpoint found at {load_dir}")
        return True

    # Check if model_checkpoint.txt exists
    if (load_dir / "model_checkpoint.txt").exists():
        logging.info(f"Model checkpoint found at {load_dir}")
        return True

    return False


def load_checkpoint(model, rank, cfg, with_meta=False, optimizer=None):
    """load checkpoint based on the folder's saved file"""
    load_dir = Path.cwd() / cfg.output_dir / cfg.resume_checkpoint_folder
    
    # if peft_checkpoint.txt found, call load_peft_checkpoint
    if (load_dir / "peft_checkpoint.txt").exists():
        logging.info(f"PEFT checkpoint found at {load_dir}")
        load_peft_checkpoint(model, rank, cfg, with_meta)
        return

    # if sharded_checkpoint.txt found, call load_model_sharded, else call load_model_checkpoint
    if (load_dir / "sharded_checkpoint.txt").exists():
        logging.info(f"Sharded checkpoint found at {load_dir}")

        if not (Path(load_dir) / ".metadata").exists():
            # try to load from pytorch_model_fsdp_0 to handle the case sharded checkpoint is saved by transformers's trainer, smooth transition to use native trainer to continue the training
            logging.info(f"fsdp_checkpoint_path {load_dir} does not have .metadata, checking if {load_dir} / pytorch_model_fsdp_0 has it")
            fsdp_checkpoint_path_fsdp_0 = Path(load_dir) / "pytorch_model_fsdp_0"
            if not (Path(fsdp_checkpoint_path_fsdp_0) / ".metadata").exists():
                raise FileNotFoundError(f"fsdp_checkpoint_path {fsdp_checkpoint_path_fsdp_0} does not have .metadata")
            logging.info(f"Loading sharded checkpoint from {fsdp_checkpoint_path_fsdp_0} ...")
            load_sharded_model_single_gpu(model, fsdp_checkpoint_path_fsdp_0, with_meta)
        else:
            logging.info(f"Loading sharded checkpoint from {load_dir} ...")
            load_sharded_model_single_gpu(model, load_dir, with_meta)
        return

    # if model_checkpoint.txt found, call load_model_checkpoint
    if (load_dir / "model_checkpoint.txt").exists():
        logging.info(f"Model checkpoint found at {load_dir}")
        load_model_checkpoint(model, rank, cfg, with_meta)
        return

    # default to load_model_checkpoint
    logging.info(f"No checkpoint type file found at {load_dir}. Loading by huggingface from_pretrained assuming the checkpoint is in huggingface format...")
    load_model_checkpoint(model, rank, cfg, with_meta)


def load_optimizer(model, optimizer, rank, cfg):
    """load checkpoint based on the folder's saved file"""
    load_dir = Path.cwd() / cfg.output_dir / cfg.resume_checkpoint_folder

    # if optimizer_checkpoint.txt found, call load_optimizer_checkpoint
    if (load_dir / "optimizer_checkpoint.txt").exists():
        load_optimizer_checkpoint(model,optimizer, rank, cfg)

    if (load_dir / "shared optimizer_checkpoint.txt").exists():
        load_sharded_model_single_gpu(model, load_dir, with_meta=False, optimizer=optimizer)


def load_model_sharded(model, rank, cfg, optimizer=None):
    load_dir = Path.cwd() / cfg.output_dir / cfg.resume_checkpoint_folder

    if not load_dir.exists():
        if rank == 0:
            logging.info(f"No sharded_state_dict checkpoint directory found...skipping")
        return
    if rank == 0:
         logging.info(f"Loading model from model path: {load_dir} ")
    reader = FileSystemReader(load_dir)

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        checkpoint = {"model": model.state_dict()}
        if rank == 0:
            ck = checkpoint.keys()
            logging.info(f"Checkpoint key len = {len(ck)} and \n keys =  {ck}")
      
        dist_cp.load_state_dict(
            state_dict=checkpoint,
            storage_reader=reader,
        )
        if rank == 0:
            logging.info(f"checkpoint after load_state_dict()")
            ck = checkpoint.keys()
            logging.info(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])

        # Load the optimizer state if it exists and the optimizer is provided
        if "optim" in checkpoint and optimizer:
            optimizer.load_state_dict(checkpoint["optim"])
            if rank == 0:
                logging.info("Optimizer state loaded successfully.")

    if rank == 0:
        logging.info(f"Sharded state checkpoint loaded from {load_dir}")


def save_model_and_optimizer_sharded(model, rank, cfg, optim=None, tokenizer=None, global_step=-1):
    """save model and optimizer via sharded_state_dict to save_dir"""
    save_dir = Path.cwd() / cfg.output_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    save_dir = Path.cwd() / cfg.output_dir / f"checkpoint-{global_step}"
    save_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        logging.info(f"--> saving sharded model to {save_dir}")

    distributed_writer = dist_cp.FileSystemWriter(
        save_dir,
    )
    t0 = time.perf_counter()

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):

        logging.info(f"--> retrieving state dict on rank {rank}\n")
        state_dict = {"model": model.state_dict()}
        if optim:
            state_dict["optim"] = FSDP.optim_state_dict(model, optim)

        # new method torch.distributed._shard.checkpoint.load and save is not stable yet in multi-node training, fallback to load_state_dict and save_state_dict
        logging.info(f"--> saving sharded state dict on rank {rank}\n")
        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=distributed_writer,
            planner=DefaultSavePlanner(),
        )

    # Post-save synchronization
    logging.info(f"--> waiting for all ranks to finish saving...\n")
    dist.barrier()

    t1 = time.perf_counter()
    if rank == 0:
        logging.info(f"Sharded state checkpoint saved to {save_dir}")
        logging.info(
            f"Checkpoint Time = {t1-t0:.4f}\n"
        )
        # save config
        logging.info(f"--> saving model config... to {save_dir}")
        model.config.save_pretrained(save_dir)

        # save tokenizer
        if tokenizer:
            logging.info(f"--> saving tokenizer... to {save_dir}")
            tokenizer.save_pretrained(save_dir)
        
        # write a file to indicate the checkpoint is for FSDP
        with open(save_dir / "sharded_checkpoint.txt", "w") as f:
            f.write("sharded checkpoint")


def load_peft_checkpoint(model, rank, cfg, with_meta=False):
    """load peft checkpoint to rank0 cpu
    must be called * before * passing to FSDP"""

    load_dir = Path.cwd() / cfg.output_dir / cfg.resume_checkpoint_folder
    # follow https://github.com/huggingface/peft/blob/02ae6bcb373d9d9d3bec9ba920d63316418ff64a/src/peft/peft_model.py#L296
    state_dict = model.from_pretrained(load_dir, device_map=torch.device('cpu')).state_dict()
    # this is in case the model is not a PEFT model
    model.load_state_dict(state_dict, assign=with_meta)
    state_dict = None

    logging.info(f"PEFT checkpoint loaded to cpu")


def save_peft_checkpoint(
    model,
    optimizer,
    rank,
    cfg,
    epoch=1,
    tokenizer=None,
    global_step=-1,
):
    """saving model of peft trained enabled"""
    save_dir = Path.cwd() / cfg.output_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    save_dir = Path.cwd() / cfg.output_dir / f"checkpoint-{global_step}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # use the new DCP api to avoid OOM
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    state_dict = get_model_state_dict(model, options=options)
    model.save_pretrained(save_dir, state_dict=state_dict)
    # save config
    logging.info(f"--> saving model config... to {save_dir}")
    model.config.save_pretrained(save_dir)
    # save tokenizer
    if tokenizer:
        logging.info(f"--> saving tokenizer... to {save_dir}")
        tokenizer.save_pretrained(save_dir)

    # write a file to indicate the checkpoint is for PEFT
    with open(save_dir / "peft_checkpoint.txt", "w") as f:
        f.write("PEFT checkpoint")


def save_model_checkpoint(
    model,
    optimizer,
    rank,
    cfg,
    epoch=1,
    tokenizer=None,
    global_step=-1,
):
    """
    Save the model checkpoint only from the main process (rank 0) in a distributed setting.

    Args:
        model (FSDP): The fully sharded model.
        optimizer (Optimizer): The optimizer used during training.
        rank (int): The rank of the current process in the distributed setting.
        cfg (Config): Configuration object with saving and model parameters.
        epoch (int, optional): Current epoch of training. Defaults to 1.
        tokenizer (Tokenizer, optional): The tokenizer to be saved alongside the model. Defaults to None.
        global_step (int, optional): The global step count at the time of saving. Defaults to -1.

    This function saves the model's state dictionary, configuration, and tokenizer
    (if provided) to the designated directory.
    """

    logging.info(f"Rank {rank}: Preparing to retrieve state dictionary for saving...")

    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
    ):
        cpu_state = model.state_dict()

        logging.info(f"saving process: rank {rank}  done w model state_dict\n")

    if rank == 0:
        logging.info(f"--> saving model ...")

        save_dir = Path.cwd() / cfg.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        save_dir = Path.cwd() / cfg.output_dir / f"checkpoint-{global_step}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model.save_pretrained(save_dir, state_dict=cpu_state)

        # save config
        logging.info(f"--> saving model config... to {save_dir}")
        model.config.save_pretrained(save_dir)

        # save tokenizer
        if tokenizer:
            logging.info(f"--> saving tokenizer... to {save_dir}")
            tokenizer.save_pretrained(save_dir)

        # write a file to indicate the checkpoint is for model checkpoint
        with open(save_dir / "model_checkpoint.txt", "w") as f:
            f.write("Model checkpoint")

        logging.info(f"model checkpoint saved for epoch {epoch}, global step {global_step} at {save_dir}\n")


def load_model_checkpoint(model, rank, cfg, with_meta=False):
    """if FSDP: load local checkpoint to rank0 cpu must be called * before * passing to FSDP"""

    full_state_dict_model_path = Path.cwd() / cfg.output_dir / cfg.resume_checkpoint_folder

    logging.info(f"--> loading model checkpoint from {full_state_dict_model_path}")
    state_dict = model.from_pretrained(str(full_state_dict_model_path), device_map=torch.device('cpu')).state_dict()
    model.load_state_dict(state_dict, assign=with_meta)
    state_dict = None

    logging.info(f"model checkpoint loaded to cpu from {full_state_dict_model_path}")


def save_optimizer_checkpoint(model, optimizer, rank, cfg, epoch=1, global_step=-1):
    """save optimizer state via full state dict"""
    logging.info(f"--> optim state call on rank {rank}\n")

    # pull all sharded optimizer states to rank0 cpu...
    optim_state = FSDP.full_optim_state_dict(model, optimizer)
    
    logging.info(f"optim state dict ready on {rank} and len of {len(optim_state)}\n")

    if rank == 0:
        save_dir = Path.cwd() / cfg.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        save_dir = Path.cwd() / cfg.output_dir / f"checkpoint-{global_step}"
        save_dir.mkdir(parents=True, exist_ok=True)

        opt_save_name = "optimizer.pt"
        opt_save_full_path = save_dir / opt_save_name

        logging.info(f"--> saving optimizer state...")

        torch.save(optim_state, opt_save_full_path)

        # write a file to indicate the checkpoint is for optimizer checkpoint
        with open(save_dir / "optimizer_checkpoint.txt", "w") as f:
            f.write("Optimizer checkpoint")

        logging.info(f"--> saved {opt_save_full_path} to disk")


def load_optimizer_checkpoint(model, optimizer, rank, cfg):
    """load an fsdp optimizer full_state checkpoint using scatter method
    this ensures only rank 0 loads the optimizer state dict and scatters to other ranks
    """

    optimizer_checkpoint_path = Path.cwd() / cfg.output_dir / cfg.resume_checkpoint_folder / "optimizer.pt"

    if not optimizer_checkpoint_path.is_file():
        logging.info(
            f"warning - optimizer checkpoint not present {optimizer_checkpoint_path}. Returning. "
        )
        return

    full_osd = None

    if rank == 0:
        full_osd = torch.load(optimizer_checkpoint_path)

    # called from all ranks, though only rank0 has a valid param for full_osd
    sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)
    # this is to handle possibly different world size
    # follow https://github.com/pytorch/pytorch/blob/b3821f1da1b348d71e90aaca1b29e99e27b24449/torch/distributed/fsdp/fully_sharded_data_parallel.py#L1618
    optimizer.load_state_dict(sharded_osd)

    logging.info(f"optimizer shard loaded on rank {rank} from {optimizer_checkpoint_path}")


def load_sharded_model_single_gpu(model, model_path, with_meta=False, optimizer=None):
    """
    Load a sharded model from a checkpoint file in current GPU.
    Less efficient than load_model_sharded, but useful if number of shards changes(different world size) between different runs of training.

    Loading optimizer state is optional and only done if optimizer is provided.
    """
    # If with_meta is True, return early without loading the state dict
    if with_meta:
        logging.info("Model is on 'meta' device, skipping state dict loading.")
        return model

    if optimizer is None:
        state_dict = {
            "model": model.state_dict()
        }
    else:
        state_dict = {
            "model": model.state_dict(),
            "optim": optimizer.state_dict()
        }

    # new method torch.distributed._shard.checkpoint.load and save is not stable yet in multi-node training, fallback to load_state_dict and save_state_dict
    logging.info(f"--> loading sharded model from model path: {model_path} ")
    dist_cp.load_state_dict(
                state_dict=state_dict,
                storage_reader=FileSystemReader(model_path),
                no_dist=True
            )

    # this is to handle possibly different world size
    # follow https://pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html#:~:text=Using%20load_state_dict(assign%3DTrue)&text=load_state_dict()%20is%20usually%20implemented,parameter%2Fbuffer%20in%20the%20nn.
    model.load_state_dict(state_dict["model"], assign=with_meta)

    # Load the optimizer state if it exists and the optimizer is provided
    if "optim" in state_dict and optimizer:
        # called from all ranks, though only rank0 has a valid param for full_osd
        sharded_osd = FSDP.scatter_full_optim_state_dict(state_dict["optim"], model)
        # this is to handle possibly different world size
        # follow https://github.com/pytorch/pytorch/blob/b3821f1da1b348d71e90aaca1b29e99e27b24449/torch/distributed/fsdp/fully_sharded_data_parallel.py#L1618
        optimizer.load_state_dict(sharded_osd)
        logging.info("Optimizer state loaded successfully.")

    logging.info(f"Sharded state checkpoint loaded from {model_path}")

    # # Restore checkpoint for disruption handling
    # logging.info("Restoring checkpoint for disruption handling...")
    # checkpoint_manager = CheckpointManager(primary_checkpoint_path="/dev/shm/controlllm/ckpt", secondary_checkpoint_path="hdfs://jobs/controlllm/ckpt")
    # checkpoint_manager.restore(checkpoint_id="latest")
    # checkpoint = torch.load(checkpoint_manager.primary_checkpoint_path[1])
    # model.load_state_dict(checkpoint["model_state_dict"])

    return model
