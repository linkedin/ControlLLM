"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import sys
import logging
import time
import random
import numpy as np
from collections import defaultdict, deque
import datetime
import dataclasses
import torch
import torch.distributed as dist
from accelerate.utils import is_xpu_available
from controlllm.configs import TrainConfig
from controlllm.configs.datasets import AbstractDataset as DatasetConfig


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logging.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    logging.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only available in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    if rank == 0:
        logging.info(f"--> Running with torch dist debug set to detail")


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        logging.info(f"Clearing GPU cache for all ranks")
    if is_xpu_available():
        torch.xpu_empty_cache()
    else:
        torch.cuda.empty_cache()


def setup_for_distributed(is_master):
    """
    This function disables printing and logging.info when not in the master process
    """
    import builtins as __builtin__
    import logging

    builtin_print = __builtin__.print

    def custom_print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    # Assign custom_print to the module
    sys.modules[__name__].custom_print = custom_print

    __builtin__.print = custom_print

    root = logging.getLogger()

    if TrainConfig().debug:
        level = logging.INFO
    else:
        level = logging.ERROR

    if not is_master:
        root.setLevel(level)
        root = logging.getLogger()
        for handler in root.handlers:
            handler.setLevel(level)


def apply_custom_load_dataset():
    """
    This function applies monkey patching to load_dataset to set the default name to 'default' and cache_dir to the provided cache_dir.
    It is to ensure load_dataset works in a training environment without internet access but with dataset cached already.
    """
    import datasets

    # Save a reference to the original load_dataset function
    original_load_dataset = datasets.load_dataset

    # Define the custom load_dataset function
    def custom_load_dataset(*args, **kwargs):
        """
        Custom wrapper for `load_dataset` that ensures:
        - If no second positional argument is provided, `name` defaults to "default".
        - If `cache_dir` is not explicitly provided in args or kwargs, it is injected from `os.environ['HF_HOME']`.

        Args:
            *args: Positional arguments for `load_dataset`
            **kwargs: Keyword arguments for `load_dataset`

        Returns:
            Dataset or DatasetDict: The loaded dataset.
        """

        # Ensure 'name' is set to 'default' if not provided in args or kwargs
        if len(args) < 2 and 'name' not in kwargs:
            kwargs['name'] = 'default'

        # Ensure 'cache_dir' is set if it's not already provided
        # cache_dir is the 6th argument (index 5), so we check if args has >=6 elements
        if len(args) < 6 and 'cache_dir' not in kwargs:
            kwargs['cache_dir'] = DatasetConfig("AbstractDataset").hf_hub_dataset_cache_dir

        return original_load_dataset(*args, **kwargs)

    # Replace the original load_dataset function with the custom one
    datasets.load_dataset = custom_load_dataset


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    # fix the seed for reproducibility, TODO: move it to other method
    args.seed = args.random_seed + get_rank()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Check if CUDA is available, otherwise use CPU
    if args.device == '':
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # uncomment this when if you want to use cpu for debugging locally
    # device = torch.device("cpu")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Manually set LOCAL_RANK if not present (useful for debugging in VSCode in single process)
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = '0'  # Default to 0 for debugging
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.local_rank = args.rank % torch.cuda.device_count()
    else:
        logging.info('Not using distributed mode')
        # When not using torchrun, set the local rank to 0 and the world size to 1.
        args.local_rank = 0
        # Deepspeed needs to know the world size and rank
        os.environ['LOCAL_RANK'] = str(args.local_rank)
        args.world_size = 1
        os.environ['WORLD_SIZE'] = str(args.world_size)
        args.rank = 0
        os.environ['RANK'] = str(args.rank)
        # Set the master address and port for local debugging
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        args.distributed = False
        return

    # args.distributed = True

    # The default timeout for NCCL operations is 600 seconds (10 minutes). 
    # Initial data processing might take a long time to complete so increasing the timeout.
    # os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    # os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

    # set ACCELERATE_USE_FSDP and FSDP_CPU_RAM_EFFICIENT_LOADING to 1 for efficent ram usage
    # os.environ["ACCELERATE_USE_FSDP"] = "1"
    # os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "1"

    torch.cuda.set_device(args.local_rank)
    # args.dist_backend = 'nccl'
    # args.dist_url = 'env://'
    logging.info(f'| distributed init (rank {args.rank}, local rank {args.local_rank}): {args.dist_backend}, {args.dist_url}, world size: {args.world_size}, {datetime.timedelta(hours=10)}')
    if args.device.type == "cuda" and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                             world_size=args.world_size, rank=args.rank,
                                              # FIXME: Setting a longer timeout 10 hours! to enable data preprocessing for extremely large datasets
                                             timeout=datetime.timedelta(hours=10)
                                             )
        torch.distributed.barrier()

    if torch.distributed.is_initialized():
        if is_xpu_available():
            torch.xpu.set_device(args.local_rank)
        else:
            torch.cuda.set_device(args.local_rank)
        clear_gpu_cache(args.local_rank)
        setup_environ_flags(args.rank)


def freeze(module):
    for p in module.parameters():
        p.requires_grad = False


def flag_pretrain(module, pretrained=True):
    setattr(module, "pretrain", pretrained)
    return module


def is_pretrain(module):
    is_pretrain = hasattr(module, 'pretrain') and module.pretrain
    return is_pretrain


def load_pretrain_model(model, model_path, finetune=False):
    checkpoint = torch.load(os.path.expanduser(model_path), map_location='cpu')
    if 'model' in checkpoint.keys():
        checkpoint_model = checkpoint['model']
    else:
        checkpoint_model = checkpoint

    if finetune:
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias',
                  'trans_cls_head.weight', 'trans_cls_head.bias', 'conv_cls_head.weight', 'conv_cls_head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                logging.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        if 'pos_embed' in checkpoint_model.keys():
            # interpolate position embedding
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

    model.load_state_dict(checkpoint_model, strict=False)
    return model


def log_metrics(writer, metrics, epoch):
    for k, v in metrics.items():
        writer.add_scalar(k, v, epoch)
    writer.add_scalars('Summary', metrics, epoch)


def log_weight_histograms(model, writer, epoch):
    layer = 'conv_trans_'
    for name, param in model.named_parameters():
        if layer in name and 'weight' in name and param.requires_grad == True:
            flattened_weights = param.flatten()
            writer.add_histogram(name, flattened_weights, global_step=epoch, bins='tensorflow')


def log_ftr_map_histograms(writer, x, x_t, i, global_step):
    flattened_x = x.flatten()
    flattened_x_t = x_t.flatten()
    writer.add_histogram(f"conv_trans_{i}.feature_map.conv_tower", flattened_x, global_step=global_step, bins='tensorflow')
    writer.add_histogram(f"conv_trans_{i}.feature_map.trans_tower", flattened_x_t, global_step=global_step, bins='tensorflow')


def setup_wandb(wandb_config, train_config, fsdp_config, rank=0):
    if train_config.use_wandb and (not train_config.enable_fsdp or rank == 0):
        try: 
            import wandb
        except ImportError:
            raise ImportError(
                "You are trying to use wandb which is not currently installed. "
                "Please install it using pip install wandb"
            )
        init_dict = dataclasses.asdict(wandb_config)
        run = wandb.init(**init_dict)
        run.config.update(train_config)
        run.config.update(fsdp_config, allow_val_change=True)
        return run
    else:
        return None


def setup_tensorboard(train_config, rank=0):
    if train_config.enable_tensorboard and (not train_config.enable_fsdp or rank == 0):
        try: 
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError(
                "You are trying to use tensorboard which is not currently installed. "
                "Please install it using pip install tensorboard"
            )
        tb_writer = SummaryWriter(log_dir=train_config.logging_dir)
        return tb_writer
    else:
        return None
