import contextlib
import logging
import os
import time
import torch
from transformers import LlamaTokenizer
from controlllm.utils.custom_llama_recipes.flop_utils import FlopMeasure


def custom_trace_handler(dir_name, worker_name=None, use_gzip=False):
    """
    Outputs tracing files to directory of ``dir_name``, then that directory can be
    directly delivered to tensorboard as logdir.
    ``worker_name`` should be unique for each worker in distributed scenario,
    it will be set to '[hostname]_[pid]' by default.
    """
    import socket

    def handler_fn(prof) -> None:
        nonlocal worker_name
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception as e:
                raise RuntimeError("Can't create directory: " + dir_name) from e
        if not worker_name:
            worker_name = f"{socket.gethostname()}_{os.getpid()}"

        _id = f"{worker_name}.{time.time_ns()}"
        # Use nanosecond here to avoid naming clash when exporting the trace
        file_name = f"{_id}.pt.trace.json"
        if use_gzip:
            file_name = file_name + ".gz"
        # prof.export_chrome_trace(os.path.join(dir_name, file_name))
        tb_file_name = f"{_id}.html"
        prof.export_memory_timeline(os.path.join(dir_name, tb_file_name))
        logging.info(f"Profiler data saved to {os.path.join(dir_name, file_name)}")
        logging.info(f"Memory timeline data saved to {os.path.join(dir_name, tb_file_name)}")
        snapshot_file_name = f"{_id}.pickle"
        # torch.cuda.memory._dump_snapshot(os.path.join(dir_name, snapshot_file_name))
        logging.info(f"Memory snapshot saved to {os.path.join(dir_name, snapshot_file_name)}")
        # torch.cuda.memory._record_memory_history(enabled=None)

    return handler_fn


def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"


@contextlib.contextmanager
def profile(cfg, local_rank=None):
    use_profiler: bool = cfg.use_profiler
    use_flop_counter: bool = cfg.flop_counter  # to be compliant with the original code of llama_recipes
    if use_flop_counter and use_profiler:
        raise ValueError("Cannot use both profiler and flop counter")

    if use_profiler:
        # profiler needs a warmup stage to get the accurate profiling results
        wait_step, warmup_step, active_step = cfg.wait_step, cfg.warmup_step, cfg.active_step
        min_step = wait_step + warmup_step + active_step + 1
        if cfg.max_train_step > 0 and cfg.max_train_step < min_step:
            raise ValueError(f"pytorch profiler requires at least {min_step} train steps to finish the warm-up and recording stage, {wait_step} for wait_step, {warmup_step} for warmup_step, {active_step} for profiling step, please increase the max_train_step, current max_train_step {cfg.max_train_step}")
        logging.info(f"pytorch profiling is activated and results will be saved in {cfg.profiler_dir}")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait_step, warmup=warmup_step, active=active_step, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                cfg.profiler_dir
            ),
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            record_shapes=True,
        ) as torch_profiler:
            yield torch_profiler
    elif use_flop_counter:
        if cfg.max_train_step > 0 and cfg.max_train_step <= cfg.flop_counter_start:
            raise ValueError(f"flop counter requires at least {cfg.flop_counter_start + 1} train steps, please increase the max_train_step, current max_train_step {cfg.max_train_step}")
        with FlopMeasure(rank=local_rank, warmup_step=cfg.flop_counter_start) as flop_counter:
            yield flop_counter
    else:
        yield contextlib.nullcontext()
