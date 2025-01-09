# code of profiling callback is borrowed from liger, thanks to Pin-Lun Hsu and the team
import os
import json
import socket
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Dict

import torch
import transformers
from accelerate.utils.constants import FSDP_SHARDING_STRATEGY
from transformers import TrainerControl, TrainerState, TrainingArguments, TrainerCallback


# https://simple.wikipedia.org/wiki/Byte
# For memory, we use binary system
M_BIN_UNIT = 2**20
# For metrics (tflops), we use decimal system
T_DEC_UNIT = 10**12


def round_to_n_decimal(x, n):
    return round(x, n)


# Define ANSI color codes as constants
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


# model save callback for multi-node training
class SaveModelCallback(TrainerCallback):
    def __init__(self, trainer, save_steps=500):
        self.trainer = trainer
        self.save_steps = save_steps

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logging.info(f"Global step {state.global_step} reached. Saving model configuration to {args.output_dir}")
        self.trainer.accelerator.wait_for_everyone()
        unwrapped_model = self.trainer.accelerator.unwrap_model(self.trainer.model)
        if self.trainer.accelerator.is_main_process:
            save_dir = Path.cwd() / args.output_dir / f"checkpoint-{state.global_step}"
            unwrapped_model.config.save_pretrained(save_dir)
            logging.info(f"Global step {state.global_step} reached. Configuration saved to {save_dir}")
            self.trainer.tokenizer.save_pretrained(save_dir)
            logging.info(f"Global step {state.global_step} reached. Tokenizer saved to {save_dir}")

    def _save_model(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Dict[str, Any]):
        logging.info(f"Global step {state.global_step} reached. Model saving to {args.output_dir}")
        self.trainer.accelerator.wait_for_everyone()
        state_dict = self.trainer.accelerator.get_state_dict(self.trainer.model)
        unwrapped_model = self.trainer.accelerator.unwrap_model(self.trainer.model)
        if self.trainer.accelerator.is_main_process:
            save_dir = Path.cwd() / args.output_dir / f"checkpoint-{state.global_step}"
            save_dir.mkdir(parents=True, exist_ok=True)
            unwrapped_model.save_pretrained(save_dir, state_dict=state_dict)
            logging.info(f"Global step {state.global_step} reached. Model saved to {save_dir}")
            unwrapped_model.config.save_pretrained(save_dir)
            logging.info(f"Global step {state.global_step} reached. Configuration saved to {save_dir}")
            self.trainer.tokenizer.save_pretrained(save_dir)
            logging.info(f"Global step {state.global_step} reached. Tokenizer saved to {save_dir}")
        self.trainer.accelerator.wait_for_everyone()

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Dict[str, Any]):
        if args.save_model:
            self._save_model(args, state, control, **kwargs)
        # TODO: save optmizer and scheduler states
        return control


# model callback for FullyShardedDataParallel
class SaveFSDPModelCallback(SaveModelCallback):
    def __init__(self, trainer, save_steps=500):
        self.trainer = trainer
        self.save_steps = save_steps

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_save(args, state, control, **kwargs)
        if self.trainer.accelerator.is_main_process:
            # write a file to indicate the checkpoint is for model checkpoint
            save_dir = Path.cwd() / args.output_dir / f"checkpoint-{state.global_step}"
            with open(save_dir / "sharded_checkpoint.txt", "w") as f:
                f.write("Model checkpoint")

            # save metrics to checkpoint folder
            if hasattr(self.trainer, "eval_metrics") and state.global_step in self.trainer.eval_metrics:
                metrics = self.trainer.eval_metrics[state.global_step]  # use state.global_step to make sure the correct metrics is saved
                with open(os.path.join(save_dir, "evaluation_results.json"), "w") as f:
                    json.dump(metrics, f, indent=4)
                logging.info(f"Saved evaluation results to {save_dir}/evaluation_results.json")

    def _save_model(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Dict[str, Any]):
        logging.info(f"Global step {state.global_step} reached. Model saving to {args.output_dir}")
        self.trainer.accelerator.wait_for_everyone()
        unwrapped_model = self.trainer.accelerator.unwrap_model(self.trainer.model)

        # SHARDED_STATE_DICT saves shard per GPU separately which makes it quick to save or resume training from intermediate checkpoint.
        # When FULL_STATE_DICT is used, first process(rank 0) gathers the whole model on CPU and then saving it in a standard format.
        save_dir = Path.cwd() / args.output_dir / f"checkpoint-{state.global_step}"
        save_dir.mkdir(parents=True, exist_ok=True)
        # set this explicitly for FSDP to save full state dict for end of training according to https://huggingface.co/docs/transformers/main/en/fsdp
        self.trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        self.trainer.save_model(save_dir)
        logging.info(f"Global step {state.global_step} reached. Model saved to {save_dir}")

        if self.trainer.accelerator.is_main_process:
            unwrapped_model.config.save_pretrained(save_dir)
            logging.info(f"Global step {state.global_step} reached. Configuration saved to {save_dir}")
            self.trainer.tokenizer.save_pretrained(save_dir)
            logging.info(f"Global step {state.global_step} reached. Tokenizer saved to {save_dir}")  
        self.trainer.accelerator.wait_for_everyone()

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Dict[str, Any]):
        if args.save_model:
            self._save_model(args, state, control, **kwargs)
        # TODO: save optmizer and scheduler states
        return control


# model callback for DeepSpeed
class SaveDeepSpeedModelCallback(SaveModelCallback):
    def __init__(self, trainer, save_steps=500):
        self.trainer = trainer
        self.save_steps = save_steps

    def _save_model(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Dict[str, Any]):
        logging.info(f"Global step {state.global_step} reached. Model saving to {args.output_dir}")
        self.trainer.accelerator.wait_for_everyone()
        state_dict = self.trainer.accelerator.get_state_dict(self.trainer.deepspeed)
        unwrapped_model = self.trainer.accelerator.unwrap_model(self.trainer.deepspeed)
        if self.trainer.accelerator.is_main_process:
            save_dir = Path.cwd() / args.output_dir / f"checkpoint-{state.global_step}"
            save_dir.mkdir(parents=True, exist_ok=True)
            unwrapped_model.save_pretrained(save_dir, state_dict=state_dict)
            logging.info(f"Global step {state.global_step} reached. Model saved to {save_dir}")
            unwrapped_model.config.save_pretrained(save_dir)
            logging.info(f"Global step {state.global_step} reached. Configuration saved to {save_dir}")
            self.trainer.tokenizer.save_pretrained(save_dir)
            logging.info(f"Global step {state.global_step} reached. Tokenizer saved to {save_dir}")        
        self.trainer.accelerator.wait_for_everyone()

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Dict[str, Any]):
        if args.save_model:
            self._save_model(args, state, control, **kwargs)
        # TODO: save optmizer and scheduler states
        return control


class EvaluationCallback(TrainerCallback):
    def __init__(self, trainer, eval_steps=500):
        self.trainer = trainer
        self.eval_steps = eval_steps

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Dict[str, Any]):
        # use on_step_end to evaluate the model
        if args.run_validation and (state.global_step) % state.eval_steps == 0:
            # This callback should not change the control state, e.g. should_evaluate
            control_copy = deepcopy(control)
            self.trainer.evaluate()
            return control_copy

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Dict[str, Any]):
        if args.run_validation:
            # This callback should not change the control state, e.g. should_evaluate
            control_copy = deepcopy(control)
            self.trainer.evaluate()
            return control_copy


@dataclass
class Precision:
    """
    Precision is a dataclass to store the number of decimal points for each metric.
    """

    n_decimal_time: int
    n_decimal_memory: int
    n_decimal_TPS: int
    n_decimal_MFU: int


@dataclass
class State:
    """
    State is a dataclass to store the internal state of the efficiency callback.
    """

    n_warmup_steps: int = 0
    total_peak_memory_allocated: float = float("-inf")
    total_peak_memory_reserved: float = float("-inf")

    step_start_time: float = 0.0
    elapsed_time: float = 0.0

    elapsed_step: int = 0

    step_start_tokens_seen: int = 0
    elapsed_tokens_seen: int = 0

    step_start_flos: float = 0.0
    elapsed_flos: float = 0.0

    global_start_step: int = 0


@dataclass
class Time:
    """
    Time is a dataclass to store the time-related metrics.
    """

    step: int = 0
    step_time_sec: float = 0.0
    avg_step_time_sec: float = 0.0
    time_to_completion_sec: float = 0.0
    estimated_total_time_sec: float = 0.0


@dataclass
class Memory:
    """
    Memory is a dataclass to store the memory-related metrics.
    """

    step_peak_memory_allocated_MB: float = 0.0
    total_peak_memory_allocated_MB: float = 0.0


@dataclass
class TPS:
    """
    TPS is a dataclass to store the tokens per second metrics.
    """

    step_tokens_per_second: float = 0.0
    avg_tokens_per_second: float = 0.0


@dataclass
class MFU:
    """
    MFU is a dataclass to store the MFU metrics.
    """

    step_MFU: float = 0.0
    avg_MFU: float = 0.0


class EfficiencyCallback(TrainerCallback):
    """
    EfficiencyCallback is a callback to track the efficiency of the training process.
    The tracked stats include: step time, memory, throughput, and MFU.

    It requires including `--include_num_input_tokens_seen` and `logging_steps=1` in the training arguments.

    Args:
        n_warmup_steps: number of warmup steps
            The stats in the first n_warmup_steps will not be added into the aggregated stats
            This is because the first few steps might take longer due to jit compliation and other initialization overheads
        n_decimal_time: number of decimal points for time
        n_decimal_memory: number of decimal points for memory
        n_decimal_TPS: number of decimal points for TPS
        n_decimal_MFU: number of decimal points for MFU in percentage
    """

    def __init__(
        self,
        n_warmup_steps=2,
        n_decimal_time=2,
        n_decimal_memory=2,
        n_decimal_TPS=2,
        n_decimal_MFU=4,
    ):
        self.state = State(
            n_warmup_steps,
        )

        self.precision = Precision(
            n_decimal_time,
            n_decimal_memory,
            n_decimal_TPS,
            n_decimal_MFU,
        )

        self.time = Time()
        self.memory = Memory()
        self.tps = TPS()
        self.mfu = MFU()

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of the initialization of the [`Trainer`].
        """
        if not args.include_num_input_tokens_seen:
            raise Exception(
                'Please pass training argument "--include_num_input_tokens_seen" to track tokens per second'
            )
        if args.logging_steps != 1:
            raise Exception(
                "Please set logging_steps=1 to track the efficiency metrics accurately"
            )

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # if loaded from checkpoints, global_start_step is not 1 but state.global_step
        self.state.global_start_step = state.global_step

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float],
        **kwargs,
    ):
        if state.global_step < (
            self.state.global_start_step + self.state.n_warmup_steps
        ):
            return
        else:
            # spread self.time, self.memory, self.tps, self.mfu to logs
            logs.update(self.time.__dict__)
            logs.update(self.memory.__dict__)
            logs.update(self.tps.__dict__)
            logs.update(self.mfu.__dict__)

            logging.info(logs)

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        # memory
        torch.cuda.reset_peak_memory_stats()

        # time
        self.state.step_start_time = time.perf_counter()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step < (
            self.state.global_start_step + self.state.n_warmup_steps
        ):
            # The end the current step_start_tokens_seen and step_start_flos are the start of next iteration

            # tokens
            self.state.step_start_tokens_seen = state.num_input_tokens_seen
            # flos
            self.state.step_start_flos = state.total_flos
            return

        # time
        current_time = time.perf_counter()
        step_time = current_time - self.state.step_start_time
        self.state.elapsed_time += step_time

        # step
        global_step = state.global_step
        self.state.elapsed_step += 1
        avg_step_time = self.state.elapsed_time / self.state.elapsed_step

        self.time.step = global_step
        self.time.step_time_sec = round_to_n_decimal(
            step_time, self.precision.n_decimal_time
        )
        self.time.avg_step_time_sec = round_to_n_decimal(
            avg_step_time, self.precision.n_decimal_time
        )
        self.time.time_to_completion_sec = round_to_n_decimal(
            avg_step_time * (state.max_steps - global_step),
            self.precision.n_decimal_time,
        )
        self.time.estimated_total_time_sec = round_to_n_decimal(
            avg_step_time * state.max_steps, self.precision.n_decimal_time
        )

        # memory
        step_peak_memory_allocated = torch.cuda.memory.max_memory_allocated()
        step_peak_memory_reserved = torch.cuda.memory.max_memory_reserved()

        self.memory.step_peak_memory_allocated_MB = round_to_n_decimal(
            step_peak_memory_allocated / M_BIN_UNIT, self.precision.n_decimal_memory
        )
        self.state.total_peak_memory_allocated = max(
            self.state.total_peak_memory_allocated, step_peak_memory_allocated
        )
        self.memory.total_peak_memory_allocated_MB = round_to_n_decimal(
            self.state.total_peak_memory_allocated / M_BIN_UNIT,
            self.precision.n_decimal_memory,
        )

        self.memory.step_peak_memory_reserved_MB = round_to_n_decimal(
            step_peak_memory_reserved / M_BIN_UNIT, self.precision.n_decimal_memory
        )

        self.state.total_peak_memory_reserved = max(
            self.state.total_peak_memory_reserved, step_peak_memory_reserved
        )

        self.memory.total_peak_memory_reserved_MB = round_to_n_decimal(
            self.state.total_peak_memory_reserved / M_BIN_UNIT,
            self.precision.n_decimal_memory,
        )

        # tokens
        step_tokens_seen = (
            state.num_input_tokens_seen - self.state.step_start_tokens_seen
        )

        self.state.elapsed_tokens_seen += step_tokens_seen

        self.tps.step_tokens_per_second = round_to_n_decimal(
            step_tokens_seen / step_time,
            self.precision.n_decimal_TPS,
        )

        self.tps.avg_tokens_per_second = round_to_n_decimal(
            self.state.elapsed_tokens_seen / self.state.elapsed_time,
            self.precision.n_decimal_TPS,
        )

        # flos
        step_flos = state.total_flos - self.state.step_start_flos
        self.state.elapsed_flos += step_flos

        # MFU
        # 1. Definition
        #
        # MFU is defined as (achieved TPS) / (theoretical maximum TPS) = (achieved floating point operations per sec) / (theoretical maximum floating point operations per sec)
        # Crucially, the "theoretical maximum" throughput only accounts for the required operations to compute the forward+backward passes, and not rematerialization. MFU therefore allows fair comparisons
        # between training runs on different systems, as the numerator is simply the observed tokens-per-second, and the denominator is only dependent on the model architecture and published maximum FLOPs for a given system.
        # Ref: https://arxiv.org/pdf/2204.02311
        # The benefit of MFU is that it
        #
        # 2. Implementation in huggingface
        #
        # current_flos = 6 * estimate_tokens(input_dict) * num_parameters()
        # total_flos = sum(current_flos) # across all GPUs
        # Ref: https://github.com/huggingface/transformers/blob/616bb11d487aabc231bb230b245c42214ea4b254/src/transformers/modeling_utils.py#L1196
        #
        # 3. Derive MFU on rank 0
        #
        # rank_0_flos = tatal_flos / n_gpus = measured_flos / effecitve_n_gpus
        # rank_0_MFU = rank_0_flos / step_time
        #
        # For FSDP, num_parameters() is (1 / n_gpus) of the total parameters. So, the effective_n_gpus = 1
        # For HSDP, num_parameters() is (1 / local_world_size) of the total parameters. So, the effective_n_gpus = n_nodes
        # For no sharding and zero-2, num_parameters() is the total parameters. So, the effective_n_gpus = n_gpus

        num_gpus = EfficiencyCallback._get_effective_num_gpus()
        step_achieved_tflops = step_flos / step_time / num_gpus / T_DEC_UNIT

        avg_achieved_tflops = (
            self.state.elapsed_flos / self.state.elapsed_time / num_gpus / T_DEC_UNIT
        )

        precision_bits = 16 if args.bf16 or args.fp16 else 32
        gpu_peak_tflops = EfficiencyCallback._get_gpu_peak_tflops(precision_bits)

        self.mfu.step_MFU = round_to_n_decimal(
            step_achieved_tflops / gpu_peak_tflops, self.precision.n_decimal_MFU
        )

        self.mfu.avg_MFU = round_to_n_decimal(
            avg_achieved_tflops / gpu_peak_tflops, self.precision.n_decimal_MFU
        )

        # The end the current step_start_tokens_seen and step_start_flos are the start of next iteration

        # tokens
        self.state.step_start_tokens_seen = state.num_input_tokens_seen
        # flos
        self.state.step_start_flos = state.total_flos

    @staticmethod
    def _get_effective_num_gpus():
        # Calculate the number of effective GPUs for the total FLOPs in order to calculate the single GPU FLOP
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        if transformers.utils.strtobool(os.environ.get("ACCELERATE_USE_FSDP", "false")):
            sharding_strategy = os.environ.get(
                "FSDP_SHARDING_STRATEGY", FSDP_SHARDING_STRATEGY[0]
            ).upper()

            # Either specified as string or enum number
            if sharding_strategy in {
                "FULL_SHARD",
                str(FSDP_SHARDING_STRATEGY.index("FULL_SHARD") + 1),
            }:
                return 1

            elif sharding_strategy in {
                "HYBRID_SHARD",
                str(FSDP_SHARDING_STRATEGY.index("HYBRID_SHARD") + 1),
            }:
                return world_size // int(os.environ.get("LOCAL_WORLD_SIZE", 1))
            else:
                return world_size

        assert (
            world_size != 0
        ), "WORLD_SIZE should be set to a positive integer. For single GPU training, please explicitly set WORLD_SIZE=1."

        # TODO: add deepspeed support
        return world_size

    @staticmethod
    def _get_gpu_peak_tflops(precision_bits: int = 16):
        if precision_bits not in {16, 32}:
            raise Exception(f"Precision bits {precision_bits} is not supported")

        device_name = torch.cuda.get_device_name()

        if "A100" in device_name:
            # data from https://www.nvidia.com/en-us/data-center/a100/
            return 312 if precision_bits == 16 else 156
        elif "H100" in device_name:
            # data from https://www.nvidia.com/en-us/data-center/h100/
            # NOTE: Specifications are one-half lower without sparsity.
            if "NVL" in device_name:
                return 1979 if precision_bits == 16 else 989
            elif "PCIe" in device_name:
                return 756 if precision_bits == 16 else 378
            else:  # for SXM and other variants
                return 989 if precision_bits == 16 else 494
        elif "V100" in device_name:
            if "NVL" in device_name:
                return 125
            else:
                return 112
        return None


def trace_handler(
    dir_name: str,
    enable_trace: bool = True,
    enable_memory_timeline: bool = True,
):

    # make dir_name a absolute path
    dir_name = os.path.abspath(dir_name)

    def handler_fn(prof) -> None:

        os.makedirs(dir_name, exist_ok=True)

        file_prefix = f"{socket.gethostname()}_{os.getpid()}_{time.time_ns()}"

        if enable_trace is True:
            # Export chrome trace
            trace_name = f"{file_prefix}.pt.trace.json"
            trace_path = os.path.join(dir_name, trace_name)
            prof.export_chrome_trace(trace_path)
            logging.info(f"{GREEN}Profiler data saved to {trace_path}{RESET}")
            logging.info(
                f"{YELLOW}Please `pip install torch-tb-profiler` and run `tensorboard --logdir {dir_name}` to view the trace{RESET}"
            )

        if enable_memory_timeline is True:
            # Export memory timeline
            memory_timeline_name = f"{file_prefix}.html"
            memory_timeline_path = os.path.join(dir_name, memory_timeline_name)
            prof.export_memory_timeline(memory_timeline_path)
            logging.info(f"{GREEN}Memory timeline data saved to {memory_timeline_path}{RESET}")
            logging.info(
                f"{YELLOW}Please download {memory_timeline_path} and open with browser to view the memory timeline.{RESET}"
            )

    return handler_fn


class ProfilerCallback(transformers.TrainerCallback):
    """
    ProfilerCallback uses PyTorch Profiler to profile the training process. It skips the first `warmup` steps and profiles the next `active` steps.
    It is recommended to only enable ProfilerCallback on rank 0 process for distributed training.
    For example,

    .. code-block:: python

        if not torch.distributed.distributed_c10d.is_initialized() or torch.distributed.get_rank() == 0:
            callbacks.append(ProfilerCallback())


    If enable_trace is True, it will export the trace data to `output_dir`.
    If enable_memory_timeline is True, it will export the memory timeline data to `output_dir`.
    At least one of enable_trace and enable_memory_timeline should be True.

    Args:
        warmup: Number of steps to warmup the profiler
        active: Number of steps to profile
        enable_trace: Enable trace output
        enable_memory_timeline: Enable memory timeline output
        output_dir: Output directory for profiler data
    """

    def __init__(
        self,
        wait: int = 0,
        warmup: int = 2,
        active: int = 2,
        output_dir: str = "./profiler-output",
        enable_trace: bool = True,
        enable_memory_timeline: bool = True
    ):

        assert (
            enable_trace or enable_memory_timeline
        ), "At least one of enable_trace and enable_memory_timeline should be True."

        self.prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=wait, warmup=warmup, active=active, repeat=1, skip_first=0
            ),
            on_trace_ready=trace_handler(
                output_dir, enable_trace, enable_memory_timeline
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

    def on_train_begin(self, args, state, control, **kwargs):
        self.prof.start()

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()
