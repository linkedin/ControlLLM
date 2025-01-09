import os
import logging
from typing import List, Optional, Union
from dataclasses import dataclass, field
import torch
from transformers import Seq2SeqTrainingArguments
from transformers.utils import add_start_docstrings
from transformers.debug_utils import DebugOption
from controlllm.configs.custom_llama_recipes.training import train_config as TrainConfigRecipes


# factory function that returns the appropriate class based on the configuration trainer: str = "transformers" or "native"
def TrainConfig() -> "TrainConfigCommon":
    if os.getenv('TRAINER'):
        trainer: str = os.getenv('TRAINER')  # type: ignore
    else:
        trainer = "native"  # default to native, alternative: transformers

    logging.info(f"Using trainer: {trainer}")

    if trainer == "transformers":
        return TrainConfigTransformers()
    elif trainer == "native":
        return TrainConfigNative()
    else:
        raise ValueError(f"Invalid trainer: {trainer}")


# # for SFT
# def default_datasets():
#     return ['OpenCoderSFTStage1']


# # for SFT
# def default_datasets():
#     return ['OpenCoderSFTStage2']


# for SFT
def default_datasets():
    return ['OpenMathInstruct2Dataset']


# # for pretraining
# def default_datasets():
#     return ["LlamaSynEZhDataset", "Llama3ChineseDataset", "AlpacaZhDataset", "AlpacaGPT4ZhDataset", "RuozhibaZhDataset", "OaastSFTZhDataset", "StemZhDataset"]


@dataclass
class TrainConfigCommon:
    # specify the dataset to train on, choose the name of the dataset from configs/dataset.py
    # samsum_dataset is a small dataset good for debugging the code when new development is done
    dataset: List[str] = field(default_factory=default_datasets)
    num_train_epochs: int = 20
    max_train_step: int = -1  # max training step, set to -1 to disable it
    # adjust the batch size of maximize the GPU usage by default:
    # full parameter: (1 node, 4k, 70B): max 8. (1 node, 4k, 7B): max 20. (1 node, 8k, 8B): max 4
    # expansion of 8 layers: (1 node, 8k, 8B, freezing 32 layers): max 12 for pretraining, 4 for sft, (1 node, 8k, 7B): max 10
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 1
    context_length: int = 8192
    # gradient accumulation for single-node training: allow to train with a larger effective batch size with limited by GPU memory
    # gradient accumulation for multi-node training: reduce communication overhead, because model parameters are updated less frequently
    gradient_accumulation_steps: int = 8  # normally 2~16, let's be a bit more aggressive with this on single node, set to 1 to disable it

    # save model and optimizer end of epoch as well as every save_steps
    save_model: bool = True  # this includes saving model state, configuration and tokenizer
    save_optimizer: bool = False
    # how frequent you want the model to be saved
    save_steps: int = 2000  # 1000 iterations is about 8 hours for 7B, set to 500 for 70B which is about 10 hours
    save_epoch: int = 1  # default to every 1 epoch, in native trainer, it is every save_step OR save_epoch
    # dir to save checkpoint and optimizer state
    output_dir: str = '/shared/user/fine-tune/controlllm/model/llama3-10b-hf-checkpoint-sft-padding-openmath-train/'
    # whether to run validation during training, suggest to disable it for pretraining and enable it for SFT
    run_validation: bool = True
    evaluation_strategy: str = "steps"
    # how frequent you want the model to be evaluated or set run_validation to False if you don't want to evaluate
    eval_steps: int = 1000  # default to every 500 iterations
    eval_epoch: int = 1  # default to every 1 epoch, in native trainer, it is every eval_step OR eval_epoch
    eval_in_memory: bool = False  # evaluate in memory, set to True to reduce I/O and avoid torch.distributed.barrier(), however, it is accurate only for addictive metrics
    # stop by max_eval_step for eval, set to 0 or negative to disable it
    max_eval_step: int = 500
    hf_hub_metrics_cache_dir: str = "/shared/public/data/controlllm/metrics/"  # cache for huggingface metrics

    # observe your initial training steps e.g. loss and make adjustments, 1e-4 can lead to nan loss
    learning_rate: float = 5e-5
    weight_decay: float = 1e-2  # for AdamW optimizer, takes default betas, eps, but set fixed weight decay if not None, works from 1e-2 to 1e-4
    weight_decay_ratio: float = 0.1  # Dynamically compute weight decay by weight_decay_ratio times the learning rate at each step. None or 0 to disable it. For native trainer, it is higher priority than weight_decay if both are set.
    # if True, use WarmupCosineAnnealingLR - cosine annealing learning rate scheduler updated per iteration, else use StepLR(only supported in native trainer) - decaying by gamma every step_size epochs
    lr_scheduler_per_iter: bool = True  # if True, lr updated per training iteration, else per epoch
    # StepLR learning rate scheduler: decays the learning rate by multiplied with gamma once every step_size epochs
    gamma: float = 0.85  # used in StepLR, needs lr_scheduler_per_iter == False, only support in native trainer
    step_size: int = 1  # used in StepLR, needs lr_scheduler_per_iter == False, only support in native trainer
    # WarmupCosineAnnealingLR learning rate scheduler: warm up and then decays the learning rate by cosine annealing schedule
    warmup_steps: int = 1000  # used in WarmupCosineAnnealingLR, supported in both trainers
    decay_steps: int = None  # used WarmupCosineAnnealingLR, supported in both trainers. Set to None or 0 to be initialized dynamically with total_steps  - train_config.warmup_steps
    eta_min: float = 1e-5  # minimum learning rate - used in WarmupCosineAnnealingLR, supported in both trainers
    # gradient clipping to prevent exploding gradients
    max_grad_norm: float = 1.0  # set to None or <=0.0 to disable gradient clipping

    enable_tensorboard: bool = True  # enable tensorboard for training
    logging_dir: Optional[str] = field(default=None, metadata={"help": "Tensorboard log dir."})
    logging_steps: int = 500  # how frequent you want the training metrics to be logged, best is a multiply of gradient_accumulation_steps
    # saves training metrics to a json file for later plotting, note that this will save metrics per step so it can slow down training
    save_metrics: bool = False
    load_best_model_at_end: bool = False  # TODO: implement final model loading and do prediction with the best model
    no_cuda: bool = False if torch.cuda.is_available() else True

    # default to main process(0), for fully cached datasets with minimal I/O, lower number like 1 , 2 is usually enough
    # for datasets with heavy I/O, higher number like 4, 8, 16 is usually better, less context switch when number is smaller
    num_workers_dataloader: int = 0  # set it to 0 if https://github.com/pytorch/pytorch/issues/8976
    # packing concatenates the tokenized samples into long sequences filling up the context length of the model
    # note: packing could reduce number of records by ~5x of course depending on length distribution of dataset, suggest to use padding for SFT, packing for pretrain
    batching_strategy: str = "padding"  # padding or packing
    drop_last: bool = True  # drop the last incomplete batch, used when dynamic_batch_size is False or batching_strategy is packing
    dynamic_batch_size: bool = False  # set to True to enable dynamic batch size, this will override per_device_train_batch_size. Used for padding strategy
    max_tokens_per_batch: int = -1  # maximum tokens per batch, enabled when dynamic_batch_size is True, used for padding strategy, -1 means context_length * per_device_train_batch_size
    memory_per_token: int = -1  # memory used per token in bytes, used when dynamic_batch_size is True to calculate max_tokens_per_batch, depending the infra and model size etc.. -1 to compute it automatically by max_tokens_per_batch = context_length * per_device_train_batch_size
    handle_long_sequences: bool = True  # enable handling for long sequences, recommend to set to True as always to avoid OOM for batching_strategy packing
    long_sequence_threshold: int = 8192 * 2  # sequences longer than threshold are dropped, sequences longer than context_length are treated specially with smaller batch size, used for packing strategy and handle_long_sequences is True
    curriculum_learning: bool = False  # enable curriculum learning
    curriculum_phases: int = 3  # number of phases in curriculum learning
    # precompute batches in data preprocessing, combined means precompute dynamic batches for all datasets together with variable sequence length and variable batch size to NOT exceed max_tokens_per_batch
    precompute_batches: str = None  # None, "per_dataset" or "combined", None is to disable it

    # sharding the model weights across multiple GPUs, this is useful for large models that do not fit on a single GPU
    # note: only one of fsdp or deepspeed can be enabled at a time, for transformers trainer, config TrainConfigTransformers->"deepspeed, fsdp, fsdp_config"
    enable_fsdp: bool = True  # enable fsdp for training, for native trainer, fsdp config is in ./configs/fsdp.py
    enable_deepspeed: bool = False  # enable deep speed for training
    quantization: bool = False  # enable quantization for training
    # True to save memory at the expense of slower backward pass(70B can train on single node by this)
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict = field(default_factory=lambda: {"use_reentrant": False})
    # peft
    use_peft: bool = False  # enable PEFT for training
    peft_method: str = "lora"  # None, lora, llama_adapter, prefix
    freeze_layers: bool = False  # enable layer freezing for training
    num_unfrozen_layers: int = 8  # number of layers to unfrozen, used when freeze_layers is True
    unfrozen_strategy: str = "interweave"  # top, bottom or interweave, interweave is to unfreeze one layer every few layers until num_unfrozen_layers

    # resume_from_checkpoint
    resume_from_latest: bool = True  # resume from latest checkpoint of the output_dir, TODO: resume_from_best
    # which model to load for training, this is useful for continuing training from a checkpoint
    resume_checkpoint_folder: str = None  # "checkpoint-3", change 3 to the global step of the checkpoint you want to load, None to respect resume_from_latest
    # starting iteration after restarting the training process, useful for long running training
    consumed_iters = 0
    overwrite_output_dir: bool = False  # overwrite the output directory if it exists
    # wandb
    use_wandb: bool = False  # enable wandb to log the experiment
    # timeout for initiate_process_group, defaults to 1800 seconds = 30 minutes
    ddp_timeout: int = 36000  # 10 hours

    # following profiler is supported in native trainer only, TODO: add support for transformers trainer
    enable_memory_trace: bool = False  # enable memory tracing in the training loop, this will slow down the training a bit
    enable_memory_profiling: bool = False  # enable profiling of memory in the training loop, this will slow down the training a lot
    flop_counter: bool = False  # Enable flop counter to measure model throughput, can not be used with pytorch profiler at the same time. For transformers trainer, set to True requires `--include_num_input_tokens_seen` and `logging_steps=1`.
    flop_counter_start: int = 3  # The step to start profiling, default is 3, which means after 3 steps of warmup stage, the profiler will start to count flops.
    use_profiler: bool = False  # Enable pytorch profiler, can not be used with flop counter at the same time.
    wait_step, warmup_step, active_step = 1, 2, 4  # The steps to wait, warmup, active, used for pytorch profiler.
    profiler_dir: Optional[str] = field(default=None, metadata={"help": "Profile results dir."})  # will be used if use_profiler is True
    # debugging for both trainers
    debug: bool = False  # False by default, logging level is set to INFO for rank == 0, ERROR otherwise. Set to True, to set all ranks at INFO level. This helps to debug failures in different ranks in distributed training.

    def __post_init__(self):
        # output_dir is mandatory to save the model, optimizer state, logging, profiling results etc.
        if self.output_dir:
            self.output_dir = os.path.expanduser(self.output_dir)
            if not os.path.exists(self.output_dir):
                logging.info(f"Output directory: {self.output_dir} does not exist, creating it.")
            os.makedirs(self.output_dir, exist_ok=True)

        # Set logging_dir based on output_dir
        if self.logging_dir is None and self.output_dir:
            def default_logdir() -> str:
                import socket
                from datetime import datetime

                current_time = datetime.now().strftime("%b%d_%H-%M-%S")
                return os.path.join("runs", current_time + "_" + socket.gethostname())

            self.logging_dir = os.path.join(self.output_dir, default_logdir())
        if self.logging_dir:
            self.logging_dir = os.path.expanduser(self.logging_dir)

        # Set profiler_dir based on output_dir
        if self.profiler_dir is None and self.output_dir:
            def default_profilerdir() -> str:
                import socket
                from datetime import datetime

                current_time = datetime.now().strftime("%b%d_%H-%M-%S")
                return os.path.join("profiler", current_time + "_" + socket.gethostname())

            self.profiler_dir = os.path.join(self.output_dir, default_profilerdir())
        if self.profiler_dir:
            self.profiler_dir = os.path.expanduser(self.profiler_dir)


@dataclass
@add_start_docstrings(Seq2SeqTrainingArguments.__doc__)
class TrainConfigTransformers(TrainConfigCommon, Seq2SeqTrainingArguments):
    # Training arguments specific for transfomers trainer
    # Important note: argument has to be typed to take effect in huggingface TrainingArguments initialization

    # we disable the default saving strategy of transformers and use our own save strategy
    # save_strategy: str = "no"  # "steps" or "epoch" or no
    # note that default learning rate scheduler of transformer is linear warmup and then linear decay, we replaced it with cosine annealing
    # warmup_steps: int = 500
    # Whether or not to group together samples of roughly the same length in the training dataset (to minimize padding applied and be more efficient)
    group_by_length: bool = True  # Not used for now as we have customized data loader which will do the same
    report_to: str = "none"  # trainer knows how to format the output, to print/log training loss during training loop, `"all"`, `"tensorboard"`, `"wandb", `"flyte"`, `"mlflow"`, or `"none"`

    # for transformers trainer, setting flop_counter to True requires `--include_num_input_tokens_seen` and `logging_steps=1` or 8
    # include_num_input_tokens_seen: bool = True  # include the number of input tokens seen in the training loop
    # logging_steps: int = 1  # how frequent you want the training metrics to be logged, best is a multiply of gradient_accumulation_steps

    # deepspeed: str = "/home/jobuser/controlllm/configs/z3_++.json"
    fsdp: str = "hybrid_shard auto_wrap"
    fsdp_config: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fsdp.json")
    # enable mixed precision training benefits of reduced memory usage and faster computation
    bf16: bool = True
    # another way of doing mixed precision
    fp16: bool = False
    save_safetensors: bool = True
    debug: Union[str, List[DebugOption]] = field(
        default="",
        metadata={
            "help": (
                "Whether or not to enable debug mode. Current options: "
                "`underflow_overflow` (Detect underflow and overflow in activations and weights), "
                "`tpu_metrics_debug` (print debug metrics on TPU)."
            )
        },
    )
    max_steps: int = -1
    save_strategy: str = "steps"
    eval_strategy: str = "steps"
    include_num_input_tokens_seen: bool = False
    metric_for_best_model: str = "eval_bleu"  # metric to use for state.best_metric, state.best_model_checkpoint
    # Additional trainer arguments
    trainer: str = "transformers"

    def __post_init__(self):
        # Ensure __post_init__ of both base classes are called
        super().__post_init__()  # Calls TrainConfigCommon's __post_init__

        # When using FSDP full shard, instead of using `gradient_checkpointing` in TrainingArguments, please use `activation_checkpointing` in `fsdp_config`.
        # The former introduces a redundant AllGather operation in backward pass. Reference: https://github.com/huggingface/transformers/issues/30404
        # This is for transformers trainer only
        if self.enable_fsdp:
            self.gradient_checkpointing = False  # disable gradient checkpointing when fsdp is enabled, config it in fsdp.json

        # To make it consistent with native trainer, save and eval every save_steps and eval_steps without considering gradient_accumulation_steps
        logging.info(f"Setting eval_steps to {self.eval_steps // self.gradient_accumulation_steps + 1} which is {self.eval_steps} // {self.gradient_accumulation_steps} + 1 to be consistent with native trainer")
        self.eval_steps = self.eval_steps // self.gradient_accumulation_steps + 1
        logging.info(f"Setting save_steps to {self.save_steps // self.gradient_accumulation_steps + 1} which is {self.save_steps} // {self.gradient_accumulation_steps} + 1 to be consistent with native trainer")
        self.save_steps = self.save_steps // self.gradient_accumulation_steps + 1

        Seq2SeqTrainingArguments.__post_init__(self)  # Explicitly call Seq2SeqTrainingArguments' __post_init__


@dataclass
class TrainConfigNative(TrainConfigCommon, TrainConfigRecipes):
    # Training arguments specific for native pytorch trainer
    # num_epochs: int = 3
    # mixed precision training using gradient scaler to handle gradient update when fp16 is enabled
    # when using bf16 instead of fp16, this could be set to False which will disable the scaler during training, 
    # so False means direct optimizer steps are performed without loss scaling, leveraging BF16's broader dynamic range.
    use_fp16: bool = False
    mixed_precision: bool = True
    one_gpu: bool = False

    # Additional trainer arguments
    trainer: str = "native"
