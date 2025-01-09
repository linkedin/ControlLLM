import os
import yaml
import logging
from pathlib import Path
from typing import Any
from functools import partial
from dataclasses import asdict
import accelerate
from accelerate.utils import is_xpu_available
import torch
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from torch.distributed.fsdp import StateDictType, FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_pt_utils import get_model_param_count
from peft import get_peft_model, prepare_model_for_kbit_training

from controlllm.utils.config_utils import Configs
from controlllm.utils.custom_llama_recipes import AnyPrecisionAdamW, freeze_transformer_layers
from controlllm.utils.custom_llama_recipes import load_checkpoint, load_optimizer, is_load_checkpoint_needed
from controlllm.utils.custom_llama_recipes import apply_fsdp_checkpointing, fsdp_auto_wrap_policy, hsdp_device_mesh, get_policies
from controlllm.utils.model_expander import ModelExpander
from controlllm.configs.loading import ModelLoadingConfig
from controlllm.utils.callbacks import SaveModelCallback, SaveFSDPModelCallback, SaveDeepSpeedModelCallback, EvaluationCallback, EfficiencyCallback, ProfilerCallback


class ModelLoader:
    def __init__(self, configs: Any):
        self.configs: Configs = configs

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(**asdict(self.configs.tokenizer_loading_config))

        # load model with cpu memory efficient loading if required to avoid OOM for large models
        self._load_model()

        # log the trainable model size
        self._log_model_size()

        # prepare the loaded model for fsdp, quantization, peft, layer freeze, layer explansion and to the right device for training
        self._prepare_model()

        # prepare the optimizer and learning rate scheduler
        self._prepare_optimizer()

        # prepare callbacks for saving model, evaluation and profiling
        self._prepare_callbacks()

    def _load_model(self):
        # set the resume_checkpoint_folder to latest checkpoint if required
        self._handle_resume_checkpoint_folder()

        # set the 'trained_from' attribute in the model configuration to indicate the source model. This helps trace the origin of the saved checkpoint.
        logging.info(f"Setting the 'trained_from' attribute in the model configuration to {self.configs.model_loading_config.pretrained_model_name_or_path}")
        trained_from = self.configs.model_loading_config.pretrained_model_name_or_path

        # expand the transformer block if required
        if self.configs.model_loading_config.num_exp_layers > 0:
            self._expand_model()

        logging.info(f"Model loading: Loading model from {self.configs.model_loading_config.pretrained_model_name_or_path}")
        with_meta = False
        if self.configs.train_config.trainer == "native":
            if self.configs.train_config.enable_fsdp and self.configs.fsdp_config.fsdp_cpu_ram_efficient_loading:
                """
                for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
                this avoids cpu oom when loading large models like llama 70B, in which case
                model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
                overhead and currently requires latest torch version(2.2.0).
                """
                if self.configs.setup_config.local_rank == 0:
                    self.model = AutoModelForCausalLM.from_pretrained(**asdict(self.configs.model_loading_config))
                else:
                    auto_config = AutoConfig.from_pretrained(**asdict(self.configs.model_loading_config))
                    with torch.device("meta"):
                        self.model = AutoModelForCausalLM.from_config(auto_config)
                    with_meta = True
            else:
                self.model = AutoModelForCausalLM.from_pretrained(**asdict(self.configs.model_loading_config))
            # all ranks finished loading
            torch.distributed.barrier()
        else:
            # for training with transformers, if training argument is initialized with fsdp, it will be low cpu fsdp automatically
            # however, when trainer is native, transformer's TrainingArguments is not used so we need the code from above
            # source: https://github.com/huggingface/transformers/pull/25107
            self.model = AutoModelForCausalLM.from_pretrained(**asdict(self.configs.model_loading_config))

        # add special tokens and resize the embedding matrix if required
        self._add_special_tokens_resize_embedding()

        # resume the weights trained and saved by native trainer - peft or sharding TODO resume lr scheduler
        if self.resume_from_native_checkpoint and self.configs.train_config.resume_checkpoint_folder:
            logging.info(f"Post model loading: Resuming training with peft or sharded state dict from {self.configs.train_config.resume_checkpoint_folder}. "
                         "Alternatively, use checkpoint_converter.py to convert the checkpoint to transformers format which is recommended.")
            if self.configs.train_config.trainer == "transformers":
                logging.info(f"Resuming checkpoint {self.resume_checkpoint_path} were saved by native trainer, but transformers trainer is used. Setting resume_checkpoint_path to None since no {TRAINER_STATE_NAME} is to resume from.")
                self.resume_checkpoint_path = None
                logging.info(f"However, state dict will be loaded to resume the training. Loading weights from {self.configs.train_config.resume_checkpoint_folder} ...")
            load_checkpoint(model=self.model, rank=self.configs.setup_config.rank, cfg=self.configs.train_config, with_meta=with_meta)

        # resume the weights trained and saved by transformers trainer, note that we are relying on transformers trainer's own resume logic
        if self.resume_from_transformers_checkpoint:
            if self.configs.train_config.trainer == "transformers":
                logging.info(f"Post model loading: Will be resuming training with transformers checkpoint from {self.configs.train_config.resume_checkpoint_folder} before training starts")
                # check if there is trainer_state.json in the checkpoint folder, if not, can't resume training, the checkpoint is saved by end of training
                if not Path(self.resume_checkpoint_path).joinpath(TRAINER_STATE_NAME).exists():
                    raise ValueError(f"{TRAINER_STATE_NAME} not found in {self.resume_checkpoint_path}. Can't resume training. Please put the path in configs/loading.py to load from pretrained")
            else:
                logging.info(
                    "Post model loading: Resuming training by native trainer with state dict saved by transformers trainer from "
                    f"{self.configs.train_config.resume_checkpoint_folder}. Note that it will raise an error if the model is not saved with full state dict or shared state dict in /pytorch_model_fsdp_0, "
                    "switch to Transformers trainer or call checkpoint_converter.py to convert checkpoint for resuming."
                )
                # transformers trainer counts every gradient_accumulation_steps as 1 step while native trainer counts every step as 1 step
                self.configs.train_config.consumed_iters = ((self.configs.train_config.consumed_iters - 1) * self.configs.train_config.gradient_accumulation_steps) + 1
                logging.info(f"Consumed iters has been adjusted to {self.configs.train_config.consumed_iters}, match the native trainer's consumed iters by multiplying with gradient_accumulation_steps {self.configs.train_config.gradient_accumulation_steps} ...")
                load_checkpoint(model=self.model, rank=self.configs.setup_config.rank, cfg=self.configs.train_config, with_meta=with_meta)

        # self.configs.model_loading_config.pretrained_model_name_or_path may have been updated during model expansion, so update config with the original pretrained_model_name_or_path
        self.model.config.trained_from = trained_from
        self.model.config.trained_by = self.configs.train_config.dataset
        logging.info(
            f"Model loaded from {self.configs.model_loading_config.pretrained_model_name_or_path}"
            f"{'. Resumed from ' + self.configs.train_config.resume_checkpoint_folder + '!' if self.resume_from_native_checkpoint or self.resume_from_transformers_checkpoint else ''}"
        )

    def _expand_model(self):
        #  "checkpoint-seed" folder in output_dir is a special folder to save the expanded model, don't create the folder if it doesn't exist yet
        self.exp_model_save_dir = Path.cwd() / self.configs.train_config.output_dir / f"seed-checkpoint-expand-{self.configs.model_loading_config.num_exp_layers}-{self.configs.model_loading_config.expand_type}"

        # make sure system loads the model from the expanded model checkpoint folder if it was expanded, expand it if not expanded yet
        if not self.exp_model_save_dir.exists():
            if self.configs.setup_config.rank == 0:
                import subprocess
                # this code block calls a script to expand and save the model in a subprocess.
                # it is done in a subprocess because deepspeed can only be initialized once,
                # and this ensures that the model expansion and saving process does not interfere with the main process.
                # self.configs.model_loading_config.num_ori_layers is used to check if the model has already been expanded in subprocess
                logging.info(f"Expanding the model by interweaving {self.configs.model_loading_config.num_exp_layers} new layers using a subprocess to avoid DeepSpeed initialization issues.")
                process = subprocess.run(
                    ['python', 'controlllm/utils/model_expander.py', ModelLoadingConfig.serialize_model_config(self.configs.model_loading_config),
                    self.exp_model_save_dir, self.resume_checkpoint_path if self.configs.train_config.resume_checkpoint_folder else ""],
                    check=True
                )
                process.check_returncode()  # check if the process was successful
                if self.exp_model_save_dir.exists():
                    logging.info(f"Model expanded and saved successfully to {self.configs.train_config.output_dir}")
                    self.tokenizer.save_pretrained(self.exp_model_save_dir)
                    logging.info(f"Tokenizer saved to {self.exp_model_save_dir}")
                else:
                    logging.info(f"Model expansion skipped because {self.configs.model_loading_config.pretrained_model_name_or_path} was already expanded.")
                torch.distributed.barrier()
            else:
                logging.info("Waiting for main process to perform expand and save")
                # all ranks wait for rank 0 to finish expanding the model
                torch.distributed.barrier()
        else:
            logging.warning(f"Model has already been expanded in {self.exp_model_save_dir}. Skipped expansion.")

        # if the expanded model checkpoint is found, load the model from the expanded model checkpoint, note that this seemly redundant check is necessary because pretrained model might be expanded already
        if self.exp_model_save_dir.exists():
            logging.info(f"Model expansion: Setting the model_loading_config.pretrained_model_name_or_path to {self.exp_model_save_dir}")
            self.configs.model_loading_config.pretrained_model_name_or_path = str(self.exp_model_save_dir)

        # register the expanded model classes with new model architecture
        logging.info(f"Model expansion: Registering the expanded model classes with new model architecture by {self.configs.model_loading_config.pretrained_model_name_or_path}")
        ModelExpander.register_expansion_classes(self.configs.model_loading_config.pretrained_model_name_or_path)

    def _add_special_tokens_resize_embedding(self):
        # make sure to have a pad_token_id which is different from eos_token_id for padding
        # pad_token_id = eos_token_id can result in the model not properly predicting EOS (End of Sentence) tokens during generation.
        special_tokens_dict = dict()
        if self.tokenizer.pad_token is None:
            logging.info(f"Adding pad_token {self.configs.tokenizer_loading_config.default_pad_token} to the tokenizer")
            special_tokens_dict["pad_token"] = self.configs.tokenizer_loading_config.default_pad_token
        if self.tokenizer.eos_token is None:
            logging.info(f"Adding eos_token {self.configs.tokenizer_loading_config.default_eos_token} to the tokenizer")
            special_tokens_dict["eos_token"] = self.configs.tokenizer_loading_config.default_eos_token
        if self.tokenizer.bos_token is None:
            logging.info(f"Adding bos_token {self.configs.tokenizer_loading_config.default_bos_token} to the tokenizer")
            special_tokens_dict["bos_token"] = self.configs.tokenizer_loading_config.default_bos_token
        if self.tokenizer.unk_token is None:
            logging.info(f"Adding unk_token {self.configs.tokenizer_loading_config.default_unk_token} to the tokenizer")
            special_tokens_dict["unk_token"] = self.configs.tokenizer_loading_config.default_unk_token

        n_added = self.tokenizer.add_special_tokens(special_tokens_dict)
        logging.info(f"Added {n_added} special tokens to the tokenizer vocab. {special_tokens_dict}")

        # final safeguard: make sure to have a pad_token_id which is different from eos_token_id for padding
        if self.tokenizer.pad_token == self.tokenizer.eos_token:
            logging.warning(f"pad_token conflicts with eos_token, changing pad_token from {self.tokenizer.pad_token} to {self.tokenizer.unk_token}")
            self.tokenizer.pad_token = self.tokenizer.unk_token
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        # if there is a mismatch between tokenizer vocab size and embedding matrix, 
        # throw a warning and then expand the embedding matrix to match the tokenizer vocab size
        if len(self.tokenizer) > self.model.get_input_embeddings().weight.shape[0]:
            logging.warning(f"WARNING: Resizing the embedding matrix with size {self.model.get_input_embeddings().weight.shape[0]} to match the tokenizer vocab size {len(self.tokenizer)}.")            
            # resize token embedding by newly added special token
            self.model.resize_token_embeddings(len(self.tokenizer))

    def _prepare_model(self):
        if self.configs.train_config.quantization:
            self.model = prepare_model_for_kbit_training(self.model)
            logging.info("Model prepared for int8 quantization training")

        # convert the model to bfloat16 if fsdp and pure_bf16 is enabled
        if self.configs.train_config.enable_fsdp and self.configs.fsdp_config.pure_bf16:
            logging.info("Converting the model to bfloat16 for pure half precision training")
            self.model.to(torch.bfloat16)

        # prepare the model for PEFT training
        if self.configs.train_config.use_peft:
            self.model = get_peft_model(self.model, self.configs.peft_config)
            logging.info(f"Model prepared for PEFT training")
            self.model.print_trainable_parameters()

        # freeze orginal layers for the expanded model
        if self.configs.model_loading_config.num_exp_layers > 0:
            if self.configs.model_loading_config.freeze_ori_layers:
                logging.info(f"Model expansion: Freezing original layers at {self.exp_model_save_dir}")                
                if self.exp_model_save_dir.exists():
                    ModelExpander.freeze_layers(self.model, self.exp_model_save_dir)
                else:
                    logging.warning(f"Model expansion: Expanded model checkpoint not found at {self.exp_model_save_dir}")
                    logging.info(f"Model expansion: Freezing original layers at {self.configs.model_loading_config.pretrained_model_name_or_path}")
                    ModelExpander.freeze_layers(self.model, self.configs.model_loading_config.pretrained_model_name_or_path)
            if self.configs.model_loading_config.freeze_ori_non_layers:
                logging.info(f"Model expansion: Freezing original none transformer layers")
                ModelExpander.freeze_none_layers(self.model)
            self._log_model_size()

        # freeze layers if required, peft can't be used with layer freezing
        if not self.configs.train_config.use_peft and self.configs.train_config.freeze_layers:
            freeze_transformer_layers(self.model, self.configs.train_config.num_unfrozen_layers, self.configs.train_config.unfrozen_strategy)
            logging.info(f"Model prepared for training with {self.configs.train_config.num_unfrozen_layers} unfrozen layers out of {len(self.model.model.layers)} layers by strategy {self.configs.train_config.unfrozen_strategy}")
            self._log_model_size()

        # extend context window for the model if required
        rope_factor_config = (getattr(self.model.config, "rope_scaling", None) or {}).get("factor")
        if self.configs.model_loading_config.rope_factor and rope_factor_config and rope_factor_config != self.configs.model_loading_config.rope_factor:
            logging.info(f"Extending the context window by {self.configs.model_loading_config.rope_factor}x with rope scaling {self.configs.model_loading_config.rope_scaling} theta {self.configs.model_loading_config.rope_theta}")
            self.model.config.rope_scaling = self.configs.model_loading_config.rope_scaling
            if self.configs.model_loading_config.rope_theta:
                self.model.config.rope_theta = self.configs.model_loading_config.rope_theta
        else:
            logging.info(f"Context window extension change is skipped as the model's rope factor is already set to {rope_factor_config}")

        # move the model to the device
        if self.configs.train_config.trainer == "native":
            # setting up FSDP if enable_fsdp is enabled
            if self.configs.train_config.enable_fsdp:
                self._prepare_fsdp()
            elif not self.configs.train_config.quantization:
                self.model.to(self.configs.setup_config.device)
                logging.info(f"Model moved to {self.configs.setup_config.device}")
                self.model = DDP(self.model, device_ids=[torch.cuda.current_device()])
                logging.info(f"Model wrapped with DDP")
            else:
                logging.info("Quantization enabled, model is already moved to right device")
                self.model = DDP(self.model, device_ids=[torch.cuda.current_device()])
                logging.info(f"Model wrapped with DDP")
        else:  # trainer is transformers
            if self.configs.train_config.enable_fsdp:
                os.environ["FSDP_STATE_DICT_TYPE"] = self.configs.train_config.fsdp_config.get("state_dict_type", StateDictType.FULL_STATE_DICT.name)

            self.model.to(self.configs.setup_config.device)
            logging.info(f"Model moved to {self.configs.setup_config.device}")

        logging.info(f"Model prepared for training")

    def _prepare_fsdp(self):
        # hsdp(hybrid sharding) essentially means that the model is sharded across both the data and model dimensions
        # similar to deepspeed's ZeRO++
        hsdp = None
        if self.configs.fsdp_config.hsdp and self.configs.fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD:
            hsdp = hsdp_device_mesh(self.configs.fsdp_config, self.configs.setup_config)
            logging.info(f"Enabling HSDP training... HSDP device mesh is ready: {hsdp}")
  
        # setting up FSDP if enable_fsdp is enabled
        mixed_precision_policy, wrapping_policy = get_policies(self.configs.fsdp_config, self.configs.setup_config, self.configs.model_loading_config.decoder_layer)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(self.model, self.configs.model_loading_config.decoder_layer)

        self.model = FSDP(
            self.model,
            auto_wrap_policy=my_auto_wrapping_policy if self.configs.train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if self.configs.fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not self.configs.fsdp_config.pure_bf16 else None,
            sharding_strategy=self.configs.fsdp_config.sharding_strategy,
            # only latest torch version supports hsdp with device mesh
            device_mesh=hsdp,
            device_id=torch.cuda.current_device() if torch.cuda.is_available() else (torch.xpu.current_device() if is_xpu_available() else 0),
            limit_all_gathers=True,
            sync_module_states=self.configs.fsdp_config.fsdp_cpu_ram_efficient_loading,
            param_init_fn=lambda module: module.to_empty(device=self.configs.setup_config.device, recurse=False)
            if self.configs.fsdp_config.fsdp_cpu_ram_efficient_loading and self.configs.setup_config.local_rank != 0 else None,
            # note that expand_type == "concat" added one side car transformer block to the decoder layer with pretrained block frozen and expanded block not, so we need to use the original params for sharding all params correctly
            # TODO: For ``use_orig_params=True``, FSDP supports mixing frozen and non-frozen, but we recommend not doing so since then the gradient
            # memory usage will be higher than expected (namely, equivalent to not freezing those parameters). This means that ideally, frozen parameters
            # should be isolated into their own ``nn.Module`` s and wrapped separately with FSDP.
            use_orig_params=True if self.configs.model_loading_config.num_exp_layers > 0 and self.configs.model_loading_config.expand_type in ["concat", "hybrid"] and self.configs.model_loading_config.freeze_ori_non_layers else False,
        )
        logging.info(f"Model wrapped with FSDP")
        if self.configs.fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(self.model, self.configs.model_loading_config.decoder_layer)
            logging.info(f"Model applied with FSDP activation checkpointing")

    def _prepare_optimizer(self):
        # initialize the optimizer with default AdamW with weight_decay
        if self.configs.train_config.enable_fsdp and self.configs.fsdp_config.pure_bf16 and self.configs.fsdp_config.optimizer == "anyprecision":
            self.optimizer = AnyPrecisionAdamW(
                self.model.parameters(),
                lr=self.configs.train_config.learning_rate,
                momentum_dtype=torch.bfloat16,
                variance_dtype=torch.bfloat16,
                use_kahan_summation=False,
                weight_decay=self.configs.train_config.weight_decay,
            )
            logging.info(f"Using AnyPrecisionAdamW with momentum_dtype and variance_dtype as bfloat16 - pure_bf16: {self.configs.fsdp_config.pure_bf16}")
        elif self.configs.train_config.enable_deepspeed:  # follow https://github.com/huggingface/blog/blob/main/accelerate-deepspeed.md
            self.optimizer = accelerate.utils.DummyOptim(self.model.parameters(), lr=self.configs.train_config.learning_rate)
            logging.info(f"Using DummyOptim for DeepSpeed with lr {self.configs.train_config.learning_rate}")
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.configs.train_config.learning_rate,
                weight_decay=self.configs.train_config.weight_decay,
            )
            logging.info(f"Using AdamW with lr {self.configs.train_config.learning_rate} and weight_decay {self.configs.train_config.weight_decay}")

        # special handling of native trainer to resume optimizer state etc. TODO: scheduler
        if self.configs.train_config.trainer == "native" and self.configs.train_config.resume_checkpoint_folder:
            load_optimizer(model=self.model, optimizer=self.optimizer, rank=self.configs.setup_config.rank, cfg=self.configs.train_config)

        # initialize the learning rate scheduler with warmup and cosine annealing by step for pretraining
        if (not self.configs.train_config.lr_scheduler_per_iter) and self.configs.train_config.trainer == "native":
            # decay by gamma with each epoch for SFT
            self.scheduler = StepLR(self.optimizer, step_size=self.configs.train_config.step_size, gamma=self.configs.train_config.gamma)
            logging.info(f"Using StepLR with step_size {self.configs.train_config.step_size} and gamma {self.configs.train_config.gamma}")
        # purposely commented out elif, uncomment if you want to use transformers's default get_linear_schedule_with_warmup
        # elif not self.configs.train_config.pretrain and self.configs.train_config.trainer == "transformers":
        #   using transformers's default linear warmup and linear decay to 0            
        #   pass
        elif self.configs.train_config.enable_deepspeed:  # follow https://github.com/huggingface/blog/blob/main/accelerate-deepspeed.md
            self.scheduler = accelerate.utils.DummyScheduler(
                self.optimizer,
                warmup_num_steps=self.configs.train_config.warmup_steps
                )
            logging.info(f"Using DummyScheduler with warmup_num_steps {self.configs.train_config.warmup_steps}")
        else:
            # better than get_linear_schedule_with_warmup as it has minimum learning rate after decay
            # it is hard to overfit with such a small lr for such a large dataset
            self.scheduler = WarmupCosineAnnealingLR(
                optimizer=self.optimizer,
                warmup_iterations=self.configs.train_config.warmup_steps,
                decay_iterations=self.configs.train_config.decay_steps,
                eta_min=self.configs.train_config.eta_min,
            )
            logging.info(f"Using WarmupCosineAnnealingLR with warmup_iterations {self.configs.train_config.warmup_steps}, decay_iterations {self.configs.train_config.decay_steps} and eta_min {self.configs.train_config.eta_min}")

    def _log_model_size(self, trainable_only: bool = True) -> None:
        if self.configs.setup_config.rank == 0:
            # print model size even when training with zero 3: https://github.com/huggingface/transformers/pull/22193
            total_params = get_model_param_count(self.model, trainable_only=trainable_only)
            logging.info(f"Model expanded from {self.configs.model_loading_config.pretrained_model_name_or_path} has {total_params / 1e6} Million trainable params")

    def _prepare_callbacks(self):
        # prepare the callback class
        if self.configs.train_config.trainer == "transformers":
            # save model
            if self.configs.train_config.enable_fsdp:
                self.save_model_callback = SaveFSDPModelCallback
            elif self.configs.train_config.enable_deepspeed:
                self.save_model_callback = SaveDeepSpeedModelCallback
            else:
                self.save_model_callback = SaveModelCallback
            # evaluation
            self.eval_callback = EvaluationCallback

        use_flop_counter: bool = self.configs.train_config.flop_counter  # to be compliant with the original code of llama_recipes
        use_profiler: bool = self.configs.train_config.use_profiler  # to be compliant with the original code of llama_recipes
        if use_flop_counter and use_profiler:
            raise ValueError("Cannot use both profiler and flop counter")
        if use_flop_counter:
            # profiling by custom logic of computing step time, memory, throughput, and MFU
            self.profiler_callback = partial(EfficiencyCallback, n_warmup_steps=self.configs.train_config.flop_counter_start)
        elif use_profiler and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            # profiling by pytorch native profiler wait_step, warmup_step, active_step
            # recommended to only enable ProfilerCallback on rank 0 process for distributed training
            self.profiler_callback = partial(ProfilerCallback, self.configs.train_config.wait_step, self.configs.train_config.warmup_step, self.configs.train_config.active_step, self.configs.train_config.profiler_dir)
        else:
            self.profiler_callback = None

        # for native trainer, the callbacks are handled within the trainer, so leave it empty here TODO: native trainer supports similar callback design as well

    def _handle_resume_checkpoint_folder(self):
        """
        Check if we need to resume from a checkpoint, and if so, load the model and optimizer state from the checkpoint
        """
        logging.info(f"Checking if we need to resume from checkpoint saved in: {self.configs.train_config.output_dir}") 
        if self.configs.train_config.resume_checkpoint_folder:
            logging.info(f"Resuming from the resume_checkpoint_folder: {self.configs.train_config.resume_checkpoint_folder}")
        elif self.configs.train_config.resume_from_latest:
            logging.info(f"Resuming from the latest checkpoint in: {self.configs.train_config.output_dir}")
            if not Path(self.configs.train_config.output_dir).exists():
                raise ValueError(
                    f"Output directory ({self.configs.train_config.output_dir}) does not exist."
                    "So it can't resume from the latest checkpoint from there."
                    "You may set resume_from_latest to False to train it from scratch which will create the output dir automatically."
                )
            latest_checkpoint_dir = get_last_checkpoint(self.configs.train_config.output_dir)
            if latest_checkpoint_dir is None:
                logging.info(f"No checkpoint found in {self.configs.train_config.output_dir}, training from scratch by checkpoint from {self.configs.model_loading_config.pretrained_model_name_or_path}")
            else:
                logging.info(f"Found the latest checkpoint in: {latest_checkpoint_dir}, changing resume_checkpoint_folder to {Path(latest_checkpoint_dir).name}")
                self.configs.train_config.resume_checkpoint_folder = Path(latest_checkpoint_dir).name                  
        elif self.configs.train_config.resume_checkpoint_folder is None and Path(self.configs.train_config.output_dir).exists() \
            and len(list(Path(self.configs.train_config.output_dir).iterdir())) > 0 and not self.configs.train_config.overwrite_output_dir:
            raise ValueError(
                f"Output directory ({self.configs.train_config.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        else:
            logging.info(f"According to configs: Output Directory: {self.configs.train_config.output_dir}, Resume_from_latest: {self.configs.train_config.resume_from_latest}"
                         " and resume_checkpoint_folder: {self.configs.train_config.resume_checkpoint_folder}, found nothing to resume from, training from scratch")

        # flag to indicate if there is post model loading operations like resume sharded weights, peft, optimizer etc.
        self.resume_from_native_checkpoint = False
        self.resume_from_transformers_checkpoint = False
        if self.configs.train_config.resume_checkpoint_folder:
            # in case user does not config ./configs/loading.py consistently with what was used to train the resumed checkpoint
            self.resume_checkpoint_path = str(Path.cwd() / self.configs.train_config.output_dir / self.configs.train_config.resume_checkpoint_folder)

            # self.configs.train_config.resume_checkpoint_folder is in pattern of checkpoint-x where x is the global step
            self.configs.train_config.consumed_iters = int(self.configs.train_config.resume_checkpoint_folder.split("-")[1]) + 1
            if is_load_checkpoint_needed(self.configs.train_config) and not Path(self.resume_checkpoint_path).joinpath(TRAINER_STATE_NAME).exists():
                # for shared model or peft, need to update the model state dict from the checkpoint folder which requires first loading initial state from pretrained folder
                self.resume_from_native_checkpoint = True
            else:
                self.resume_from_transformers_checkpoint = True

            self.configs.model_loading_config = ModelExpander.restore_expansion_configs(self.resume_checkpoint_path, self.configs.model_loading_config)
            logging.info(f"Restored model loading expansion config from 'expansion' in config.json of {self.resume_checkpoint_path} to: {ModelExpander.get_expansion_configs(self.resume_checkpoint_path)}")
            self.configs.model_loading_config.pretrained_model_name_or_path = self.get_trained_from_model_name_or_path(self.resume_checkpoint_path)
            logging.info(f"Restored model loading pretrained_model_name_or_path from 'trained_from' in config.json of {self.resume_checkpoint_path} to: {self.configs.model_loading_config.pretrained_model_name_or_path}")

    @classmethod
    def get_trained_from_model_name_or_path(cls, model_checkpoint_path: str) -> str:
        """
        Load the model configuration to get the trained_from attribute which is set during model loading of this class before training the model
        trained_from: the base model name or path from which the model was trained from
        """
        if not os.path.exists(os.path.join(model_checkpoint_path, "config.json")):
            raise FileNotFoundError(f"config.json not found in {model_checkpoint_path}")

        # Load the config from fsdp_checkpoint_path/config.json into dictionary
        with open(os.path.join(model_checkpoint_path, "config.json"), "r") as f:
            config_dict = yaml.safe_load(f)
        if "trained_from" in config_dict:
            return config_dict["trained_from"]
        else:
            return None

    @classmethod
    def get_trained_by_datasets(cls, model_checkpoint_path: str) -> str:
        """
        Load the model configuration to get the trained_by: dataset names used to train the model from ./controlllm/configs/datasets.py
        """
        if not os.path.exists(os.path.join(model_checkpoint_path, "config.json")):
            raise FileNotFoundError(f"config.json not found in {model_checkpoint_path}")

        # Load the config from fsdp_checkpoint_path/config.json into dictionary
        with open(os.path.join(model_checkpoint_path, "config.json"), "r") as f:
            config_dict = yaml.safe_load(f)
        if "trained_by" in config_dict:
            return config_dict["trained_by"]
        else:
            return None


class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_iterations: int,
        decay_iterations: int,
        eta_min: float = 0.0,
        # note that here we define it as last_epoch to be compatible with parent class's init method
        # but it actually means last_iteration
        last_epoch: int = -1,
    ) -> None:
        self.warmup_iterations: int = warmup_iterations
        self.decay_iterations: int = decay_iterations
        self.eta_min: float = eta_min
        # run parent class's __init__ method
        super().__init__(optimizer=optimizer, last_epoch=last_epoch, verbose=False)     

    def get_lr(self):
        if self.last_epoch < self.warmup_iterations:
            # Linear warmup
            warmup_ratio: float = self.last_epoch / self.warmup_iterations # type: ignore
            return [
                max(self.eta_min + (base_lr - self.eta_min) * warmup_ratio, self.eta_min)
                for base_lr in self.base_lrs
            ]
        elif self.last_epoch < self.decay_iterations:
            # Cosine annealing
            progress: float = (self.last_epoch - self.warmup_iterations) / (self.decay_iterations - self.warmup_iterations)   # type: ignore
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress) * 3.14159))
            return [
                max(self.eta_min + (base_lr - self.eta_min) * cosine_decay.item(), self.eta_min)
                for base_lr in self.base_lrs
            ]
        else:
            # Constant learning rate after decay_iterations
            return [self.eta_min for _ in self.base_lrs]
