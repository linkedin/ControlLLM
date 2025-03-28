import logging
import time
import shutil
from pathlib import Path
from typing import Callable, Union, Optional, Dict, List, Tuple, TYPE_CHECKING, Any
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.utils.data import DataLoader, Dataset

from transformers import TrainerCallback
from transformers import Trainer, TrainingArguments, DataCollator
from transformers.processing_utils import ProcessorMixin
from transformers.image_processing_utils import BaseImageProcessor
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers import PreTrainedModel, PreTrainedTokenizerBase, EvalPrediction

from controlllm.utils.custom_llama_recipes import train
from controlllm.utils.custom_llama_recipes import evaluate as run_evaluation, initialize_metrics_modules
from controlllm.utils.custom_llama_recipes import save_model_checkpoint, save_peft_checkpoint, save_model_and_optimizer_sharded, save_optimizer_checkpoint
from controlllm.configs.training import TrainConfigCommon
from controlllm.utils.model_expander import ModelExpander


if TYPE_CHECKING:
    import optuna


class CustomTrainerNative(Trainer):
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        lr_scheduler,
        train_config,
        fsdp_config=None,
        local_rank=None,
        rank=None,
        wandb_run=None,
        tb_writer=None,
    ):
        # don't call super().__init__ as we are not using transformers.Trainer.train, using callbacks directly
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_config = train_config
        self.fsdp_config = fsdp_config
        self.local_rank = local_rank
        self.rank = rank
        self.wandb_run = wandb_run
        self.tb_writer = tb_writer

        # Initialize decay steps for the learning rate scheduler here as we need to know the total steps for the scheduler
        initialize_decay_steps(self.lr_scheduler, self.train_config, self.train_dataloader, self.train_config.gradient_accumulation_steps)

    def train(self):
        # FIXME: move this into a separate utils file when complexity grows
        def save_checkpoint(model, optimizer, rank, train_config, fsdp_config, epoch, checkpoint_times, tokenizer=None, global_step=-1):
            checkpoint_start_time = time.perf_counter()

            if train_config.enable_fsdp:
                dist.barrier()

            if train_config.use_peft:
                if not train_config.enable_fsdp or rank == 0:
                    logging.info(f"we are about to save the PEFT modules")

                save_peft_checkpoint(
                    model, optimizer, rank, train_config, epoch=epoch, tokenizer=tokenizer, global_step=global_step
                )

                if not train_config.enable_fsdp or rank == 0:
                    logging.info(f"PEFT modules are saved in {train_config.output_dir} directory")

            else:
                if fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
                    logging.info(" Saving the FSDP model checkpoints using FULL_STATE_DICT")
                    logging.info("=====================================================")
                    save_model_checkpoint(
                        model, optimizer, rank, train_config, epoch=epoch, tokenizer=tokenizer, global_step=global_step
                    )
                    if train_config.save_optimizer:
                        logging.info(" Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
                        logging.info("=====================================================")
                        save_optimizer_checkpoint(
                            model, optimizer, rank, train_config, epoch=epoch, global_step=global_step
                        )
                elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                    if train_config.save_optimizer:
                        logging.info(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                        logging.info("=====================================================")
                        save_model_and_optimizer_sharded(model, rank, train_config, tokenizer=tokenizer, optim=optimizer, global_step=global_step)
                    else:
                        logging.info("Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                        logging.info("=====================================================")
                        save_model_and_optimizer_sharded(model, rank, train_config, tokenizer=tokenizer, global_step=global_step)
                else:
                    raise ValueError(f"checkpoint_type {fsdp_config.checkpoint_type} is not supported, please use one of {StateDictType.FULL_STATE_DICT} or {StateDictType.SHARDED_STATE_DICT}")

            if train_config.enable_fsdp:
                dist.barrier()

            checkpoint_end_time = time.perf_counter() - checkpoint_start_time
            checkpoint_times.append(checkpoint_end_time)

        def evaluate(model, train_config, eval_dataloader, local_rank, tokenizer, val_step_loss, val_step_perplexity, rank, val_loss, val_ppl, wandb_run, tb_writer, global_step):
            try:
                eval_ppl, eval_loss, eval_bleu, eval_rougeLsum, temp_val_loss, temp_step_perplexity, eval_avg_roc_auc, eval_avg_pr_auc = run_evaluation(model, train_config, eval_dataloader, local_rank, rank, tokenizer, wandb_run, tb_writer, global_step, initialize_metrics_modules(train_config, rank))
            except Exception as e:
                logging.exception(f"Error during evaluation at rank {rank}: {e}, continue training...")
                # Clean up cache_dir to be able to recover from failures using rmtrees
                cache_dir = str(Path(train_config.output_dir) / "evaluate")
                shutil.rmtree(cache_dir, ignore_errors=True)
                torch.cuda.empty_cache()  # This can help release unoccupied memory back to the GPU
                model.train()  # Make sure the model is in training mode
                eval_ppl, eval_loss, eval_bleu, eval_rougeLsum, temp_val_loss, temp_step_perplexity, eval_avg_roc_auc, eval_avg_pr_auc = float('inf'), float('inf'), float('inf'), float('inf'), [], [], float('inf'), float('inf')

            dist.barrier()  # Ensure all processes are synchronized before returning for training, avoid NCCL timeout in resuming the training
            if train_config.save_metrics and train_config.debug:
                val_step_loss.extend(temp_val_loss)
                val_step_perplexity.extend(temp_step_perplexity)
            val_loss.append(float(eval_loss))
            val_ppl.append(float(eval_ppl))
            return eval_ppl, eval_loss, eval_bleu, eval_rougeLsum, eval_avg_roc_auc, eval_avg_pr_auc

        try:
            dist.barrier()  # Ensure all processes are synchronized starting the forward/backward pass, avoid NCCL timeout when starting the training
            results = train(
                model=self.model,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                tokenizer=self.tokenizer,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                train_config=self.train_config,
                fsdp_config=self.fsdp_config,
                local_rank=self.local_rank,
                rank=self.rank,
                wandb_run=self.wandb_run,
                tb_writer=self.tb_writer,
                save_checkpoint=save_checkpoint,
                evaluate=evaluate
            )
            if (not self.train_config.enable_fsdp or self.rank == 0) and results is not None:
                [logging.info(f'Key: {k}, Value: {v}') for k, v in results.items()]
        except Exception as e:
            logging.exception(f"Error during training at rank {self.rank}: {e}")
            raise
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

class CustomTrainerTransformers(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
        local_rank: Optional[int] = None,
        rank: Optional[int] = None,
        train_config: Optional[TrainConfigCommon] = None,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        wandb_run=None,
        tb_writer=None,
        **kwargs,
    ):
        # call parent class constructor
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            # optimizers=optimizers,  # purposely not passing this to the parent class as we customize it in create_optimizer_and_scheduler
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **kwargs,
        )
        # FIXME: Trainer does not have those attributes for now, but can introduce them in the future which may cause issues
        self.custom_optimizer = optimizers
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.local_rank = local_rank
        self.rank = rank
        self.train_config = train_config
        self.resume_from_checkpoint = resume_from_checkpoint
        # FIXME: It is a temporary fix to use separate wandb_rn and tb_writer for evaluation, need to refactor this to share it with training
        self.wandb_run = wandb_run
        self.tb_writer = tb_writer

        # Initialize decay steps for the learning rate scheduler here as we need to know the total steps for the scheduler
        lr_scheduler = optimizers[1]
        initialize_decay_steps(lr_scheduler, self.train_config, self.train_dataloader, self.train_config.gradient_accumulation_steps)

    # override dataloader methods to enable packing or padding of the data
    def get_train_dataloader(self) -> DataLoader:
        return self.train_dataloader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        return self.eval_dataloader

    def train(
        self,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        # consumed_iters > 0 and resume_from_checkpoint is None means we are resuming from a checkpoint saved by native trainer
        if self.resume_from_checkpoint is None and self.args.consumed_iters > 0:
            logging.info(f"Resuming training from {self.resume_from_checkpoint}: global step {self.args.consumed_iters} and epoch {self.state.global_step // len(self.train_dataloader)}")

        try:
            dist.barrier()  # Ensure all processes are synchronized starting the forward/backward pass, avoid NCCL timeout when starting the training
            super().train(self.resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
        except Exception as e:
            logging.exception(f"Error during training at rank {self.rank}: {e}")
            raise
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior: add divergence loss to the cross-entropy loss, report to wandb and tensorboard.
        """
        if return_outputs:
            cross_entropy_loss, outputs = super().compute_loss(model, inputs, return_outputs=return_outputs)
        else:
            cross_entropy_loss = super().compute_loss(model, inputs, return_outputs=return_outputs)

        additional_loss = ModelExpander.get_additional_loss(model, cross_entropy_loss)
        loss = cross_entropy_loss + additional_loss

        if self.wandb_run:
            self.wandb_run.log({
                'train/epoch': self.state.epoch + 1,
                'train/step': self.state.global_step,
                'train/loss': loss.detach().float(),
                'train/cross_entropy_loss': cross_entropy_loss.detach().float() / self.args.gradient_accumulation_steps,
                'train/additional_loss': additional_loss.detach().float() / self.args.gradient_accumulation_steps,
            })

        if self.tb_writer:
            self.tb_writer.add_scalar("TrainLoss/GlobalStep", loss.detach().float(), self.state.global_step)
            self.tb_writer.add_scalar("CrossEntropyLoss/GlobalStep", cross_entropy_loss.detach().float() / self.args.gradient_accumulation_steps, self.state.global_step)
            self.tb_writer.add_scalar("DivergenceLoss/GlobalStep", additional_loss.detach().float() / self.args.gradient_accumulation_steps, self.state.global_step)

        return (loss, outputs) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        # evaluation function has parameters: model,train_config, eval_dataloader, local_rank, tokenizer, wandb_run
        if self.eval_dataloader is None:
            return {}

        try:
            eval_ppl, eval_loss, eval_bleu, eval_rougeLsum, val_step_loss, val_step_perplexity, eval_avg_roc_auc, eval_avg_pr_auc = run_evaluation(
                model=self.model, train_config=self.args, eval_dataloader=self.eval_dataloader,
                local_rank=self.local_rank, rank=self.rank, tokenizer=self.tokenizer, wandb_run=self.wandb_run, tb_writer=self.tb_writer, global_step=self.state.global_step, metrics_modules=initialize_metrics_modules(self.train_config, self.rank))
        except Exception as e:
            logging.exception(f"Error during evaluation at rank {self.rank}: {e}, continue training...")
            eval_ppl, eval_loss, eval_bleu, eval_rougeLsum, val_step_loss, val_step_perplexity = float('inf'), float('inf'), float('inf'), float('inf'), [], []
            torch.cuda.empty_cache()  # This can help release unoccupied memory back to the GPU

        dist.barrier()  # Ensure all processes are synchronized before returning for training, avoid NCCL timeout in resuming the training

        # convert the evaluation results to a dictionary
        metrics = {f"{metric_key_prefix}_ppl": eval_ppl, f"{metric_key_prefix}_loss": eval_loss,
                   f"{metric_key_prefix}_step_loss": val_step_loss, f"{metric_key_prefix}_step_perplexity": val_step_perplexity,
                   f"{metric_key_prefix}_bleu": eval_bleu, f"{metric_key_prefix}_rougeLsum": eval_rougeLsum,
                   f"{metric_key_prefix}_avg_roc_auc": eval_avg_roc_auc, f"{metric_key_prefix}_avg_pr_auc": eval_avg_pr_auc}

        # assign to self.metrics to let the save callback to save into checkpoint folder, no need to cumulate the metrics for all steps so overwrite it
        self.eval_metrics = {self.state.global_step: metrics}

        return metrics

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        # Important: the following two lines are necessary to properly set the scheduler and optimizer
        self.optimizer = self.custom_optimizer[0]
        self.lr_scheduler = self.custom_optimizer[1]


def initialize_decay_steps(lr_scheduler, train_config, train_dataloader, gradient_accumulation_steps=1):
    """Initialize decay steps for the learning rate scheduler WarmupCosineAnnealingLR"""
    if isinstance(lr_scheduler, torch.optim.lr_scheduler.StepLR):
        logging.info(f"Decay steps are not initialized dynamically for StepLR. train_config.step_size {train_config.step_size} is used")
        return
    # if train_config.decay_steps is not None or 0, then decay steps are not initialized dynamically
    if train_config.decay_steps:
        logging.info(f"Decay steps are set to {train_config.decay_steps} in train_config, skipping dynamic initialization. lr scheduler uses decay steps: {lr_scheduler.decay_iterations}")
        return

    # set lr_scheduler.decay_iterations with total_steps - train_config.warmup_steps
    if lr_scheduler.decay_iterations is None or lr_scheduler.decay_iterations == 0:  # double confirm if decay_iterations is not set
        total_length = (len(train_dataloader) // gradient_accumulation_steps) + 1  # make sure total_length is at least 1
        total_steps = train_config.num_train_epochs * total_length
        lr_scheduler.decay_iterations = max(total_steps - train_config.warmup_steps, 1)  # make sure decay_iterations is at least 1
        logging.info(f"Decay steps are initialized dynamically with {lr_scheduler.decay_iterations} = total_steps {total_steps}: epoches {train_config.num_train_epochs} * ((per_device_data_len {len(train_dataloader)} // gradient accumulation steps {gradient_accumulation_steps}) + 1) - warmup_steps {train_config.warmup_steps}")
