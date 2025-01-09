# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import logging
import time
import yaml
import json
import numpy as np
import contextlib
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from accelerate.utils import is_xpu_available, is_ccl_available

from controlllm.utils.custom_llama_recipes.memory_utils import MemoryTrace
from controlllm.utils.custom_llama_recipes.profile_utils import custom_trace_handler, profile
from controlllm.utils.custom_llama_recipes.eval_utils import save_eval_result
from controlllm.utils.model_expander import ModelExpander
# import linkedin.dllib.torch.disruption.disruption_handler as disruption_handler
# from linkedin.dllib.common.checkpoint_manger import CheckpointManager

# checkpoint_manager = CheckpointManager(primary_checkpoint_path="/dev/shm/controlllm/ckpt", secondary_checkpoint_path="hdfs://jobs/controlllm/ckpt")


def update_weigth_decay(optimizer, train_config):
    """Update the weight decay dynamically based on the learning rate"""
    # If train_config.weight_decay_ratio is None or 0, then weight decay is not updated dynamically
    if train_config.weight_decay_ratio:
        # Update weight decay dynamically
        for param_group in optimizer.param_groups:
            # Calculate dynamic weight decay
            learning_rate = param_group['lr']
            dynamic_weight_decay = train_config.weight_decay_ratio * learning_rate

            # Update weight decay for the current parameter group
            param_group['weight_decay'] = dynamic_weight_decay


def train(model, train_dataloader, eval_dataloader, tokenizer, optimizer, lr_scheduler, train_config, fsdp_config=None, local_rank=None, rank=None, wandb_run=None, tb_writer=None, save_checkpoint=None, evaluate=None):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        num_train_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        rank: The rank of the current process in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    # For bf16: FSDP’s internal mechanisms manage to maintain gradient integrity effectively thanks to the properties of BF16, reducing or eliminating the need for explicit gradient scaling.
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else contextlib.nullcontext
    train_ppl, train_loss, val_ppl, val_loss = [], [], [], []

    train_step_perplexity, train_step_loss, eval_step_loss, eval_step_perplexity = None, None, None, None
    if train_config.save_metrics:
        os.makedirs(train_config.output_dir, exist_ok=True)
        metrics_filename = f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        if train_config.debug:  # add per step loss and perplexity for debugging purpose
            train_step_perplexity, train_step_loss, eval_step_loss, eval_step_perplexity = [], [], [], []

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    consumed_iters = train_config.consumed_iters
    starting_epoch = consumed_iters // len(train_dataloader)
    starting_step = consumed_iters % len(train_dataloader)
    if consumed_iters > 0 and rank == 0:
        logging.warning(
            f"Consumed number of iterations is {consumed_iters}. "
            f"We will skip {starting_epoch} epoch(s) and {starting_step} step(s) for training. "
            f"Resuming training from epoch {starting_epoch + 1} and step {starting_step + 1}. "
            f"{'Quitting because starting_epoch + 1 > train_config.num_train_epochs' if starting_epoch >= train_config.num_train_epochs else ''}"
        )
    if starting_epoch >= train_config.num_train_epochs:
        return

    max_steps_reached = False  # Flag to indicate max training steps reached

    if rank == 0 and train_config.enable_memory_profiling:
        torch.cuda.memory._record_memory_history()
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=train_config.wait_step, warmup=train_config.warmup_step, active=train_config.active_step, repeat=1, skip_first=0),
            on_trace_ready=custom_trace_handler(train_config.profiler_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()

        torch.cuda.reset_peak_memory_stats()

    # Start the training loop
    for epoch in range(train_config.num_train_epochs):
        # stop when the maximum number of training steps is reached
        if max_steps_reached:
            break

        # Skip epochs that have already been completed for resumed training
        if epoch < starting_epoch:
            # resume lr from where it was left off
            # TODO: remove this once save_lr is implemented, in fact this is more flexible to allow a different lr scheduler setting for resumed training
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.StepLR):  # StepLR for per epoch lr decay
                lr_scheduler.step()
            else:  # use WarmupCosineAnnealingLR from ./controlllm/loading_utils.py for per step lr decay
                for _ in tqdm(range(len(train_dataloader)), colour="blue", desc=f"[{rank=}] Training Epoch - resuming to epoch {starting_epoch}: {epoch + 1}/{train_config.num_train_epochs}") if local_rank == 0 or train_config.debug else range(len(train_dataloader)):
                    lr_scheduler.step()
            continue
        else:
            # update weight decay dynamically according to lr, Note: this is not updated per lr_scheduler.step(), in resuming case, update it just once after lr is resumed
            update_weigth_decay(optimizer, train_config)

        # Initialize the epoch
        epoch_start_time = time.perf_counter()
        total_loss = 0.0
        gradient_accumulation_steps = train_config.gradient_accumulation_steps
        total_length = len(train_dataloader) // gradient_accumulation_steps + 1  # make sure total_length is at least 1
        pbar = tqdm(colour="blue", desc=f"[{rank=}] Training Epoch: {epoch + 1}", total=total_length, dynamic_ncols=True) if local_rank == 0 or train_config.debug else None
        # set the data sampler to the current epoch to ensure the data is shuffled differently for each epoch, refer to controlllm/utils/dataset_utils/DataLoaderWrapper.get_dataloader_kwargs()
        # set the data sampler to the starting epoch to ensure the data is shuffled differently for each epoch and set the starting step to ensure steps are skipped once once the training is resumed
        if train_config.batching_strategy == "padding":
            if train_config.enable_fsdp:  # DistributedLengthBasedBatchSampler
                train_dataloader.batch_sampler.batch_sampler.set_epoch(epoch)
                train_dataloader.batch_sampler.batch_sampler.set_starting_step(starting_step * world_size)  # starting_step implementation is global(not per device)
            else:  # LengthBasedBatchSampler
                train_dataloader.batch_sampler.set_epoch(epoch)
                train_dataloader.batch_sampler.set_starting_step(starting_step)
        elif train_config.batching_strategy == "packing":  # DistributedSampler
            if train_config.enable_fsdp:
                train_dataloader.sampler.set_epoch(epoch)
                train_dataloader.sampler.set_starting_step(starting_step)  # in this sampler, starting_step implementation is per device
        else:
            raise ValueError(f"Invalid batching strategy {train_config.batching_strategy}")

        # Skip steps that have already been completed for resumed training
        for step in range(len(train_dataloader)):
            # Skip steps that have already been completed for resumed training
            if step < starting_step:
                if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:       
                    # resume lr from where it was left off
                    # TODO: remove this once save_lr is implemented, in fact this is more flexible to allow a different lr scheduler setting for resumed training
                    if not isinstance(lr_scheduler, torch.optim.lr_scheduler.StepLR):  # use WarmupCosineAnnealingLR from ./controlllm/loading_utils.py
                        lr_scheduler.step()

                    if pbar:
                        pbar.update(1)
                if pbar:
                    pbar.set_description(f"[{rank=}] Training Epoch - resuming to step {starting_step}: {epoch + 1}/{train_config.num_train_epochs}, step {step}/{len(train_dataloader)} completed")
            else:
                # update weight decay dynamically according to lr, Note: this is not updated per lr_scheduler.step(), in resuming case, update it just once after lr is resumed
                update_weigth_decay(optimizer, train_config)
                break

        # Initialize MemoryTrace based on configuration
        memory_trace_context = MemoryTrace() if train_config.enable_memory_trace else contextlib.nullcontext()
        with memory_trace_context as memtrace:  # track the memory usage
            model.train()
            with profile(train_config, local_rank) as profile_context:
                for step, batch in enumerate(train_dataloader, start=starting_step):  # train_dataloader.batchsampler.batchsampler handles the skip starting_step internally
                    if step >= len(train_dataloader):  # this is to compensate start=starting_step if epoch == starting_epoch, otherwise it will go beyond len(train_dataloader) because of start=xyz
                        logging.info(f"step {step} is beyond the length of train_dataloader {len(train_dataloader)}, breaking the loop for epoch {epoch + 1}")
                        break
                    total_train_steps = epoch * len(train_dataloader) + step
                    # stop when the maximum number of training steps is reached
                    if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                        max_steps_reached = True
                        logging.info(f"max training steps reached, stopping training, total_train_steps: {total_train_steps-1}")
                        break

                    if rank == 0 and train_config.enable_memory_profiling:
                        prof.step()
                        logging.info(f"Peak memory usage {torch.cuda.memory.max_memory_allocated()}")

                    device_batch = {}
                    for key, value in batch.items():
                        if train_config.enable_fsdp:
                            if is_xpu_available():
                                device_batch[key] = value.to(torch.device(f"xpu:{local_rank}"))
                            else:
                                device_batch[key] = value.to(local_rank)
                        else:
                            if is_xpu_available():
                                device_batch[key] = value.to(f"xpu:{local_rank}")
                            else:
                                device_batch[key] = value.to(f"cuda:{local_rank}")

                    with autocast():
                        cross_entropy_loss = model(**device_batch).loss
                        additional_loss = ModelExpander.get_additional_loss(model, cross_entropy_loss)
                        loss = cross_entropy_loss + additional_loss
                    total_loss += loss.detach().float()
                    loss = loss / gradient_accumulation_steps
                    if train_config.save_metrics and train_config.debug:
                        train_step_loss.append(loss.detach().float().item())
                        train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                    if train_config.use_fp16:
                        # if fp16 is enabled, use gradient scaler to handle gradient update
                        scaler.scale(loss).backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.max_grad_norm and train_config.max_grad_norm > 0.0:
                                scaler.unscale_(optimizer)
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.max_grad_norm)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)

                            # Adjust gradients for incomplete accumulation (if necessary)
                            if (step + 1) % gradient_accumulation_steps != 0:
                                for p in model.parameters():
                                    if p.grad is not None:
                                        p.grad /= ((step + 1) % gradient_accumulation_steps / gradient_accumulation_steps)

                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            # TODO: remove this or implement the step progress per gradient_accumulation_steps which is how transformers's trainer does
                            if pbar:
                                pbar.update(1)
                    else:
                        # regular backpropagation when fp16 is not used
                        loss.backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.max_grad_norm and train_config.max_grad_norm > 0.0:
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.max_grad_norm)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)

                            # Adjust gradients for incomplete accumulation (if necessary)
                            if (step + 1) % gradient_accumulation_steps != 0:
                                for p in model.parameters():
                                    if p.grad is not None:
                                        p.grad /= ((step + 1) % gradient_accumulation_steps / gradient_accumulation_steps)

                            optimizer.step()
                            optimizer.zero_grad()
                            # TODO: remove this or implement the step progress per gradient_accumulation_steps which is how transformers's trainer does
                            if pbar:
                                pbar.update(1)
                    if train_config.use_profiler or train_config.flop_counter:
                        profile_context.step()
                    if train_config.flop_counter and profile_context.is_done():
                        TFlops = profile_context.get_flops_per_sec() / 1e12
                        logging.info(f"TFlops - flops per sec: {TFlops}")
                    else:
                        TFlops = None
                    if wandb_run:
                        wandb_run.log({
                            'train/epoch': epoch + 1,
                            'train/step': epoch * len(train_dataloader) + step,
                            'train/loss': loss.detach().float(),
                            'train/cross_entropy_loss': cross_entropy_loss.detach().float() / gradient_accumulation_steps,
                            'train/additional_loss': additional_loss.detach().float() / gradient_accumulation_steps,
                        })

                    if tb_writer:
                        tb_writer.add_scalar("TrainLoss/GlobalStep", loss.detach().float(), epoch * len(train_dataloader) + step)
                        tb_writer.add_scalar("CrossEntropyLoss/GlobalStep", cross_entropy_loss.detach().float() / gradient_accumulation_steps, epoch * len(train_dataloader) + step)
                        tb_writer.add_scalar("DivergenceLoss/GlobalStep", additional_loss.detach().float() / gradient_accumulation_steps, epoch * len(train_dataloader) + step)

                    if pbar:
                        pbar.set_description(f"[{rank=}] Training Epoch: {epoch + 1}/{train_config.num_train_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")

                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:       
                        # note that from torch.optim.lr_scheduler.StepLR is designed for per epoch learning rate decay while WarmupCosineAnnealingLR is used for per step lr decay, recommended for pretraining
                        if not isinstance(lr_scheduler, torch.optim.lr_scheduler.StepLR):  # use WarmupCosineAnnealingLR from ./controlllm/loading_utils.py
                            update_weigth_decay(optimizer, train_config)  # update weight decay dynamically according to lr, Note: this is after optimizer.step(). Still works because, although it doesn't affect the current update, it adjusts the regularization strength for subsequent updates.
                            lr_scheduler.step()

                    global_step = epoch * len(train_dataloader) + step
                    # save the model, optimizer states every save_steps using save_checkpoint function
                    if train_config.save_model and global_step % train_config.save_steps == 0:
                        save_checkpoint(model, optimizer, rank, train_config, fsdp_config, epoch, checkpoint_times, tokenizer=tokenizer, global_step=global_step)
                    # # Disruption handling here
                    # if disruption_handler.need_checkpoint_now_and_shutdown():
                    #     if rank == 0:
                    #         torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint_manager.primary_checkpoint_path[1])
                    #         checkpoint_manager.write(blocking=True)
                    #         disruption_handler.shutdown_job()

                    # evaluate the model every eval_steps using evaluation function
                    if train_config.run_validation and global_step % train_config.eval_steps == 0:
                        # eval_step_loss, eval_step_perplexity will be updated only when save_metrics is True as it is memory expensive
                        _, eval_loss, _, _ = evaluate(model, train_config, eval_dataloader, local_rank, tokenizer, eval_step_loss, eval_step_perplexity, rank, val_loss, val_ppl, wandb_run, tb_writer, global_step)
                        if eval_loss < best_val_loss:
                            best_val_loss = eval_loss
                            logging.info(f"--> best eval loss on step {global_step} is {best_val_loss}")

                    if train_config.save_metrics and train_config.debug:
                        save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_ppl, eval_step_loss, val_loss, eval_step_perplexity, val_ppl, rank)

                    del loss, cross_entropy_loss, additional_loss, device_batch

                if pbar:
                    pbar.close()
                starting_step = 0  # reset starting step to 0 after the epoch

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)

        # if total_loss is not a tensor(a float), it means that the model is trained yet
        if isinstance(total_loss, torch.Tensor):
            # Reducing total_loss across all devices if there's more than one CUDA device
            if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            elif torch.cuda.device_count() > 1 and train_config.enable_fsdp:
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)

            train_epoch_loss = total_loss / len(train_dataloader)
            if train_config.enable_fsdp:
                train_epoch_loss = train_epoch_loss/world_size
            train_perplexity = torch.exp(train_epoch_loss)
        else:
            train_epoch_loss = total_loss
            train_perplexity = np.nan

        train_ppl.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if (not train_config.enable_fsdp or rank == 0) and memtrace:
            memtrace.print_stats()

        # note that from torch.optim.lr_scheduler.StepLR is designed for per epoch learning rate decay which is recommended for SFT
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.StepLR):
            update_weigth_decay(optimizer, train_config)
            lr_scheduler.step()

        global_step = (epoch + 1) * len(train_dataloader)
        if train_config.run_validation and (epoch + 1) % train_config.eval_epoch == 0:
            # eval_step_loss, eval_step_perplexity will be updated only when save_metrics is True as it is memory expensive
            eval_ppl, eval_epoch_loss, eval_bleu, eval_rougeLsum = evaluate(model, train_config, eval_dataloader, local_rank, tokenizer, eval_step_loss, eval_step_perplexity, rank, val_loss, val_ppl, wandb_run, tb_writer, global_step)

            # if run_validation is True, save the model checkpoint if the validation loss is better than the best validation loss, this ensures that the best model is saved
            if train_config.save_model and eval_epoch_loss < best_val_loss:
                logging.info(f"Epoch {epoch + 1}: Saving the model checkpoint because the validation loss {eval_epoch_loss} is better than the best validation loss {best_val_loss}")
                save_checkpoint(model, optimizer, rank, train_config, fsdp_config, epoch, checkpoint_times, tokenizer=tokenizer, global_step=global_step)
                save_eval_result(eval_ppl=eval_ppl, eval_loss=eval_epoch_loss, eval_bleu=eval_bleu, eval_rougeLsum=eval_rougeLsum, eval_step_loss=None, eval_step_perplexity=None, train_config=train_config, global_step=global_step, rank=rank)
            elif train_config.save_model and eval_epoch_loss >= best_val_loss:
                logging.info(f"Epoch {epoch + 1}: Validation loss {eval_epoch_loss} is not better than the best validation loss {eval_epoch_loss}, skipping the checkpoint saving")
            else:
                logging.info(f"Epoch {epoch + 1}: Note that no checkpoint is saved. Please make sure this is intended.")

            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                logging.info(f"--> best eval loss on step {global_step} - epoch {epoch + 1} is {best_val_loss}")

            logging.info(f"Epoch {epoch + 1}: eval_perplexity={eval_ppl:.4f}, eval_epoch_loss={eval_epoch_loss:.4f}, eval_bleu={eval_bleu:.4f}, eval_rougeLsum={eval_rougeLsum:.4f}, epoch time {epoch_end_time}s")
            if tb_writer:
                tb_writer.add_scalar("EvalLoss/Epoch", eval_epoch_loss, epoch + 1)
                tb_writer.add_scalar("EvalPerplexity/Epoch", eval_ppl, epoch + 1)
                tb_writer.add_scalar("EvalBleu/Epoch", eval_bleu, epoch + 1)
                tb_writer.add_scalar("EvalRougeLsum/Epoch", eval_rougeLsum, epoch + 1)

            if wandb_run:
                wandb_run.log({
                    'eval/epoch': epoch + 1,
                    'eval/loss': eval_epoch_loss,
                    'eval/perplexity': eval_ppl,
                    'eval/bleu': eval_bleu,
                    'eval/rougeLsum': eval_rougeLsum,
                    'eval/epoch_time': epoch_end_time,
                })
        elif train_config.save_model and (epoch + 1) % train_config.save_epoch == 0:  # save the checkpoint for every save_epoch
            logging.info(f"Epoch {epoch + 1}: Saving the model checkpoint for every {train_config.save_epoch} epoch")
            save_checkpoint(model, optimizer, rank, train_config, fsdp_config, epoch, checkpoint_times, tokenizer=tokenizer, global_step=global_step)
        else:  # no save, no validation
            logging.info(f"Epoch {epoch + 1}: Note that no validation is performed and no checkpoint is saved. Please make sure this is intended.")

        logging.info(f"Epoch {epoch + 1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        if tb_writer:
            tb_writer.add_scalar("TrainLoss/Epoch", train_epoch_loss, epoch + 1)
            tb_writer.add_scalar("TrainPerplexity/Epoch", train_perplexity, epoch + 1)
            tb_writer.add_scalar("EpochTime/Epoch", epoch_end_time, epoch + 1)
            tb_writer.add_scalar("LearningRate/Epoch", optimizer.param_groups[0]['lr'], epoch + 1)

        if wandb_run:
            wandb_run.log({
                'train/epoch': epoch + 1,
                'train/loss': train_epoch_loss,
                'train/perplexity': train_perplexity,
                'train/epoch_time': epoch_end_time,
                'train/lr': optimizer.param_groups[0]['lr'],
            })

        # Saving the results every epoch to plot later
        if train_config.save_metrics and train_config.debug:
            save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_ppl, eval_step_loss, val_loss, eval_step_perplexity, val_ppl, rank)

    avg_epoch_time = sum(epoch_times)/ len(epoch_times) if epoch_times else 0
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else -1
    avg_train_ppl = sum(train_ppl)/len(train_ppl) if train_ppl else -1
    avg_train_loss = sum(train_loss)/len(train_loss) if train_loss else -1
    if train_config.run_validation and val_ppl and val_loss:
        avg_eval_ppl = sum(val_ppl)/len(val_ppl)
        avg_eval_loss = sum(val_loss)/len(val_loss)
    else:
        avg_eval_ppl = -1
        avg_eval_loss = -1

    results['avg_train_ppl'] = avg_train_ppl
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_ppl'] = avg_eval_ppl
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics and train_config.debug:
        results["metrics_filename"] = metrics_filename
    if train_config.flop_counter:
        results["model_tflops"]= TFlops
    #saving the training params including fsdp setting for reference.
    if train_config.enable_fsdp and not train_config.use_peft and rank == 0:
        save_train_params(train_config, fsdp_config, rank)

    return results


def freeze_transformer_layers(model, num_unfrozen_layers: int, unfrozen_strategy: str):
    """
    num_unfrozen_layers: int = default(8)  # number of layers to unfrozen, used when freeze_layers is True
    unfrozen_strategy: str = "interweave"  # top, bottom or interweave, interweave is to unfreeze one layer every few layers until num_unfrozen_layers
    """
    num_of_layers = len(model.model.layers)
    freezed_layers = set()
    for i, layer in enumerate(model.model.layers):
        if unfrozen_strategy == "top":
            if i < num_of_layers - num_unfrozen_layers:
                logging.info(f"Freezing layer {i}")
                freezed_layers.add(i)
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                logging.info(f"Unfreezing layer {i}")
                for param in layer.parameters():
                    param.requires_grad = True
        elif unfrozen_strategy == "bottom":
            if i >= num_unfrozen_layers:
                logging.info(f"Freezing layer {i}")
                freezed_layers.add(i)
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                logging.info(f"Unfreezing layer {i}")
                for param in layer.parameters():
                    param.requires_grad = True
        elif unfrozen_strategy == "interweave":
            split = max(int(num_of_layers // num_unfrozen_layers), 1)
            if (i + 1) % split == 0:
                logging.info(f"Unfreezing layer {i}")
                for param in layer.parameters():
                    param.requires_grad = True
            else:
                logging.info(f"Freezing layer {i}")
                freezed_layers.add(i)
                for param in layer.parameters():
                    param.requires_grad = False
        else:
            raise ValueError(f"Invalid unfrozen_strategy: {unfrozen_strategy}. Please use 'top', 'bottom' or 'interweave'")

    model.config.freezed_layers = list(freezed_layers)


def check_frozen_layers_peft_model(model):
    for i, layer in enumerate(model.base_model.model.model.layers):
        for name, param in layer.named_parameters():
            logging.info(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")


def setup():
    """Initialize the process group for distributed training"""
    if is_ccl_available():
        # distributed training on xpus
        dist.init_process_group("ccl")
    else:
        dist.init_process_group("nccl")


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


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        logging.info(f"Clearing GPU cache for all ranks")
    if is_xpu_available():
        torch.xpu_empty_cache()
    else:
        torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes


def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        logging.info(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")


def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries,
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    # folder_name = (
    # train_config.dist_checkpoint_root_folder
    # + "/"
    # + train_config.dist_checkpoint_folder
    # + "-"
    # + train_config.model_name
    # )

    # save_dir = Path.cwd() / folder_name
    save_dir = Path.cwd() / train_config.output_dir
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir,'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        logging.info(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank == 0:
            logging.info(f"training params are saved in {file_name}")


def save_to_json(output_filename, train_step_loss, train_epoch_loss, train_step_ppl, train_epoch_ppl, eval_step_loss, val_epoch_loss, val_step_ppl, val_epoch_ppl, rank=0):
    """
    Save the metrics data to a JSON file
    """
    if rank != 0:
        return
    metrics_data = {
        "train_step_loss": train_step_loss,
        "train_epoch_loss": train_epoch_loss,
        "train_step_perplexity": train_step_ppl,
        "train_epoch_perplexity": train_epoch_ppl,
        "eval_step_loss": eval_step_loss,
        "val_epoch_loss": val_epoch_loss,
        "eval_step_perplexity": val_step_ppl,
        "val_epoch_perplexity": val_epoch_ppl
    }
    with open(output_filename, "w") as f:
        json.dump(metrics_data, f)
