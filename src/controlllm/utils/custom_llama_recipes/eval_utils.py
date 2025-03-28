# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
from __future__ import annotations
import os
import json
import logging
import contextlib
import numpy as np
import urllib.parse
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import torch
import torch.distributed as dist
from accelerate.utils import is_xpu_available

from evaluate.module import temp_seed
from evaluate import load, EvaluationModule
from datasets.arrow_writer import ArrowWriter
from datasets.utils.filelock import FileLock, Timeout

from controlllm.utils.model_expander import ModelExpander, ModelLogger
from controlllm.utils.custom_llama_recipes.memory_utils import MemoryTrace


# Overwrite the _create_cache_file method to avoid blocking by acquiring the file lock
# Code originally from hugginface evaluate library
# Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
def _create_cache_file(metrics_module: EvaluationModule, timeout=1) -> Tuple[str, FileLock]:
    """Create a new cache file. If the default cache file is used, we generated a new hash."""
    file_path = os.path.join(metrics_module.data_dir, f"{metrics_module.experiment_id}-{metrics_module.num_process}-{metrics_module.process_id}.arrow")
    filelock = FileLock(file_path + ".lock")
    try:
        filelock.acquire(timeout=timeout)
    except Timeout:
        # If we have reached the max number of attempts or we are not allow to find a free name (distributed setup)
        # We log and continue
        if metrics_module.num_process != 1:
            logging.warning(
                f"Error in _create_cache_file: another evaluation module instance is already using the local cache file at {file_path}. "
                f"Please specify an experiment_id (currently: {metrics_module.experiment_id}) to avoid collision "
                f"between distributed evaluation module instances."
            )

    return file_path, filelock


# Overwrite the get_all_cache_files method to avoid blocking by acquiring the file lock
# Code originally from hugginface evaluate library
# Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
def _get_all_cache_files(metrics_module: EvaluationModule) -> Tuple[List[str], List[FileLock]]:
    """Get a lock on all the cache files in a distributed setup.
    We wait for timeout second to let all the distributed node finish their tasks (default is 100 seconds).
    """
    if metrics_module.num_process == 1:
        if metrics_module.cache_file_name is None:
            raise ValueError(
                "Evaluation module cache file doesn't exist. Please make sure that you call `add` or `add_batch` "
                "at least once before calling `compute`."
            )
        file_paths = [metrics_module.cache_file_name]
    else:
        file_paths = [
            os.path.join(metrics_module.data_dir, f"{metrics_module.experiment_id}-{metrics_module.num_process}-{process_id}.arrow")
            for process_id in range(metrics_module.num_process)
        ]

    # Let's acquire a lock on each process files to be sure they are finished writing
    filelocks = []
    for process_id, file_path in enumerate(file_paths):
        if process_id == 0:  # process 0 already has its lock file
            filelocks.append(metrics_module.filelock)
        else:
            filelock = FileLock(file_path + ".lock")  # no need to acquire the lock as we have torch.distributed.barrier() before metrics_module.compute()
            filelocks.append(filelock)

    return file_paths, filelocks


# Overwrite the compute method to avoid acquiring the file lock which fails due to contention of resource in multi-node training
# Code originally from hugginface evaluate library
# Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
def compute(metrics_module: EvaluationModule, *, predictions=None, references=None, **kwargs) -> Optional[dict]:
    """Compute the evaluation module.

    Usage of positional arguments is not allowed to prevent mistakes.

    Args:
        predictions (list/array/tensor, optional): Predictions.
        references (list/array/tensor, optional): References.
        **kwargs (optional): Keyword arguments that will be forwarded to the evaluation module :meth:`_compute`
            method (see details in the docstring).

    Return:
        dict or None

        - Dictionary with the results if this evaluation module is run on the main process (``process_id == 0``).
        - None if the evaluation module is not run on the main process (``process_id != 0``).
    """
    threshold = kwargs.pop("threshold", None)
    # Create binary labels: 1 if groundtruth score >= threshold, else 0.
    if references:
        references = [1 if r >= threshold else 0 for r in references]

    all_kwargs = {"predictions": predictions, "references": references, **kwargs}
    if predictions is None and references is None:
        missing_kwargs = {k: None for k in metrics_module._feature_names() if k not in all_kwargs}
        all_kwargs.update(missing_kwargs)
    else:
        missing_inputs = [k for k in metrics_module._feature_names() if k not in all_kwargs]
        if missing_inputs:
            raise ValueError(
                f"Evaluation module inputs are missing: {missing_inputs}. All required inputs are {list(metrics_module._feature_names())}"
            )
    inputs = {input_name: all_kwargs[input_name] for input_name in metrics_module._feature_names()}
    compute_kwargs = {k: kwargs[k] for k in kwargs if k not in metrics_module._feature_names()}

    if any(v is not None for v in inputs.values()):
        metrics_module.add_batch(**inputs)
    # Synchronize all processes to make sure that all the data is written before reading it
    logging.info(f"Current process finished writing scores to file system for {metrics_module.experiment_id}, waiting for all processes to finish writing data before computing metrics")
    torch.distributed.barrier()
    try:
        metrics_module._finalize()
    except Exception as e:
        logging.exception(f"Error in {metrics_module.experiment_id} -> _finalize: {e}. Continuing without finalizing the metrics.")
        return None

    metrics_module.cache_file_name = None
    metrics_module.filelock = None
    metrics_module.selected_feature_format = None

    if metrics_module.process_id == 0:
        metrics_module.data.set_format(type=metrics_module.info.format)

        inputs = {input_name: metrics_module.data[input_name] for input_name in metrics_module._feature_names()}
        with temp_seed(metrics_module.seed):
            logging.info(f"Computing metrics - {metrics_module.experiment_id} with {len(inputs[list(inputs.keys())[0]])} data points, this may take a while ...")

            # Create binary labels: 1 if groundtruth score >= threshold, else 0.
            if threshold is not None and "references" in inputs:
                logging.info(f"Creating binary labels with threshold {threshold}")
                references = inputs["references"]
                inputs["references"] = [1 if r >= threshold else 0 for r in references]
                # Only one class present in y_true. ROC AUC score is not defined in that case. So if all are 0 or 1, we flip the labels of first element of y_true
                if len(set(inputs["references"])) == 1:
                    inputs["references"][0] = 1 - inputs["references"][0]

            output = metrics_module._compute(**inputs, **compute_kwargs)

            # Don't delete the metrics_module.data and writer yet
            if threshold is not None and "references" in inputs:
                logging.info(f"output before restoring the original references: {output}")
                inputs["references"] = references  # restore the original references for next threshold
                if threshold != -1:  # -1 is used to delete data and writer(cross reference to code line 608)
                    return output

        if metrics_module.buf_writer is not None:
            metrics_module.buf_writer = None
            del metrics_module.data
            metrics_module.data = None
        else:
            # Release locks and delete all the cache files. Process 0 is released last.
            for filelock, file_path in reversed(list(zip(metrics_module.filelocks, metrics_module.file_paths))):
                logging.debug(f"Removing {file_path}")
                del metrics_module.data
                metrics_module.data = None
                del metrics_module.writer
                metrics_module.writer = None
                os.remove(file_path)
                if filelock:
                    filelock.release()

        return output
    else:
        return None


def evaluate(model, train_config, eval_dataloader, local_rank, rank, tokenizer, wandb_run, tb_writer, global_step, metrics_modules: Dict[str, EvaluationModule] = None, enable_model_debug=False, use_probe_data=False) -> Tuple[float, float, float, float, List[float], List[float]]:
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        train_config: The training configuration
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        rank: The rank of the current process in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss, eval_bleu, eval_rougeLsum, eval_step_loss, eval_step_perplexity
    """
    enable_fsdp = True if train_config.enable_deepspeed or train_config.enable_fsdp else False  # add this to support evaluation with deepspeed in transformers trainer, as this method is shared with the native trainer

    if enable_fsdp:
        world_size = dist.get_world_size()
    else:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

    model.eval()
    if enable_model_debug:
        model_logger = ModelLogger(global_step, tb_writer)  # this is to log the model debug info to tensorboard for ModelExpander
        model_debugger = ModelDebugger(train_config, eval_dataloader, local_rank, rank, tokenizer, global_step, use_probe_data)  # this is to get the data and metadata for model debugging
        data, metadata = model_debugger.data, model_debugger.metadata
        eval_dataloader = data  # replace the eval_dataloader with the probe data

    # Initialize metrics
    if metrics_modules is None:
        metrics_modules = initialize_metrics_modules(train_config, rank, world_size)
    # Prepare lists for predictions and references when train_config.eval_in_memory is True
    all_preds = []
    all_labels = []

    eval_step_loss = []
    eval_step_perplexity = []
    ts_eval_loss, ts_eval_ce_loss, ts_eval_div_loss = 0.0, 0.0, 0.0  # Initialize evaluation loss with python number first
    # Initialize MemoryTrace based on configuration
    memory_trace_context = MemoryTrace() if train_config.enable_memory_trace else contextlib.nullcontext()
    with memory_trace_context as memtrace:  # track the memory usage
        for step, batch in enumerate(tqdm(eval_dataloader, colour="green", desc=f"[{rank=}] Evaluating step", dynamic_ncols=True)) if local_rank == 0 or train_config.debug else enumerate(eval_dataloader):
            # inject the model debug info into the model so that it can report the debug info to tensorboard by metadata
            if enable_model_debug:
                ModelExpander.enable_model_debug(model, metadata[step], model_logger)

            # stop when the maximum number of eval steps is reached
            if train_config.max_eval_step > 0 and step > train_config.max_eval_step:
                logging.info(f"Max eval steps reached, stopping evaluation, total_eval_steps: {step - 1}")
                break

            # Create a new dictionary to store tensors moved to the appropriate device. This is necessary to avoid additional GPU memory usage if we move batch to GPU in place.
            device_batch = {}
            for key, value in batch.items():
                # skip is value is not a tensor
                if not isinstance(value, torch.Tensor):
                    continue

                if enable_fsdp:
                    device_batch[key] = value.to(local_rank)  # Move to the specified device based on local rank
                else:
                    if is_xpu_available():
                        device_batch[key] = value.to(f"xpu:{local_rank}")  # Move to XPU if available
                    else:
                        device_batch[key] = value.to(f"cuda:{local_rank}")  # Default to moving to CUDA device

            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**device_batch, return_verbose=True)
                cross_entropy_loss = outputs.loss
                additional_loss = ModelExpander.get_additional_loss(model, cross_entropy_loss)
                loss = cross_entropy_loss + additional_loss
                if train_config.save_metrics and train_config.debug:
                    # note that loss will still keep tensor in GPU though it is detached which means does not require grad, item() will be a standard python number
                    eval_step_loss.append(loss.item())
                    eval_step_perplexity.append(torch.exp(loss).item())

                ts_eval_loss += loss.detach().float()
                ts_eval_ce_loss += cross_entropy_loss.detach().float()
                ts_eval_div_loss += additional_loss.detach().float()

            if hasattr(outputs, "logits"):
                log_causallm_samples(outputs, batch, tokenizer, train_config, metrics_modules, all_preds, all_labels)
            elif any("similarities" in attr for attr in dir(outputs)):
                log_samples_sentence_transformer(outputs, batch, tokenizer, train_config, metrics_modules, all_preds, all_labels)
            else:
                logging.warning("No logits or similarities attributes found in the model outputs. Skipping evaluation.")

            # Clean up to release GPU memory
            del outputs, cross_entropy_loss, additional_loss, loss, device_batch
            torch.cuda.empty_cache()

        if (not enable_fsdp or local_rank == 0) and memtrace:
            memtrace.print_stats()

    if enable_model_debug:
        model_debugger.log_model_debug(model_logger)

    # No evaluation has been executed, return -1 for perplexity and loss
    if (type(ts_eval_loss) == float and ts_eval_loss == 0.0) or (enable_model_debug and use_probe_data):  # means no eval has been executed or model debug is enabled, no need to compute metrics
        logging.warning(f"Evaluation loss is {ts_eval_loss}. This is likely due to not enough data for evaluation, or per_device_eval_batch_size might be too high")
        # Ensure the model is switched back to training mode
        model.train()
        return float('inf'), float('inf'), float('inf'), float('inf'), eval_step_loss, eval_step_perplexity

    # Compute average loss, perplexity
    with torch.no_grad():
        # If there's more than one CUDA device, reduce evaluation loss and metrics across all devices
        if enable_fsdp or world_size > 1:
            logging.info("Reducing evaluation loss across all devices...")
            # Synchronize all processes to complete initialization to avoid NCCL timeout
            torch.distributed.barrier()
            dist.all_reduce(ts_eval_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(ts_eval_ce_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(ts_eval_div_loss, op=dist.ReduceOp.SUM)
            # Perform avg the metrics and loss across all processes and dataloader length after reduction
            ts_eval_loss, ts_eval_ce_loss, ts_eval_div_loss = (x / (world_size * max(min(train_config.max_eval_step if train_config.max_eval_step != -1 else float('inf'), len(eval_dataloader)), 1)) for x in (ts_eval_loss, ts_eval_ce_loss, ts_eval_div_loss))
        # Compute average loss, perplexity
        else:
            ts_eval_loss, ts_eval_ce_loss, ts_eval_div_loss = (x / max(min(train_config.max_eval_step if train_config.max_eval_step != -1 else float('inf'), len(eval_dataloader)), 1) for x in (ts_eval_loss, ts_eval_ce_loss, ts_eval_div_loss))  # Ensure this division happens outside the if-condition for single GPU case

        ts_eval_ppl = torch.exp(ts_eval_loss)
        eval_loss, eval_ce_loss, eval_div_loss, eval_ppl = ts_eval_loss.item(), ts_eval_ce_loss.item(), ts_eval_div_loss.item(), ts_eval_ppl.item()

        # interp weights for each layer used for balancing between output hidden state of pretrained and expanded layer, also used to balance between CE and KL loss
        ts_interp_factors, avg_interp_weights = ModelExpander.get_interp_weights(model)
        interp_factors = {f"layer_{layer_idx}": factor.item() if factor.numel() == 1 else factor[0].item() for layer_idx, factor in ts_interp_factors.items()}
        avg_interp_weights = {f"layer_{layer_idx}": avg_weight for layer_idx, avg_weight in avg_interp_weights.items()}

    # Clean up to release GPU memory
    ts_device, ts_dtype = ts_eval_loss.device, ts_eval_loss.dtype
    del ts_eval_loss, ts_eval_ce_loss, ts_eval_div_loss, ts_eval_ppl, ts_interp_factors

    # Compute ROUGE and BLEU scores
    eval_metrics = train_config.eval_metrics if train_config.eval_metrics else ["rouge", "sacrebleu"]
    if "rouge" in eval_metrics or "sacrebleu" in eval_metrics:
        causallm_metrics = compute_causallm_metrics(metrics_modules, all_preds, all_labels, ts_device, ts_dtype, world_size, enable_fsdp, train_config)
    else:
        causallm_metrics = {k: float('inf') for k in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bleu']}

    # Compute ROC-AUC and PR-AUC scores
    thresholds = [1, 2, 3, 4]
    if "roc_auc" in eval_metrics or "pr_auc" in eval_metrics:
        sentence_transformer_metrics = compute_sentence_transformer_metrics(metrics_modules, all_preds, all_labels, ts_device, ts_dtype, world_size, enable_fsdp, train_config, thresholds=thresholds)
    else:
        auc_results = {k: float('inf') for k in thresholds}
        sentence_transformer_metrics = {"roc_auc": auc_results, "pr_auc": auc_results}
    # Calculate avg_roc_auc, ignoring 'inf' values
    valid_metrics = [v for v in sentence_transformer_metrics["roc_auc"].values() if v != float('inf')]
    avg_roc_auc = sum(valid_metrics) / len(valid_metrics) if valid_metrics else float('inf')
    # Calculate avg_pr_auc, ignoring 'inf' values
    valid_metrics = [v for v in sentence_transformer_metrics["pr_auc"].values() if v != float('inf')]
    avg_pr_auc = sum(valid_metrics) / len(valid_metrics) if valid_metrics else float('inf')

    import gc
    gc.collect()
    torch.cuda.empty_cache()  # This can help release unoccupied memory back to the GPU

    # Print evaluation loss and metrics
    causallm_metrics_log = ' '.join([f"{key}={value}" for key, value in causallm_metrics.items()])
    sentence_transformer_metrics_log = ' '.join([f"{outer_key}.{inner_key}={value}" for outer_key, module in sentence_transformer_metrics.items() for inner_key, value in module.items()])
    logging.info(f"{global_step=} {eval_ppl=} {eval_loss=} {causallm_metrics_log}, {sentence_transformer_metrics_log}, {avg_roc_auc=}, {avg_pr_auc=}")

    if wandb_run:
        wandb_run.log({
                        'eval/perplexity': eval_ppl,
                        'eval/loss': eval_loss,
                        'eval/ce_loss': eval_ce_loss,
                        'eval/div_loss': eval_div_loss,
                        **{f'eval/interp_factors/{k}': v for k, v in interp_factors.items()},
                        **{f'eval/avg_interp_weight/{k}': v for k, v in avg_interp_weights.items()},
                        **{f'eval/{k}': v for k, v in causallm_metrics.items()},
                        **{f'eval/roc_auc/threshold_{k}': v for k, v in sentence_transformer_metrics["roc_auc"].items()},
                        **{f'eval/pr_auc/threshold_{k}': v for k, v in sentence_transformer_metrics["pr_auc"].items()},
                        'eval/roc_auc/avg': avg_roc_auc,
                        'eval/pr_auc/avg': avg_pr_auc,
                        'eval/global_step': global_step
                    }, commit=False)

    # Makes sure that only rank 0 writes to tensorboard
    if tb_writer:
        tb_writer.add_scalar("EvalPerplexity/GlobalStep", eval_ppl, global_step)
        tb_writer.add_scalar("EvalLoss/GlobalStep", eval_loss, global_step)
        tb_writer.add_scalar("EvalCrossEntropyLoss/GlobalStep", eval_ce_loss, global_step)
        tb_writer.add_scalar("EvalDivergenceLoss/GlobalStep", eval_div_loss, global_step)
        tb_writer.add_scalars("EvalInterpFactors", interp_factors, global_step)
        if avg_interp_weights:
            tb_writer.add_scalars("EvalAvgInterpWeight", avg_interp_weights, global_step)
        if "rouge" in eval_metrics:
            tb_writer.add_scalars("EvalRouge", {
                "Rouge1": causallm_metrics['rouge1'],
                "Rouge2": causallm_metrics['rouge2'],
                "RougeL": causallm_metrics['rougeL'],
                "RougeLsum": causallm_metrics['rougeLsum']
            }, global_step)
        if "sacrebleu" in eval_metrics:
            tb_writer.add_scalar("EvalBleu/GlobalStep", causallm_metrics['bleu'], global_step)
        if "roc_auc" in eval_metrics:
            tb_writer.add_scalars("EvalRocAuc", {f"Threshold_{k}": v for k, v in sentence_transformer_metrics["roc_auc"].items()}, global_step)
            tb_writer.add_scalar("EvalRocAuc/Avg", avg_roc_auc, global_step)
        if "pr_auc" in eval_metrics:
            tb_writer.add_scalars("EvalPrAuc", {f"Threshold_{k}": v for k, v in sentence_transformer_metrics["pr_auc"].items()}, global_step)
            tb_writer.add_scalar("EvalPrAuc/Avg", avg_pr_auc, global_step)

    results = {
        "eval_ppl": eval_ppl,
        "eval_loss": eval_loss,
        "eval_bleu": causallm_metrics['bleu'],
        "eval_rougeLsum": causallm_metrics['rougeLsum'],
        "eval_step_loss": eval_step_loss,
        "eval_step_perplexity": eval_step_perplexity,
        "eval_avg_interp_weights": avg_interp_weights,
        "global_step": global_step
    }
    results.update({f"eval_roc_acu_threshold_{k}": v for k, v in sentence_transformer_metrics["roc_auc"].items()})
    results.update({f"eval_pr_acu_threshold_{k}": v for k, v in sentence_transformer_metrics["pr_auc"].items()})
    save_eval_result(results, train_config, global_step, rank)

    # Ensure the model is switched back to training mode
    model.train()

    return eval_ppl, eval_loss, causallm_metrics['bleu'], causallm_metrics['rougeLsum'], eval_step_loss, eval_step_perplexity, avg_roc_auc, avg_pr_auc


def compute_causallm_metrics(metrics_modules, all_preds, all_labels, ts_device, ts_dtype, world_size, enable_fsdp, train_config):
    """
    Compute the BLEU and ROUGE metrics for causal language model
    """
    # Compute ROUGE and BLEU scores
    rouge, bleu = metrics_modules["rouge"], metrics_modules["bleu"]
    if not train_config.eval_in_memory:
        # Compute ROUGE and BLEU scores, return none for ranks other than 0, note that metrics_modules handles the distributed computation of metrics
        logging.info(f"Computing ROUGE and BLEU scores for all ranks, this may take a while ...")
        rouge_score = rouge.compute(use_stemmer=False, rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"])  # Specifies which ROUGE scores to calculate
        bleu_score = bleu.compute()
        # extract all metrics
        if rouge_score:
            eval_rouge1, eval_rouge2, eval_rougeL, eval_rougeLsum = rouge_score["rouge1"], rouge_score["rouge2"], rouge_score["rougeL"], rouge_score["rougeLsum"]
        else:
            eval_rouge1, eval_rouge2, eval_rougeL, eval_rougeLsum = float('inf'), float('inf'), float('inf'), float('inf')
        eval_bleu = bleu_score['score'] if bleu_score else float('inf')
    else:
        # Compute ROUGE and BLEU scores
        logging.info(f"Computing ROUGE and BLEU scores for {len(all_preds)} predictions and references...")
        rouge_score = rouge.compute(predictions=all_preds,
                                    references=all_labels,
                                    use_stemmer=False,  # Optional: Adds stemming for evaluation, which can be useful for languages like English
                                    rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"]  # Specifies which ROUGE scores to calculate
                                    )
        bleu_score = bleu.compute(predictions=all_preds, references=[[ref] for ref in all_labels])

        # Convert metric scores to the same dtype as ts_eval_loss to ensure consistency
        ts_eval_metrics = {
            'rouge1': torch.tensor(rouge_score['rouge1'], device=ts_device, dtype=ts_dtype),
            'rouge2': torch.tensor(rouge_score['rouge2'], device=ts_device, dtype=ts_dtype),
            'rougeL': torch.tensor(rouge_score['rougeL'], device=ts_device, dtype=ts_dtype),
            'rougeLsum': torch.tensor(rouge_score['rougeLsum'], device=ts_device, dtype=ts_dtype),
            'bleu': torch.tensor(bleu_score['score'], device=ts_device, dtype=ts_dtype)
        }

        # Ensure no gradients are computed for this scope to save memory
        with torch.no_grad():
            # If there's more than one CUDA device, reduce evaluation metrics across all devices
            if enable_fsdp and world_size > 1:
                logging.info("Reducing evaluation metrics across all devices...")
                # Synchronize all processes to complete initialization to avoid NCCL timeout
                torch.distributed.barrier()
                for metric in ts_eval_metrics.keys():
                    dist.all_reduce(ts_eval_metrics[metric], op=dist.ReduceOp.SUM)

                # Perform avg the metrics across all processes and dataloader length after reduction
                for key in ts_eval_metrics:
                    ts_eval_metrics[key] /= world_size  # Here assuming averaging only over world_size for metrics if necessary

            eval_rouge1, eval_rouge2, eval_rougeL, eval_rougeLsum, eval_bleu = (ts_eval_metrics[key].item() for key in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bleu'])

        # Clean up to release GPU memory
        del ts_eval_metrics

    return {'rouge1': eval_rouge1, 'rouge2': eval_rouge2, 'rougeL': eval_rougeL, 'rougeLsum': eval_rougeLsum, 'bleu': eval_bleu}


def log_causallm_samples(outputs, batch, tokenizer, train_config, metrics_modules, all_preds, all_labels):
    """
    Log the predicted tokens for causal language model

    Parameters:
        outputs: model output with logits
        batch: input batch containing tokenized inputs and groundtruth labels
        tokenizer: the tokenizer for decoding the input ids
        train_config: configuration object; used here to check if debug logging is enabled
        metrics_modules: dictionary containing the evaluation modules
        all_preds: list to which the decoded predictions will be appended
        all_labels: list to which the decoded groundtruth labels will be
    """
    # Decode predictions and add to evaluation predictions list
    preds = torch.argmax(outputs.logits, -1)
    # prints the first five decoded predictions and corresponding ground truth labels for comparison
    # Replace same place in preds_np of labels_np[labels_np == -100] with 0, note that labels -100 means prompt tokens
    preds_np, labels_np = preds.detach().cpu().numpy(), batch["labels"].cpu().numpy()
    mask = np.roll(labels_np == -100, shift=-1)
    mask[:, -1] = False  # Ensure the last element is False after rolling
    preds_np[mask] = tokenizer.pad_token_id
    # Replace -100 labels with pad_token_id which is a special token for padding
    labels_np[labels_np == -100] = tokenizer.pad_token_id
    # zip the predictions(preds_np) and labels(labels_np) together and print them to compare
    for i in range(len(preds_np)):
        if i < 1 or train_config.debug:
            logging.info(f"Predicted output: {tokenizer.decode(preds_np[i], skip_special_tokens=True)}")
            logging.info(f"Groundtruth: {tokenizer.decode(labels_np[i], skip_special_tokens=True)}")

    batch_decoded_preds = [tokenizer.decode(pred_np, skip_special_tokens=True) for pred_np in preds_np]
    batch_decoded_labels = [tokenizer.decode(label_np, skip_special_tokens=True) for label_np in labels_np]

    if not train_config.eval_in_memory:
        for metrics_module in metrics_modules.values():
            metrics_module.add_batch(predictions=batch_decoded_preds, references=batch_decoded_labels)
    else:
        # there is a unsolved issue running it in distributed multi-node shared file system, TODO: remove this when the issue is resolved
        # https://github.com/huggingface/evaluate/issues/481
        # temp workaround, simply sum up individual BLEU scores from subsets of data and then average them which is NOT accurate
        all_preds.extend(batch_decoded_preds)
        all_labels.extend(batch_decoded_labels)


def log_samples_sentence_transformer(outputs, batch, tokenizer, train_config, metrics_modules, all_preds, all_labels):
    """
    Log the predicted similarity scores for a SentenceTransformer model and accumulate
    the ground truth scores (references) and predicted similarity scores for ROC-AUC computation.
    
    Parameters:
        outputs: model output with various similarity scores (e.g. similarity, similarity_chosen, etc.)
        batch: input batch containing tokenized inputs and groundtruth scores (e.g. "score", "score_chosen")
        tokenizer: the tokenizer for decoding the input ids
        train_config: configuration object; used here to check if debug logging is enabled
        metrics_modules: dictionary containing the evaluation modules
        all_labels: list to which the groundtruth scores will be appended
        all_preds: list to which the predicted similarity scores will be appended
    """
    # Loop over each attribute in outputs that contains "similarity"
    for similarity_key, similarity_val in outputs.similarities.items():
        if similarity_val is None:
            continue

        # Determine which keys to use from the batch based on the attribute name.
        if "chosen" in similarity_key:
            input_key = "chosen_input_ids"
            score_key = "score_chosen"
        elif "rejected" in similarity_key:
            input_key = "rejected_input_ids"
            score_key = "score_reject"
        else:
            # Note that this is to log data input for cosent margin loss, refer to ./utils/custom_sentence_transformers/cosent_margin_loss.py
            # Rigourously input_key named as "chosen_input_ids" is not accurate, it should be "input_ids" as it needs just query and single doc compared to ./utils/custom_sentence_transformers/pairwise_cosine_loss.py
            # Keep it as "chosen_input_ids" to make it compatible to pairwise_cosine_loss. Refer to data preprocessing in ./utils/data/semantic_search_cosent_dataset.py
            input_key = "chosen_input_ids"
            score_key = "label"

        # Loop over the batch samples
        for i in range(similarity_val.shape[0]):
            # Optionally log detailed sample information
            if i < 1 or train_config.debug:
                decoded_prompt = tokenizer.decode(batch["prompt_input_ids"][i], skip_special_tokens=True)
                decoded_doc = tokenizer.decode(batch[input_key][i], skip_special_tokens=True)
                # Use .item() to extract a Python number from tensor/numpy scalar
                pred_score = similarity_val[i].item()
                true_score = batch[score_key][i].item()
                logging.info(f"Predicted {similarity_key}: {pred_score}. Groundtruth {score_key}: {true_score}. "
                             f"Decoded prompt: {decoded_prompt}. Decoded {input_key}: {decoded_doc}")

        if not train_config.eval_in_memory:
            for metrics_module in metrics_modules.values():
                metrics_module.add_batch(predictions=similarity_val, references=batch[score_key])
        else:
            # there is a unsolved issue running it in distributed multi-node shared file system, TODO: remove this when the issue is resolved
            # https://github.com/huggingface/evaluate/issues/481
            # Accumulate groundtruth and prediction for later ROC-AUC computation.
            all_labels.extend(batch[score_key])
            all_preds.extend(similarity_val)


def compute_sentence_transformer_metrics(metrics_modules, all_preds, all_labels, ts_device, ts_dtype, world_size, enable_fsdp, train_config, thresholds=[1, 2, 3, 4]):
    """
    Compute ROC-AUC for various binary splits by thresholding the groundtruth scores.

    For each threshold, samples with a groundtruth score >= threshold are labeled as 1, and 0 otherwise.

    Parameters:
        all_labels: list of accumulated groundtruth relevance scores
        all_preds: list of accumulated predicted similarity scores
        metrics_modules: dictionary containing metric modules (including "roc_auc")
        ts_device: torch device to create evaluation tensors on
        ts_dtype: torch dtype to ensure consistency across metrics
        world_size: number of distributed processes
        enable_fsdp: flag indicating if Fully Sharded Data Parallel is enabled
        train_config: training configuration object (with attribute eval_in_memory)
        thresholds: list of thresholds to compute binary splits

    Returns:
        For eval_in_memory=True: dictionary mapping each threshold to its computed ROC-AUC score.
        For eval_in_memory=False: metrics are computed (and reduced) internally (no return value).
    """
    roc_auc, pr_auc = metrics_modules["roc_auc"], metrics_modules["pr_auc"]
    roc_auc_results, pr_auc_results = {}, {}
    if not train_config.eval_in_memory:
        # Distributed computation: assume the metric module already has accumulated
        # predictions and references, so we simply call compute without extra inputs.
        logging.info("Computing ROC-AUC scores for all ranks, this may take a while ...")
        for thresh in thresholds:
            # Note: In non in-memory mode, the module is expected to have already aggregated
            # the predictions/references across devices; thus, we do not pass them here.
            # ROC-AUC
            results_roc_auc = roc_auc.compute(threshold=thresh)
            # If results exist, extract roc_auc; otherwise, assign infinity.
            roc_auc_results[thresh] = round(results_roc_auc['roc_auc'], 2) if results_roc_auc else float('inf')
            # PR-AUC
            results_pr_auc = pr_auc.compute(threshold=thresh)
            # If results exist, extract pr_auc; otherwise, assign infinity.
            pr_auc_results[thresh] = round(results_pr_auc['pr_auc'], 2) if results_pr_auc else float('inf')
        # after all thresholds are computed, delete the data in the metrics module and release the lock, this is done via not passing threshold with -1(refer to code line 170)
        _, _ = roc_auc.compute(threshold=-1), pr_auc.compute(threshold=-1)
    else:
        # In-memory computation: explicitly pass predictions and references.
        logging.info(f"Computing ROC-AUC scores for {len(all_labels)} samples...")
        for thresh in thresholds:
            binary_refs = [1 if r >= thresh else 0 for r in all_labels]
            # ROC-AUC
            roc_auc_results = roc_auc.compute(predictions=all_preds, references=binary_refs)
            roc_auc_results[thresh] = round(roc_auc_results['roc_auc'], 2)
            # PR-AUC
            pr_auc_results = pr_auc.compute(predictions=all_preds, references=binary_refs)
            pr_auc_results[thresh] = round(pr_auc_results['pr_auc'], 2)

        # Convert the computed scores into torch tensors for consistency.
        # ROC-AUC
        ts_roc_auc_metrics = {thresh: torch.tensor(roc_auc_results[thresh], device=ts_device, dtype=ts_dtype) for thresh in thresholds}
        # PR-AUC
        ts_pr_auc_metrics = {thresh: torch.tensor(pr_auc_results[thresh], device=ts_device, dtype=ts_dtype) for thresh in thresholds}
        with torch.no_grad():
            # If using FSDP across multiple devices, reduce the metrics.
            if enable_fsdp and world_size > 1:
                logging.info("Reducing evaluation ROC-AUC scores across all devices...")
                torch.distributed.barrier()
                for thresh in thresholds:
                    dist.all_reduce(ts_roc_auc_metrics[thresh], op=dist.ReduceOp.SUM)
                    dist.all_reduce(ts_pr_auc_metrics[thresh], op=dist.ReduceOp.SUM)
                for thresh in thresholds:
                    ts_roc_auc_metrics[thresh] /= world_size
                    ts_pr_auc_metrics[thresh] /= world_size

            # Extract the reduced values.
            roc_auc_results = {thresh: ts_roc_auc_metrics[thresh].item() for thresh in thresholds}
            pr_auc_results = {thresh: ts_pr_auc_metrics[thresh].item() for thresh in thresholds}
        del ts_roc_auc_metrics, ts_pr_auc_metrics

    return {"roc_auc": roc_auc_results, "pr_auc": pr_auc_results}


def init_writer(metrics_module: EvaluationModule):
    """
    Initialize the metrics writer for the given metrics module
    Solve https://github.com/huggingface/evaluate/issues/481
    """
    metrics_module.selected_feature_format = metrics_module._infer_feature_from_batch({"predictions": ["dummy"], "references": ["dummy"]})
    metrics_module.buf_writer = None
    # Get cache file name and lock it
    if metrics_module.cache_file_name is None or metrics_module.filelock is None:
        cache_file_name, filelock = metrics_module._create_cache_file()  # get ready
        metrics_module.cache_file_name = cache_file_name
        metrics_module.filelock = filelock

    metrics_module.writer = ArrowWriter(
        features=metrics_module.selected_feature_format,
        path=metrics_module.cache_file_name,
        writer_batch_size=metrics_module.writer_batch_size,
    )


def save_eval_result(results, train_config, global_step, rank=0):
    if rank != 0:
        return
    # save all returned result as a json file in the output directory / resume_checkpoint_folder
    output_path = Path(train_config.output_dir) / "checkpoint-{}".format(global_step)
    if output_path.exists():
        with open(os.path.join(output_path, "evaluation_results.json"), "w") as f:
            json.dump(results, f, indent=4)
        logging.info(f"Saved evaluation results to {output_path}/evaluation_results.json")
    else:
        logging.warning(f"Checkpoint directory {output_path} does not exist, skipping saving evaluation results")


def initialize_metrics_modules(train_config, rank: int = 0, world_size=None) -> Dict[str, EvaluationModule]:
    """
    Initialize the metrics modules for evaluation
    """
    # Initialize world size, TODO: consolidate this logic in one place
    if world_size is None:
        enable_fsdp = True if train_config.enable_deepspeed or train_config.enable_fsdp else False  # add this to support evaluation with deepspeed in transformers trainer, as this method is shared with the native trainer
        if enable_fsdp:
            world_size = dist.get_world_size()
        else:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # Initialize cache directory and script path
    cache_dir = str(Path(train_config.output_dir) / "evaluate")
    os.makedirs(cache_dir, exist_ok=True)
    eval_metrics = train_config.eval_metrics if train_config.eval_metrics else ["rouge", "sacrebleu"]
    if "rouge" in eval_metrics:
        rough_script_path = str(Path(train_config.hf_hub_metrics_cache_dir) / "rouge/rouge.py")
    if "sacrebleu" in eval_metrics:
        sacrebleu_script_path = str(Path(train_config.hf_hub_metrics_cache_dir) / "sacrebleu/sacrebleu.py")
    if "roc_auc" in eval_metrics:
        roc_auc_script_path = str(Path(train_config.hf_hub_metrics_cache_dir) / "auc/roc_auc.py")
    if "pr_auc" in eval_metrics:
        pr_auc_script_path = str(Path(train_config.hf_hub_metrics_cache_dir) / "auc/pr_auc.py")

    # Initialize metrics modules
    if not train_config.eval_in_memory:  # distrbuted computation fails in multi-node training with nfs https://github.com/huggingface/evaluate/issues/481
        metrics_modules = {}
        if "rouge" in eval_metrics:
            rouge: EvaluationModule = load(path=rough_script_path, num_process=world_size, process_id=rank, cache_dir=cache_dir, experiment_id="rouge", timeout=1, trust_remote_code=True)
            metrics_modules["rouge"] = rouge
        if "sacrebleu" in eval_metrics:
            bleu: EvaluationModule = load(path=sacrebleu_script_path, num_process=world_size, process_id=rank, cache_dir=cache_dir, experiment_id="sacrebleu", timeout=1, trust_remote_code=True)
            metrics_modules["bleu"] = bleu
        if "roc_auc" in eval_metrics:
            roc_auc: EvaluationModule = load(path=roc_auc_script_path, num_process=world_size, process_id=rank, cache_dir=cache_dir, experiment_id="roc_auc", timeout=1, trust_remote_code=True)
            metrics_modules["roc_auc"] = roc_auc
        if "pr_auc" in eval_metrics:
            pr_auc: EvaluationModule = load(path=pr_auc_script_path, num_process=world_size, process_id=rank, cache_dir=cache_dir, experiment_id="pr_auc", timeout=1, trust_remote_code=True)
            metrics_modules["pr_auc"] = pr_auc

        for metrics_module in metrics_modules.values():  # solve https://github.com/huggingface/evaluate/issues/481
            logging.info(f"{train_config.eval_in_memory=}: initializing distributed file writer for {metrics_module.experiment_id}")
            # customized code to avoid acquiring the file lock which is far less reliable in multi-node training than torch.distributed.barrier()
            metrics_module._create_cache_file = _create_cache_file.__get__(metrics_module)
            metrics_module._get_all_cache_files = _get_all_cache_files.__get__(metrics_module)
            metrics_module.compute = compute.__get__(metrics_module)
            init_writer(metrics_module)
    else:
        if "rouge" in eval_metrics:
            rouge: EvaluationModule = load(path=rough_script_path, cache_dir=cache_dir, experiment_id=f"rouge-{world_size}-{rank}", keep_in_memory=True, trust_remote_code=True)
            metrics_modules["rouge"] = rouge
        if "sacrebleu" in eval_metrics:
            bleu: EvaluationModule = load(path=sacrebleu_script_path, cache_dir=cache_dir, experiment_id=f"sacrebleu-{world_size}-{rank}", keep_in_memory=True, trust_remote_code=True)
            metrics_modules["bleu"] = bleu
        if "roc_auc" in eval_metrics:
            roc_auc: EvaluationModule = load(path=roc_auc_script_path, cache_dir=cache_dir, experiment_id=f"roc_auc-{world_size}-{rank}", keep_in_memory=True, trust_remote_code=True)
            metrics_modules["roc_auc"] = roc_auc
        if "pr_auc" in eval_metrics:
            pr_auc: EvaluationModule = load(path=pr_auc_script_path, cache_dir=cache_dir, experiment_id=f"pr_auc-{world_size}-{rank}", keep_in_memory=True, trust_remote_code=True)
            metrics_modules["pr_auc"] = pr_auc

    return metrics_modules


class ModelDebugger:
    def __init__(self, train_config, eval_dataloader, local_rank, rank, tokenizer, global_step, use_probe_data=False, pretrain=False):
        """
        Model Debugger is to prepare the probe data and metadata for model debugging. TODO: refactor this to be more general for other debugging purpose, and move it out of eval_utils.py
        """
        self.train_config = train_config
        self.eval_dataloader = eval_dataloader
        self.local_rank = local_rank
        self.rank = rank
        self.tokenizer = tokenizer
        self.global_step = global_step
        self.use_probe_data = use_probe_data
        self.pretrain = pretrain
        self.init_model_debug_info()  # this will prepare self.data, self.metadata, self.meta_data_dict

    def log_model_debug(self, model_logger: ModelLogger):
        logging.info("Logging model debug info to tensorboard, this may take a while ...")
        token_importance = model_logger.log_to_tensorboard()  # list of dict, each dict contains top_k keys with {'tag': tag, 'token_importance': token_importance, 'labels': labels}

        # Log token importance
        from colorama import init, Fore
        init(autoreset=True)

        for token_importance_per_tag in tqdm(token_importance, desc="Logging top important keys in attention matrix", dynamic_ncols=True):
            tag, batch_importance_weight, meta_labels = token_importance_per_tag['tag'], token_importance_per_tag['token_importance'], token_importance_per_tag['labels']
            if batch_importance_weight is None:
                break

            # labels: [f"{label}-layer{self.layer_idx}" for label in self.metadata["labels"]] -> meta_labels: [f"{label}"]
            meta_labels_wo_layer_idx = [label.rsplit("-", 1)[0] for label in meta_labels]  # (2N)
            # get original prompt by meta_labels, self.meta_data_dict: {label: (prompt, response, input_ids, labels)}
            meta_data_by_label = [self.meta_data_dict[meta_labels_wo_layer_idx[i]] for i in range(len(meta_labels_wo_layer_idx))]  # (2N)

            # batch_importance_weight is tensor of shape (2N, seq_len)
            batch_importance_weight = batch_importance_weight.float().cpu().numpy()
            for i in range(len(batch_importance_weight)):
                (_, _, input_ids, labels) = meta_data_by_label[i]  # input_ids shape: (seq_len), overall: (prompt, response, input_ids, labels)
                importance_weight = batch_importance_weight[i]  # importance_weight shape: (seq_len)

                # Filter out unmasked tokens for both input_ids and importance_weight by looking up positions of -100 in labels
                filtered_input_ids = [token_id for token_id, label in zip(input_ids, labels) if label == -100]
                filtered_importance_weight = [weight for weight, label in zip(importance_weight, labels) if label == -100]
                # Filter out special tokens for both input_ids and importance_weight
                special_tokens = set(self.tokenizer.all_special_ids)
                filtered_input_ids = [token_id for token_id in filtered_input_ids if token_id not in special_tokens]
                filtered_importance_weight = [filtered_importance_weight[j] for j, token_id in enumerate(filtered_input_ids) if token_id not in special_tokens]
                filtered_decoded_tokens = [self.tokenizer.decode([token_id], skip_special_tokens=True) for token_id in filtered_input_ids]

                # Apply logarithmic scaling if weights span several orders of magnitude
                filtered_importance_weight = np.log1p(filtered_importance_weight - np.min(filtered_importance_weight) + 1)

                # Define ANSI color functions
                def get_color(weight, mode='ansi'):
                    # Calculate percentiles: [90, 75, 50, 25, 10, 5]
                    percentiles = np.percentile(filtered_importance_weight, [90, 50, 25, 5, 2, 1])
                    
                    if weight > percentiles[0]:
                        color = ('red', Fore.RED)        # Top 10%
                    elif weight > percentiles[1]:
                        color = ('orange', Fore.YELLOW)  # 10%-50%
                    elif weight > percentiles[2]:
                        color = ('blue', Fore.BLUE)      # 50%-25%
                    elif weight > percentiles[3]:
                        color = ('cyan', Fore.CYAN)      # 25%-5%
                    elif weight > percentiles[4]:
                        color = ('green', Fore.GREEN)    # 5%-2%
                    else:
                        color = ('darkgreen', Fore.LIGHTBLACK_EX)   # Bottom 1%

                    return color[1] if mode == 'ansi' else color[0]

                # Prepare console messages with ANSI colors
                console_message = "".join([f"{get_color(weight, 'ansi')}{token}{Fore.RESET} " for token, weight in zip(filtered_decoded_tokens, filtered_importance_weight)])
                # Prepare HTML messages with ANSI colors, manually paste the logged result into an HTML viewer to view
                html_message = "".join([f'<span style="color:{get_color(weight, "html")}">{token}</span> ' for token, weight in zip(filtered_decoded_tokens, filtered_importance_weight)])
                logging.info(f"-->Attention Heatmap for {tag}:")
                logging.info(f"[Console]Attention Heatmap of {urllib.parse.unquote(meta_labels[i])}: {console_message}")
                logging.info(f"[HTML]Attention Heatmap of {urllib.parse.unquote(meta_labels[i])}: {html_message}")

    def init_model_debug_info(self) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        """
        Get the data and metadata for model debugging

        data: List of dictionaries containing input_ids, attention_mask, and labels
        metadata: Dictionary containing the sentences and labels for each analogy relationship, e.g. {'labels': ['king-queen', "man-woman"], 'global_step': 0}
        """
        if self.use_probe_data:
            # Example sentences and labels for direct comparison of analogy relationships
            prompts = [
                "king is to queen",
                "man is to woman",
                "Paris is to France",
                "Tokyo is to Japan",
                "swim is to swam",
                "fly is to flew",
                "doctor is to nurse",
                "teacher is to student",
                "dog is to puppy",
                "cat is to kitten",
                "Apple is to iPhone",
                "Microsoft is to Windows",
                "Russia is to ruble",
                "Japan is to yen",
                "actor is to actress",
                "waiter is to waitress",
                "see is to saw",
                "go is to went",
                "Spain is to Spanish",
                "Italy is to Italian",
                "amazing is to amazed",
                "terrifying is to terrified",
                "ability is to abilities",
                "country is to countries",
                "uncle is to aunt",
                "nephew is to niece",
                "good is to better",
                "cold is to colder",
                "Brazil is to Brasilia",
                "Canada is to Ottawa",
                "tree is to forest",
                "building is to city",
                "walking is to walked",
                "swimming is to swam",
                "mouse is to mice",
                "goose is to geese",
                "United States is to dollar",
                "United Kingdom is to pound",
                "child is to children",
                "person is to people"
            ]

            responses = [""] * len(prompts)  # leave it to empty to debug end of sentence, fill this for other debugging purpose

            assert len(prompts) == len(responses), "The number of prompts and responses should be the same"

            meta_labels = [
                "king-queen",
                "man-woman",
                "Paris-France",
                "Tokyo-Japan",
                "swim-swam",
                "fly-flew",
                "doctor-nurse",
                "teacher-student",
                "dog-puppy",
                "cat-kitten",
                "Apple-iPhone",
                "Microsoft-Windows",
                "Russia-ruble",
                "Japan-yen",
                "actor-actress",
                "waiter-waitress",
                "see-saw",
                "go-went",
                "Spain-Spanish",
                "Italy-Italian",
                "amazing-amazed",
                "terrifying-terrified",
                "ability-abilities",
                "country-countries",
                "uncle-aunt",
                "nephew-niece",
                "good-better",
                "cold-colder",
                "Brazil-Brasilia",
                "Canada-Ottawa",
                "tree-forest",
                "building-city",
                "walking-walked",
                "swimming-swam",
                "mouse-mice",
                "goose-geese",
                "US-dollar",
                "UK-pound",
                "child-children",
                "person-people"
            ]

            assert len(prompts) == len(meta_labels), "The number of prompts and labels should be the same"

        else:
            metadata = self._get_metadata()
            prompts, responses, meta_labels = [m["prompts"] for m in metadata], [m["responses"] for m in metadata], [m["labels"] for m in metadata]
            # make prompts, responses, labels even by dropping the last one if the number of prompts etc. is odd
            if len(prompts) % 2 != 0:
                prompts, responses, meta_labels = prompts[:-1], responses[:-1], meta_labels[:-1]

        # Encode meta_labels to be URL safe to ensure tensorboard can display it correctly
        def encode_meta_label(meta_label):
            encoded_meta_label = urllib.parse.quote(meta_label)
            return encoded_meta_label
        meta_labels = [encode_meta_label(meta_label) for meta_label in meta_labels]

        data = []
        self.meta_data_dict = {}  # {label: (prompt, response, input_ids, labels)} for easy lookup
        # len(prompts) should be even numbers
        assert len(prompts) % 2 == 0, "The number of sentences should be even"
        for i in range(0, len(prompts), 2):
            # Encode pair of prompts and labels into one batch
            prompt_ids1 = self.tokenizer.encode(self.tokenizer.bos_token + prompts[i], add_special_tokens=False)
            prompt_ids2 = self.tokenizer.encode(self.tokenizer.bos_token + prompts[i + 1], add_special_tokens=False)
            label_ids1 = self.tokenizer.encode(responses[i], add_special_tokens=False)
            label_ids2 = self.tokenizer.encode(responses[i + 1], add_special_tokens=False)

            # Prepare the batch without labels
            batch = {
                "input_ids": [prompt_ids1 + label_ids1, prompt_ids2 + label_ids2],
                "attention_mask": [[1] * len(prompt_ids1 + label_ids1), [1] * len(prompt_ids2 + label_ids2)]
            }

            # Pad the batch and convert to tensors
            padded_batch = self.tokenizer.pad(batch, padding=True, return_tensors="pt")

            # Prepare labels separately
            if self.pretrain:
                labels = [prompt_ids1 + label_ids1, prompt_ids2 + label_ids2]
            else:
                labels = [[-100] * len(prompt_ids1) + label_ids1, [-100] * len(prompt_ids2) + label_ids2]

            # Pad labels manually to match the length of input_ids
            max_length = padded_batch['input_ids'].size(1)
            padding_side = self.tokenizer.padding_side

            padded_labels = []
            for label in labels:
                padding_length = max_length - len(label)
                if padding_side == 'right':
                    padded_label = label + [-100] * padding_length
                else:  # padding_side == 'left'
                    padded_label = [-100] * padding_length + label
                padded_labels.append(padded_label)

            # Convert padded labels to tensor and add to batch
            padded_batch['labels'] = torch.tensor(padded_labels)

            # Add the meta labels to the dictionary for easy lookup, assuming meta_labels[i] is globally unique
            self.meta_data_dict.update({meta_labels[i]: (prompts[i], responses[i], padded_batch['input_ids'][0], padded_batch['labels'][0])})
            self.meta_data_dict.update({meta_labels[i+1]: (prompts[i], responses[i+1], padded_batch['input_ids'][1], padded_batch['labels'][1])})

            # Append the batch to the data list
            data.append(padded_batch)

        # make prompts, response, labels into batch as well, batch size is 2
        prompts = [prompts[i:i + 2] for i in range(0, len(prompts), 2)]
        responses = [responses[i:i + 2] for i in range(0, len(responses), 2)]
        meta_labels = [meta_labels[i:i + 2] for i in range(0, len(meta_labels), 2)]

        metadata = [{"prompts": prompts[i], "responses": responses[i], "labels": meta_labels[i], "global_step": self.global_step} for i in range(len(prompts))]

        assert len(data) == len(metadata), "The number of data and metadata should be the same"

        self.data, self.metadata = data, metadata

    def _get_metadata(self) -> List[Dict[str, torch.Tensor]]:
        """
        Get metadata for model debugging from the evaluation dataloader
        """
        prompts = []
        responses = []
        if self.pretrain is False:
            # remove the response part of input_ids from each batch of eval_dataloader
            for step, batch in enumerate(tqdm(self.eval_dataloader, colour="green", desc=f"[{self.rank=}] Preparing eval data step", dynamic_ncols=True)) if self.local_rank == 0 or self.train_config.debug else enumerate(self.eval_dataloader):
                # stop when the maximum number of eval steps is reached
                if self.train_config.max_eval_step > 0 and step > self.train_config.max_eval_step:
                    logging.info(f"Max eval steps reached, stopping evaluation, total_eval_steps: {step - 1}")
                    break

                # Keep only the input_ids for all position of corresponding labels being last -100
                for i in range(batch['labels'].size(0)):
                    labels = batch['labels'][i]
                    input_ids = batch['input_ids'][i]

                    # Find the index of the last -100 in the labels
                    masked_indices = (labels == -100).nonzero(as_tuple=True)[0]
                    if masked_indices.numel() > 0:
                        last_masked_idx = masked_indices.max().item()
                    else:
                        last_masked_idx = -1  # Default value if -100 is not found

                    # Slice the input_ids and labels up to the last -100 position
                    input_decoded = "".join([self.tokenizer.decode(input_id, skip_special_tokens=True) for input_id in input_ids[:last_masked_idx + 1]])
                    # Keep only the labels for all positions after the last -100
                    label_decoded = "".join([self.tokenizer.decode(label, skip_special_tokens=True) for label in labels[last_masked_idx + 1:]])

                    prompts.append(input_decoded)
                    responses.append(label_decoded)
        else:  # take all input_ids as prompts and no responses
            for step, batch in enumerate(tqdm(self.eval_dataloader, colour="green", desc=f"[{self.rank=}] Preparing eval data step", dynamic_ncols=True)) if self.local_rank == 0 or self.train_config.debug else enumerate(self.eval_dataloader):
                # stop when the maximum number of eval steps is reached
                if self.train_config.max_eval_step > 0 and step > self.train_config.max_eval_step:
                    logging.info(f"Max eval steps reached, stopping evaluation, total_eval_steps: {step - 1}")
                    break

                # Keep only the input_ids for all position of corresponding labels being last -100
                # Get labels tensor
                labels = batch['input_ids']
                inputs = [self.tokenizer.decode(input_id, skip_special_tokens=True) for input_id in batch['input_ids']]
                labels = [""] * len(inputs)  # leave it to empty to debug end of sentence, fill this for other debugging purpose

                prompts.extend(inputs)
                responses.extend(labels)

        # labels being last 10 words of the prompt
        labels = [f'[{step}]{"-".join(prompt.split()[-10:])}' for step, prompt in enumerate(prompts)]

        metadata = [{"prompts": prompts[i], "responses": responses[i], "labels": labels[i], "global_step": self.global_step} for i in range(len(prompts))]

        return metadata
