# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import argparse
import json
import logging
import os
import re
import sys
import glob
from pathlib import Path

if "HF_DATASETS_OFFLINE" not in os.environ:
    os.environ["HF_DATASETS_OFFLINE"] = "1"  # Disable datasets downloading
if "HF_HOME" not in os.environ:
    os.environ["HF_HUB_OFFLINE"] = "1"  # Disable model downloading
if "HF_DATASETS_CACHE" not in os.environ:
    os.environ["HF_DATASETS_CACHE"] = "/shared/public/data/controlllm/datasets"  # Set the cache directory for datasets
if "HF_METRICS_CACHE" not in os.environ:
    os.environ["HF_METRICS_CACHE"] = "/shared/public/data/controlllm/metrics"  # Set the cache directory for metrcs modules, copy the metrics code from ./controlllm/metrics to this folder
if "HF_ALLOW_CODE_EVAL" not in os.environ:
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"  # Enable code evaluation
if "HF_ALLOW_CODE_EVAL_DEBUG" not in os.environ:
    os.environ["HF_ALLOW_CODE_EVAL_DEBUG"] = "0"  # Set this to 1 to enable debug mode for coding task evaluation, the code being executed and the correctness result will be printed and logged
if "EVAL_PATH" not in os.environ:
    os.environ["EVAL_PATH"] = "/home/jobuser/resources/lm-evaluation-harness"  # Path of the cloned repo of lm-evaluation-harness
os.environ['MODULE'] = 'controlllm_eval'
if "MODEL_PATH" not in os.environ:  # Path of the model to be evaluated in huggingface model format, use the ./src/controlllm/utils/checkpoint_converter.py to convert the checkpoint to huggingface model format before putting it here, or alternatively, put the pretrained model path here and set the checkpoint path in os.environ["CHECKPOINT_PATH"]
    os.environ["MODEL_PATH"] = ""  # Set MODEL_PATH here, otherwise, run with MODEL_PATH=<model checkpoint path> accelerate launch ./src/controlllm/inference/llm_eval_harness/eval.py
from controlllm.inference.llm_eval_harness import control_llm_filter_init  # Money patch the lm_eval.filter package's __init__.py

import torch
from transformers import PreTrainedModel

import numpy as np
import lm_eval
from lm_eval import tasks, utils
from lm_eval.utils import make_table
from lm_eval.api.task import ConfigurableTask

# noqa: E402, put it after os.environ to make sure we set os.environ["HF_DATASETS_CACHE"] after importing the datasets library
from controlllm.inference.llm_eval_harness.control_llm_task import ControlLLMConfigTask
from controlllm.inference.llm_eval_harness.control_llm_evaluator import evaluate
from controlllm.inference.llm_eval_harness.control_llm_hf import ControlLLMWrapper as ControlLLMWrapperHF
from controlllm.inference.llm_eval_harness.control_llm_vllm import ControlLLMWrapper as ControlLLMWrapperVLLM
from controlllm.utils.model_expander import ModelExpander


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def setup_logging(verbosity):
    logging.basicConfig(
        level=verbosity.upper(), format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def handle_output(args, results, logger):
    if results is None:
        logger.error("No results were returned.")
        return

    if not args.output_path:
        if args.log_samples:
            logger.error("Specify --output_path for logging samples.")
            sys.exit(1)
        logger.info(json.dumps(results, indent=2, default=_handle_non_serializable))
        return

    path = Path(args.output_path)
    file_name = f"benchmark_results_{args.tasks.replace(',', '_')}.json"
    if (path / file_name).exists():
        logger.warning(f"File already exists at {path}. Results will be overwritten.")

    output_dir = path.parent if path.suffix in (".json", ".jsonl") else path
    output_dir.mkdir(parents=True, exist_ok=True)

    results_str = json.dumps(results, indent=2, default=_handle_non_serializable)
    if args.show_config:
        logger.info(results_str)

    file_path = os.path.join(args.output_path, file_name)
    with open(file_path , "w", encoding="utf-8") as f:
        f.write(results_str)
    logging.info(f"Benchmark results saved to {file_path}")

    if args.log_samples:
        samples = results.pop("samples", {})
        for task_name, _ in results.get("configs", {}).items():
            output_name = re.sub(r"/|=", "__", args.model_args) + "_" + task_name
            sample_file = output_dir.joinpath(f"{output_name}.jsonl")
            sample_data = json.dumps(
                samples.get(task_name, {}), indent=2, default=_handle_non_serializable
            )
            sample_file.write_text(sample_data, encoding="utf-8")

    batch_sizes = ",".join(map(str, results.get("config", {}).get("batch_sizes", [])))
    summary = f"{args.model} ({args.model_args}), gen_kwargs: ({args.gen_kwargs}), limit: {args.limit}, num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
    logger.info(summary)
    logger.info(make_table(results))
    if "groups" in results:
        logger.info(make_table(results, "groups"))


def load_tasks(args):
    # monkey patching to customize ConfigurableTask to support additional coding tasks
    ConfigurableTask.process_results = ControlLLMConfigTask.process_results
    lm_eval.evaluator.evaluate = evaluate

    from datasets.utils.logging import set_verbosity_error
    # Set the logging level to ERROR to suppress warnings when loading task specifc data
    set_verbosity_error()

    if args.open_llm_leaderboard_tasks:
        # To reproduce Meta 3.1 Evaluation Metrics Using LM-Evaluation-Harness, we set apply_chat_template and include tasks with few shots
        logging.info("Loading tasks from HF open LLM-leaderboard with chat template applied and few-shot tasks...")
        args.apply_chat_template = True
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.join(current_dir, "open_llm_leaderboard")
        task_manager = tasks.TaskManager(include_path=config_dir)
        args.tasks = "arc_challenge_25_shot,hellaswag_10_shot,truthfulqa_mc2,winogrande_5_shot,gsm8k,mmlu"
        return task_manager, args.tasks.split(",")

    logging.info("Loading additional tasks that are not supported in default tasks folder of lm-evaluation-harness...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    additional_tasks_dir = os.path.join(current_dir, "additional_tasks")
    task_manager = tasks.TaskManager(include_path=additional_tasks_dir)
    return task_manager, args.tasks.split(",") if args.tasks else []


def parse_eval_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--model", "-m", default="hf", help="Name of model, e.g., `hf`, `vllm`."
    )
    parser.add_argument(
        "--tasks",
        "-t",
        # default="arc_challenge,hellaswag,truthfulqa_mc2,winogrande,gsm8k,mathqa,meta_math_hard,mmlu,offline_mmlu_pro,ceval-valid,cmmlu,mbpp,humaneval,humaneval_greedy,meta_instruct,meta_pretrain",
        default="arc_challenge,hellaswag,truthfulqa_mc2,winogrande,gsm8k,mmlu",
        help="Comma-separated list of tasks, or 'list' to display available tasks.",
    )
    parser.add_argument(
        "--model_args",
        "-a",
        default=f"pretrained={os.environ['MODEL_PATH']},dtype=bfloat16",
        help="Comma-separated string arguments for model, e.g., `pretrained=EleutherAI/pythia-160m`.",
    )
    parser.add_argument(
        "--open_llm_leaderboard_tasks",
        "-oplm",
        action="store_true",
        default=False,
        help="Choose the list of tasks with specification in HF open LLM-leaderboard.",
    )
    parser.add_argument(
        "--num_fewshot",
        "-f",
        type=int,
        default=None,
        help="Number of examples in few-shot context.",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        default=32,
        help="Batch size, can be 'auto', 'auto:N', or an integer.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Maximal batch size with 'auto' batch size.",
    )
    parser.add_argument(
        "--device", default=None, help="Device for evaluation, e.g., 'cuda', 'cpu'."
    )
    parser.add_argument(
        "--output_path", "-o", type=str,
        default=os.environ["MODEL_PATH"],
        help="Path for saving results."
    )
    parser.add_argument(
        "--limit",
        "-L",
        type=float,
        default=None,
        help="Limit number of examples per task.",
    )
    parser.add_argument(
        "--use_cache", "-c", default=None, help="Path to cache db file, if used."
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        default="INFO",
        help="Logging level: CRITICAL, ERROR, WARNING, INFO, DEBUG.",
    )
    parser.add_argument(
        "--gen_kwargs",
        default=None,
        help="Generation kwargs for tasks that support it.",
    )
    parser.add_argument(
        "--check_integrity",
        action="store_true",
        help="Whether to run the relevant part of the test suite for the tasks.",
    )
    parser.add_argument(
        "--write_out",
        "-w",
        action="store_true",
        default=False,
        help="logging.infos the prompt for the first few documents.",
    )
    parser.add_argument(
        "--log_samples",
        "-s",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis.",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        help="Additional path to include if there are external tasks.",
    )
    parser.add_argument(
        "--apply_chat_template",
        "-chtplt",
        action="store_true",
        default=False,
        help="Apply chat template to data of the tasks.",
    )
    parser.add_argument(
        "--fewshot_as_multiturn",
        "-fsmt",
        action="store_true",
        default=False,
        help="Treat few-shot as multi-turn conversation.",
    )
    return parser.parse_args()


def evaluate_model(args, logger=None, loaded_model: PreTrainedModel=None, rank=0):
    # Set up tasks
    setup_open_llm_eval()

    # Register the expanded model classes with new model architecture
    if loaded_model:
        control_llm_model = loaded_model.name_or_path  # make sure this is the path to the model in config.json, assuming the model is already registered by ModelExpander.register_expansion_classes
    else:
        # Extract pretrained as a string from model_args and register the expanded model classe in huggingface transformers, assuming only one model is configured!
        control_llm_model = next((arg.split("=")[1] for arg in args.model_args.split(",") if arg.startswith("pretrained=")), None)
        if not control_llm_model:
            raise ValueError(f"Could not find a valid model name or path from the provided model_args: {args.model_args}")

        logging.info(f"Registering the expanded model classes with new model architecture from {control_llm_model}...")
        ModelExpander.register_expansion_classes(control_llm_model, use_vllm=(args.model=="vllm"))

    task_manager, task_list = load_tasks(args)
    logging.info(f"Running benchmark {task_list} on model: {control_llm_model}...")
    os.environ['MODEL_PATH'] = control_llm_model  # tasks in additional_tasks needs this environment variable to load tokenizer for apply_chat_template, refer to additional_tasks/tokenizer.py
    # Customized model such as Quantized model etc.
    # In case you are working with a custom model, you can use the following guide to add it here:
    # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage

    # Prepare the args.model_args and args.model for benchmark
    # evaluate with customized lm-evaluation-harness/vLLM class by DDP
    if args.model == "vllm":
        # DDP by num_gpus, tensor_parallel_size=1, without model parallelism
        # note: trained_from is required for vLLM to register the expanded model classes, for none expanded model class, it is not required but still okay to provide. TODO: check if it is required, remove trained_from={control_llm_model} if not required
        # note: As for `add_bos_token=True`, since our prompts in the evals dataset has already included all the special tokens required by instruct model, such as `<|start_header_id|>user<|end_header_id|>`, we will not use `--apply_chat_template` argument for instruct models anymore. However, we need to use `add_bos_token=True` flag to add the BOS_token back during VLLM inference, as the BOS_token is removed by default in [this PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/1465).
        if args.model_args == f"pretrained={os.environ['MODEL_PATH']},dtype=bfloat16":
            num_gpus = torch.cuda.device_count()
            # add vllm parameter if args.model_args is default, so it is still configurable TODO: make this as default in parse_eval_args for vLLM instead of here
            args.model_args += f",tensor_parallel_size=1,gpu_memory_utilization=0.8,data_parallel_size={num_gpus},max_model_len=8192,enable_prefix_caching=True,{'' if args.apply_chat_template else 'add_bos_token=True'},seed=42,trained_from={control_llm_model}"
        model_args_dict = utils.simple_parse_args_string(args.model_args)
        lm_control_llm_vllm = ControlLLMWrapperVLLM.create_from_arg_obj(
            model_args_dict,
            {
                "batch_size": args.batch_size,
                "max_batch_size": args.max_batch_size,
                "device": args.device,
            },
        )
        # lm_control_llm_vllm = ControlLLMWrapperVLLM(**model_args_dict)
        args.model = lm_control_llm_vllm
    # evaluate with customized lm-evaluation-harness/HF(huggingface transformers lib) class by DDP
    elif args.model == "hf":
        if args.model_args == f"pretrained={os.environ['MODEL_PATH']},dtype=bfloat16":
            args.model_args += f",{'' if args.apply_chat_template else 'add_bos_token=True'}"
        model_args_dict = utils.simple_parse_args_string(args.model_args)

        # evaluate with huggerface interface for already loaded model
        if loaded_model:
            model_args_dict["pretrained"] = loaded_model

        lm_control_llm_hf = ControlLLMWrapperHF.create_from_arg_obj(
            model_args_dict,
            {
                "batch_size": args.batch_size,
                "max_batch_size": args.max_batch_size,
                "device": args.device,
            },
        )
        # lm_control_llm_hf = ControlLLMWrapperHF(**model_args_dict)
        args.model = lm_control_llm_hf

    # Run benchmark
    results = lm_eval.simple_evaluate(
        model=args.model,
        model_args=model_args_dict,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        use_cache=args.use_cache,
        limit=args.limit,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        log_samples=args.log_samples,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        gen_kwargs=args.gen_kwargs,
        task_manager=task_manager,
    )

    if rank == 0:
        handle_output(args, results, logger)

    return results


def setup_open_llm_eval():
    """
    Set up the tasks to be evaluated. This is the same as open_llm_eval_prep.sh but run it in python automatically
    """
    logging.info("Setting up the tasks for evaluation with lm-evaluation-harness...")

    # Get EVAL_PATH from the environment variable or prompt the user
    EVAL_PATH = os.environ.get('EVAL_PATH')
    if not EVAL_PATH:
        EVAL_PATH = input("Enter the absolute path to the lm-evaluation-harness: ")

    # Get the directory where the script is located
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

    # Use the script's directory to find the open_llm_leaderboard directory
    DIR_LEADERBOARD = os.path.join(SCRIPT_DIR, 'open_llm_leaderboard')
    # Use the script's directory to find the open_llm_leaderboard directory
    DIR_ADDITIONAL_TASKS = os.path.join(SCRIPT_DIR, 'additional_tasks')

    for DIR in [DIR_LEADERBOARD, DIR_ADDITIONAL_TASKS]:
        # Check if the directory exists
        if not os.path.isdir(DIR):
            logging.info(f"Error: Directory '{DIR}' not found.")
            continue

        # Function to replace the placeholder '{$EVAL_PATH}' with the actual path recursively
        def replace_eval_path(dir):
            # Flag to track whether any replacements were made
            replacements_made = False

            # Iterate over YAML files in the directory and update them
            for yaml_file in [f for pattern in ('*.yaml', '*_yaml') for f in glob.glob(os.path.join(dir, pattern))]:
                with open(yaml_file, 'r') as file:
                    content = file.read()

                if '{$EVAL_PATH}' in content:
                    # Placeholder found, perform the replacement
                    new_content = content.replace('{$EVAL_PATH}', EVAL_PATH)
                    with open(yaml_file, 'w') as file:
                        file.write(new_content)
                    logging.info(f"Updated {yaml_file} with EVAL_PATH: {EVAL_PATH}")
                    replacements_made = True
                elif '{$CURRENT_FILE_PATH}' in content:
                    # Placeholder found, perform the replacement
                    new_content = content.replace('{$CURRENT_FILE_PATH}', os.path.dirname(yaml_file))
                    with open(yaml_file, 'w') as file:
                        file.write(new_content)
                    logging.info(f"Updated {yaml_file} with CURRENT_FILE_PATH: {os.path.dirname(yaml_file)}")
                    replacements_made = True
                else:
                    # Placeholder not found, possibly already replaced
                    logging.debug(f"No placeholder found in {yaml_file}; skipping.")

            # Iterate over subdirectories and update YAML files in them
            for sub_dir in [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]:
                replace_eval_path(os.path.join(dir, sub_dir))

            if replacements_made:
                logging.info("Replacements were made. The placeholder '{$EVAL_PATH}' was found in any files " + f"of {DIR}")

        # Perform the replacements recursively
        replace_eval_path(DIR)


if __name__ == "__main__":
    args = parse_eval_args()
    logger = setup_logging(args.verbosity)
    evaluate_model(args, logger, rank=int(os.environ.get("RANK", "0")))
