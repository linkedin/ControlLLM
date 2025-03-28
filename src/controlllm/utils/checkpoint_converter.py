# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# This is a utility script to convert FSDP sharded model checkpoints to HuggingFace format
# Model is prefered to be saved in shared state dict format, so it does not cause NCCL timeout in multiple node training
# So this script is designed as a post-processing step after training
import re
import os
import torch
import shutil
from tqdm import tqdm
from typing import List
from pathlib import Path
from sentence_transformers.util import is_sentence_transformer_model
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, PretrainedConfig

from controlllm.utils.model_expander import ModelExpander
from controlllm.utils.custom_llama_recipes import load_sharded_model_single_gpu
from controlllm.utils.custom_sentence_transformers import CustomSentenceTransformer as SentenceTransformer
if "CHECKPOINT_PATH" not in os.environ:
    os.environ['CHECKPOINT_PATH'] = ""  # set the model path here, otherwise run with CHECKPOINT_PATH=<model checkpoint path> accelerate launch ./src/controlllm/inference/batch_eval.py

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Checkpoint Converter FSDP to HF")
    parser.add_argument("--fsdp_checkpoint_path", type=str, default="", help="Path to FSDP Sharded model checkpoints")
    parser.add_argument("--consolidated_model_path", type=str, default="", help="Path to save the HF converted model checkpoints")
    parser.add_argument("--merge_layers", action="store_true", default=False, help="Merge layers with concat operation")
    parser.add_argument("--merge_back", action="store_true", default=False, help="Merge back the layers with split operation")  # action="store_true" sets the default value to False
    parser.add_argument("--merge_from", type=str, default="", help="Path to FSDP Sharded model checkpoints to merge from, if not provided, merge from the same model path")
    return parser.parse_args()


# Loading the model from config to load FSDP checkpoints into that
def load_model_from_config(fsdp_checkpoint_path):
    """
    Register the expanded model classes with new model architecture before load the model from config.

    Args:

    hf_model_path_or_name: the original HF model path or name which are not expanded yet
    fsdp_checkpoint_path: the path to the FSDP sharded checkpoints with expanded model classes
    
    Note that huggingface needs a special registeration for the expanded model classes so that it can load the model from config.
    """
    print(f"Registering the expanded model classes with new model architecture from {fsdp_checkpoint_path}")
    ModelExpander.register_expansion_classes(fsdp_checkpoint_path)

    model_config = AutoConfig.from_pretrained(fsdp_checkpoint_path, trust_remote_code=True)

    if is_sentence_transformer_model(model_config.trained_from):
        # Note that _name_or_path in config.json is the path to the model pretrained model after expansion saved by ./utils/model_expander.py
        config_dict, _ = PretrainedConfig.get_config_dict(fsdp_checkpoint_path)
        # this loads the pretrained model from the expanded folder with expansion but without fine tuning
        model = SentenceTransformer(model_name_or_path=config_dict["_name_or_path"])
    else:
        # this does not load the model weights, weights are initialized randomly
        model = AutoModelForCausalLM.from_config(model_config)
    return model


def main():
    args = parse_args()

    # the command line arguments has higher priority than CHECKPOINT_PATH in os.environ if it is not empty
    if args.fsdp_checkpoint_path == "":
        args = handle_model_path_in_os_env(args)
    else:
        if args.consolidated_model_path == "":
            args.consolidated_model_path = Path(args.fsdp_checkpoint_path).parent / ("hf-merged" if args.merge_layers else "hf") / Path(args.fsdp_checkpoint_path).name

    if isinstance(args.fsdp_checkpoint_path, list):
        print(f"Converting multiple FSDP sharded model checkpoints to HuggingFace format one by one: {args.fsdp_checkpoint_path}...")
        for fsdp_checkpoint, consolidated_model in tqdm(zip(args.fsdp_checkpoint_path, args.consolidated_model_path), total=len(args.fsdp_checkpoint_path), desc="Converting Checkpoints"):
            print(f"Converting FSDP sharded model checkpoint: {fsdp_checkpoint} to HuggingFace format: {consolidated_model}...")
            convert_checkpoint(fsdp_checkpoint, consolidated_model, args.merge_layers, args.merge_back, args.merge_from)
    else:
        convert_checkpoint(args.fsdp_checkpoint_path, args.consolidated_model_path, args.merge_layers, args.merge_back, args.merge_from)


def load_checkpoint(fsdp_checkpoint_path: str):
    """
    Load the model from config to load FSDP checkpoints into that.
    """
    # load the HF model definition from config
    print(f"Loading model from config {fsdp_checkpoint_path}")
    model_def = load_model_from_config(fsdp_checkpoint_path)
    print(f"Model is loaded from config {fsdp_checkpoint_path}")
    # load the FSDP sharded checkpoints into the model
    print(f"Loading model from FSDP sharded checkpoints {fsdp_checkpoint_path}")

    # check if fsdp_checkpoint_path folder has .metadata, __0_0.distcp, __1_0.distcp, __2_0.distcp, ... etc, if not, check if Path(fsdp_checkpoint_path) / pytorch_model_fsdp_0 has it
    if not (Path(fsdp_checkpoint_path) / ".metadata").exists():
        print(f"fsdp_checkpoint_path {fsdp_checkpoint_path} does not have .metadata, checking if {fsdp_checkpoint_path} / pytorch_model_fsdp_0 has it")
        fsdp_checkpoint_path_fsdp_0 = Path(fsdp_checkpoint_path) / "pytorch_model_fsdp_0"
        if not (Path(fsdp_checkpoint_path_fsdp_0) / ".metadata").exists():
            raise FileNotFoundError(f"fsdp_checkpoint_path {fsdp_checkpoint_path_fsdp_0} does not have .metadata")
        print(f"Loading sharded checkpoint from {fsdp_checkpoint_path_fsdp_0} ...")
        model = load_sharded_model_single_gpu(model_def, fsdp_checkpoint_path_fsdp_0)
    else:
        print(f"Loading sharded checkpoint from {fsdp_checkpoint_path} ...")
        model = load_sharded_model_single_gpu(model_def, fsdp_checkpoint_path)
    print("Model is loaded from FSDP sharded checkpoints")

    return model


def convert_checkpoint(fsdp_checkpoint_path: str, consolidated_model_path: str, merge_layers: bool = False, merge_back: bool = True, merge_from: str = ""):
    """
    Convert the FSDP sharded model checkpoints to HuggingFace format.
    """

    # load the model from FSDP sharded checkpoint path
    print(f"Loading model from FSDP sharded checkpoints {fsdp_checkpoint_path}")
    model = load_checkpoint(fsdp_checkpoint_path)

    # load and save the tokenizer from the model_path to the consolidated_model_path
    print(f"Loading tokenizer from {fsdp_checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(fsdp_checkpoint_path)
    tokenizer.save_pretrained(consolidated_model_path)
    print(f"Tokenizer has been saved in {consolidated_model_path}")

    # merge layers with concat operation
    if merge_layers:
        if merge_from:
            print(f"Merging layers of {fsdp_checkpoint_path} with concat operation from another checkpoint {merge_from}") 
            # load the model from merge_from
            print(f"Loading model from config {merge_from}")
            model_merge_from = load_checkpoint(merge_from)
            if model.config.expansion["expand_type"] == "concat":
                ModelExpander.merge_models_concat(model_a=model, model_b=model_merge_from, merge_back=merge_back)
            else:
                ModelExpander.merge_models(model_a=model, model_b=model_merge_from, interp_weight=0.5)
            model.config.merge_from = merge_from
        else:
            assert merge_back, "merge_back should be True when merge_layers is True and merge_from is not provided"
            print(f"Merging layers of {fsdp_checkpoint_path} with concat operation back to pretrain_layers")
            ModelExpander.merge_layers(model=model, model_path_or_name=fsdp_checkpoint_path)
            model.config.merge_from = fsdp_checkpoint_path

    model = model.to(torch.bfloat16)  # Convert model parameters to bfloat16
    # save the FSDP sharded checkpoints in HF format
    model.save_pretrained(consolidated_model_path)
    print(f"HuggingFace model checkpoints has been saved in {consolidated_model_path}")

    # Check if evaluation_results.json exists and copy it over to the new model path, this is for evaluation metrics based on the eval dataset
    if (Path(fsdp_checkpoint_path) / "evaluation_results.json").exists():
        print(f"evaluation_results.json found at {fsdp_checkpoint_path}, copying it to {consolidated_model_path}")
        shutil.copy(Path(fsdp_checkpoint_path) / "evaluation_results.json", Path(consolidated_model_path) / "evaluation_results.json")

    # Check if train_params.yaml exists and copy it over to the new model path
    if (Path(fsdp_checkpoint_path).parent / "train_params.yaml").exists():
        print(f"train_params.yaml found at {Path(fsdp_checkpoint_path).parent}, copying it to {consolidated_model_path}")
        shutil.copy(Path(fsdp_checkpoint_path).parent / "train_params.yaml", Path(consolidated_model_path) / "train_params.yaml")

    # # Check if runs folder exists under the parent directory and copy it over to the new model path, this is for auto eval metrics
    # if (Path(fsdp_checkpoint_path).parent / "runs").exists():
    #     print(f"runs folder found at {Path(fsdp_checkpoint_path).parent}, copying it to {consolidated_model_path}")
    #     shutil.copytree(Path(fsdp_checkpoint_path).parent / "runs", Path(consolidated_model_path) / "runs", dirs_exist_ok=True)

    # Check if benchmark_results.json exists and copy it over to the new model path, this is for llm benchmark metrics
    if (Path(fsdp_checkpoint_path).parent / "benchmark_results.json").exists():
        print(f"benchmark_results.json found at {Path(fsdp_checkpoint_path).parent}, copying it to {consolidated_model_path}")
        shutil.copy(Path(fsdp_checkpoint_path).parent / "benchmark_results.json", Path(consolidated_model_path) / "benchmark_results.json")


def handle_model_path_in_os_env(args):
    """
    Handle the CHECKPOINT_PATH in os.environ to support CHECKPOINT_PATH=<model checkpoint path> accelerate launch ./src/controlllm/inference/batch_eval.py
    """
    # split os.environ["CHECKPOINT_PATH"] into the output_dir and resume checkpoint folder
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "")
    if checkpoint_path:
        print(f"CHECKPOINT_PATH in os.environ detected and takes priority, checkpoint_path: {checkpoint_path}")
        # if checkpoint_path is ending with "checkpoint-<global_step>" <global_step> is a number
        if re.match(r".*checkpoint-\d+$", checkpoint_path):
            output_dir = checkpoint_path.rsplit("checkpoint-", 1)[0]
            output_folder = f"checkpoint-{checkpoint_path.rsplit('checkpoint-', 1)[1]}"
            if Path(output_dir) / output_folder != Path(checkpoint_path):  # not in format of <output_dir>/checkpoint-<global_step>
                args.fsdp_checkpoint_path = checkpoint_path
                print(f"CHECKPOINT_PATH in os.environ detected, converting from checkpoint_path: {args.fsdp_checkpoint_path}")
                if args.consolidated_model_path == "":
                    args.consolidated_model_path = output_dir + ("-hf-merged-" if args.merge_layers else "-hf-") + output_folder  # don't use Path and / here as it is output_dir / resume_checkpoint_folder
                print(f"CHECKPOINT_PATH in os.environ detected, converting to consolidated_model_path: {args.consolidated_model_path}")
            else:
                args.fsdp_checkpoint_path = checkpoint_path
                print(f"CHECKPOINT_PATH in os.environ detected, converting from checkpoint_path: {args.fsdp_checkpoint_path}")
                if args.consolidated_model_path == "":
                    args.consolidated_model_path = Path(output_dir) / ("hf-merged" if args.merge_layers else "hf") / output_folder
                print(f"CHECKPOINT_PATH in os.environ detected, converting to consolidated_model_path: {args.consolidated_model_path}")
        else:
            all_checkpoints = get_all_checkpoints(checkpoint_path)
            args.fsdp_checkpoint_path = []
            args.consolidated_model_path = []
            for checkpoint in all_checkpoints:
                args.fsdp_checkpoint_path.append(Path(checkpoint_path) / checkpoint)
                print(f"CHECKPOINT_PATH in os.environ detected, converting from checkpoint_path: {args.fsdp_checkpoint_path}")
                output_dir = checkpoint_path
                output_folder = checkpoint
                consolidated_model_path = Path(output_dir) / ("hf-merged" if args.merge_layers else "hf") / output_folder
                args.consolidated_model_path.append(consolidated_model_path)
                print(f"CHECKPOINT_PATH in os.environ detected, converting to consolidated_model_path: {consolidated_model_path}")
    else:
        print("CHECKPOINT_PATH not found in os.environ, please run with CHECKPOINT_PATH=<model checkpoint path> accelerate launch ./src/controlllm/inference/batch_eval.py")

    return args


def get_all_checkpoints(folder) -> List[str]:
    """
    Get all the checkpoints in the folder.
    """
    PREFIX_CHECKPOINT_DIR = "checkpoint"
    _re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")

    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]), reverse=True)

    return checkpoints


if __name__ == "__main__":
    main()
