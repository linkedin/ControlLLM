import os
import logging
import datasets
from pathlib import Path
from functools import partial

from controlllm.data.utils import tokenize_dialog, sample_dataset, load_arrow_dirs_as_dataset


def get_dataset(dataset_config, tokenizer, split):
    # Check if split is a shared nfs file path(e.g. /shared/public/data/controlllm/datasets/...) or not
    if os.path.isdir(split):
        dataset = load_arrow_dirs_as_dataset(arrow_dir=split, arrow_dataset_cache_dir=dataset_config.dataset_map_cache_dir, dataset_config=dataset_config)
    else:
        # note that the dataset name is different from the one in the original code, to enable loading from cache
        dataset = datasets.load_dataset(dataset_config.dataset, split=split, cache_dir=dataset_config.hf_hub_dataset_cache_dir)

    logging.info(f"Sampling the dataset: {dataset_config.dataset} - {split}")
    dataset = sample_dataset(dataset, dataset_config, split)
    logging.info(f"Finished sampling the dataset: {dataset_config.dataset} - {split}")

    per_dp_cache_folder = Path(split).name
    dataset_map_cache_dir = Path(dataset_config.dataset_map_cache_dir) / per_dp_cache_folder
    # create the folder is it doesn't exist
    if not dataset_map_cache_dir.exists():
        dataset_map_cache_dir.resolve().mkdir(parents=True, exist_ok=True)

    # filter out 564 questions (roughly 0.1%) were longer than 1024 Llama tokens, refer to: https://huggingface.co/datasets/nvidia/OpenMathInstruct-2
    # ++max_problem_length=1024 \
    # ++filters.remove_len_outlier_solutions=true \
    # ++use_chars_for_min_length=true \
    # ++min_solution_length=200 \
    # ++hf_model_name="meta-llama/Meta-Llama-3.1-8B" \
    # ++max_solution_length=1024 \
    logging.info(f"Filtering the dataset: {dataset_config.dataset}. Maximum problem length: 1024 tokens, Minimum solution length: 200 characters, Maximum solution length: 1024 tokens.")
    encode = partial(tokenizer.encode, add_special_tokens=False, truncation=dataset_config.truncation, max_length=dataset_config.max_length)
    dataset = dataset.filter(lambda x: len(encode(x[dataset_config.prompt_columns])) <= 1024 and len(x[dataset_config.response_column]) >= 200 and len(encode(x[dataset_config.response_column])) <= 1024, num_proc=dataset_config.max_workers)

    def to_dialog(user_prompt, response, problem_source):
        user_prompt = apply_prompt_training(user_prompt, problem_source)

        dialog = []
        dialog.append({
                "role": "user",
                "content": user_prompt,
        })
        dialog.append({
                "role": "assistant",
                "content": response,
        })
        return {"dialog": dialog}

    logging.info(f"Converting the dataset to dialog format: {dataset_config.dataset}")
    dataset = dataset.map(lambda x: to_dialog(x[dataset_config.prompt_columns], x[dataset_config.response_column], x[dataset_config.problem_source_column]), remove_columns=list(dataset.features), num_proc=dataset_config.max_workers)
    # take the name of the tokenizer as the cache folder, if it is a path, take the last part, if it is a model name, as is
    per_tokz_cache_folder = Path(tokenizer.name_or_path).name if Path(tokenizer.name_or_path).exists() else tokenizer.name_or_path
    per_tokz_cache_dir = dataset_map_cache_dir / per_tokz_cache_folder
    logging.info(f"Transforming the dataset with tokenization by {per_tokz_cache_folder}: {dataset_config.dataset}")
    # create the folder is it doesn't exist
    if not per_tokz_cache_dir.exists():
        per_tokz_cache_dir.resolve().mkdir(parents=True, exist_ok=True)
    dataset = dataset.map(lambda x: tokenize_dialog(x["dialog"],tokenizer, dataset_config),
                          remove_columns=list(dataset.features),
                          num_proc=dataset_config.max_workers,
                          cache_file_name=str(per_tokz_cache_dir / "tokenize_add_label.arrow"),
                          load_from_cache_file=False if dataset_config.force_refresh else True)

    return dataset


def apply_prompt_training(user_prompt, problem_source):
    """
    Apply the prompt template to the user prompt for training. Note that problem_source is not used here in this prompt template for training.
    TODO: find the best prompt template that may need problem_source for training the best model.
    """
    # Solve the following math problem. Make sure to put the answer (and only answer) inside \boxed{{}}.

    # {examples}{problem}
    return f"Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{{}}.\n\n{user_prompt}"


def apply_prompt(user_prompt, problem_source):
    """
    Apply the prompt template to the user prompt for inference.
    """
    if "gsm8k" in problem_source:
        #Given the following problem, reason and give a final answer to the problem.
        #Problem: {user_prompt}
        #Your response should end with "The final answer is \boxed{{[answer]}}" where [answer] is the response to the problem.
        return f"Given the following problem, reason and give a final answer to the problem.\nProblem: {user_prompt}\nYour response should end with \"The final answer is \\boxed{{[answer]}}\" where [answer] is the response to the problem."
    elif "math" in problem_source:
        #   Solve the following math problem efficiently and clearly:
        #   - For simple problems (2 steps or fewer):
        #   Provide a concise solution with minimal explanation.
        #   - For complex problems (3 steps or more):
        #   Use this step-by-step format:
        #   ## Step 1: [Concise description]
        #   [Brief explanation and calculations]
        #   ## Step 2: [Concise description]
        #   [Brief explanation and calculations]
        #   ...
        #   Regardless of the approach, always conclude with:
        #   Therefore, the final answer is: $\boxed{{answer}}$. I hope it is correct.
        #   Where [answer] is just the final number or expression that solves the problem.
        #   Problem: {problem}
        return f"Solve the following math problem efficiently and clearly:\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n...\nRegardless of the approach, always conclude with:\nTherefore, the final answer is: $\\boxed{{answer}}$. I hope it is correct.\nWhere [answer] is just the final number or expression that solves the problem.\nProblem: {user_prompt}"
    else:
        raise ValueError(f"Unknown problem source: {problem_source}. It should be either '*gsm8k' or '*math'.")
