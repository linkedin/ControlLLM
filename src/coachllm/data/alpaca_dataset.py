import os
import logging
import datasets
from pathlib import Path

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

    def to_dialog(system_prompt, user_prompt, response, history=None):
        dialog = []

        if history is not None:
            for i, h in enumerate(history):
                # h must be a list of two str, first one is user prompt, second one is response
                if len(h) != 2 or not all(isinstance(item, str) and item.strip() for item in h):
                    continue
                dialog.append({
                        "role": "user",
                        "content": h[0],
                })
                dialog.append({
                        "role": "assistant",
                        "content": h[1],
                })

        if user_prompt.strip() != "":  # prompt with input
            dialog.append({
                    "role": "system",
                    "content": system_prompt,
            })
            dialog.append({
                    "role": "user",
                    "content": user_prompt,
            })
        else:  # prompt without input
            dialog.append({
                    "role": "user",
                    "content": system_prompt,
            })
        dialog.append({
                "role": "assistant",
                "content": response,
        })        
        return {"dialog": dialog}

    logging.info(f"Converting the dataset to dialog format: {dataset_config.dataset}")
    dataset = dataset.map(lambda x: to_dialog(x[dataset_config.system_column], x[dataset_config.prompt_columns], x[dataset_config.response_column], x[dataset_config.history_column] if hasattr(dataset_config, 'history_column') else None), remove_columns=list(dataset.features), num_proc=dataset_config.max_workers)
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
