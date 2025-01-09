import logging
import datasets
from pathlib import Path

from controlllm.data.utils import tokenize_dialog, sample_dataset


def get_dataset(dataset_config, tokenizer, split):
    # note that the dataset name is different from the one in the original code, to enable loading from cache, Magpie-Pro-MT-300K-v0.1->magpie-pro-mt-300_k-v0.1
    dataset = datasets.load_dataset(dataset_config.dataset, name="default", split=split, cache_dir=dataset_config.hf_hub_dataset_cache_dir)

    logging.info(f"Sampling the dataset: {dataset_config.dataset} - {split}")
    dataset = sample_dataset(dataset, dataset_config, split)
    logging.info(f"Finished sampling the dataset: {dataset_config.dataset} - {split}")

    per_dp_cache_folder = Path(split).name
    dataset_map_cache_dir = Path(dataset_config.dataset_map_cache_dir) / per_dp_cache_folder
    # create the folder is it doesn't exist
    if not dataset_map_cache_dir.exists():
        dataset_map_cache_dir.resolve().mkdir(parents=True, exist_ok=True)

    def to_dialog(conversations):
        dialog = []
        for i, conversation in enumerate(conversations):
            dialog.append({
                "role": "user" if conversation["from"] == "human" else "assistant",
                "content": conversation["value"],
            })
        return {"dialog": dialog}

    logging.info(f"Converting the dataset to dialog format: {dataset_config.dataset}")
    dataset = dataset.map(lambda x: to_dialog(x[dataset_config.prompt_columns]), remove_columns=list(dataset.features), num_proc=dataset_config.max_workers)
    # take the name of the tokenizer as the cache folder, if it is a path, take the last part, if it is a model name, as is
    per_tokz_cache_folder = Path(tokenizer.name_or_path).name if Path(tokenizer.name_or_path).exists() else tokenizer.name_or_path
    per_tokz_cache_dir = dataset_map_cache_dir / per_tokz_cache_folder
    logging.info(f"Transforming the dataset with tokenization by {per_tokz_cache_folder}: {dataset_config.dataset}")
    # create the folder is it doesn't exist
    if not per_tokz_cache_dir.exists():
        per_tokz_cache_dir.resolve().mkdir(parents=True, exist_ok=True)
    dataset = dataset.map(lambda x: tokenize_dialog(x["dialog"], tokenizer, dataset_config), remove_columns=list(dataset.features), num_proc=dataset_config.max_workers)

    return dataset
