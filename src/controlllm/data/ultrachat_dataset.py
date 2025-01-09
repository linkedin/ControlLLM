import logging
import datasets
from pathlib import Path

from controlllm.data.utils import tokenize_dialog, sample_dataset


def get_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset(dataset_config.dataset, name="default", split=split, cache_dir=dataset_config.hf_hub_dataset_cache_dir)

    logging.info(f"Sampling the dataset: {dataset_config.dataset} - {split}")
    dataset = sample_dataset(dataset, dataset_config, split)
    logging.info(f"Finished sampling the dataset: {dataset_config.dataset} - {split}")

    per_dp_cache_folder = Path(split).name
    dataset_map_cache_dir = Path(dataset_config.dataset_map_cache_dir) / per_dp_cache_folder
    # create the folder is it doesn't exist
    if not dataset_map_cache_dir.exists():
        dataset_map_cache_dir.resolve().mkdir(parents=True, exist_ok=True)

    # tokenize the dialog
    def tokenize_add_label(sample):
        return tokenize_dialog(dialog=sample[dataset_config.prompt_columns], tokenizer=tokenizer, dataset_config=dataset_config)

    # take the name of the tokenizer as the cache folder, if it is a path, take the last part, if it is a model name, as is
    per_tokz_cache_folder = Path(tokenizer.name_or_path).name if Path(tokenizer.name_or_path).exists() else tokenizer.name_or_path
    per_tokz_cache_dir = dataset_map_cache_dir / per_tokz_cache_folder
    logging.info(f"Transforming the dataset with tokenization by {per_tokz_cache_folder}: {dataset_config.dataset}")
    # create the folder is it doesn't exist
    if not per_tokz_cache_dir.exists():
        per_tokz_cache_dir.resolve().mkdir(parents=True, exist_ok=True)
    dataset = dataset.map(tokenize_add_label,
                          num_proc=dataset_config.max_workers,
                          remove_columns=list(dataset.features),
                          cache_file_name=str(per_tokz_cache_dir / "tokenize_add_label.arrow"),
                          load_from_cache_file=False if dataset_config.force_refresh else True)

    return dataset
