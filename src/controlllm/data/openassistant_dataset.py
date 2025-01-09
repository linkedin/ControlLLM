# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import copy
import logging
import datasets
from pathlib import Path

from controlllm.data.utils import tokenize_dialog, sample_dataset


def get_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("OpenAssistant/oasst1", name="default-40df6ffd164a3969", split=split, cache_dir=dataset_config.hf_hub_dataset_cache_dir)

    dataset = dataset.map(lambda sample: {
        "message_id": sample["message_id"],
        "parent_id": sample["parent_id"],
        "text": sample["text"],
        },
        batched=True,
        remove_columns=list(dataset.features),)

    nodes = {}

    messages = {}
    root_ids = []

    for data in dataset:
        if data["parent_id"]:
            nodes[data["parent_id"]] = nodes.get(data["parent_id"], []) + [data["message_id"]]
        else:
            root_ids.append(data["message_id"])
        messages[data["message_id"]]=data["text"]

    def follow(thread, current_id):
        thread = copy.copy(thread) + [messages[current_id]]
        if current_id in nodes:
            new_threads = []
            for next_id in nodes[current_id]:
                new_threads += follow(thread, next_id)
            return new_threads
        else:
            return [thread]

    def get_threads_from_root(root_id):
        all_threads = []
        thread = [messages[root_id]]
        for cid in nodes[root_id]:
            all_threads += follow(thread, cid)
        return all_threads

    logging.info(f"Filtering and mapping thread with message_id: {dataset_config.dataset}")
    dataset = dataset.filter(lambda x: x["message_id"] in root_ids)
    dataset = dataset.map(lambda x: {"thread": get_threads_from_root(x["message_id"])}, remove_columns=list(dataset.features))
    dataset = dataset.map(lambda x: {"thread": [i for row in x["thread"] for i in row]}, batched=True)

    logging.info(f"Sampling the dataset: {dataset_config.dataset} - {split}")
    dataset = sample_dataset(dataset, dataset_config, split)
    logging.info(f"Finished sampling the dataset: {dataset_config.dataset} - {split}")

    per_dp_cache_folder = Path(split).name
    dataset_map_cache_dir = Path(dataset_config.dataset_map_cache_dir) / per_dp_cache_folder
    # create the folder is it doesn't exist
    if not dataset_map_cache_dir.exists():
        dataset_map_cache_dir.resolve().mkdir(parents=True, exist_ok=True)

    def to_dialog(thread):
        dialog = []
        for i, content in enumerate(thread):
            dialog.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": content,
            })
        return {"dialog": dialog}

    logging.info(f"Converting the dataset to dialog format: {dataset_config.dataset}")
    dataset = dataset.map(lambda x: to_dialog(x["thread"]), remove_columns=list(dataset.features), num_proc=dataset_config.max_workers)
    # take the name of the tokenizer as the cache folder, if it is a path, take the last part, if it is a model name, as is
    per_tokz_cache_folder = Path(tokenizer.name_or_path).name if Path(tokenizer.name_or_path).exists() else tokenizer.name_or_path
    per_tokz_cache_dir = dataset_map_cache_dir / per_tokz_cache_folder
    logging.info(f"Transforming the dataset with tokenization by {per_tokz_cache_folder}: {dataset_config.dataset}")
    # create the folder is it doesn't exist
    if not per_tokz_cache_dir.exists():
        per_tokz_cache_dir.resolve().mkdir(parents=True, exist_ok=True)
    dataset = dataset.map(lambda x: tokenize_dialog(x["dialog"], tokenizer, dataset_config), remove_columns=list(dataset.features), num_proc=dataset_config.max_workers)

    return dataset
