import random
import logging
import itertools
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, load_from_disk, concatenate_datasets
from transformers import PreTrainedTokenizer

from controlllm.configs.datasets import AbstractDataset as DatasetConfig
from controlllm.data.utils import load_files_as_dataset, sample_dataset


class SemanticSearchCosent(Dataset):
    # list of important attributes: hf_dataset (HFDataset), tokenizer (PreTrainedTokenizer), dataset_config (DatasetConfig)
    def __init__(
        self,
        dataset_config: DatasetConfig,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
    ):
        self.dataset_config = dataset_config
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.print_text = dataset_config.print_text
        self.data_path = data_path
        # take the split of dataset as the cache folder
        per_dp_cache_folder = Path(self.data_path).name
        self.dataset_map_cache_dir = Path(self.dataset_config.dataset_map_cache_dir) / per_dp_cache_folder
        self.dataset_raw_cache_dir = Path(self.dataset_config.avro_cache_dir) / per_dp_cache_folder

        # create the folder is it doesn't exist
        def ensure_dir_exists(dir_path):
            dir_path.resolve().mkdir(parents=True, exist_ok=True)

        ensure_dir_exists(self.dataset_map_cache_dir)
        ensure_dir_exists(self.dataset_raw_cache_dir)
        try:
            logging.info(f"Loading data from path: {data_path}")
            # the dataset is not big enough to be necessary to load with streaming
            self.hf_dataset: HFDataset = load_files_as_dataset(
                data_dir=self.data_path, cache_dir=self.dataset_raw_cache_dir, max_workers=self.dataset_config.max_workers)
            # check if the dataset is loaded
            if self.hf_dataset is None:
                raise ValueError(f"No data loaded from path: {self.data_path}")
            logging.info(f"Finished loading data from path: {self.data_path}")
            if self.dataset_config.sample_before_preprocessing:
                logging.info(f"Sampling the dataset: {self.dataset_config.dataset} - {self.data_path}")
                self.hf_dataset = sample_dataset(self.hf_dataset, self.dataset_config, data_path)
                logging.info(f"Finished sampling the dataset: {self.dataset_config.dataset} - {self.data_path}")
            else:
                logging.info(f"Sample before preprocessing is turned off for the dataset: {self.dataset_config.dataset} - {self.data_path}")
            logging.info(f"Converting the dataset to features: {self.dataset_config.dataset}")
            self.convert_to_features()
            logging.info(f"Finished converting the dataset to features: {self.dataset_config.dataset}")
        except Exception as e:
            logging.exception(f"Loading of dataset failed!: {e}")
            raise

    def __len__(self):
        return self.hf_dataset.shape[0]

    # TODO: Add text cleaning by hf_dataset transform(e.g. mapping)
    def _apply_prompt_template(self, batch, K=0):
        """
        Map function that processes one batch of size 1 at a time.

        - If K=0, set permute=False, returns a SINGLE row (dictionary of length-1 lists).
        - If K>0 or K=-1, set permute=True, returns MULTIPLE rows (dictionary of length-K lists),
        one for each permutation or random sample of permutations.

        This is done so that Hugging Face expands those lists into multiple rows.
        """
        permute = K != 0  # if K=0, don't permute

        # Each field in `batch` is a list of length == batch_size (which we'll set to 1).
        # So we extract the single sample's value. For example:
        sample = {}
        for key, values in batch.items():
            sample[key] = values[0]  # take the first (and only) item

        # Data setup
        group_id = str(sample[self.dataset_config.group_column])
        relevance_score = float(sample[self.dataset_config.relevance_column])

        doc_cols = self.dataset_config.response_column
        query_cols = self.dataset_config.prompt_columns

        # We'll accumulate the final prompts/responses in these lists
        prompts = []
        responses = []
        relevance_scores = []
        group_ids = []

        # -----------------------------------------
        # CASE 1: No permutations => exactly 1 output row
        # -----------------------------------------
        if not permute:
            input_text = "".join(
                f"# {col}: {sample[col] if col in sample else 'N/A'}\n"
                for col in doc_cols
            )
            response_ = f'{input_text}'
            # prompt_ = f'Query: {sample[query_cols]}'  # query is tagged with prefix
            query_col = query_cols[0]  # else query_cols means if it is configured as a string

            prompt_ = f'Query: {sample[query_col]}'  # query is tagged with prefix

            # Append to our lists (one item)
            prompts.append(prompt_)
            responses.append(response_)
            relevance_scores.append(relevance_score)
            group_ids.append(group_id)

            # Return a dict of lists, each list has length=1
            return {"group_id": group_ids, "prompt": prompts, "response": responses, "relevance_score": relevance_scores}

        # -----------------------------------------
        # CASE 2: Permutations => multiple output rows
        # -----------------------------------------
        # If K != -1, pick K random permutations. Otherwise, use all permutations.
        all_perms = list(itertools.permutations(doc_cols))
        if K != -1 and K < len(all_perms):
            chosen_perms = random.sample(all_perms, K)
        else:
            chosen_perms = all_perms

        # Build one row for each permutation
        for permutation in chosen_perms:
            input_text = "".join(
                f"# {col}: {sample[col] if col in sample else 'N/A'}\n"
                for col in permutation
            )
            response_ = f'{input_text}'
            # prompt_ = f'Query: {sample[query_cols]}'  # query is tagged with prefix
            query_col = query_cols[0]  # else query_cols means if it is configured as a string

            prompt_ = f'Query: {sample[query_col]}'  # query is tagged with prefix

            prompts.append(prompt_)
            responses.append(response_)
            relevance_scores.append(relevance_score)
            group_ids.append(group_id)

        # Return a dict of lists (length=K). HF will create K rows automatically.
        return {"group_id": group_ids, "prompt": prompts, "response": responses, "relevance_score": relevance_scores}


    def _tokenize_add_label(self, sample, add_special_tokens=True):
        # don't do padding here, it will be done in the dataloader. don't need to do truncation if packing is true
        prompt_input_ids = self.tokenizer.encode(sample["prompt"],
                                            truncation=self.dataset_config.truncation,
                                            max_length=self.dataset_config.max_length,
                                            add_special_tokens=False)

        chosen_input_ids = self.tokenizer.encode(sample["response"],
                                            truncation=self.dataset_config.truncation,
                                            max_length=self.dataset_config.max_length,
                                            add_special_tokens=False)

        # Add special tokens if needed
        prompt_attention_mask = [1] * len(prompt_input_ids)
        chosen_attention_mask = [1] * len(chosen_input_ids)
        if add_special_tokens:
            if self.tokenizer.bos_token_id is not None:
                prompt_input_ids = [self.tokenizer.bos_token_id] + prompt_input_ids
                # According to the llama paper, better to mask the special token for contrastive learning. TODO: check by experiment
                prompt_attention_mask = [1] + prompt_attention_mask
                chosen_input_ids = [self.tokenizer.bos_token_id] + chosen_input_ids
                chosen_attention_mask = [1] + chosen_attention_mask
            if self.tokenizer.eos_token_id is not None:
                prompt_input_ids = prompt_input_ids + [self.tokenizer.eos_token_id]
                prompt_attention_mask = prompt_attention_mask + [1]
                chosen_input_ids = chosen_input_ids + [self.tokenizer.eos_token_id]
                chosen_attention_mask = chosen_attention_mask + [1]

        sample = {
            "prompt_input_ids": prompt_input_ids,  # follow sentence-transformers's convention
            "chosen_input_ids": chosen_input_ids,  # follow sentence-transformers's convention
            "prompt_attention_mask": prompt_attention_mask,  # follow sentence-transformers's convention
            "chosen_attention_mask": chosen_attention_mask,  # follow sentence-transformers's convention
            "label": sample["relevance_score"],  # follow sentence-transformers's convention
            "group_id": sample["group_id"],
        }
        return sample

    @classmethod
    def _precompute_response_embeddings(cls, batch, tokenizer, model):
        """
        Precompute embeddings for the response pairs in 'batch'.
        With sharded data, each rank processes its own batch.
        """
        # Unpack and pad sequences
        chosen_input_ids = batch["chosen_input_ids"]
        chosen_attention_mask = batch["chosen_attention_mask"]

        max_len_chosen = max(len(ids) for ids in chosen_input_ids)
        chosen_input_ids = [
            [tokenizer.pad_token_id] * (max_len_chosen - len(ids)) + ids
            for ids in chosen_input_ids
        ]
        chosen_attention_mask = [
            [0] * (max_len_chosen - len(mask)) + mask
            for mask in chosen_attention_mask
        ]

        # Convert to tensors on model.device
        chosen_input_ids = torch.tensor(chosen_input_ids, dtype=torch.long, device=model.device)
        chosen_attention_mask = torch.tensor(chosen_attention_mask, dtype=torch.long, device=model.device)

        model.eval()
        with torch.no_grad():
            chosen_features = {"input_ids": chosen_input_ids, "attention_mask": chosen_attention_mask}
            chosen_output = model(chosen_features)

        batch["chosen_embedding"] = [emb.cpu().tolist() for emb in chosen_output["sentence_embedding"]]
        return batch

    @classmethod
    def post_process(cls, hf_dataset: HFDataset, dataset_config, tokenizer, model, data_path, rank_0_only=True) -> HFDataset:
        """
        Post-processes the dataset by computing embeddings for the response pairs.
        Args:
            hf_dataset: The dataset to process.
            dataset_config: The dataset configuration.
            tokenizer: The tokenizer.
            model: The model.
            data_path: The path to the data.
            rank_0_only: If True, only rank 0 will return the processed dataset.
        """
        logging.info(f"Precompute the embeddings for the response pairs: {dataset_config.dataset}. Length={len(hf_dataset)}")

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Shard the dataset so that each process gets a unique subset.
        if dist.is_initialized():
            hf_dataset = hf_dataset.shard(num_shards=world_size, index=rank)

        per_dp_cache_folder = Path(data_path).name
        # Make a cache folder that is unique per model and per process.
        dataset_map_cache_dir = Path(dataset_config.dataset_map_cache_dir) / per_dp_cache_folder / model.config._name_or_path.replace("/", "__")
        dataset_map_cache_dir.resolve().mkdir(parents=True, exist_ok=True)
        cache_file_name = str(dataset_map_cache_dir / f"precompute_embedding_rank{rank}.arrow")

        # Adjust batch size: now each process only processes its shard.
        per_device_batch_size = dataset_config.post_process_per_device_batch_size
        batch_size_for_map = per_device_batch_size

        processed_dataset = hf_dataset.map(
            function=cls._precompute_response_embeddings,
            fn_kwargs={"tokenizer": tokenizer, "model": model},
            batched=True,
            batch_size=batch_size_for_map,
            num_proc=1,
            cache_file_name=cache_file_name,
            load_from_cache_file=False if dataset_config.force_refresh else True,
        )

        if world_size == 1:
            return processed_dataset

        # Save the processed shard to disk.
        shard_dir = dataset_map_cache_dir / f"shard_{rank}"
        processed_dataset.save_to_disk(str(shard_dir))
        # Wait for all processes to finish writing their shards.
        if dist.is_initialized():
            dist.barrier()

        if rank_0_only and rank != 0:
            return None
        else:
            shards = []
            for r in range(world_size):
                shard_path = dataset_map_cache_dir / f"shard_{r}"
                shard_ds = load_from_disk(str(shard_path))
                shards.append(shard_ds)
            merged_dataset = concatenate_datasets(shards)

        return merged_dataset

    def convert_to_features(self):
        logging.info(f"Transforming the dataset with prompt template: {self.dataset_config.dataset}. Length: {len(self.hf_dataset)}")
        self.hf_dataset = self.hf_dataset.map(lambda batch: self._apply_prompt_template(batch, K=self.dataset_config.K_doc),
                                              batched=True,
                                              batch_size=1,  # Process one sample at a time
                                              num_proc=self.dataset_config.max_workers,
                                              cache_file_name=str(self.dataset_map_cache_dir / "apply_prompt_template.arrow"),
                                              load_from_cache_file=False if self.dataset_config.force_refresh else True,
                                              remove_columns=list(self.hf_dataset.features.keys())
                                              )

        # take the name of the tokenizer as the cache folder, if it is a path, take the last part, if it is a model name, as is
        per_tokz_cache_folder = Path(self.tokenizer.name_or_path).name if Path(self.tokenizer.name_or_path).exists() else self.tokenizer.name_or_path
        per_tokz_cache_dir = self.dataset_map_cache_dir / per_tokz_cache_folder
        logging.info(f"Transforming the dataset with tokenization {per_tokz_cache_folder}: {self.dataset_config.dataset}. Length: {len(self.hf_dataset)}")
        # create the folder is it doesn't exist
        if not per_tokz_cache_dir.exists():
            per_tokz_cache_dir.resolve().mkdir(parents=True, exist_ok=True)
        self.hf_dataset = self.hf_dataset.map(lambda sample: self._tokenize_add_label(sample, add_special_tokens=self.dataset_config.add_special_tokens),
                                              num_proc=self.dataset_config.max_workers,
                                              cache_file_name=str(per_tokz_cache_dir / "tokenize_add_label.arrow"),
                                              load_from_cache_file=False if self.dataset_config.force_refresh else True,
                                              remove_columns=list(self.hf_dataset.features.keys())
                                              )

    def __getitem__(self, index):
        return self.hf_dataset[int(index)]


# added this to allow testing with mock data
# !!! don't forget to map the dataset to its get function such as this one in DATASET_PREPROC of /data/__init__.py !!!
def get_dataset(
    dataset_config: DatasetConfig, tokenizer: PreTrainedTokenizer, data_path: Optional[str] = None
) -> Dataset:
    """cover function for handling loading the working dataset"""
    """dataset loading"""
    if data_path is None or dataset_config.local_test:
        # this is for local testing
        currPath = Path.cwd() / "controlllm" / "data" / "mock_data"
        logging.info(f"Loading dataset {currPath}")
        data_path = str(currPath)
    dataset = SemanticSearchCosent(
        dataset_config,
        tokenizer=tokenizer,
        data_path=data_path
    )

    return dataset.hf_dataset


def postprocess_dataset(hf_dataset, dataset_config, tokenizer, model, data_path) -> HFDataset:
    """
    Post-processes the dataset by computing embeddings for the response pairs.
    """
    return SemanticSearchCosent.post_process(hf_dataset, dataset_config, tokenizer, model, data_path)
